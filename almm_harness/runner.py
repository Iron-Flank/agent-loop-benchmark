"""Chronological fixture execution and crash-safe, unscored run artifacts."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import threading
import time

from almm_adapter.contract import validate_manifest, validate_response
from almm_fixture.engine import canonical_json
from almm_fixture.validation import SchemaValidator

from .errors import AdapterFailure, HarnessFailure, TimeoutFailure
from .isolation import validate_adapter_input


# Keep strong references: timed-out work must never share state with another run.
_TIMED_OUT = []
_STREAMS = ('events.jsonl', 'probes.jsonl', 'requests.jsonl', 'dead-letters.jsonl')
_CATEGORIES = ('budget', 'provider', 'adapter', 'timeout')


def _timestamp():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _hash(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _harness_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(path.name.encode())
        digest.update(b'\0')
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    try:
        with temporary.open('wb') as handle:
            handle.write(canonical_json(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _failure(error):
    return error if isinstance(error, HarnessFailure) else AdapterFailure(str(error))


def _request_identity(request):
    # Compare the complete native interaction, not just claimed tier text.
    identity = deepcopy({key: request[key] for key in (
        'requestId', 'messages', 'tools', 'toolCalls', 'toolResults',
        'tierSegments', 'model', 'decodingSettings', 'seed') if key in request})
    for segment in identity['tierSegments']:
        segment.pop('tokenCount', None)
    return identity


class ProbeRunner:
    """Bound a probe wall-clock call; timeout makes the adapter non-reusable.

    Python cannot safely kill an arbitrary adapter thread. On timeout the entire
    run stops, the proxy is aborted, and the adapter/proxy are permanently barred
    from further runs. Resume requires fresh instances and checkpoint replay.
    """

    def __init__(self, adapter, proxy, timeout):
        self.adapter = adapter
        self.proxy = proxy
        self.timeout = timeout

    def call(self, probe):
        outcomes = queue.Queue(maxsize=1)

        def invoke():
            try:
                outcomes.put((True, self.adapter.answerProbe(deepcopy(probe))))
            except BaseException as error:
                outcomes.put((False, error))

        worker = threading.Thread(target=invoke, name='almm-probe', daemon=True)
        worker.start()
        try:
            succeeded, value = outcomes.get(timeout=self.timeout)
        except queue.Empty:
            _TIMED_OUT.append((self.adapter, self.proxy))
            abort = getattr(self.proxy, 'abort_run', None)
            if abort is not None:
                abort()
            raise TimeoutFailure(f'probe exceeded {self.timeout:g} seconds') from None
        if not succeeded:
            raise value
        return value


class FixtureRunner:
    """Run an adapter against a full fixture without exposing scorer-only data.

    Final artifacts are immutable. Interrupted sessions are rolled back to byte
    offsets stored with the last fully completed session, then completed sessions
    (including probes) are replayed through a newly initialized adapter. Replay
    uses verified archived native responses, without provider calls or rescoring.
    """

    def __init__(self, manifest, adapter, proxy, artifact_dir):
        validate_manifest(manifest)
        self.manifest = deepcopy(manifest)
        self.adapter = adapter
        self.proxy = proxy
        self.artifact_root = Path(artifact_dir)
        timeout = self.manifest.get('probeTimeoutSeconds', 60)
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError('probeTimeoutSeconds must be positive and finite')
        concurrency = self.manifest.setdefault('concurrency', {'mode': 'solo', 'factor': 1})
        if (not isinstance(concurrency, dict) or concurrency.get('mode') not in {'solo', 'concurrent'}
                or type(concurrency.get('factor')) is not int or concurrency['factor'] < 1
                or (concurrency['mode'] == 'solo' and concurrency['factor'] != 1)):
            raise ValueError('concurrency requires solo/factor 1 or concurrent/positive integer factor')
        self.probes = ProbeRunner(adapter, proxy, timeout)
        self._handles = {}
        self._redact = proxy.redact

    def _append(self, name, value):
        self._handles[name].write(canonical_json(self._redact(deepcopy(value))))

    def _checkpoint(self, completed, last_session_id=None):
        offsets = {}
        for name, handle in self._handles.items():
            handle.flush()
            os.fsync(handle.fileno())
            offsets[name] = handle.tell()
        checkpoint = {'runId': self.manifest['runId'], 'identity': self.identity,
                      'lastCompletedSession': completed, 'lastSessionId': last_session_id,
                      'offsets': offsets, 'timestamp': _timestamp()}
        _atomic_json(self.directory / 'checkpoint.json', checkpoint)
        return checkpoint

    def _read_checkpoint(self, session_count):
        path = self.directory / 'checkpoint.json'
        try:
            checkpoint = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        if not isinstance(checkpoint, dict):
            return None
        if not isinstance(checkpoint.get('identity'), str):
            return None
        if checkpoint.get('identity') != self.identity:
            raise ValueError('checkpoint run identity does not match fixture/configuration')
        completed = checkpoint.get('lastCompletedSession')
        offsets = checkpoint.get('offsets')
        if (type(completed) is not int or not 0 <= completed <= session_count
                or not isinstance(offsets, dict)):
            return None
        expected_session = f's-{completed:04d}' if completed else None
        if checkpoint.get('lastSessionId') != expected_session:
            return None
        for name in _STREAMS:
            offset = offsets.get(name)
            path = self.directory / name
            if (type(offset) is not int or offset < 0 or not path.exists()
                    or path.stat().st_size < offset):
                return None
            if offset:
                with path.open('rb') as handle:
                    handle.seek(offset - 1)
                    if handle.read(1) != b'\n':
                        return None
        return checkpoint

    def _execute(self, kind, value, session_id, *, replay=False):
        started = time.monotonic()
        telemetry_start = len(self.proxy.telemetry)
        dead_letter_start = len(self.proxy.dead_letters)
        result = None
        error = None
        try:
            if kind == 'probe':
                result = self.probes.call(value)
            else:
                result = self.adapter.handleTurn(deepcopy(value))
            field = 'answer' if kind == 'probe' else 'response'
            validate_response(result, field)
            telemetry = deepcopy(self.proxy.telemetry[telemetry_start:])
            for request in telemetry:
                if request.get('status') != 'ok':
                    category = request.get('category', 'adapter')
                    failure = HarnessFailure(request.get('error', 'model request failed'))
                    failure.category = category if category in _CATEGORIES else 'adapter'
                    raise failure
            observed = [_request_identity(request) for request in telemetry]
            declared = [_request_identity(request) for request in result['requests']]
            if not observed or observed != declared:
                raise AdapterFailure('adapter requests must match requests actually sent through the proxy')
        except Exception as exception:
            error = _failure(exception)
            if error.category == 'timeout':
                if not any(self.adapter is adapter for adapter, _ in _TIMED_OUT):
                    _TIMED_OUT.append((self.adapter, self.proxy))
                self.proxy.abort_run()
        telemetry = deepcopy(self.proxy.telemetry[telemetry_start:])
        latency = (time.monotonic() - started) * 1000
        if replay:
            if error is not None:
                raise error
            return None
        identity_key = 'probeId' if kind == 'probe' else 'turnId'
        event = {'type': kind, 'sessionId': session_id, identity_key: value[identity_key],
                 'timestamp': _timestamp(), 'latencyMs': latency,
                 'status': 'error' if error else 'ok',
                 'requestIds': [request.get('requestId') for request in telemetry]}
        if error is not None:
            event.update(category=error.category, error=str(error))
        elif kind == 'turn':
            event['response'] = result['response']
        if kind == 'probe':
            sources = list(dict.fromkeys(
                source for request in telemetry
                for segment in (request.get('tierSegments') if isinstance(request.get('tierSegments'), list)
                                else [])
                if isinstance(segment, dict) and isinstance(segment.get('sourceIds', []), list)
                for source in segment.get('sourceIds', []) if isinstance(source, str)))
            probe = dict(event, answer=result.get('answer') if isinstance(result, dict) else None,
                         eligibleForAccuracy=error is None, requests=telemetry,
                         sourceIds=sources)
            self._append('probes.jsonl', probe)
        self._append('events.jsonl', event)
        for request in telemetry:
            self._append('requests.jsonl', dict(request, sessionId=session_id,
                                              **{identity_key: value[identity_key]}))
        for letter in self.proxy.dead_letters[dead_letter_start:]:
            self._append('dead-letters.jsonl', dict(letter, sessionId=session_id,
                                                  **{identity_key: value[identity_key]}))
        return error

    def _initialize(self):
        try:
            self.proxy.begin_run()
            # Adapter manifests are allowlisted as well; no fixture gold is passed.
            manifest = {key: deepcopy(self.manifest[key]) for key in
                        ('runId', 'adapter', 'model', 'tokenizer', 'stablePrefix',
                         'nativeModelEnvelopeVersion')}
            self.adapter.initialize(manifest)
        except Exception as error:
            raise _failure(error) from error

    def run(self, fixture, resume=False, adapter_input_path=None):
        if any(self.adapter is adapter or self.proxy is proxy for adapter, proxy in _TIMED_OUT):
            raise AdapterFailure('timed-out adapter/proxy cannot be reused; create fresh instances')
        SchemaValidator.validate(fixture)
        fixture_hash = _hash({key: value for key, value in fixture.items() if key != 'contentHash'})
        if 'contentHash' in fixture and fixture['contentHash'] != fixture_hash:
            raise ValueError('fixture contentHash does not match fixture content')
        view = validate_adapter_input(fixture, adapter_input_path)
        configuration = {key: deepcopy(self.manifest[key]) for key in (
            'runId', 'adapter', 'model', 'tokenizer', 'stablePrefix',
            'nativeModelEnvelopeVersion', 'seed', 'scorerVersion', 'concurrency')}
        configuration.update(harnessHash=_harness_hash(), fixtureHash=fixture_hash,
                             rateLimitRpm=self.manifest.get('rateLimitRpm', 60),
                             probeTimeoutSeconds=self.manifest.get('probeTimeoutSeconds', 60))
        self.identity = _hash(configuration)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        safe_run_id = hashlib.sha256(self.manifest['runId'].encode()).hexdigest()[:16]
        self.directory = self.artifact_root / f'{safe_run_id}-{self.identity}'
        if (self.directory / 'manifest.json').exists():
            raise FileExistsError('final run artifacts are immutable; choose a new runId')
        if resume and not self.directory.exists():
            # A different hash under the same runId is not permission to resume.
            for path in self.artifact_root.glob(f'{safe_run_id}-*/checkpoint.json'):
                try:
                    previous = json.loads(path.read_text())
                except (OSError, ValueError):
                    continue
                if previous.get('runId') == self.manifest['runId']:
                    raise ValueError('checkpoint run identity does not match fixture/configuration')
        self.directory.mkdir(exist_ok=True)
        checkpoint = self._read_checkpoint(len(view['sessions'])) if resume else None
        if not resume and any((self.directory / name).exists() for name in _STREAMS):
            raise FileExistsError('interrupted artifacts exist; use resume=True or a new runId')
        completed = checkpoint['lastCompletedSession'] if checkpoint else 0
        recovery = {'resumed': bool(resume), 'checkpointSession': completed,
                    'nextSession': completed + 1}
        if resume:
            try:
                recovery['interruption'] = json.loads(
                    (self.directory / 'interruption.json').read_text())
            except (OSError, ValueError):
                pass
        interrupted_sizes = {name: (self.directory / name).stat().st_size
                             for name in _STREAMS if (self.directory / name).exists()}
        try:
            for name in _STREAMS:
                path = self.directory / name
                handle = path.open('r+b') if path.exists() else path.open('w+b')
                offset = checkpoint['offsets'][name] if checkpoint else 0
                handle.truncate(offset)
                handle.seek(offset)
                self._handles[name] = handle
            if checkpoint is None:
                self._append('events.jsonl', {'type': 'run_started', 'timestamp': _timestamp(),
                                             'runId': self.manifest['runId'], 'identity': self.identity})
                self._checkpoint(0)
            if resume:
                self._append('events.jsonl', {'type': 'resume', 'timestamp': _timestamp(),
                                             'lastCompletedSession': completed,
                                             'nextSession': completed + 1,
                                             'discardedBytes': {name: size - (checkpoint['offsets'][name]
                                                                          if checkpoint else 0)
                                                                for name, size in interrupted_sizes.items()},
                                             'reconstruction': 'fresh-initialize-and-replay'})
            self._initialize()
            by_session = {}
            for probe in view['probes']:
                by_session.setdefault(probe['afterSessionId'], []).append(
                    {key: probe[key] for key in ('probeId', 'question')})
            if completed:
                try:
                    archived = [json.loads(line) for line in
                                (self.directory / 'requests.jsonl').read_text().splitlines()]
                    self.proxy.begin_replay(archived)
                    for session in view['sessions'][:completed]:
                        for turn in session['turns']:
                            self._execute('turn', dict(turn, sessionId=session['sessionId']),
                                          session['sessionId'], replay=True)
                        for probe in by_session.get(session['sessionId'], []):
                            self._execute('probe', probe, session['sessionId'], replay=True)
                    self.proxy.end_replay()
                    self.proxy.begin_run()
                except Exception as error:
                    if isinstance(error, TimeoutFailure):
                        raise
                    # An unverifiable transcript must never become adapter memory.
                    completed = 0
                    recovery.update(restarted=True, reason=str(error), nextSession=1)
                    for handle in self._handles.values():
                        handle.seek(0)
                        handle.truncate()
                    self._initialize()
                    self._append('events.jsonl', {'type': 'resume_restart',
                                                 'timestamp': _timestamp(), 'nextSession': 1,
                                                 'reason': str(error)})
                    self._checkpoint(0)
            terminated = False
            for index, session in enumerate(view['sessions'][completed:], completed + 1):
                self._append('events.jsonl', {'type': 'session_started',
                                             'sessionId': session['sessionId'],
                                             'timestamp': _timestamp(), 'sessionIndex': index})
                for turn in session['turns']:
                    error = self._execute('turn', dict(turn, sessionId=session['sessionId']),
                                          session['sessionId'])
                    if error is not None and error.category == 'timeout':
                        terminated = True
                        break
                if not terminated:
                    for probe in by_session.get(session['sessionId'], []):
                        error = self._execute('probe', probe, session['sessionId'])
                        if error is not None and error.category == 'timeout':
                            terminated = True
                            break
                self._append('events.jsonl', {
                    'type': 'session_terminated' if terminated else 'session_completed',
                    'sessionId': session['sessionId'], 'sessionIndex': index,
                    'timestamp': _timestamp()})
                if terminated:
                    break
                completed = index
                self._checkpoint(completed, session['sessionId'])
            self._append('events.jsonl', {'type': 'run_terminated' if terminated else 'run_completed',
                                         'timestamp': _timestamp(), 'lastCompletedSession': completed})
            for handle in self._handles.values():
                handle.flush()
                os.fsync(handle.fileno())
            probe_records = [json.loads(line) for line in
                             (self.directory / 'probes.jsonl').read_text().splitlines()]
            requests = [json.loads(line) for line in
                        (self.directory / 'requests.jsonl').read_text().splitlines()]
            answered = sum(record['eligibleForAccuracy'] for record in probe_records)
            total = len(fixture['probes'])
            failures = Counter(record['category'] for record in probe_records
                               if not record['eligibleForAccuracy'])
            incomplete = sum(failures[category] for category in ('provider', 'adapter', 'timeout'))
            results = {'status': 'terminated' if terminated else 'completed', 'valid': True,
                       'totalCount': total, 'answeredCount': answered,
                       'accuracyDenominator': answered, 'incompleteCount': incomplete,
                       'completionRate': answered / total if total else 1,
                       'investigationRequired': incomplete / total > 0.05 if total else False,
                       'failureCounts': {category: failures[category] for category in _CATEGORIES},
                       'unattemptedCount': total - len(probe_records),
                       'lastCompletedSession': completed, 'requestCount': len(requests)}
            final_manifest = dict(configuration, timestamp=_timestamp(), configHash=self.identity,
                                  fixtureId=fixture['fixtureId'], fixtureSeed=fixture['seed'],
                                  generatorVersion=fixture['generatorVersion'],
                                  stablePrefixHash=_hash(self.manifest['stablePrefix']),
                                  results=results, recovery=recovery,
                                  everyRequestTierTotals=[{'requestId': request.get('requestId'),
                                                          'tierTokens': request.get('tierTokens'),
                                                          'totalTokens': request.get('totalTokens')}
                                                         for request in requests])
            final_manifest = self._redact(final_manifest)
            with (self.directory / 'manifest.json').open('xb') as handle:
                handle.write(canonical_json(self._redact(final_manifest)))
                handle.flush()
                os.fsync(handle.fileno())
            for name in (*_STREAMS, 'manifest.json'):
                (self.directory / name).chmod(0o444)
            return {'manifest': final_manifest, 'results': results, 'artifactDir': str(self.directory)}
        except BaseException as error:
            # Never advance the checkpoint: resume rolls back partial-session work.
            interruption = {'type': 'interruption', 'timestamp': _timestamp(),
                            'lastCompletedSession': completed, 'nextSession': completed + 1,
                            'category': getattr(error, 'category', 'interruption'),
                            'error': str(error)}
            _atomic_json(self.directory / 'interruption.json', self._redact(interruption))
            if self._handles:
                self._append('events.jsonl', interruption)
            raise
        finally:
            for handle in self._handles.values():
                handle.close()
            self._handles = {}
