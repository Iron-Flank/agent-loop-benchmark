"""Validated, content-addressed reports over immutable runner/scorer archives."""
from copy import deepcopy
import ctypes
from datetime import datetime
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

from almm_adapter.contract import TIERS, validate_manifest
from almm_fixture.engine import canonical_json
from almm_scorer.pipeline import digest, preflight

from .metrics import build_report


_ARTIFACTS = ('probes.jsonl', 'requests.jsonl', 'scores.jsonl', 'report.json')
_CONFIG_KEYS = ('runId', 'adapter', 'model', 'tokenizer', 'stablePrefix', 'seed',
                'scorerVersion', 'harnessHash', 'fixtureHash', 'rateLimitRpm',
                'probeTimeoutSeconds')
_IDENTITY_KEYS = ('fixtureHash', 'harnessHash', 'adapter', 'model', 'tokenizer',
                  'scorerVersion', 'scorerHash', 'judge', 'calibrationHashes',
                  'seed', 'concurrency')
_REQUIRED = (
    'schemaVersion', 'runId', 'adapter', 'runtime', 'model', 'tokenizer',
    'stablePrefix', 'stablePrefixHash', 'seed', 'timestamp', 'harnessHash',
    'fixtureHash', 'fixtureId', 'fixtureSeed', 'generatorVersion', 'configHash',
    'rateLimitRpm', 'probeTimeoutSeconds', 'results', 'recovery',
    'everyRequestTierTotals', 'sourceManifestHash', 'sourceScorerVersion',
    'scorerVersion', 'scorerHash', 'scoringHash', 'judge', 'canonical',
    'calibration', 'calibrationHashes', 'scoredAt', 'sourceConfiguration',
    'sessionCount', 'concurrency', 'actualRunTimeSeconds',
    'effectiveThroughputRequestsPerSecond', 'artifactSizeBytes',
    'interruptionResumeHistory', 'providerVersionPinned', 'artifactHashes', 'warnings',
)


def _string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name}: nonempty string required')


def _hash(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'{name}: SHA-256 required')


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name}: integer >= {minimum} required')


def _number(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f'{name}: finite number >= {minimum} required')


def _timestamp(value, name):
    _string(value, name)
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if parsed.tzinfo is None:
            raise ValueError('timezone missing')
        return parsed
    except ValueError as error:
        raise ValueError(f'{name}: ISO timestamp with timezone required') from error


def footprint_warning(size):
    """Disk footprint is operational information, never an accuracy penalty."""
    _integer(size, 'artifactSizeBytes')
    return ['Artifact footprint exceeds 5GB; investigate telemetry storage.'] if size > 5_000_000_000 else []


def _scoring_identity(manifest):
    return {key: manifest[key] for key in (
        'sourceManifestHash', 'fixtureHash', 'scorerVersion', 'scorerHash',
        'judge', 'calibrationHashes', 'canonical')}


def validate_report_manifest(manifest):
    """Reject incomplete/nonfinite identities and inconsistent request accounting."""
    if not isinstance(manifest, dict):
        raise ValueError('report manifest: object required')
    missing = [key for key in _REQUIRED if key not in manifest]
    if missing:
        raise ValueError(f'report manifest missing fields: {", ".join(missing)}')
    try:
        canonical_json(manifest)
    except (ValueError, TypeError) as error:
        raise ValueError('report manifest must contain finite JSON values') from error
    validate_manifest(manifest)
    if manifest['schemaVersion'] != 'report-1.0':
        raise ValueError('unsupported report schemaVersion')
    for key in ('fixtureId', 'generatorVersion', 'scorerVersion', 'sourceScorerVersion'):
        _string(manifest[key], key)
    for key in ('fixtureHash', 'harnessHash', 'configHash', 'stablePrefixHash',
                'sourceManifestHash', 'scorerHash', 'scoringHash'):
        _hash(manifest[key], key)
    for key in ('timestamp', 'scoredAt'):
        _timestamp(manifest[key], key)
    for key in ('seed', 'fixtureSeed'):
        if type(manifest[key]) is not int:
            raise ValueError(f'{key}: integer required')
    _integer(manifest['sessionCount'], 'sessionCount', 1)
    runtime = manifest['runtime']
    if not isinstance(runtime, dict) or runtime != {k: manifest['adapter'][k] for k in ('name', 'revision')}:
        raise ValueError('runtime must identify adapter name/revision')
    for key in ('name', 'version'):
        _string(manifest['tokenizer'].get(key), f'tokenizer.{key}')
    _integer(manifest['tokenizer'].get('requestOverheadTokens', 0), 'tokenizer.requestOverheadTokens')
    model = manifest['model']
    for key in ('provider', 'name', 'version'):
        _string(model.get(key), f'model.{key}')
    if not isinstance(model.get('decoding'), dict):
        raise ValueError('model.decoding: pinned object required')
    if 'deterministic' not in model or (model['deterministic'] is not None and type(model['deterministic']) is not bool):
        raise ValueError('model.deterministic: boolean or null required')
    pinning = manifest['providerVersionPinned']
    if pinning is not None and type(pinning) is not bool:
        raise ValueError('providerVersionPinned: boolean or null required')
    if pinning != model.get('versionPinned'):
        raise ValueError('providerVersionPinned differs from declared model guarantee')
    concurrency = manifest['concurrency']
    if not isinstance(concurrency, dict):
        raise ValueError('concurrency: object required')
    if concurrency.get('mode') not in {'solo', 'concurrent'}:
        raise ValueError('concurrency.mode: solo or concurrent required')
    _integer(concurrency.get('factor'), 'concurrency.factor', 1)
    if concurrency['mode'] == 'solo' and concurrency['factor'] != 1:
        raise ValueError('solo concurrency factor must be 1')
    for key in ('rateLimitRpm', 'probeTimeoutSeconds'):
        _number(manifest[key], key)
        if manifest[key] == 0:
            raise ValueError(f'{key}: positive number required')
    configuration = manifest['sourceConfiguration']
    if (not isinstance(configuration, dict)
            or set(configuration) not in (set(_CONFIG_KEYS), set(_CONFIG_KEYS) | {'concurrency'})):
        raise ValueError('sourceConfiguration: complete runner configuration required')
    if digest(configuration) != manifest['configHash']:
        raise ValueError('sourceConfiguration/configHash mismatch')
    for key in configuration:
        expected = deepcopy(configuration[key])
        if key == 'model':
            if not isinstance(expected, dict):
                raise ValueError('sourceConfiguration.model: object required')
            expected.setdefault('deterministic', None)
        elif key == 'scorerVersion':
            expected = manifest['sourceScorerVersion']
        if (configuration[key] != expected if key == 'scorerVersion' else manifest[key] != expected):
            raise ValueError(f'report/source configuration mismatch: {key}')
    if digest(manifest['stablePrefix']) != manifest['stablePrefixHash']:
        raise ValueError('stablePrefixHash mismatch')
    results = manifest['results']
    if not isinstance(results, dict) or results.get('status') not in {'completed', 'terminated'}:
        raise ValueError('results: finalized runner results required')
    for key in ('requestCount', 'totalCount', 'answeredCount', 'accuracyDenominator', 'unattemptedCount'):
        _integer(results.get(key), f'results.{key}')
    if (results['answeredCount'] > results['totalCount'] or
            results['unattemptedCount'] > results['totalCount'] - results['answeredCount'] or
            results['accuracyDenominator'] != results['answeredCount']):
        raise ValueError('results: inconsistent probe counts')
    for key in ('incompleteCount', 'lastCompletedSession'):
        _integer(results.get(key), f'results.{key}')
    if results['lastCompletedSession'] > manifest['sessionCount']:
        raise ValueError('results.lastCompletedSession exceeds sessionCount')
    for key in ('valid', 'investigationRequired'):
        if type(results.get(key)) is not bool:
            raise ValueError(f'results.{key}: boolean required')
    failures = results.get('failureCounts')
    if not isinstance(failures, dict) or set(failures) != {'budget', 'provider', 'adapter', 'timeout'}:
        raise ValueError('results.failureCounts: all failure categories required')
    for key, value in failures.items():
        _integer(value, f'results.failureCounts.{key}')
    if (sum(failures.values()) + results['answeredCount'] + results['unattemptedCount'] != results['totalCount']
            or results['incompleteCount'] != sum(failures[key] for key in ('provider', 'adapter', 'timeout'))):
        raise ValueError('results: inconsistent failure accounting')
    completion = results['answeredCount'] / results['totalCount'] if results['totalCount'] else 1
    _number(results.get('completionRate'), 'results.completionRate')
    if results['completionRate'] != completion:
        raise ValueError('results.completionRate differs from probe counts')
    investigation = results['incompleteCount'] / results['totalCount'] > 0.05 if results['totalCount'] else False
    if results['investigationRequired'] != investigation:
        raise ValueError('results.investigationRequired differs from incomplete rate')
    totals = manifest['everyRequestTierTotals']
    if not isinstance(totals, list) or len(totals) != results['requestCount']:
        raise ValueError('everyRequestTierTotals: request count mismatch')
    seen = set()
    for row in totals:
        if not isinstance(row, dict):
            raise ValueError('everyRequestTierTotals: object rows required')
        _string(row.get('requestId'), 'requestId')
        if row['requestId'] in seen:
            raise ValueError('everyRequestTierTotals: duplicate requestId')
        seen.add(row['requestId'])
        _integer(row.get('totalTokens'), 'totalTokens')
        tiers = row.get('tierTokens')
        if not isinstance(tiers, dict) or set(tiers) != set(TIERS):
            raise ValueError('tierTokens: all three tiers required')
        for tier in TIERS:
            _integer(tiers[tier], f'tierTokens.{tier}')
    elapsed = manifest['actualRunTimeSeconds']
    _number(elapsed, 'actualRunTimeSeconds')
    throughput = manifest['effectiveThroughputRequestsPerSecond']
    if elapsed == 0:
        if throughput is not None:
            raise ValueError('throughput undefined for zero run time')
    else:
        _number(throughput, 'effectiveThroughputRequestsPerSecond')
        if not math.isclose(throughput, results['requestCount'] / elapsed, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('effective throughput differs from request count/runtime')
    _integer(manifest['artifactSizeBytes'], 'artifactSizeBytes')
    if manifest['warnings'] != footprint_warning(manifest['artifactSizeBytes']):
        raise ValueError('warnings differ from footprint policy')
    if not isinstance(manifest['recovery'], dict):
        raise ValueError('recovery: object required')
    recovery = manifest['recovery']
    if type(recovery.get('resumed')) is not bool:
        raise ValueError('recovery.resumed: boolean required')
    for key in ('checkpointSession', 'nextSession'):
        _integer(recovery.get(key), f'recovery.{key}')
    if 'interruption' in recovery:
        if not isinstance(recovery['interruption'], dict):
            raise ValueError('recovery.interruption: object required')
        _timestamp(recovery['interruption'].get('timestamp'), 'recovery.interruption.timestamp')
    history = manifest['interruptionResumeHistory']
    if not isinstance(history, list):
        raise ValueError('interruptionResumeHistory: list required')
    for entry in history:
        if not isinstance(entry, dict):
            raise ValueError('interruptionResumeHistory: object entries required')
        if entry.get('type') == 'recovery':
            if entry.get('details') != manifest['recovery']:
                raise ValueError('recovery history differs from runner recovery')
        elif entry.get('type') in {'interruption', 'resume', 'resume_restart'}:
            _timestamp(entry.get('timestamp'), 'history.timestamp')
        else:
            raise ValueError('invalid interruption/resume history type')
    if type(manifest['canonical']) is not bool:
        raise ValueError('canonical: boolean required')
    calibration = manifest['calibration']
    hashes = manifest['calibrationHashes']
    if not isinstance(calibration, dict) or calibration.get('approved') is not True:
        raise ValueError('calibration: approved scorer required')
    if not isinstance(hashes, dict) or set(hashes) != {'manifest.json', 'probes.json', 'review.json'}:
        raise ValueError('calibrationHashes: complete frozen calibration hashes required')
    for key, value in hashes.items():
        _hash(value, f'calibrationHashes.{key}')
    if calibration.get('calibrationHashes') != hashes or calibration.get('scorerVersion') != manifest['scorerVersion']:
        raise ValueError('calibration identity mismatch')
    judge = manifest['judge']
    if judge is not None:
        if not isinstance(judge, dict) or not judge:
            raise ValueError('judge: pinned configuration or null required')
        for key in ('provider', 'model', 'version', 'rubricVersion', 'endpoint', 'keyEnv'):
            _string(judge.get(key), f'judge.{key}')
        if type(judge.get('temperatureUnavailable')) is not bool:
            raise ValueError('judge.temperatureUnavailable: explicit boolean required')
        if judge['temperatureUnavailable']:
            if 'temperature' in judge:
                raise ValueError('unavailable judge temperature cannot be specified')
        elif type(judge.get('temperature')) not in (int, float) or judge['temperature'] != 0:
            raise ValueError('judge.temperature: zero required')
        _number(judge.get('timeout'), 'judge.timeout')
        if judge['timeout'] == 0:
            raise ValueError('judge.timeout: positive required')
        if 'resolvedModel' in judge:
            _string(judge['resolvedModel'], 'judge.resolvedModel')
    if digest(_scoring_identity(manifest)) != manifest['scoringHash']:
        raise ValueError('scoringHash mismatch')
    artifact_hashes = manifest['artifactHashes']
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != set(_ARTIFACTS):
        raise ValueError('artifactHashes: complete artifact hashes required')
    for key, value in artifact_hashes.items():
        _hash(value, f'artifactHashes.{key}')


def comparability(manifest_a, manifest_b):
    """Compare pinned configurations; observations and nondeterministic variance differ."""
    mismatches = []
    for name, manifest in (('manifest_a', manifest_a), ('manifest_b', manifest_b)):
        try:
            validate_report_manifest(manifest)
        except (ValueError, TypeError, KeyError) as error:
            mismatches.append(f'{name}: {error}')
    if not mismatches:
        mismatches.extend(key for key in _IDENTITY_KEYS if manifest_a[key] != manifest_b[key])
    variance_required = any(
        not isinstance(manifest, dict) or not isinstance(manifest.get('model'), dict)
        or manifest['model'].get('deterministic') is not True
        for manifest in (manifest_a, manifest_b))
    return {'comparable': not mismatches, 'mismatches': mismatches, 'varianceRequired': variance_required}


def _read_rows(path):
    rows = []
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for line in handle:
            hasher.update(line)
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'{path.name}: object rows required')
            rows.append(row)
    return rows, hasher.hexdigest()


def _file_hash(path):
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def _copy(source, destination):
    hasher = hashlib.sha256()
    with source.open('rb') as incoming, destination.open('xb') as outgoing:
        for chunk in iter(lambda: incoming.read(1024 * 1024), b''):
            outgoing.write(chunk)
            hasher.update(chunk)
    return hasher.hexdigest()


def _score_inputs(fixture, inputs, scored):
    manifest = json.loads((scored / 'manifest.json').read_bytes())
    if not isinstance(manifest, dict) or not isinstance(manifest.get('calibration'), dict):
        raise ValueError('scorer manifest: object with calibration required')
    if manifest.get('sourceManifestHash') != inputs['manifestHash'] or manifest.get('fixtureHash') != inputs['fixtureHash']:
        raise ValueError('scorer source/fixture identity hash mismatch')
    changed_fields = {'scorerVersion', 'sourceManifestHash', 'scoringHash', 'scorerHash',
                      'judge', 'canonical', 'calibration', 'scoredAt', 'sourceScorerVersion'}
    for key, value in inputs['manifest'].items():
        if key not in changed_fields and manifest.get(key) != value:
            raise ValueError(f'scorer/source manifest mismatch: {key}')
    if manifest.get('sourceScorerVersion') != inputs['manifest']['scorerVersion']:
        raise ValueError('scorer sourceScorerVersion mismatch')
    rows, score_hash = _read_rows(scored / 'scores.jsonl')
    gold = {probe['probeId']: probe for probe in fixture['probes']}
    seen = set()
    for row in rows:
        pid = row.get('probeId')
        if pid not in gold or pid in seen:
            raise ValueError('score archive has unknown/duplicate probe')
        seen.add(pid)
        for key, expected in (('manifestHash', inputs['manifestHash']), ('fixtureHash', inputs['fixtureHash']),
                              ('scorerVersion', manifest.get('scorerVersion'))):
            if row.get(key) != expected:
                raise ValueError(f'score {key} identity mismatch')
        captured = inputs['captured'].get(pid)
        eligible = captured is not None and captured['eligibleForAccuracy']
        if type(row.get('eligibleForAccuracy')) is not bool or row['eligibleForAccuracy'] != eligible:
            raise ValueError('score eligibility differs from source')
        if row.get('rawAnswer') != (captured.get('answer') if captured else None):
            raise ValueError('score raw answer differs from source')
        if row.get('requestTokenTelemetry') != (captured['requests'] if captured else []):
            raise ValueError('score request telemetry differs from source')
        if row.get('failureCategory') != (captured.get('category') if captured else 'unattempted'):
            raise ValueError('score failure category differs from source')
        result = row.get('normalizedResult')
        if not isinstance(result, dict) or type(result.get('correct')) is not bool:
            raise ValueError('score normalized correctness must be boolean')
        if not eligible:
            if (row.get('matchingMethod') != 'not-scored' or result !=
                    {'judgment': 'incomplete', 'correct': False, 'normalizedAnswer': None}
                    or row.get('judgeOutput') is not None):
                raise ValueError('incomplete probe cannot be scored correct')
        else:
            method = gold[pid]['expected']['matchType']
            judgment = result.get('judgment')
            if row.get('matchingMethod') != method or judgment not in {'pass', 'fail', 'abstain'}:
                raise ValueError('score matching method/judgment malformed')
            if method == 'semantic':
                expected_correct = ((judgment == 'pass' and gold[pid]['answerability']) or
                                    (judgment == 'abstain' and not gold[pid]['answerability']))
                if not isinstance(row.get('judgeOutput'), dict):
                    raise ValueError('semantic score requires judge output')
            else:
                expected_correct = (judgment == 'pass' or
                                    (method == 'abstain' and judgment == 'abstain' and
                                     not gold[pid]['answerability']))
                if row.get('judgeOutput') is not None:
                    raise ValueError('atomic score must not have judge output')
            if result['correct'] != expected_correct:
                raise ValueError('score correctness disagrees with judgment/answerability')
    if seen != set(gold):
        raise ValueError('score archive is missing fixture probes')
    return manifest, rows, score_hash


def _timing(source, recovery):
    if not isinstance(recovery, dict):
        raise ValueError('runner recovery metadata required')
    start = end = None
    history = []
    if recovery.get('resumed') or recovery.get('interruption'):
        history.append({'type': 'recovery', 'details': deepcopy(recovery)})
    with (source / 'events.jsonl').open() as handle:
        for line in handle:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError('events.jsonl: object events required')
            kind = event.get('type')
            if kind in {'run_started', 'resume_restart'} and start is None:
                start = _timestamp(event.get('timestamp'), 'events.start.timestamp')
            if kind in {'run_completed', 'run_terminated'}:
                end = _timestamp(event.get('timestamp'), 'events.end.timestamp')
            if kind in {'interruption', 'resume', 'resume_restart'}:
                history.append(event)
    if start is None or end is None or end < start:
        raise ValueError('archive requires ordered run start/final events for timing')
    return (end - start).total_seconds(), history


def publish_directory(staging, destination):
    """Native exclusive rename prevents replacing even an existing empty directory."""
    libc = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(staging)
    target = os.fsencode(destination)
    if sys.platform == 'darwin':
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        result = rename(source, target, 4)  # RENAME_EXCL
    elif sys.platform.startswith('linux'):
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        result = rename(-100, source, -100, target, 1)  # AT_FDCWD, RENAME_NOREPLACE
    elif os.name == 'nt':
        os.rename(staging, destination)  # Windows rename never replaces destinations.
        return
    else:
        raise OSError('atomic exclusive directory publication is unsupported on this platform')
    if result != 0:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(error, os.strerror(error), str(destination))
        raise OSError(error, os.strerror(error), str(destination))


def _verify_existing(destination, manifest_bytes, manifest):
    if destination.is_symlink() or not destination.is_dir():
        raise FileExistsError('report identity exists but is not an immutable artifact directory')
    if {path.name for path in destination.iterdir()} != {'manifest.json', *_ARTIFACTS}:
        raise FileExistsError('existing report artifact files differ')
    expected_hashes = dict(manifest['artifactHashes'], **{'manifest.json': hashlib.sha256(manifest_bytes).hexdigest()})
    for name, expected in expected_hashes.items():
        path = destination / name
        if path.is_symlink() or not path.is_file() or _file_hash(path) != expected:
            raise FileExistsError(f'existing report artifact differs: {name}')
        if path.stat().st_mode & 0o222:
            raise FileExistsError(f'existing report artifact is writable: {name}')
    if sum((destination / name).stat().st_size for name in expected_hashes) != manifest['artifactSizeBytes']:
        raise FileExistsError('existing report artifact size differs')


class ReportWriter:
    def write(self, fixture, source, scored, output):
        source, scored, output = Path(source), Path(scored), Path(output)
        inputs = preflight(fixture, source)
        manifest, scores, score_hash = _score_inputs(fixture, inputs, scored)
        requests, request_hash = _read_rows(source / 'requests.jsonl')
        probe_hash = _file_hash(source / 'probes.jsonl')
        elapsed, history = _timing(source, manifest.get('recovery'))
        manifest = deepcopy(manifest)
        manifest.update(
            schemaVersion='report-1.0',
            runtime={key: manifest['adapter'][key] for key in ('name', 'revision')},
            sessionCount=len(fixture['sessions']),
            concurrency=deepcopy(inputs['manifest'].get('concurrency', {'mode': 'solo', 'factor': 1})),
            sourceConfiguration={key: deepcopy(inputs['manifest'][key]) for key in _CONFIG_KEYS},
            actualRunTimeSeconds=elapsed,
            effectiveThroughputRequestsPerSecond=inputs['requestCount'] / elapsed if elapsed else None,
            interruptionResumeHistory=history,
            providerVersionPinned=manifest['model'].get('versionPinned'),
            calibrationHashes=deepcopy(manifest['calibration'].get('calibrationHashes')),
            artifactHashes=dict.fromkeys(_ARTIFACTS, '0' * 64), artifactSizeBytes=0, warnings=[],
        )
        if 'concurrency' in inputs['manifest']:
            manifest['sourceConfiguration']['concurrency'] = deepcopy(inputs['manifest']['concurrency'])
        manifest['model'].setdefault('deterministic', None)
        validate_report_manifest(manifest)
        report = build_report(fixture, manifest, list(inputs['captured'].values()), requests, scores)
        report_bytes = canonical_json(report)
        output.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.report-', dir=output))
        try:
            for name, origin, expected in (
                ('probes.jsonl', source, probe_hash), ('requests.jsonl', source, request_hash),
                ('scores.jsonl', scored, score_hash),
            ):
                copied = _copy(origin / name, staging / name)
                if copied != expected:
                    raise ValueError(f'archive changed during reporting: {name}')
                manifest['artifactHashes'][name] = copied
            # Revalidate the exact copied runner telemetry, not just a prior read.
            (staging / 'manifest.json').write_bytes((source / 'manifest.json').read_bytes())
            copied_inputs = preflight(fixture, staging)
            if copied_inputs['manifestHash'] != inputs['manifestHash']:
                raise ValueError('source manifest changed during reporting')
            (staging / 'manifest.json').unlink()
            (staging / 'report.json').write_bytes(report_bytes)
            manifest['artifactHashes']['report.json'] = hashlib.sha256(report_bytes).hexdigest()
            artifact_size = sum((staging / name).stat().st_size for name in _ARTIFACTS)
            # Decimal digit growth is the only size dependency (plus one warning).
            while True:
                manifest_bytes = canonical_json(manifest)
                size = artifact_size + len(manifest_bytes)
                warnings = footprint_warning(size)
                if manifest['artifactSizeBytes'] == size and manifest['warnings'] == warnings:
                    break
                manifest.update(artifactSizeBytes=size, warnings=warnings)
            validate_report_manifest(manifest)
            (staging / 'manifest.json').write_bytes(manifest_bytes)
            for path in staging.iterdir():
                path.chmod(0o444)
            destination = output / hashlib.sha256(manifest_bytes).hexdigest()
            try:
                publish_directory(staging, destination)
            except FileExistsError:
                _verify_existing(destination, manifest_bytes, manifest)
            return {'artifactDir': str(destination), 'manifest': manifest, 'report': report}
        finally:
            if staging.exists():
                shutil.rmtree(staging)
