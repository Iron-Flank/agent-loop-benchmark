"""Validate the complete archive before evaluating or publishing any score."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

from almm_adapter.contract import validate_manifest
from almm_fixture.engine import canonical_json
from almm_fixture.validation import SchemaValidator
from almm_harness.budget import BudgetVerifier
from almm_harness.errors import BudgetFailure, HarnessFailure
from almm_harness.proxy import redact
from almm_harness.tokenizers import load_tokenizer

from . import SCORER_VERSION
from .calibration import CalibrationGate
from .matcher import match_atomic


def digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name}: required nonempty string')


def _rows(path):
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'{path.name}: expected JSON objects')
            yield row


def preflight(fixture, source):
    """Return validated inputs; no scorer or judge is invoked in this stage."""
    source = Path(source)
    SchemaValidator.validate(fixture)
    fixture_hash = digest({key: value for key, value in fixture.items() if key != 'contentHash'})
    if fixture.get('contentHash') != fixture_hash:
        raise ValueError('fixture contentHash missing or differs from fixture hash')
    manifest_bytes = (source / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    validate_manifest(manifest)
    if redact(manifest) != manifest:
        raise ValueError('manifest contains credentials; keys must remain in environment')
    for key in ('timestamp', 'harnessHash', 'scorerVersion', 'configHash', 'fixtureHash', 'stablePrefixHash'):
        _string(manifest.get(key), f'manifest.{key}')
    try:
        timestamp = datetime.fromisoformat(manifest['timestamp'].replace('Z', '+00:00'))
        if timestamp.tzinfo is None:
            raise ValueError('missing timezone')
    except ValueError as error:
        raise ValueError('manifest.timestamp must be an ISO timestamp with timezone') from error
    if manifest['fixtureHash'] != fixture_hash or manifest.get('fixtureId') != fixture['fixtureId']:
        raise ValueError('manifest fixture hash/identity mismatch')
    if manifest.get('generatorVersion') != fixture['generatorVersion'] or manifest.get('fixtureSeed') != fixture['seed']:
        raise ValueError('manifest fixture version/seed mismatch')
    if type(manifest.get('seed')) is not int:
        raise ValueError('manifest.seed must be pinned')
    model = manifest['model']
    for key in ('provider', 'name', 'version'):
        _string(model.get(key), f'model.{key}')
    decoding = model.get('decoding')
    if not isinstance(decoding, dict):
        raise ValueError('model.decoding must be pinned')
    json.dumps(decoding, allow_nan=False)
    for key in ('rateLimitRpm', 'probeTimeoutSeconds'):
        value = manifest.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'manifest.{key}: positive finite number required')
    configuration = {key: manifest[key] for key in (
        'runId', 'adapter', 'model', 'tokenizer', 'stablePrefix', 'seed', 'scorerVersion',
        'harnessHash', 'fixtureHash', 'rateLimitRpm', 'probeTimeoutSeconds')}
    # Retain support for immutable archives created before concurrency metadata.
    if 'concurrency' in manifest:
        configuration['concurrency'] = manifest['concurrency']
    if digest(configuration) != manifest['configHash'] or digest(manifest['stablePrefix']) != manifest['stablePrefixHash']:
        raise ValueError('manifest configuration hash mismatch')
    verifier = BudgetVerifier(manifest, load_tokenizer(manifest['tokenizer']))
    by_request = {}
    totals = []
    for request in _rows(source / 'requests.jsonl'):
        try:
            normalized = verifier.verify(request)
        except BudgetFailure as error:
            # An honestly rejected request is execution telemetry, not invalid gold.
            if request.get('status') != 'error' or request.get('category') != 'budget':
                raise ValueError(f'over-budget request was not rejected: {request.get("requestId")}') from error
            normalized = error.request
        except (HarnessFailure, ValueError) as error:
            raise ValueError(f'request budget/telemetry validation failed: {error}') from error
        rid = request['requestId']
        if rid in by_request:
            raise ValueError('duplicate requestId in request telemetry')
        for field in ('totalTokens', 'tierTokens', 'requestOverheadTokens', 'stablePrefixHash', 'segments'):
            if normalized[field] != request.get(field):
                raise ValueError(f'request telemetry differs from retokenization: {rid}.{field}')
        if request.get('status') not in {'ok', 'error'}:
            raise ValueError('request telemetry requires ok/error status')
        # Retain compact identities, not full turn contexts, at 1000-session scale.
        without_checkpoint = {k: v for k, v in request.items() if k not in {'sessionId', 'probeId', 'turnId'}}
        by_request[rid] = {'probeId': request.get('probeId'), 'status': request['status'],
                           'hash': digest(without_checkpoint)}
        totals.append({'requestId': rid, 'tierTokens': request['tierTokens'], 'totalTokens': request['totalTokens']})
    if manifest.get('everyRequestTierTotals') != totals:
        raise ValueError('manifest request-token totals differ from archive')
    results = manifest.get('results')
    if not isinstance(results, dict) or results.get('requestCount') != len(by_request):
        raise ValueError('manifest results/request count missing or inconsistent')
    gold = {probe['probeId']: probe for probe in fixture['probes']}
    captured = list(_rows(source / 'probes.jsonl'))
    by_probe = {}
    for record in captured:
        pid = record.get('probeId')
        if pid not in gold or pid in by_probe:
            raise ValueError('unknown or duplicate captured probe')
        if record.get('sessionId') != gold[pid]['afterSessionId']:
            raise ValueError('probe checkpoint mismatch')
        eligible = record.get('eligibleForAccuracy')
        if type(eligible) is not bool or record.get('status') != ('ok' if eligible else 'error'):
            raise ValueError('probe eligibility/status mismatch')
        if eligible and not isinstance(record.get('answer'), str):
            raise ValueError('answered probe requires raw answer string')
        if not eligible and record.get('category') not in {'budget', 'provider', 'adapter', 'timeout'}:
            raise ValueError('incomplete probe requires failure category')
        linked = record.get('requests')
        if not isinstance(linked, list) or (eligible and not linked):
            raise ValueError('probe requires request telemetry')
        for request in linked:
            rid = request.get('requestId')
            archived = by_request.get(rid)
            if archived is None or archived.get('probeId') != pid:
                raise ValueError('probe references unarchived request')
            if archived['hash'] != digest(request):
                raise ValueError('probe request telemetry differs from request archive')
            if eligible and archived['status'] != 'ok':
                raise ValueError('answered probe includes failed request')
        if record.get('requestIds') != [r['requestId'] for r in linked]:
            raise ValueError('probe requestIds differ from telemetry')
        by_probe[pid] = record
    answered = sum(record['eligibleForAccuracy'] for record in captured)
    if (results.get('totalCount') != len(gold) or results.get('answeredCount') != answered
            or results.get('accuracyDenominator') != answered
            or results.get('unattemptedCount') != len(gold) - len(captured)):
        raise ValueError('manifest probe counts differ from archive')
    if results.get('status') == 'completed' and set(by_probe) != set(gold):
        raise ValueError('completed run missing captured probes')
    if results.get('status') not in {'completed', 'terminated'}:
        raise ValueError('run manifest is not finalized')
    return {'manifest': manifest, 'manifestHash': hashlib.sha256(manifest_bytes).hexdigest(),
            'fixtureHash': fixture_hash, 'captured': by_probe, 'requestCount': len(by_request)}


class ScoringPipeline:
    def __init__(self, judge=None, calibration_dir='calibration/scorer-v1.0.0', version=SCORER_VERSION):
        _string(version, 'scorerVersion')
        if '/' in version or '\\' in version or version in {'.', '..'}:
            raise ValueError('invalid scorerVersion')
        self.version = version
        self.judge = judge
        self.calibration = CalibrationGate(calibration_dir)

    def score(self, probe, answer):
        if probe['expected']['matchType'] == 'semantic':
            if self.judge is None:
                raise ValueError('semantic records require a configured pinned judge')
            return self.judge.score(probe, answer)
        return match_atomic(probe, answer)

    def run(self, fixture, source, output, *, canonical=True):
        inputs = preflight(fixture, source)
        if canonical and inputs['manifest']['tokenizer']['name'] == 'characters-smoke':
            raise ValueError('characters-smoke tokenizer cannot produce canonical results')
        approval = self.calibration.evaluate(self.score, self.version,
                                             judge_config=self.judge.config if self.judge else None,
                                             require_human=canonical)
        if not approval.get('approved'):
            raise ValueError('calibration did not approve scorer')
        rows = []
        for probe in fixture['probes']:
            captured = inputs['captured'].get(probe['probeId'])
            eligible = captured is not None and captured['eligibleForAccuracy']
            raw_answer = captured.get('answer') if captured else None
            if eligible:
                score = self.score(probe, raw_answer)
            else:
                score = {'matchingMethod': 'not-scored', 'judgment': 'incomplete', 'correct': False,
                         'normalizedAnswer': None, 'judgeOutput': None}
            rows.append({'probeId': probe['probeId'], 'rawAnswer': raw_answer,
                         'matchingMethod': score['matchingMethod'],
                         'normalizedResult': {key: score[key] for key in ('judgment', 'correct', 'normalizedAnswer')},
                         'judgeOutput': score['judgeOutput'], 'eligibleForAccuracy': eligible,
                         'failureCategory': captured.get('category') if captured else 'unattempted',
                         'requestTokenTelemetry': captured['requests'] if captured else [],
                         'fixtureHash': inputs['fixtureHash'], 'manifestHash': inputs['manifestHash'],
                         'scorerVersion': self.version})
        judge_config = deepcopy(self.judge.config) if self.judge else None
        scorer_hash = hashlib.sha256(b''.join(
            path.relative_to(Path(__file__).parent).as_posix().encode() + b'\0' + path.read_bytes()
            for path in sorted(Path(__file__).parent.rglob('*.py')))).hexdigest()
        identity = {'sourceManifestHash': inputs['manifestHash'], 'fixtureHash': inputs['fixtureHash'],
                    'scorerVersion': self.version, 'scorerHash': scorer_hash, 'judge': judge_config,
                    'calibrationHashes': approval.get('calibrationHashes'), 'canonical': canonical}
        manifest = deepcopy(inputs['manifest'])
        manifest.update(scorerVersion=self.version, sourceManifestHash=inputs['manifestHash'],
                        scoringHash=digest(identity), scorerHash=scorer_hash, judge=judge_config, canonical=canonical,
                        calibration=approval, scoredAt=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'))
        manifest['sourceScorerVersion'] = inputs['manifest']['scorerVersion']
        clean = self.judge.redact if self.judge else redact
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        destination = output / inputs['manifestHash'] / f'{self.version}-{manifest["scoringHash"]}'
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Publish only complete directories. Existing identities are never replaced.
        temporary = Path(tempfile.mkdtemp(prefix='.scoring-', dir=destination.parent))
        try:
            (temporary / 'scores.jsonl').write_bytes(b''.join(canonical_json(clean(row)) for row in rows))
            (temporary / 'manifest.json').write_bytes(canonical_json(clean(manifest)))
            for path in temporary.iterdir():
                path.chmod(0o444)
            if destination.exists():
                raise FileExistsError('scoring artifacts are immutable; identity already published')
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return {'artifactDir': str(destination), 'manifest': clean(manifest), 'probeCount': len(rows)}
