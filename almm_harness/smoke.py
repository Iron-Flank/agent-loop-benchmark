"""Persistent ten-session CI proof, never canonical benchmark evidence."""
import json
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

from almm_adapter.conformance import check_native_harness
from almm_adapter.reference import ReferenceAdapter
from almm_fixture.engine import canonical_json, generate
from almm_report.__main__ import load_report
from almm_report.smoke import _judge_transport
from almm_report.writer import ReportWriter
from almm_scorer.pipeline import ScoringPipeline
from almm_scorer.semantic import SemanticJudge
from almm_scorer.smoke import _calibration

from .budget import BudgetVerifier
from .proxy import ModelProxy, redact
from .runner import FixtureRunner


_SESSIONS = 10
_TURNS = 100
_PROBES = 5
_REQUESTS = _TURNS + _PROBES
_MAX_SECONDS = 120
_PROVENANCE = 'synthetic smoke only; mock labels and judge, not human agreement'


def _model_transport(payload):
    """Deterministic offline provider; not a benchmark model."""
    return {'content': 'I do not know.', 'finishReason': 'stop',
            'usage': {'promptTokens': 0, 'completionTokens': 0},
            'modelVersion': 'offline-abstention-1'}


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _validate_execution(fixture, run):
    expected_turns = [(session['sessionId'], turn['turnId'])
                      for session in fixture['sessions'] for turn in session['turns']]
    if (len(fixture['sessions']), len(expected_turns), len(fixture['probes'])) != (
            _SESSIONS, _TURNS, _PROBES):
        raise ValueError('smoke: fixture must contain exactly 10 sessions, 100 turns and 5 probes')
    directory = Path(run['artifactDir'])
    events = _rows(directory / 'events.jsonl')
    turns = [event for event in events if event['type'] == 'turn']
    if ([(event['sessionId'], event['turnId']) for event in turns] != expected_turns or
            any(event['status'] != 'ok' for event in turns)):
        raise ValueError('smoke: failed or incomplete turn chronology')
    results = run['results']
    if (results['status'] != 'completed' or results['lastCompletedSession'] != _SESSIONS or
            results['answeredCount'] != _PROBES or results['unattemptedCount'] != 0 or
            any(results['failureCounts'].values())):
        raise ValueError('smoke: failed or incomplete probe execution')
    requests = _rows(directory / 'requests.jsonl')
    if (results['requestCount'] != _REQUESTS or len(requests) != _REQUESTS or
            any(request['status'] != 'ok' for request in requests)):
        raise ValueError('smoke: full successful request count must be 105')
    if _rows(directory / 'dead-letters.jsonl'):
        raise ValueError('smoke: unexpected request failures')


def _validate_report(fixture, scored, published):
    # Reload and hash-verify the persisted report, rather than trust a return value.
    loaded = load_report(published['artifactDir'])
    report = loaded['report']
    scores = _rows(Path(scored['artifactDir']) / 'scores.jsonl')
    if ({row['probeId'] for row in scores} != {probe['probeId'] for probe in fixture['probes']} or
            len(scores) != _PROBES or not all(row['eligibleForAccuracy'] for row in scores)):
        raise ValueError('smoke: all five probes must be captured and scored')
    expected_correct = sum(probe['expected']['matchType'] == 'abstain' for probe in fixture['probes'])
    if (expected_correct != 1 or sum(row['normalizedResult']['correct'] for row in scores) != 1 or
            report['accuracy'] != {'correct': 1, 'answered': _PROBES, 'total': _PROBES,
                                   'accuracy': 0.2, 'completionRate': 1.0} or
            report['abstention']['correctAbstentions'] != 1 or
            report['abstention']['falseAbstentions'] != 4):
        raise ValueError('smoke: expected one correct and four false abstentions')
    if report['scale'] != _SESSIONS or report['operationalTelemetry']['requestCount'] != _REQUESTS:
        raise ValueError('smoke: report scale/request accounting differs from execution')
    if loaded['manifest']['canonical'] or scored['manifest']['canonical']:
        raise ValueError('smoke: synthetic results must remain noncanonical')
    if scored['manifest']['calibration']['agreement'] != 1.0:
        raise ValueError('smoke: mock semantic calibration agreement must be 1.0')
    return report


def smoke(output='artifacts/smoke'):
    """Run the real offline pipeline and retain a fresh diagnostic bundle per call.

    Fixed fixture seed, pinned mock identities and a character tokenizer keep
    accounting reproducible. Timestamps, measured timings and archive identities
    vary intentionally; a new bundle preserves earlier immutable runs.
    """
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='smoke-', dir=output))
    stage = 'native-conformance'
    try:
        native_checks = check_native_harness()
        stage = 'fixture'
        fixture = generate(42, _SESSIONS)
        fixture_path = root / 'fixture.json'
        fixture_path.write_bytes(canonical_json(fixture))
        calibration = root / 'calibration'
        calibration.mkdir()
        _calibration(calibration)
        manifest = {
            'runId': 'ci-smoke-10',
            'adapter': {'name': 'reference-offline-smoke', 'revision': '2', 'contractVersion': '2.0'},
            'nativeModelEnvelopeVersion': '1.0',
            'model': {'provider': 'offline-smoke', 'name': 'abstention-mock',
                      'version': 'offline-abstention-1', 'decoding': {'temperature': 0},
                      'deterministic': True},
            'tokenizer': {'name': 'characters-smoke', 'version': '1'},
            'stablePrefix': ['ALMM offline noncanonical CI smoke'],
            'seed': 42, 'scorerVersion': 'unscored-1',
            'rateLimitRpm': 1000000, 'probeTimeoutSeconds': 10,
        }
        proxy = ModelProxy(manifest, BudgetVerifier(manifest, len), _model_transport)
        adapter = ReferenceAdapter(proxy, len)
        stage = 'execution'
        run = FixtureRunner(manifest, adapter, proxy, root / 'runs').run(fixture)
        _validate_execution(fixture, run)
        judge = SemanticJudge({
            'provider': 'offline-smoke', 'model': 'offline-judge', 'version': 'offline-judge-1',
            'rubricVersion': '1.0', 'endpoint': 'http://127.0.0.1:1/v1/chat/completions',
            'keyEnv': 'ALMM_CI_SMOKE_JUDGE_API_KEY',
        }, transport=_judge_transport)
        stage = 'scoring'
        # A local mock credential satisfies the real judge boundary without ever
        # consuming, replacing or requiring a user's runtime/judge credentials.
        with patch.dict(os.environ, {'ALMM_CI_SMOKE_JUDGE_API_KEY': 'ci-smoke-mock-credential'}):
            scored = ScoringPipeline(judge, calibration).run(
                fixture, run['artifactDir'], root / 'scores', canonical=False)
        stage = 'reporting'
        published = ReportWriter().write(fixture, run['artifactDir'], scored['artifactDir'], root / 'reports')
        report = _validate_report(fixture, scored, published)
        elapsed = time.perf_counter() - started
        if not 0 <= elapsed < _MAX_SECONDS:
            raise ValueError(f'smoke: elapsed {elapsed:.6f}s must be below 120s')
        summary = {
            'canonical': False, 'calibrationProvenance': _PROVENANCE,
            'nativeEnvelopeChecks': native_checks,
            'sessions': _SESSIONS, 'turns': _TURNS, 'probes': _PROBES, 'requests': _REQUESTS,
            'correctAbstentions': report['abstention']['correctAbstentions'],
            'falseAbstentions': report['abstention']['falseAbstentions'],
            'accuracy': report['accuracy'], 'elapsedSeconds': elapsed,
            'artifacts': {'directory': str(root), 'fixture': str(fixture_path),
                          'calibration': str(calibration), 'run': run['artifactDir'],
                          'scores': scored['artifactDir'], 'report': published['artifactDir'],
                          'summary': str(root / 'summary.json')},
        }
        (root / 'summary.json').write_bytes(canonical_json(summary))
        return summary
    except Exception as error:
        failure = {'canonical': False, 'stage': stage, 'error': redact(str(error)),
                   'elapsedSeconds': time.perf_counter() - started, 'artifactDir': str(root)}
        (root / 'failure.json').write_bytes(canonical_json(failure))
        raise RuntimeError(f'smoke {stage} failed; diagnostics retained at {root}: {failure["error"]}') from error
