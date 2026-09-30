"""Offline observed full pipeline at all scales; never canonical benchmark evidence."""
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

from almm_fixture.engine import generate
from almm_harness.budget import BudgetVerifier
from almm_harness.proxy import ModelProxy
from almm_harness.runner import FixtureRunner
from almm_scorer.pipeline import ScoringPipeline
from almm_scorer.semantic import SemanticJudge
from almm_scorer.smoke import _Adapter, _calibration
from .__main__ import load_report, publish_curves
from .writer import ReportWriter, comparability


class _ProvenanceAdapter(_Adapter):
    def answerProbe(self, probe):
        return self._respond(probe['question'], 'answer', self.turns[-1:])


def _judge_transport(request, timeout):
    body = json.loads(request.data)
    candidate = json.loads(body['messages'][-1]['content'].split('\n', 1)[1].rsplit('\n', 1)[0])['candidate']
    expected = json.loads(body['messages'][0]['content'].split('TRUSTED_RUBRIC\n', 1)[1])
    covered = [claim.lower() in candidate.lower() for claim in expected['requiredClaims']]
    contradicted = [claim.lower() in candidate.lower() for claim in expected['disallowedContradictions']]
    judgment = 'pass' if all(covered) and not any(contradicted) else 'fail'
    if candidate == 'I do not know.':
        judgment = 'abstain'
    return {'model': 'offline-judge-1', 'choices': [{'message': {'content': json.dumps({
        'judgment': judgment, 'requiredClaimCoverage': covered,
        'contradictionFlags': contradicted, 'confidence': 1.0})}}]}


def smoke():
    with tempfile.TemporaryDirectory(prefix='almm-report-smoke-') as temporary:
        root = Path(temporary)
        calibration = root / 'calibration'
        calibration.mkdir()
        _calibration(calibration)
        judge = SemanticJudge({'provider': 'offline-smoke', 'model': 'offline-judge',
                               'version': 'offline-judge-1', 'rubricVersion': '1.0',
                               'endpoint': 'http://127.0.0.1:1/v1/chat/completions'},
                              transport=_judge_transport)
        directories, observations = [], []
        with patch.dict(os.environ, {'ALMM_JUDGE_API_KEY': 'offline-smoke-credential'}):
            for scale in (10, 100, 500, 1000):
                manifest = {'runId': f'report-smoke-{scale}',
                            'adapter': {'name': 'report-smoke', 'revision': '2', 'contractVersion': '2.0'},
                            'nativeModelEnvelopeVersion': '1.0',
                            'model': {'provider': 'offline-smoke', 'name': 'offline', 'version': '1',
                                      'decoding': {'temperature': 0}, 'deterministic': True},
                            'tokenizer': {'name': 'characters-smoke', 'version': '1'},
                            'stablePrefix': ['ALMM smoke'], 'seed': 42, 'scorerVersion': 'unscored-1',
                            'rateLimitRpm': 1000000, 'probeTimeoutSeconds': 10}
                proxy = ModelProxy(manifest, BudgetVerifier(manifest, len),
                                   lambda payload: {'content': 'I do not know.', 'finishReason': 'stop',
                                                    'usage': {'promptTokens': 0, 'completionTokens': 0}})
                adapter = _ProvenanceAdapter(proxy)
                fixture = generate(42, scale)
                run = FixtureRunner(manifest, adapter, proxy, root / 'runs').run(fixture)
                scored = ScoringPipeline(judge, calibration).run(fixture, run['artifactDir'], root / 'scores', canonical=False)
                result = ReportWriter().write(fixture, run['artifactDir'], scored['artifactDir'], root / 'reports')
                replay = ReportWriter().write(fixture, run['artifactDir'], scored['artifactDir'], root / 'reports')
                if result['artifactDir'] != replay['artifactDir']:
                    raise ValueError('smoke: immutable replay changed destination')
                loaded = load_report(result['artifactDir'])
                report = loaded['report']
                expected_correct = sum(probe['expected']['matchType'] == 'abstain' for probe in fixture['probes'])
                if report['accuracy']['answered'] != len(fixture['probes']) or report['accuracy']['correct'] != expected_correct:
                    raise ValueError('smoke: abstention accuracy denominator/result mismatch')
                if report['operationalTelemetry']['requestCount'] != scale * 10 + len(fixture['probes']):
                    raise ValueError('smoke: full request count mismatch')
                if not comparability(result['manifest'], replay['manifest'])['comparable']:
                    raise ValueError('smoke: identical archived runs not comparable')
                directories.append(result['artifactDir'])
                observations.append({'scale': scale, 'accuracy': report['accuracy'],
                                     'artifactSizeBytes': result['manifest']['artifactSizeBytes'],
                                     'contextRelevance': report['contextRelevance']['status']})
        curves = publish_curves(directories, root / 'curves')
        plotted = json.loads((Path(curves['artifactDir']) / 'curves.json').read_text())
        if len(plotted['series']) != 1 or plotted['series'][0]['missingScales']:
            raise ValueError('smoke: four-scale curve not complete')
        return {'canonical': False, 'calibrationProvenance': 'synthetic smoke only; not human agreement',
                'immutableReplay': True, 'fourScaleCurve': True, 'observations': observations}
