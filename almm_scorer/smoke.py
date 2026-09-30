"""Offline full-pipeline proof; synthetic calibration is never canonical approval."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
from urllib.request import urlopen

from almm_fixture.engine import canonical_json, generate
from almm_harness.budget import BudgetVerifier
from almm_harness.proxy import ModelProxy
from almm_harness.runner import FixtureRunner
from .pipeline import ScoringPipeline
from .semantic import SemanticJudge


class _Adapter:
    def __init__(self, proxy):
        self.proxy = proxy

    def initialize(self, manifest):
        self.manifest = manifest
        self.prefix = manifest['stablePrefix'][0]
        self.sequence = 0
        self.turns = []

    def _respond(self, text, field, source_ids=None):
        self.sequence += 1
        request = {'requestId': str(self.sequence),
                   'messages': [{'role': 'system', 'content': self.prefix},
                                {'role': 'user', 'content': text}],
                   'model': self.manifest['model'],
                   'decodingSettings': self.manifest['model']['decoding'],
                   'tierSegments': [
                       {'tier': 'stable', 'content': self.prefix,
                        'tokenCount': len(self.prefix), 'messageIndices': [0]},
                       {'tier': 'unstable', 'content': text,
                        'tokenCount': len(text), 'messageIndices': [1]}]}
        if source_ids is not None:
            request['tierSegments'][-1]['sourceIds'] = source_ids
        answer = self.proxy(request)['content']
        return {field: answer, 'requests': [request]}

    def handleTurn(self, turn):
        self.turns.append(turn['turnId'])
        return self._respond(turn['text'], 'response')

    def answerProbe(self, probe):
        return self._respond(probe['question'], 'answer')


def _calibration(directory):
    """Generate independent mock labels, explicitly not human annotation."""
    abilities = ('information-extraction', 'cross-session-reasoning', 'temporal-reasoning',
                 'knowledge-update', 'abstention')
    kinds = ('semantic',) * 4 + ('exact',) * 2 + ('numeric',) * 2 + ('ordered-list', 'abstain')
    rows = []
    for index in range(100):
        kind = kinds[index % 10]
        expected = {'matchType': kind, 'acceptedAnswers': ['blue'],
                    'requiredFactIds': ['smoke-fact'], 'forbiddenFactIds': []}
        answer, gold = ('blue', 'pass') if index % 3 else ('red', 'fail')
        if kind == 'semantic':
            expected.update(rubric='State blue; red contradicts blue.',
                            requiredClaims=['blue'], disallowedContradictions=['red'])
        elif kind == 'numeric':
            expected.update(targetNumber='10', tolerance='0.5', toleranceMode='absolute-inclusive',
                            acceptedAnswers=['10'])
            answer = '10' if gold == 'pass' else '11'
        elif kind == 'ordered-list':
            expected.update(items=['blue', 'green'], acceptedAnswers=['["blue", "green"]'])
            answer = '["blue", "green"]' if gold == 'pass' else '["green", "blue"]'
        elif kind == 'abstain':
            expected.update(acceptedAnswers=['I do not know.'], requiredFactIds=[])
            answer, gold = ('I do not know.', 'abstain') if gold == 'pass' else ('blue', 'fail')
        rows.append({'probeId': f'smoke-cal-{index}', 'ability': abilities[index // 20],
                     'question': f'Synthetic value {index}?', 'candidateAnswer': answer,
                     'expected': expected, 'answerability': kind != 'abstain',
                     'goldJudgment': gold, 'justification': f'Synthetic smoke label: {gold}.',
                     'answerAsOf': 'current', 'afterSessionIndex': 10, 'edgeCases': ['synthetic'],
                     'evidence': [{'factId': 'smoke-fact', 'sessionIndex': 1,
                                   'text': 'Synthetic smoke value.', 'state': 'active'}]})
    probes = {'schemaVersion': 'calibration-1.0', 'calibrationSetId': 'synthetic-smoke-only',
              'calibrationSetVersion': 'smoke-1', 'frozen': True,
              'labelingMethod': 'Synthetic smoke labels; not human annotations.', 'probes': rows}
    review = {'calibrationSetId': probes['calibrationSetId'], 'result': 'passed',
              'reviewMethod': 'Synthetic smoke consistency check, not human review.',
              'reviewedAt': '2026-01-01T00:00:00Z',
              'records': [{'probeId': row['probeId'], 'confirmedGoldJudgment': row['goldJudgment'],
                           'justificationConsistent': True, 'evidenceConsistent': True,
                           'reviewNote': row['justification']} for row in rows]}
    def save(name, value):
        content = canonical_json(value)
        (directory / name).write_bytes(content)
        return hashlib.sha256(content).hexdigest()
    probe_hash = save('probes.json', probes)
    review['probesSha256'] = probe_hash
    review_hash = save('review.json', review)
    save('manifest.json', {'schemaVersion': 'calibration-manifest-1.0',
                          'calibrationSetId': probes['calibrationSetId'], 'calibrationSetVersion': 'smoke-1',
                          'retainedWithScorerVersion': '1.0.0', 'frozenAt': '2026-01-01T00:00:00Z',
                          'files': {'probes.json': probe_hash, 'review.json': review_hash},
                          'requiredAgreement': 0.95})


def smoke():
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            data = json.loads(body['messages'][-1]['content'].split('\n', 1)[1].rsplit('\n', 1)[0])
            expected = json.loads(body['messages'][0]['content'].split('TRUSTED_RUBRIC\n', 1)[1])
            # This is an explicitly mock judge, not a prose-semantic implementation.
            answer = data['candidate']
            claims = expected['requiredClaims']
            contradictions = expected['disallowedContradictions']
            coverage = [claim.lower() in answer.lower() for claim in claims]
            flags = [claim.lower() in answer.lower() for claim in contradictions]
            judgment = 'pass' if all(coverage) and not any(flags) else 'fail'
            if answer == 'I do not know.':
                judgment = 'abstain'
            verdict = {'judgment': judgment, 'requiredClaimCoverage': coverage,
                       'contradictionFlags': flags, 'confidence': 1.0}
            content = json.dumps({'model': 'offline-judge-1',
                                  'choices': [{'message': {'content': json.dumps(verdict)}}]}).encode()
            calls.append(judgment)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix='almm-scoring-smoke-') as temporary:
            root = Path(temporary)
            calibration = root / 'calibration'
            calibration.mkdir()
            _calibration(calibration)
            judge = SemanticJudge({'provider': 'offline-smoke', 'model': 'offline-judge',
                                   'version': 'offline-judge-1', 'rubricVersion': '1.0',
                                   'endpoint': f'http://127.0.0.1:{server.server_port}/v1/chat/completions'},
                                  transport=urlopen)
            manifest = {'runId': 'scoring-smoke', 'adapter': {'name': 'smoke', 'revision': '2', 'contractVersion': '2.0'},
                        'nativeModelEnvelopeVersion': '1.0',
                        'model': {'provider': 'offline-smoke', 'name': 'offline', 'version': '1',
                                  'decoding': {'temperature': 0}},
                        'tokenizer': {'name': 'characters-smoke', 'version': '1'},
                        'stablePrefix': ['ALMM smoke'], 'seed': 42, 'scorerVersion': 'unscored-1',
                        'rateLimitRpm': 1000000, 'probeTimeoutSeconds': 10}
            proxy = ModelProxy(manifest, BudgetVerifier(manifest, len),
                               lambda payload: {'content': 'I do not know.', 'finishReason': 'stop',
                                                'usage': {'promptTokens': 0, 'completionTokens': 0}})
            adapter = _Adapter(proxy)
            fixture = generate(42, 10)
            semantic = next(p for p in fixture['probes'] if p['expected']['matchType'] == 'exact')
            semantic['expected'].update(matchType='semantic', rubric='State the recorded value.',
                                        requiredClaims=semantic['expected']['acceptedAnswers'],
                                        disallowedContradictions=[])
            fixture['contentHash'] = hashlib.sha256(canonical_json(
                {k: v for k, v in fixture.items() if k != 'contentHash'})).hexdigest()
            run = FixtureRunner(manifest, adapter, proxy, root / 'runs').run(fixture)
            # Mock HTTP judge needs an env-only mock credential, never a real API key.
            import os
            from unittest.mock import patch
            with patch.dict(os.environ, {'ALMM_JUDGE_API_KEY': 'offline-smoke-credential'}):
                result = ScoringPipeline(judge, calibration).run(fixture, run['artifactDir'], root / 'scores', canonical=False)
            rows = [json.loads(line) for line in (Path(result['artifactDir']) / 'scores.jsonl').read_text().splitlines()]
            if len(adapter.turns) != 100 or len(rows) != 5 or sum(row['normalizedResult']['correct'] for row in rows) != 1:
                raise ValueError('smoke: chronology/probe/abstention result mismatch')
            semantic_row = next(row for row in rows if row['matchingMethod'] == 'semantic')
            if semantic_row['judgeOutput']['judgment'] != 'abstain':
                raise ValueError('smoke: semantic verdict missing from per-probe artifact')
            if result['manifest']['calibration']['agreement'] != 1 or len(calls) != 41:
                raise ValueError('smoke: synthetic calibration did not exercise semantic judge')
            return {'sessions': 10, 'turns': 100, 'requests': len(proxy.telemetry),
                    'probes': len(rows), 'correctAbstentions': 1, 'falseAbstentions': 4,
                    'semanticJudgeHTTPCalls': len(calls), 'syntheticCalibrationAgreement': 1.0,
                    'canonical': False, 'scorerVersion': '1.0.0',
                    'calibrationProvenance': 'synthetic smoke only; not human agreement'}
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
