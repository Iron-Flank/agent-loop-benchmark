import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from almm_fixture.engine import canonical_json, generate
from almm_fixture.validation import SchemaValidator
from almm_scorer.pipeline import ScoringPipeline, preflight
from test_harness import make_runner


def archive(directory, fixture=None):
    fixture = fixture or generate(42, 10)
    runner, _, _ = make_runner(directory)
    result = runner.run(fixture)
    return fixture, Path(result['artifactDir'])


def rewrite(path, value):
    path.chmod(0o644)
    path.write_bytes(canonical_json(value))


class OverBudgetAdapter:
    """Delegate turns normally but assemble an over-budget probe request."""
    def __new__(cls, proxy):
        from test_harness import CompactAdapter
        class Adapter(CompactAdapter):
            def answerProbe(self, probe):
                return self.respond('x' * 25001, probe['probeId'], 'answer')
        return Adapter(proxy)


class ScoringTests(unittest.TestCase):
    def test_entire_archive_validated_before_calibration_or_judge(self):
        with tempfile.TemporaryDirectory() as root:
            fixture, source = archive(root)
            rows = [json.loads(line) for line in (source / 'requests.jsonl').read_text().splitlines()]
            rows[-1]['tierSegments'][-1]['content'] = 'x' * 25001
            path = source / 'requests.jsonl'
            path.chmod(0o644)
            path.write_bytes(b''.join(canonical_json(row) for row in rows))
            with patch('almm_scorer.pipeline.CalibrationGate.evaluate', side_effect=AssertionError('too early')):
                with self.assertRaisesRegex(ValueError, 'budget|tokens|telemetry'):
                    ScoringPipeline().run(fixture, source, Path(root) / 'scores', canonical=False)

    def test_preflight_rejects_hash_manifest_and_probe_corruption(self):
        with tempfile.TemporaryDirectory() as root:
            fixture, source = archive(root)
            bad = copy.deepcopy(fixture)
            bad['contentHash'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'hash'):
                preflight(bad, source)
            manifest = json.loads((source / 'manifest.json').read_text())
            for field in ('timestamp', 'harnessHash', 'scorerVersion'):
                altered = copy.deepcopy(manifest)
                altered.pop(field)
                rewrite(source / 'manifest.json', altered)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    preflight(fixture, source)
            rewrite(source / 'manifest.json', manifest)
            rows = [json.loads(line) for line in (source / 'probes.jsonl').read_text().splitlines()]
            rows[0]['probeId'] = 'unknown'
            path = source / 'probes.jsonl'
            path.chmod(0o644)
            path.write_bytes(b''.join(canonical_json(row) for row in rows))
            with self.assertRaisesRegex(ValueError, 'probe'):
                preflight(fixture, source)

    def test_immutable_rescoring_and_complete_probe_records(self):
        with tempfile.TemporaryDirectory() as root:
            fixture, source = archive(root)
            before = (source / 'manifest.json').read_bytes()
            approval = {'approved': True, 'agreement': 1.0, 'canonical': False,
                        'scorerVersion': '1.0.0', 'labelingMethod': 'synthetic test only'}
            with patch('almm_scorer.pipeline.CalibrationGate.evaluate', return_value=approval):
                first = ScoringPipeline().run(fixture, source, Path(root) / 'scores', canonical=False)
                second = ScoringPipeline(version='1.0.1').run(fixture, source, Path(root) / 'scores', canonical=False)
            self.assertNotEqual(first['artifactDir'], second['artifactDir'])
            self.assertEqual((source / 'manifest.json').read_bytes(), before)
            output = Path(first['artifactDir'])
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['scorerVersion'], '1.0.0')
            self.assertFalse(manifest['canonical'])
            rows = [json.loads(line) for line in (output / 'scores.jsonl').read_text().splitlines()]
            self.assertEqual({row['probeId'] for row in rows}, {p['probeId'] for p in fixture['probes']})
            for row in rows:
                self.assertEqual(row['rawAnswer'], 'a raw model answer')
                self.assertIn(row['matchingMethod'], ('exact', 'ordered-list', 'abstain'))
                self.assertEqual(row['normalizedResult']['judgment'], 'fail')
                self.assertEqual(row['fixtureHash'], fixture['contentHash'])
                self.assertEqual(row['manifestHash'], hashlib.sha256(before).hexdigest())
                self.assertEqual(row['scorerVersion'], '1.0.0')
                self.assertEqual(row['requestTokenTelemetry'][0]['status'], 'ok')
                self.assertIsNone(row['judgeOutput'])

    def test_provider_failure_excluded_not_scored_wrong(self):
        from almm_harness.errors import ProviderFailure
        from test_harness import manifest
        with tempfile.TemporaryDirectory() as root:
            fixture = generate(42, 10)
            def provider(payload):
                if payload['request']['tierSegments'][-1].get('sourceIds', [''])[0].startswith('p-'):
                    raise ProviderFailure('offline')
                return {'content': 'ok', 'finishReason': 'stop', 'usage': {'promptTokens': 0, 'completionTokens': 0}}
            runner, _, _ = make_runner(root, manifest(), provider)
            source = Path(runner.run(fixture)['artifactDir'])
            with patch('almm_scorer.pipeline.CalibrationGate.evaluate', return_value={'approved': True}):
                result = ScoringPipeline().run(fixture, source, Path(root) / 'scores', canonical=False)
            rows = [json.loads(line) for line in (Path(result['artifactDir']) / 'scores.jsonl').read_text().splitlines()]
            self.assertTrue(all(row['normalizedResult']['judgment'] == 'incomplete' for row in rows))
            self.assertTrue(all(not row['eligibleForAccuracy'] for row in rows))
    def test_archived_budget_failures_remain_publishable_but_not_answered(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = generate(42, 10)
            runner, _, _ = make_runner(root, adapter_class=OverBudgetAdapter)
            run = runner.run(fixture)
            validated = preflight(fixture, run['artifactDir'])
            self.assertEqual(validated['manifest']['results']['answeredCount'], 0)
            self.assertEqual(validated['manifest']['results']['failureCounts']['budget'], 5)
            with patch('almm_scorer.pipeline.CalibrationGate.evaluate', return_value={'approved': True}):
                result = ScoringPipeline().run(fixture, run['artifactDir'], Path(root) / 'scores', canonical=False)
            rows = [json.loads(line) for line in (Path(result['artifactDir']) / 'scores.jsonl').read_text().splitlines()]
            self.assertTrue(all(row['failureCategory'] == 'budget' for row in rows))
            self.assertTrue(all(not row['eligibleForAccuracy'] for row in rows))


    def test_numeric_and_semantic_fixture_gold_supported(self):
        from test_fixture_validation import valid_fixture
        fixture = valid_fixture()
        probe = next(p for p in fixture['probes'] if p['expected']['matchType'] == 'exact')
        fact = next(f for f in fixture['facts'] if f['factId'] == probe['expected']['requiredFactIds'][0])
        fact['value'] = '12.5'
        probe['expected'].update(matchType='numeric', acceptedAnswers=['12.5'], tolerance=0.5)
        SchemaValidator.validate(fixture)
        probe['expected']['tolerance'] = -1
        with self.assertRaisesRegex(ValueError, 'tolerance'):
            SchemaValidator.validate(fixture)
        probe['expected'].pop('tolerance')
        probe['expected'].update(matchType='semantic', acceptedAnswers=['12.5'], rubric='State the value.',
                                 requiredClaims=['The value is 12.5.'], disallowedContradictions=['The value is 10.'])
        SchemaValidator.validate(fixture)
        probe['expected']['requiredClaims'] = []
        with self.assertRaisesRegex(ValueError, 'requiredClaims'):
            SchemaValidator.validate(fixture)


if __name__ == '__main__':
    unittest.main()
