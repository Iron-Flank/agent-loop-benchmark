import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from almm_fixture.engine import canonical_json, generate
from almm_report.writer import (
    ReportWriter, comparability, footprint_warning, publish_directory, validate_report_manifest,
)
from almm_scorer.pipeline import ScoringPipeline
from test_harness import CompactAdapter, make_runner, manifest as runner_manifest


def score_archive(fixture, source, root):
    frozen = Path(__file__).resolve().parents[1] / 'calibration' / 'scorer-v1.0.0'
    approval = {'approved': True, 'canonical': False, 'scorerVersion': '1.0.0',
                'judgeConfig': None,
                'calibrationHashes': {name: hashlib.sha256((frozen / name).read_bytes()).hexdigest()
                                      for name in ('manifest.json', 'probes.json', 'review.json')}}
    with patch('almm_scorer.pipeline.CalibrationGate.evaluate', return_value=approval):
        return Path(ScoringPipeline().run(
            fixture, source, Path(root) / 'scores', canonical=False)['artifactDir'])


def archives(root):
    fixture = generate(42, 10)
    runner, _, _ = make_runner(Path(root) / 'runs')
    source = Path(runner.run(fixture)['artifactDir'])
    return fixture, source, score_archive(fixture, source, root)


def rewrite(path, value):
    path.chmod(0o644)
    path.write_bytes(canonical_json(value))


class ReportWriterTests(unittest.TestCase):
    def test_publication_replay_size_hashes_and_readonly(self):
        with tempfile.TemporaryDirectory() as root:
            fixture, source, scored = archives(root)
            source_bytes = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
            writer = ReportWriter()
            first = writer.write(fixture, source, scored, Path(root) / 'reports')
            second = writer.write(fixture, source, scored, Path(root) / 'reports')
            self.assertEqual(first, second)
            directory = Path(first['artifactDir'])
            self.assertEqual(directory.name, hashlib.sha256((directory / 'manifest.json').read_bytes()).hexdigest())
            self.assertEqual(first['manifest']['artifactSizeBytes'], sum(p.stat().st_size for p in directory.iterdir()))
            for filename, digest in first['manifest']['artifactHashes'].items():
                self.assertEqual(hashlib.sha256((directory / filename).read_bytes()).hexdigest(), digest)
                self.assertEqual((directory / filename).stat().st_mode & 0o222, 0)
            self.assertEqual((directory / 'probes.jsonl').read_bytes(), (source / 'probes.jsonl').read_bytes())
            self.assertEqual((directory / 'requests.jsonl').read_bytes(), (source / 'requests.jsonl').read_bytes())
            self.assertEqual((directory / 'scores.jsonl').read_bytes(), (scored / 'scores.jsonl').read_bytes())
            self.assertEqual(source_bytes, {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()})
            self.assertEqual(first['manifest']['concurrency'], {'mode': 'solo', 'factor': 1})
            self.assertIsNone(first['manifest']['model']['deterministic'])
            self.assertGreaterEqual(first['manifest']['actualRunTimeSeconds'], 0)

    def test_existing_corrupt_artifact_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            fixture, source, scored = archives(root)
            writer = ReportWriter()
            first = writer.write(fixture, source, scored, Path(root) / 'reports')
            path = Path(first['artifactDir']) / 'scores.jsonl'
            path.chmod(0o644)
            path.write_bytes(b'corrupted\n')
            with self.assertRaises((ValueError, FileExistsError)):
                writer.write(fixture, source, scored, Path(root) / 'reports')
            self.assertEqual(path.read_bytes(), b'corrupted\n')

    def test_score_archive_identity_and_eligibility_are_verified(self):
        with tempfile.TemporaryDirectory() as root:
            fixture, source, scored = archives(root)
            original = json.loads((scored / 'manifest.json').read_text())
            altered = copy.deepcopy(original)
            altered['sourceManifestHash'] = '0' * 64
            rewrite(scored / 'manifest.json', altered)
            with self.assertRaisesRegex(ValueError, 'source|identity|hash'):
                ReportWriter().write(fixture, source, scored, Path(root) / 'reports')
            rewrite(scored / 'manifest.json', original)
            rows = [json.loads(line) for line in (scored / 'scores.jsonl').read_text().splitlines()]
            rows[0]['eligibleForAccuracy'] = False
            path = scored / 'scores.jsonl'
            path.chmod(0o644)
            path.write_bytes(b''.join(canonical_json(row) for row in rows))
            with self.assertRaisesRegex(ValueError, 'eligib|score'):
                ReportWriter().write(fixture, source, scored, Path(root) / 'reports')
            self.assertFalse((Path(root) / 'reports').exists())

    def test_overbudget_errors_remain_operational_not_accuracy_failures(self):
        class OversizedProbeAdapter(CompactAdapter):
            def answerProbe(self, probe):
                return self.respond('x' * 25001, probe['probeId'], 'answer')

        with tempfile.TemporaryDirectory() as root:
            fixture = generate(42, 10)
            runner, _, _ = make_runner(Path(root) / 'runs', adapter_class=OversizedProbeAdapter)
            source = Path(runner.run(fixture)['artifactDir'])
            scored = score_archive(fixture, source, root)
            result = ReportWriter().write(fixture, source, scored, Path(root) / 'reports')
            self.assertEqual(result['report']['accuracy']['answered'], 0)
            self.assertIsNone(result['report']['accuracy']['accuracy'])
            self.assertEqual(result['report']['accuracy']['completionRate'], 0)
            self.assertGreater(result['report']['operationalTelemetry']['overBudgetFailures'], 0)

    def test_explicit_concurrency_and_determinism_preserve_configuration(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = generate(42, 10)
            configuration = runner_manifest()
            configuration['model']['deterministic'] = True
            runner, _, _ = make_runner(Path(root) / 'runs', configuration)
            source = Path(runner.run(fixture)['artifactDir'])
            manifest = json.loads((source / 'manifest.json').read_text())
            manifest['concurrency'] = {'mode': 'parallel', 'factor': 4}
            rewrite(source / 'manifest.json', manifest)
            scored = score_archive(fixture, source, root)
            result = ReportWriter().write(fixture, source, scored, Path(root) / 'reports')
            self.assertEqual(result['manifest']['concurrency'], manifest['concurrency'])
            self.assertFalse(comparability(result['manifest'], result['manifest'])['varianceRequired'])

    def test_resume_history_and_elapsed_time_use_archived_run_events(self):
        interrupted = False

        def provider(payload):
            nonlocal interrupted
            if not interrupted and payload['request']['segments'][-1]['sourceIds'] == ['t-0002-03']:
                interrupted = True
                raise KeyboardInterrupt('process interruption')
            return {'answer': 'ok'}

        with tempfile.TemporaryDirectory() as root:
            fixture = generate(42, 10)
            runner, _, _ = make_runner(Path(root) / 'runs', provider=provider)
            with self.assertRaises(KeyboardInterrupt):
                runner.run(fixture)
            resumed, _, _ = make_runner(Path(root) / 'runs')
            source = Path(resumed.run(fixture, resume=True)['artifactDir'])
            events = [json.loads(line) for line in (source / 'events.jsonl').read_text().splitlines()]
            for event in events:
                if event['type'] == 'run_started':
                    event['timestamp'] = '2026-09-29T12:00:00Z'
                elif event['type'] == 'run_completed':
                    event['timestamp'] = '2026-09-29T12:00:10Z'
            path = source / 'events.jsonl'
            path.chmod(0o644)
            path.write_bytes(b''.join(canonical_json(event) for event in events))
            scored = score_archive(fixture, source, root)
            result = ReportWriter().write(fixture, source, scored, Path(root) / 'reports')
            self.assertEqual(result['manifest']['actualRunTimeSeconds'], 10)
            history = result['manifest']['interruptionResumeHistory']
            self.assertIn('resume', {entry['type'] for entry in history})
            recovery = next(entry['details'] for entry in history if entry['type'] == 'recovery')
            self.assertEqual(recovery['interruption']['type'], 'interruption')
            self.assertEqual(result['manifest']['effectiveThroughputRequestsPerSecond'],
                             result['manifest']['results']['requestCount'] / 10)

    def test_missing_identity_and_malformed_totals_cannot_compare(self):
        with tempfile.TemporaryDirectory() as root:
            fixture, source, scored = archives(root)
            manifest = ReportWriter().write(fixture, source, scored, Path(root) / 'reports')['manifest']
            for field in tuple(manifest):
                incomplete = copy.deepcopy(manifest)
                incomplete.pop(field)
                with self.subTest(field=field):
                    self.assertFalse(comparability(manifest, incomplete)['comparable'])
            malformed = copy.deepcopy(manifest)
            malformed['everyRequestTierTotals'][0]['totalTokens'] = -1
            with self.assertRaises(ValueError):
                validate_report_manifest(malformed)
            self.assertFalse(comparability(manifest, malformed)['comparable'])

    def test_comparison_uses_configuration_not_run_identity_or_runtime(self):
        with tempfile.TemporaryDirectory() as root:
            fixture, source, scored = archives(root)
            original = ReportWriter().write(fixture, source, scored, Path(root) / 'reports')['manifest']
            other = copy.deepcopy(original)
            other['runId'] = 'other-observed-run'
            other['sourceConfiguration']['runId'] = other['runId']
            other['configHash'] = hashlib.sha256(canonical_json(other['sourceConfiguration'])).hexdigest()
            other['timestamp'] = '2026-09-29T12:00:00Z'
            other['actualRunTimeSeconds'] *= 2
            other['effectiveThroughputRequestsPerSecond'] = (
                other['results']['requestCount'] / other['actualRunTimeSeconds']
                if other['actualRunTimeSeconds'] else None)
            comparison = comparability(original, other)
            self.assertTrue(comparison['comparable'])
            self.assertTrue(comparison['varianceRequired'])
            other['adapter']['revision'] = 'changed-revision'
            other['runtime']['revision'] = 'changed-revision'
            other['sourceConfiguration']['adapter']['revision'] = 'changed-revision'
            other['configHash'] = hashlib.sha256(canonical_json(other['sourceConfiguration'])).hexdigest()
            comparison = comparability(original, other)
            self.assertFalse(comparison['comparable'])
            self.assertIn('adapter', comparison['mismatches'])

    def test_model_version_string_does_not_claim_provider_version_pinning(self):
        with tempfile.TemporaryDirectory() as root:
            fixture, source, scored = archives(root)
            result = ReportWriter().write(fixture, source, scored, Path(root) / 'reports')
            self.assertIsNone(result['manifest']['providerVersionPinned'])

    def test_footprint_limit_is_warning_not_invalidation(self):
        self.assertEqual(footprint_warning(5_000_000_000), [])
        self.assertTrue(footprint_warning(5_000_000_001))
        with tempfile.TemporaryDirectory() as root:
            fixture, source, scored = archives(root)
            manifest = ReportWriter().write(fixture, source, scored, Path(root) / 'reports')['manifest']
            manifest['artifactSizeBytes'] = 5_000_000_001
            manifest['warnings'] = footprint_warning(manifest['artifactSizeBytes'])
            validate_report_manifest(manifest)
            self.assertTrue(comparability(manifest, manifest)['comparable'])

    def test_publication_cannot_replace_existing_empty_directory(self):
        with tempfile.TemporaryDirectory() as root:
            staging = Path(root) / 'staging'
            staging.mkdir()
            (staging / 'answer.json').write_text('{\"answer\":\"original\"}')
            destination = Path(root) / 'published'
            destination.mkdir()
            with self.assertRaises(FileExistsError):
                publish_directory(staging, destination)
            self.assertEqual(list(destination.iterdir()), [])
            self.assertEqual((staging / 'answer.json').read_text(), '{\"answer\":\"original\"}')

    def test_missing_final_run_event_does_not_fabricate_runtime(self):
        with tempfile.TemporaryDirectory() as root:
            fixture, source, scored = archives(root)
            events = [json.loads(line) for line in (source / 'events.jsonl').read_text().splitlines()]
            path = source / 'events.jsonl'
            path.chmod(0o644)
            path.write_bytes(b''.join(canonical_json(event) for event in events
                                      if event['type'] != 'run_completed'))
            with self.assertRaisesRegex(ValueError, 'timing|events'):
                ReportWriter().write(fixture, source, scored, Path(root) / 'reports')
            self.assertFalse((Path(root) / 'reports').exists())


if __name__ == '__main__':
    unittest.main()
