"""Offline CI smoke must retain complete accounting, not just exit successfully."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from almm_harness.__main__ import main
from almm_harness.smoke import smoke
from almm_report.__main__ import load_report


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class CISmokeTests(unittest.TestCase):
    def test_offline_pipeline_retains_report_and_complete_accounting(self):
        with tempfile.TemporaryDirectory() as temporary:
            # Neither the runtime nor judge requires a user-supplied credential.
            with patch.dict('os.environ', {}, clear=True):
                summary = smoke(Path(temporary) / 'output')
            artifacts = summary['artifacts']
            self.assertEqual(json.loads(Path(artifacts['summary']).read_text()), summary)
            fixture = json.loads(Path(artifacts['fixture']).read_text())
            run = Path(artifacts['run'])
            scored = rows(Path(artifacts['scores']) / 'scores.jsonl')
            report = load_report(artifacts['report'])
            turns = [event for event in rows(run / 'events.jsonl') if event['type'] == 'turn']
            self.assertEqual([event['turnId'] for event in turns], [
                turn['turnId'] for session in fixture['sessions'] for turn in session['turns']])
            self.assertTrue(all(event['status'] == 'ok' for event in turns))
            self.assertEqual(len(turns), 100)
            self.assertEqual(len(fixture['sessions']), 10)
            self.assertEqual((summary['sessions'], summary['turns'], summary['probes'],
                              summary['requests']), (10, 100, 5, 105))
            self.assertEqual(len(rows(run / 'requests.jsonl')), 105)
            self.assertEqual({row['probeId'] for row in scored},
                             {probe['probeId'] for probe in fixture['probes']})
            self.assertTrue(all(row['eligibleForAccuracy'] for row in scored))
            self.assertEqual(sum(row['normalizedResult']['correct'] for row in scored), 1)
            self.assertEqual(report['report']['accuracy'], {
                'correct': 1, 'answered': 5, 'total': 5, 'accuracy': 0.2, 'completionRate': 1.0})
            self.assertEqual(report['report']['abstention']['correctAbstentions'], 1)
            self.assertEqual(report['report']['abstention']['falseAbstentions'], 4)
            self.assertEqual(report['report']['operationalTelemetry']['requestCount'], 105)
            self.assertFalse(report['manifest']['canonical'])
            self.assertFalse(summary['canonical'])
            self.assertIn('synthetic', summary['calibrationProvenance'])
            self.assertLess(summary['elapsedSeconds'], 120)

    def test_failed_model_requests_fail_cli_and_retain_diagnostics(self):
        def failed_provider(payload):
            raise RuntimeError('offline provider failed')

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'output'
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch('almm_harness.smoke._model_transport', new=failed_provider):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    status = main(['--smoke', '--output', str(output)])
            self.assertEqual(status, 1)
            self.assertEqual(stdout.getvalue(), '')
            self.assertIn('FAIL:', stderr.getvalue())
            archives = list(output.glob('smoke-*/runs/*'))
            self.assertEqual(len(archives), 1)
            requests = rows(archives[0] / 'requests.jsonl')
            self.assertTrue(all(request['status'] == 'error' for request in requests))
            self.assertEqual(len(requests), 105)
            self.assertFalse(list(output.glob('smoke-*/reports/*/manifest.json')))
            self.assertFalse(list(output.glob('smoke-*/summary.json')))
            failure = json.loads(next(output.glob('smoke-*/failure.json')).read_text())
            self.assertEqual(failure['stage'], 'execution')
            self.assertIn('failed', failure['error'])


if __name__ == '__main__':
    unittest.main()
