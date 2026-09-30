import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from almm_fixture.engine import EVALUATION_SEED_START, adapter_view, generate
from almm_harness.budget import BudgetVerifier
from almm_harness.errors import AdapterFailure, ProviderFailure
from almm_harness.proxy import ModelProxy
from almm_harness.runner import FixtureRunner


def manifest(run_id='harness-test'):
    return {'runId': run_id,
            'adapter': {'name': 'compact-test', 'revision': 'abc123', 'contractVersion': '1.0'},
            'model': {'provider': 'local', 'name': 'test-model', 'version': 'immutable-1',
                      'decoding': {'temperature': 0}, 'seed': 42},
            'tokenizer': {'name': 'characters-smoke', 'version': '1'},
            'stablePrefix': ['contract'], 'seed': 42, 'scorerVersion': 'unscored-1',
            'harnessHash': 'harness-revision', 'rateLimitRpm': 1000000000,
            'probeTimeoutSeconds': 2}


class CompactAdapter:
    def __init__(self, proxy):
        self.proxy = proxy

    def initialize(self, run_manifest):
        self.manifest = copy.deepcopy(run_manifest)
        self.turn_ids = []
        self.probe_counts = []
        self.sequence = 0
        self.answers = []

    def respond(self, text, source_id, field):
        self.sequence += 1
        request = {'requestId': f"{self.manifest['runId']}:{self.sequence}",
                   'segments': [{'tier': 'stable', 'content': 'contract', 'tokenCount': 8},
                                {'tier': 'unstable', 'content': text, 'tokenCount': len(text),
                                 'sourceIds': [source_id]}]}
        answer = self.proxy(request)
        self.answers.append(answer)
        return {field: answer, 'requests': [request]}

    def handleTurn(self, turn):
        if set(turn) != {'turnId', 'sessionId', 'role', 'text'}:
            raise AssertionError('scorer fields reached adapter')
        result = self.respond(turn['text'], turn['turnId'], 'response')
        self.turn_ids.append(turn['turnId'])
        return result

    def answerProbe(self, probe):
        if set(probe) != {'probeId', 'question'}:
            raise AssertionError('scorer fields reached adapter')
        self.probe_counts.append(len(self.turn_ids))
        return self.respond(probe['question'], probe['probeId'], 'answer')


def make_runner(directory, configuration=None, provider=None, adapter_class=CompactAdapter):
    configuration = configuration or manifest()
    provider = provider or (lambda payload: {'answer': 'a raw model answer', 'inputTokens': 1,
                                           'modelVersion': 'immutable-1'})
    proxy = ModelProxy(configuration, BudgetVerifier(configuration, len), provider)
    adapter = adapter_class(proxy)
    return FixtureRunner(configuration, adapter, proxy, directory), adapter, proxy


def records(directory, name):
    return [json.loads(line) for line in (Path(directory) / name).read_text().splitlines()]


class HarnessTests(unittest.TestCase):
    def test_adapter_file_preflight_rejects_gold_before_initialization_or_artifacts(self):
        fixture = generate(42, 10)
        for field in ('expected', 'acceptedAnswers', 'requiredFactIds', 'forbiddenFactIds',
                      'goldEvidenceIds', 'matchType', 'facts', 'introducedFactIds'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                view = adapter_view(fixture)
                view['probes'][0][field] = {'nested': fixture['probes'][0]['expected']}
                adapter_path = root / 'adapter.json'
                adapter_path.write_text(json.dumps(view))
                artifacts = root / 'runs'
                runner, adapter, proxy = make_runner(artifacts)
                with patch.object(adapter, 'initialize') as initialize:
                    with self.assertRaises(ValueError):
                        runner.run(fixture, adapter_input_path=adapter_path)
                initialize.assert_not_called()
                self.assertEqual(proxy.telemetry, [])
                self.assertFalse(artifacts.exists())

    def test_adapter_file_must_match_fixture_and_reject_unknown_nested_fields(self):
        fixture = generate(42, 10)
        candidates = [adapter_view(generate(43, 10)), adapter_view(fixture)]
        candidates[1]['sessions'][0]['turns'][0]['metadata'] = {'matchType': 'exact'}
        for view in candidates:
            with self.subTest(view=view), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                adapter_path = root / 'adapter.json'
                adapter_path.write_text(json.dumps(view))
                runner, adapter, _ = make_runner(root / 'runs')
                with patch.object(adapter, 'initialize') as initialize:
                    with self.assertRaises(ValueError):
                        runner.run(fixture, adapter_input_path=adapter_path)
                initialize.assert_not_called()
                self.assertFalse((root / 'runs').exists())

    def test_duplicate_adapter_keys_cannot_hide_gold_and_folders_are_rejected(self):
        fixture = generate(42, 10)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter_path = root / 'adapter.json'
            clean = json.dumps(adapter_view(fixture))
            adapter_path.write_text('{"probes":[{"expected":{}}],' + clean[1:])
            for path in (adapter_path, root):
                runner, adapter, _ = make_runner(root / 'runs')
                with patch.object(adapter, 'initialize') as initialize:
                    with self.assertRaises(ValueError):
                        runner.run(fixture, adapter_input_path=path)
                initialize.assert_not_called()
                self.assertFalse((root / 'runs').exists())

    def test_private_held_out_split_runs_without_scorer_fields_reaching_adapter(self):
        fixture = generate(EVALUATION_SEED_START, 10, held_out=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            scorer_path = root / 'scorer.json'
            scorer_path.write_text(json.dumps(fixture))
            scorer_path.chmod(0o600)
            adapter_path = root / 'adapter.json'
            adapter_path.write_text(json.dumps(adapter_view(fixture)))
            adapter_path.chmod(0o600)
            runner, adapter, _ = make_runner(root / 'runs')
            result = runner.run(fixture, adapter_input_path=adapter_path)
            self.assertTrue(result['results']['valid'])
            self.assertEqual(result['results']['answeredCount'], 5)
            self.assertEqual(adapter.probe_counts, [100] * 5)

    def test_held_out_scorer_records_in_adapter_file_are_not_publishable(self):
        fixture = generate(EVALUATION_SEED_START, 10, held_out=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            for name in ('scorer.json', 'adapter.json'):
                path = root / name
                path.write_text(json.dumps(fixture))
                path.chmod(0o600)
            runner, adapter, proxy = make_runner(root / 'runs')
            with patch.object(adapter, 'initialize') as initialize:
                with self.assertRaisesRegex(ValueError, 'scorer-only'):
                    runner.run(fixture, adapter_input_path=root / 'adapter.json')
            initialize.assert_not_called()
            self.assertEqual(proxy.telemetry, [])
            self.assertFalse((root / 'runs').exists())

    def test_held_out_split_must_match_scorer_fixture_and_private_parent(self):
        fixture = generate(EVALUATION_SEED_START, 10, held_out=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            adapter_path = root / 'adapter.json'
            adapter_path.write_text(json.dumps(adapter_view(fixture)))
            adapter_path.chmod(0o600)
            scorer_path = root / 'scorer.json'
            scorer_path.write_text(json.dumps(
                generate(EVALUATION_SEED_START + 1, 10, held_out=True)))
            scorer_path.chmod(0o600)
            runner, adapter, _ = make_runner(root / 'runs')
            with patch.object(adapter, 'initialize') as initialize:
                with self.assertRaisesRegex(ValueError, 'scorer file does not match'):
                    runner.run(fixture, adapter_input_path=adapter_path)
                root.chmod(0o755)
                with self.assertRaisesRegex(ValueError, 'private permissions'):
                    runner.run(fixture, adapter_input_path=adapter_path)
            initialize.assert_not_called()
            self.assertFalse((root / 'runs').exists())

    def test_held_out_scorer_permissions_and_symlink_targets_are_checked(self):
        fixture = generate(EVALUATION_SEED_START, 10, held_out=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            adapter_path = root / 'adapter.json'
            adapter_path.write_text(json.dumps(adapter_view(fixture)))
            adapter_path.chmod(0o600)
            scorer_path = root / 'scorer.json'
            scorer_path.write_text(json.dumps(fixture))
            scorer_path.chmod(0o644)
            runner, adapter, _ = make_runner(root / 'runs')
            with patch.object(adapter, 'initialize') as initialize:
                with self.assertRaisesRegex(ValueError, 'private permissions'):
                    runner.run(fixture, adapter_input_path=adapter_path)
            initialize.assert_not_called()
            scorer_path.unlink()
            scorer_path.symlink_to(Path(__file__).resolve())
            with self.assertRaisesRegex(ValueError, 'outside the public repository'):
                runner.run(fixture, adapter_input_path=adapter_path)
            self.assertFalse((root / 'runs').exists())

    def test_run_log_preserves_session_turn_probe_failure_diagnostics_without_secrets(self):
        key = 'private-log-provider-key'
        def provider(payload):
            request_id = payload['request']['requestId']
            if request_id.endswith(':101'):
                raise RuntimeError(
                    f'provider refused request; Authorization: Basic basic-private-value; '
                    f'x-api-key=header-private-value; environment={key}')
            return {'answer': 'ok'}
        provider.api_key_env = 'CUSTOM_PROVIDER_AUTH'
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                'os.environ', {'CUSTOM_PROVIDER_AUTH': key}):
            runner, _, _ = make_runner(directory, provider=provider)
            result = runner.run(generate(42, 10))
            events = records(result['artifactDir'], 'events.jsonl')
            self.assertEqual([event['sessionId'] for event in events
                              if event['type'] == 'session_started'],
                             [f's-{index:04d}' for index in range(1, 11)])
            self.assertEqual([event['sessionId'] for event in events
                              if event['type'] == 'session_completed'],
                             [f's-{index:04d}' for index in range(1, 11)])
            self.assertEqual([event['turnId'] for event in events if event['type'] == 'turn'],
                             [f't-{session:04d}-{turn:02d}' for session in range(1, 11)
                              for turn in range(1, 11)])
            failed = next(event for event in events
                          if event['type'] == 'probe' and event['status'] == 'error')
            self.assertEqual(failed['category'], 'provider')
            self.assertEqual(failed['requestIds'], ['harness-test:101'])
            self.assertIn('provider refused request', failed['error'])
            for name in ('events.jsonl', 'probes.jsonl', 'requests.jsonl', 'dead-letters.jsonl'):
                contents = (Path(result['artifactDir']) / name).read_text()
                for secret in (key, 'basic-private-value', 'header-private-value'):
                    self.assertNotIn(secret, contents)

    def test_generated_fixture_runs_chronologically_and_preserves_raw_artifacts(self):
        fixture = generate(42, 10)
        with tempfile.TemporaryDirectory() as directory:
            runner, adapter, proxy = make_runner(directory)
            result = runner.run(fixture)
            self.assertEqual(adapter.turn_ids,
                             [turn['turnId'] for session in fixture['sessions']
                              for turn in session['turns']])
            self.assertEqual(adapter.probe_counts, [100] * 5)
            self.assertEqual(result['results']['answeredCount'], 5)
            self.assertEqual(result['results']['accuracyDenominator'], 5)
            self.assertFalse(result['results']['investigationRequired'])
            probes = records(result['artifactDir'], 'probes.jsonl')
            self.assertTrue(all(probe['answer'] == 'a raw model answer' for probe in probes))
            self.assertTrue(all(probe['eligibleForAccuracy'] for probe in probes))
            self.assertEqual(probes[0]['sourceIds'], [fixture['probes'][0]['probeId']])
            final_manifest = result['manifest']
            self.assertEqual(final_manifest['fixtureHash'], fixture['contentHash'])
            self.assertEqual(final_manifest['model'], manifest()['model'])
            self.assertEqual(final_manifest['tokenizer'], manifest()['tokenizer'])
            self.assertTrue(final_manifest['timestamp'].endswith('Z'))
            requests = records(result['artifactDir'], 'requests.jsonl')
            self.assertEqual(len(requests), 105)
            self.assertTrue(all('tierTokens' in request and 'totalTokens' in request
                                for request in requests))
            before = (Path(result['artifactDir']) / 'manifest.json').read_bytes()
            with self.assertRaises(FileExistsError):
                runner.run(fixture)
            self.assertEqual((Path(result['artifactDir']) / 'manifest.json').read_bytes(), before)
            self.assertEqual(len(proxy.telemetry), 105)

    def test_checkpoint_resume_replays_fifty_sessions_and_truncates_partial_outputs(self):
        fixture = generate(42, 100)
        interrupted = False
        def provider(payload):
            nonlocal interrupted
            if payload['request']['segments'][-1].get('sourceIds') == ['t-0051-03']:
                interrupted = True
                raise KeyboardInterrupt('simulated process interruption')
            return {'answer': 'ok'}
        with tempfile.TemporaryDirectory() as directory:
            runner, _, _ = make_runner(directory, provider=provider)
            with self.assertRaises(KeyboardInterrupt):
                runner.run(fixture)
            self.assertTrue(interrupted)
            checkpoint_path = next(Path(directory).glob('*/checkpoint.json'))
            checkpoint = json.loads(checkpoint_path.read_text())
            self.assertEqual(checkpoint['lastCompletedSession'], 50)
            resumed_calls = []
            def resumed_provider(payload):
                resumed_calls.append(payload['request']['requestId'])
                return {'answer': 'different nondeterministic answer'}
            resumed, adapter, _ = make_runner(directory, provider=resumed_provider)
            result = resumed.run(fixture, resume=True)
            self.assertEqual(adapter.turn_ids,
                             [turn['turnId'] for session in fixture['sessions']
                              for turn in session['turns']])
            events = records(result['artifactDir'], 'events.jsonl')
            turns = [event['turnId'] for event in events if event['type'] == 'turn']
            self.assertEqual(len(turns), 1000)
            self.assertEqual(len(set(turns)), 1000)
            self.assertEqual(len(records(result['artifactDir'], 'probes.jsonl')), 10)
            self.assertEqual(len(records(result['artifactDir'], 'requests.jsonl')), 1010)
            self.assertTrue(any(event['type'] == 'resume' and event['nextSession'] == 51
                                for event in events))
            self.assertEqual(result['results']['answeredCount'], 10)
            self.assertEqual(len(resumed_calls), 505)
            self.assertEqual(adapter.answers[:505], ['ok'] * 505)
            self.assertEqual(adapter.answers[505:], ['different nondeterministic answer'] * 505)

    def test_complete_request_budget_boundary_is_enforced_at_probe(self):
        class Boundary(CompactAdapter):
            def answerProbe(self, probe):
                number = int(probe['probeId'].split('-')[1])
                if number in (1, 2):
                    return self.respond('x' * (24991 + number), probe['probeId'], 'answer')
                return super().answerProbe(probe)
        with tempfile.TemporaryDirectory() as directory:
            runner, _, _ = make_runner(directory, adapter_class=Boundary)
            result = runner.run(generate(42, 10))
            probes = records(result['artifactDir'], 'probes.jsonl')
            self.assertEqual(probes[0]['status'], 'ok')
            self.assertEqual(probes[0]['requests'][0]['totalTokens'], 25000)
            self.assertEqual(probes[1]['category'], 'budget')
            self.assertEqual(probes[1]['requests'][0]['totalTokens'], 25001)
            self.assertEqual(result['results']['accuracyDenominator'], 4)
            self.assertEqual(result['results']['incompleteCount'], 0)
            self.assertFalse(result['results']['investigationRequired'])
            letters = records(result['artifactDir'], 'dead-letters.jsonl')
            self.assertEqual(letters[0]['request']['totalTokens'], 25001)
            self.assertEqual(letters[0]['error']['category'], 'budget')

    def test_corrupt_checkpoint_restarts_without_duplicate_outputs(self):
        fixture = generate(42, 10)
        def interrupted_provider(payload):
            if payload['request']['segments'][-1].get('sourceIds') == ['t-0002-03']:
                raise KeyboardInterrupt()
            return {'answer': 'ok'}
        with tempfile.TemporaryDirectory() as directory:
            runner, _, _ = make_runner(directory, provider=interrupted_provider)
            with self.assertRaises(KeyboardInterrupt):
                runner.run(fixture)
            checkpoint = next(Path(directory).glob('*/checkpoint.json'))
            checkpoint.write_text('{broken checkpoint')
            resumed, adapter, _ = make_runner(directory)
            result = resumed.run(fixture, resume=True)
            self.assertEqual(len(adapter.turn_ids), 100)
            events = records(result['artifactDir'], 'events.jsonl')
            self.assertEqual(len([event for event in events if event['type'] == 'turn']), 100)
            self.assertEqual(len(records(result['artifactDir'], 'requests.jsonl')), 105)
            self.assertEqual(result['manifest']['recovery']['checkpointSession'], 0)

    def test_resume_rejects_changed_configuration_or_fixture(self):
        fixture = generate(42, 100)
        def provider(payload):
            if payload['request']['segments'][-1].get('sourceIds') == ['t-0051-01']:
                raise KeyboardInterrupt()
            return {'answer': 'ok'}
        with tempfile.TemporaryDirectory() as directory:
            runner, _, _ = make_runner(directory, provider=provider)
            with self.assertRaises(KeyboardInterrupt):
                runner.run(fixture)
            changed = manifest()
            changed['model']['decoding']['temperature'] = 0.2
            resumed, _, _ = make_runner(directory, changed)
            with self.assertRaisesRegex(ValueError, 'identity|checkpoint'):
                resumed.run(fixture, resume=True)
            resumed, _, _ = make_runner(directory)
            with self.assertRaisesRegex(ValueError, 'identity|checkpoint'):
                resumed.run(generate(43, 100), resume=True)

    def test_failure_categories_do_not_enter_accuracy_denominator(self):
        class Failures(CompactAdapter):
            def answerProbe(self, probe):
                number = int(probe['probeId'].split('-')[1])
                if number == 1:
                    return self.respond('x' * 25000, probe['probeId'], 'answer')
                if number == 2:
                    raise ProviderFailure('provider unavailable')
                if number == 3:
                    raise RuntimeError('adapter bug')
                if number == 4:
                    return {'answer': 'unproxied', 'requests': []}
                return super().answerProbe(probe)
        with tempfile.TemporaryDirectory() as directory:
            runner, _, _ = make_runner(directory, adapter_class=Failures)
            result = runner.run(generate(42, 10))
            self.assertEqual(result['results']['failureCounts'],
                             {'budget': 1, 'provider': 1, 'adapter': 2, 'timeout': 0})
            self.assertEqual(result['results']['answeredCount'], 1)
            self.assertEqual(result['results']['accuracyDenominator'], 1)
            self.assertTrue(result['results']['investigationRequired'])
            self.assertTrue(result['results']['valid'])
            probes = records(result['artifactDir'], 'probes.jsonl')
            self.assertEqual([p['eligibleForAccuracy'] for p in probes],
                             [False, False, False, False, True])

    def test_fabricated_requests_cannot_be_scored(self):
        class Fabricated(CompactAdapter):
            def answerProbe(self, probe):
                result = super().answerProbe(probe)
                result['requests'][0]['segments'][-1]['content'] = 'not the model input'
                return result
        with tempfile.TemporaryDirectory() as directory:
            runner, _, _ = make_runner(directory, adapter_class=Fabricated)
            result = runner.run(generate(42, 10))
            self.assertEqual(result['results']['answeredCount'], 0)
            self.assertEqual(result['results']['failureCounts']['adapter'], 5)

    def test_timeout_terminates_fixture_and_forbids_reusing_mutable_adapter(self):
        release = threading.Event()
        exited = threading.Event()
        class Slow(CompactAdapter):
            def answerProbe(self, probe):
                try:
                    release.wait(2)
                    return super().answerProbe(probe)
                finally:
                    exited.set()
        configuration = manifest()
        configuration['probeTimeoutSeconds'] = 0.02
        with tempfile.TemporaryDirectory() as directory:
            runner, adapter, proxy = make_runner(directory, configuration, adapter_class=Slow)
            try:
                result = runner.run(generate(42, 100))
                self.assertEqual(len(adapter.turn_ids), 110)
                self.assertEqual(result['results']['failureCounts']['timeout'], 1)
                self.assertEqual(result['results']['answeredCount'], 0)
                self.assertEqual(result['results']['status'], 'terminated')
                with self.assertRaises(AdapterFailure):
                    runner.run(generate(42, 10))
                new_runner = FixtureRunner(manifest('different'), adapter, proxy, directory)
                with self.assertRaises(AdapterFailure):
                    new_runner.run(generate(42, 10))
            finally:
                release.set()
                self.assertTrue(exited.wait(2))


if __name__ == '__main__':
    unittest.main()
