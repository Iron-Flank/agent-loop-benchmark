import copy
import io
import json
import os
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from almm_harness.budget import BudgetVerifier

from almm_harness.errors import AdapterFailure, BudgetFailure, ProviderFailure, TimeoutFailure
from almm_harness.proxy import ModelProxy, OpenAICompatibleProvider, ProviderHTTPError


def manifest(rate=60):
    return {'runId': 'proxy-test', 'stablePrefix': ['fixed'],
            'tokenizer': {'name': 'characters-smoke', 'version': '1'},
            'model': {'provider': 'test', 'name': 'model', 'version': 'model-2026',
                      'decoding': {'temperature': 0}, 'seed': 7},
            'rateLimitRpm': rate}


def request():
    return {'requestId': 'r1', 'segments': [
        {'tier': 'stable', 'content': 'fixed', 'tokenCount': 5},
        {'tier': 'unstable', 'content': 'question', 'tokenCount': 8}]}


class FakeClock:
    def __init__(self):
        self.now = 0
        self.waits = []
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            return self.now

    def sleep(self, duration):
        with self.lock:
            self.waits.append(duration)
            self.now += duration


class ProxyTests(unittest.TestCase):
    def proxy(self, provider, rate=60):
        clock = FakeClock()
        proxy = ModelProxy(manifest(rate), BudgetVerifier(manifest(rate), len), provider,
                           clock=clock, sleep=clock.sleep)
        return proxy, clock

    def test_budget_failure_is_archived_without_provider_call(self):
        calls = []
        proxy, _ = self.proxy(lambda payload: calls.append(payload))
        value = request()
        value['segments'][1].update(content='x' * 25_000, tokenCount=1)
        with self.assertRaises(BudgetFailure):
            proxy(value)
        self.assertEqual(calls, [])
        self.assertEqual(proxy.telemetry[0]['category'], 'budget')
        self.assertEqual(proxy.dead_letters[0]['request']['totalTokens'], 25_005)
        self.assertEqual(proxy.dead_letters[0]['request']['segments'][1]['tokenCount'], 25_000)
        self.assertEqual(proxy.dead_letters[0]['error']['category'], 'budget')

    def test_changed_stable_prefix_is_adapter_failure_before_provider(self):
        calls = []
        proxy, _ = self.proxy(lambda payload: calls.append(payload))
        value = request()
        value['segments'][0]['content'] = 'changed'
        with self.assertRaises(AdapterFailure):
            proxy(value)
        self.assertEqual(calls, [])
        self.assertEqual(proxy.dead_letters[0]['error']['category'], 'adapter')

    def test_identical_requests_are_not_cached_and_config_is_pinned(self):
        calls = []
        def provider(payload):
            calls.append(copy.deepcopy(payload))
            payload['model']['version'] = 'mutated'
            return {'answer': str(len(calls)), 'inputTokens': 15,
                    'modelVersion': 'model-2026'}
        config = manifest()
        clock = FakeClock()
        proxy = ModelProxy(config, BudgetVerifier(config, len), provider, clock=clock, sleep=clock.sleep)
        config['model']['version'] = 'changed'
        self.assertEqual(proxy(request()), '1')
        self.assertEqual(proxy(request()), '2')
        self.assertEqual([call['model']['version'] for call in calls], ['model-2026'] * 2)
        self.assertEqual([row['providerTokens'] for row in proxy.telemetry], [15, 15])
        self.assertEqual(proxy.telemetry[0]['totalTokens'], 13)
        self.assertEqual(proxy.dead_letters, [])

    def test_retryable_errors_back_off_then_recover(self):
        attempts = []
        def provider(payload):
            attempts.append(payload)
            if len(attempts) < 3:
                raise ProviderHTTPError(429 if len(attempts) == 1 else 503, 'try later')
            return {'answer': 'recovered'}
        proxy, clock = self.proxy(provider)
        self.assertEqual(proxy(request()), 'recovered')
        self.assertEqual(clock.waits, [1, 2])
        self.assertEqual(proxy.telemetry[0]['attempts'], 3)
        self.assertEqual([row['status'] for row in proxy.telemetry[0]['retryHistory']],
                         [429, 503, 'ok'])
        self.assertEqual(proxy.telemetry[0]['latencyMs'], 3000)

    def test_only_three_retries_and_full_error_history_is_retained(self):
        proxy, clock = self.proxy(lambda payload: (_ for _ in ()).throw(
            ProviderHTTPError(503, 'overloaded')))
        with self.assertRaises(ProviderFailure):
            proxy(request())
        self.assertEqual(clock.waits, [1, 2, 4])
        letter = proxy.dead_letters[0]
        self.assertEqual(len(letter['retryHistory']), 4)
        self.assertEqual(letter['error']['status'], 503)
        self.assertEqual(letter['manifest']['runId'], 'proxy-test')
        self.assertEqual(letter['request']['totalTokens'], 13)

    def test_retry_rate_wait_cannot_exceed_total_wait_budget(self):
        calls = []
        def provider(payload):
            calls.append(payload)
            raise ProviderHTTPError(429, 'busy')
        proxy, clock = self.proxy(provider, rate=1)
        with self.assertRaises(ProviderFailure):
            proxy(request())
        self.assertEqual(len(calls), 1)
        self.assertLessEqual(sum(clock.waits), 30)
        self.assertIn('wait', proxy.dead_letters[0]['error']['message'])

    def test_backoff_and_rate_wait_can_recover_at_exact_wait_boundary(self):
        calls = []
        def provider(payload):
            calls.append(payload)
            if len(calls) < 3:
                raise ProviderHTTPError(429, 'busy')
            return {'answer': 'at the boundary'}
        proxy, clock = self.proxy(provider, rate=2)
        self.assertEqual(proxy(request()), 'at the boundary')
        self.assertEqual(len(calls), 3)
        self.assertAlmostEqual(sum(clock.waits), 30)
        self.assertAlmostEqual(proxy.telemetry[0]['retryHistory'][1]['rateWaitSeconds'], 27)

    def test_token_bucket_refills_and_begin_run_does_not_reset_quota(self):
        proxy, clock = self.proxy(lambda payload: {'answer': 'ok'}, rate=2)
        proxy(request())
        proxy(request())
        proxy.begin_run()
        self.assertEqual(proxy.telemetry, [])
        self.assertEqual(proxy(request()), 'ok')
        self.assertEqual(clock.waits, [30])
        self.assertEqual(proxy.telemetry[0]['waitSeconds'], 30)

    def test_initial_low_rate_wait_is_not_subject_to_retry_wait_cap(self):
        proxy, clock = self.proxy(lambda payload: {'answer': 'throttled'}, rate=1)
        self.assertEqual(proxy(request()), 'throttled')
        self.assertEqual(proxy(request()), 'throttled')
        self.assertEqual(clock.waits, [60])
        self.assertEqual(proxy.telemetry[1]['waitSeconds'], 60)
        self.assertEqual(proxy.dead_letters, [])

    def test_provider_specific_rate_and_independent_proxies(self):
        proxy, clock = self.proxy(lambda payload: {'answer': 'ok'}, rate={'test': 2})
        other, other_clock = self.proxy(lambda payload: {'answer': 'ok'}, rate=2)
        proxy(request())
        proxy(request())
        other(request())
        self.assertEqual(clock.waits, [])
        self.assertEqual(other_clock.waits, [])

    def test_nonretryable_http_and_invalid_results_are_provider_failures(self):
        cases = [lambda payload: (_ for _ in ()).throw(ProviderHTTPError(401, 'denied')),
                 lambda payload: (_ for _ in ()).throw(RuntimeError('network failed')),
                 lambda payload: {'answer': 3},
                 lambda payload: {'answer': 'ok', 'inputTokens': True},
                 lambda payload: {'answer': 'ok', 'modelVersion': 'different'}]
        for provider in cases:
            with self.subTest(provider=provider):
                proxy, clock = self.proxy(provider)
                with self.assertRaises(ProviderFailure):
                    proxy(request())
                self.assertEqual(proxy.telemetry[0]['status'], 'error')
                self.assertEqual(proxy.telemetry[0]['category'], 'provider')
                self.assertEqual(clock.waits, [])
                self.assertEqual(len(proxy.dead_letters), 1)

    def test_version_mismatch_keeps_reported_provider_usage(self):
        proxy, _ = self.proxy(lambda payload: {'answer': 'wrong', 'inputTokens': 99,
                                              'modelVersion': 'unpinned'})
        with self.assertRaises(ProviderFailure):
            proxy(request())
        self.assertEqual(proxy.telemetry[0]['providerTokens'], 99)
        self.assertEqual(proxy.telemetry[0]['modelVersion'], 'unpinned')

    def test_all_archives_and_exceptions_redact_environment_credentials(self):
        key = 'private-provider-value'
        with patch.dict('os.environ', {'TEST_API_KEY': key}):
            proxy, _ = self.proxy(lambda payload: (_ for _ in ()).throw(
                ProviderHTTPError(401, 'Bearer ' + key)))
            value = request()
            value['segments'][1]['content'] = 'credential ' + key
            value['authorization'] = key
            with self.assertRaises(ProviderFailure) as raised:
                proxy(value)
            archive = json.dumps([proxy.telemetry, proxy.dead_letters])
            self.assertNotIn(key, archive)
            self.assertNotIn(key, str(raised.exception))
            self.assertIn('[REDACTED]', archive)
            self.assertEqual(proxy.dead_letters[0]['request']['segments'][0]['content'], 'fixed')

    def test_abort_prevents_new_calls_and_poisoned_inflight_answer(self):
        entered = threading.Event()
        release = threading.Event()
        errors = []
        def provider(payload):
            entered.set()
            release.wait(2)
            return {'answer': 'late'}
        proxy, _ = self.proxy(provider)
        def worker():
            try:
                proxy(request())
            except Exception as exc:
                errors.append(exc)
        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(entered.wait(2))
        try:
            with self.assertRaises(ProviderFailure):
                proxy.begin_run()
            proxy.abort_run()
            with self.assertRaises(TimeoutFailure):
                proxy(request())
        finally:
            release.set()
            thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(errors[0], TimeoutFailure)

    def test_concurrent_calls_have_independent_payloads_and_complete_accounting(self):
        barrier = threading.Barrier(4)
        errors = []
        def provider(payload):
            barrier.wait(2)
            return {'answer': payload['request']['requestId'], 'inputTokens': 13}
        proxy, _ = self.proxy(provider)
        answers = []
        def worker(index):
            value = request()
            value['requestId'] = str(index)
            try:
                answers.append(proxy(value))
            except Exception as exc:
                errors.append(exc)
        threads = [threading.Thread(target=worker, args=(index,)) for index in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(3)
        self.assertEqual(errors, [])
        self.assertEqual(set(answers), {'0', '1', '2', '3'})
        self.assertEqual({row['requestId'] for row in proxy.telemetry}, set(answers))
        self.assertEqual(sum(row['providerTokens'] for row in proxy.telemetry), 52)

    def test_recovery_replay_reuses_answers_without_provider_or_quota(self):
        calls = []
        def provider(payload):
            calls.append(payload)
            return {'answer': 'stored answer'}
        proxy, clock = self.proxy(provider, rate=2)
        proxy(request())
        records = copy.deepcopy(proxy.telemetry)
        proxy.begin_run()
        proxy.begin_replay(records)
        self.assertEqual(proxy(request()), 'stored answer')
        proxy.end_replay()
        self.assertEqual(len(calls), 1)
        self.assertEqual(proxy.telemetry, records)
        self.assertEqual(clock.waits, [])
        proxy.begin_run()
        self.assertEqual(proxy(request()), 'stored answer')
        self.assertEqual(len(calls), 2)
        self.assertEqual(clock.waits, [])

    def test_replay_divergence_and_incomplete_consumption_are_adapter_failures(self):
        proxy, _ = self.proxy(lambda payload: {'answer': 'stored'})
        proxy(request())
        records = copy.deepcopy(proxy.telemetry)
        proxy.begin_run()
        proxy.begin_replay(records)
        with self.assertRaisesRegex(AdapterFailure, 'consume'):
            proxy.end_replay()
        changed = request()
        changed['segments'][1]['content'] = 'divergent input'
        with self.assertRaisesRegex(AdapterFailure, 'differs'):
            proxy(changed)
        proxy.begin_run()
        self.assertEqual(proxy(request()), 'stored')

    def test_replay_requires_successful_records_and_archived_answers(self):
        proxy, _ = self.proxy(lambda payload: {'answer': 'stored'})
        for records in ([{'status': 'error', 'answer': 'stored'}], [{'status': 'ok'}]):
            with self.subTest(records=records), self.assertRaises(AdapterFailure):
                proxy.begin_replay(records)

    def test_successful_answer_and_rotated_credentials_are_redacted(self):
        key = 'credential-that-rotates'
        with patch.dict('os.environ', {'TEST_API_KEY': key}):
            def provider(payload):
                os.environ['TEST_API_KEY'] = 'replacement-credential'
                raise RuntimeError('failed using ' + key)
            proxy, _ = self.proxy(provider)
            with self.assertRaises(ProviderFailure) as raised:
                proxy(request())
            self.assertNotIn(key, str(raised.exception))
            self.assertNotIn(key, json.dumps(proxy.dead_letters))
        with patch.dict('os.environ', {'TEST_API_KEY': key}):
            proxy, _ = self.proxy(lambda payload: {'answer': 'answer with ' + key})
            proxy(request())
            self.assertEqual(proxy.telemetry[0]['answer'], 'answer with [REDACTED]')


class HTTPProviderTests(unittest.TestCase):
    def payload(self):
        return {'model': manifest()['model'], 'request': request()}

    def test_real_transport_uses_exact_content_and_pinned_model(self):
        def opener(req, timeout):
            body = json.loads(req.data)
            if (body['model'] != 'model-2026' or
                    body['messages'] != [{'role': 'user', 'content': 'fixedquestion'}] or
                    req.get_header('Authorization') != 'Bearer environment-only'):
                return io.BytesIO(json.dumps({'choices': [{'message': {'content': 'bad request'}}]}).encode())
            return io.BytesIO(json.dumps({'choices': [{'message': {'content': 'answer'}}],
                                         'usage': {'prompt_tokens': 14},
                                         'model': 'model-2026'}).encode())
        with patch.dict('os.environ', {'TEST_API_KEY': 'environment-only'}):
            provider = OpenAICompatibleProvider(api_key_env='TEST_API_KEY', opener=opener)
            self.assertEqual(provider(self.payload()), {'answer': 'answer', 'inputTokens': 14,
                                                         'modelVersion': 'model-2026'})

    def test_missing_key_never_sends_a_request(self):
        calls = []
        with patch.dict('os.environ', {}, clear=True):
            provider = OpenAICompatibleProvider(api_key_env='MISSING_KEY',
                                                opener=lambda *args, **kwargs: calls.append(args))
            with self.assertRaises(ProviderFailure):
                provider(self.payload())
        self.assertEqual(calls, [])

    def test_http_error_is_retryable_at_proxy_boundary_and_redacted(self):
        def opener(req, timeout):
            raise HTTPError(req.full_url, 429, 'secret-http-key', {},
                            io.BytesIO(b'{"error":"secret-http-key"}'))
        with patch.dict('os.environ', {'TEST_API_KEY': 'secret-http-key'}):
            provider = OpenAICompatibleProvider(api_key_env='TEST_API_KEY', opener=opener)
            with self.assertRaises(ProviderHTTPError) as raised:
                provider(self.payload())
        self.assertEqual(raised.exception.status, 429)
        self.assertNotIn('secret-http-key', str(raised.exception))

    def test_network_timeout_keeps_timeout_category(self):
        def opener(req, timeout):
            raise URLError(TimeoutError('timed out'))
        with patch.dict('os.environ', {'TEST_API_KEY': 'key'}):
            provider = OpenAICompatibleProvider(api_key_env='TEST_API_KEY', opener=opener)
            proxy = ModelProxy(manifest(), BudgetVerifier(manifest(), len), provider)
            with self.assertRaises(TimeoutFailure):
                proxy(request())
        self.assertEqual(proxy.dead_letters[0]['error']['category'], 'timeout')

    def test_http_provider_rejects_secret_urls_and_insecure_remote_origins(self):
        for endpoint in ('http://remote.example/v1/chat/completions',
                         'https://user:pass@api.example/v1/chat/completions',
                         'https://api.example/v1/chat/completions?key=secret'):
            with self.subTest(endpoint=endpoint), self.assertRaises(ProviderFailure):
                OpenAICompatibleProvider(endpoint=endpoint)


if __name__ == '__main__':
    unittest.main()
