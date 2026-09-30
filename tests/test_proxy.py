import copy
import io
import json
import os
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from almm_harness.budget import BudgetVerifier
from almm_adapter.native import canonical_wire

from almm_harness.errors import AdapterFailure, BudgetFailure, ProviderFailure, TimeoutFailure
from almm_harness.proxy import ModelProxy, OpenAICompatibleProvider, ProviderHTTPError, redact


def manifest(rate=60):
    return {'runId': 'proxy-test', 'stablePrefix': ['fixed'],
            'nativeModelEnvelopeVersion': '1.0',
            'tokenizer': {'name': 'characters-smoke', 'version': '1'},
            'model': {'provider': 'test', 'name': 'model', 'version': 'model-2026',
                      'decoding': {'temperature': 0}, 'seed': 7},
            'rateLimitRpm': rate}


def request():
    config = manifest()['model']
    return {'requestId': 'r1', 'messages': [
        {'role': 'system', 'content': 'fixed'},
        {'role': 'user', 'content': 'question'}],
        'tierSegments': [
            {'tier': 'stable', 'content': 'fixed', 'tokenCount': 5, 'messageIndices': [0]},
            {'tier': 'unstable', 'content': 'question', 'tokenCount': 8, 'messageIndices': [1]}],
        'model': config, 'decodingSettings': copy.deepcopy(config['decoding']), 'seed': 7}


def response(content='ok', prompt_tokens=15, **metadata):
    return {'content': content, 'finishReason': 'stop',
            'usage': {'promptTokens': prompt_tokens, 'completionTokens': 3}, **metadata}


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
    def test_redaction_masks_header_values_in_objects_pairs_and_error_strings(self):
        value = {'headers': {'Authorization': 'Basic private-basic',
                             'X-API-Key': 'private-api-key'},
                 'pairs': [['Proxy-Authorization', 'Digest private-digest']],
                 'error': 'denied; Authorization: Basic private-basic; '
                          'api_key="private-api-key"; requestId=r-17',
                 'requestId': 'r-17', 'category': 'provider'}
        original = copy.deepcopy(value)
        masked = redact(value, [])
        self.assertEqual(masked['headers']['Authorization'], '[REDACTED]')
        self.assertEqual(masked['pairs'][0][1], '[REDACTED]')
        self.assertEqual(masked['requestId'], 'r-17')
        self.assertEqual(masked['category'], 'provider')
        self.assertIn('denied', masked['error'])
        self.assertIn('requestId=r-17', masked['error'])
        for secret in ('private-basic', 'private-api-key', 'private-digest'):
            self.assertNotIn(secret, json.dumps(masked))
        self.assertEqual(value, original)

    def test_custom_provider_key_environment_is_redacted_in_errors(self):
        key = 'custom-provider-credential'
        class Provider:
            api_key_env = 'CUSTOM_PROVIDER_AUTH'

            def __call__(self, payload):
                raise RuntimeError('denied with ' + key)

        with patch.dict('os.environ', {'CUSTOM_PROVIDER_AUTH': key}):
            proxy, _ = self.proxy(Provider())
            with self.assertRaises(ProviderFailure) as raised:
                proxy(request())
            self.assertNotIn(key, str(raised.exception))
            self.assertNotIn(key, json.dumps(proxy.telemetry))
            self.assertNotIn(key, json.dumps(proxy.dead_letters))
            self.assertEqual(proxy.telemetry[0]['requestId'], 'r1')
            self.assertEqual(proxy.telemetry[0]['category'], 'provider')
            self.assertIn('denied', proxy.telemetry[0]['error']['message'])

    def proxy(self, provider, rate=60):
        clock = FakeClock()
        proxy = ModelProxy(manifest(rate), BudgetVerifier(manifest(rate), len), provider,
                           clock=clock, sleep=clock.sleep)
        return proxy, clock

    def test_budget_failure_is_archived_without_provider_call(self):
        calls = []
        proxy, _ = self.proxy(lambda payload: calls.append(payload))
        value = request()
        value['messages'][1]['content'] = 'x' * 25_000
        value['tierSegments'][1].update(content='x' * 25_000, tokenCount=1)
        with self.assertRaises(BudgetFailure):
            proxy(value)
        self.assertEqual(calls, [])
        self.assertEqual(proxy.telemetry[0]['category'], 'budget')
        self.assertEqual(proxy.dead_letters[0]['request']['totalTokens'], len(canonical_wire(value)))
        self.assertEqual(proxy.dead_letters[0]['error']['category'], 'budget')

    def test_changed_stable_prefix_is_adapter_failure_before_provider(self):
        calls = []
        proxy, _ = self.proxy(lambda payload: calls.append(payload))
        value = request()
        value['messages'][0]['content'] = 'changed'
        value['tierSegments'][0]['content'] = 'changed'
        with self.assertRaises(AdapterFailure):
            proxy(value)
        self.assertEqual(calls, [])
        self.assertEqual(proxy.dead_letters[0]['error']['category'], 'adapter')

    def test_identical_requests_are_not_cached_and_config_is_pinned(self):
        calls = []
        def provider(payload):
            calls.append(copy.deepcopy(payload))
            payload['model']['version'] = 'mutated'
            return response(str(len(calls)), modelVersion='model-2026')
        config = manifest()
        clock = FakeClock()
        proxy = ModelProxy(config, BudgetVerifier(config, len), provider, clock=clock, sleep=clock.sleep)
        config['model']['version'] = 'changed'
        self.assertEqual(proxy(request())['content'], '1')
        self.assertEqual(proxy(request())['content'], '2')
        self.assertEqual([call['model']['version'] for call in calls], ['model-2026'] * 2)
        self.assertEqual([row['providerTokens'] for row in proxy.telemetry], [15, 15])
        self.assertEqual(proxy.dead_letters, [])

    def test_retryable_errors_back_off_then_recover(self):
        attempts = []
        def provider(payload):
            attempts.append(payload)
            if len(attempts) < 3:
                raise ProviderHTTPError(429 if len(attempts) == 1 else 503, 'try later')
            return response('recovered')
        proxy, clock = self.proxy(provider)
        self.assertEqual(proxy(request())['content'], 'recovered')
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
            return response('at the boundary')
        proxy, clock = self.proxy(provider, rate=2)
        self.assertEqual(proxy(request())['content'], 'at the boundary')
        self.assertEqual(len(calls), 3)
        self.assertAlmostEqual(sum(clock.waits), 30)
        self.assertAlmostEqual(proxy.telemetry[0]['retryHistory'][1]['rateWaitSeconds'], 27)

    def test_token_bucket_refills_and_begin_run_does_not_reset_quota(self):
        proxy, clock = self.proxy(lambda payload: response(), rate=2)
        proxy(request())
        proxy(request())
        proxy.begin_run()
        self.assertEqual(proxy.telemetry, [])
        self.assertEqual(proxy(request())['content'], 'ok')
        self.assertEqual(clock.waits, [30])
        self.assertEqual(proxy.telemetry[0]['waitSeconds'], 30)

    def test_initial_low_rate_wait_is_not_subject_to_retry_wait_cap(self):
        proxy, clock = self.proxy(lambda payload: response('throttled'), rate=1)
        self.assertEqual(proxy(request())['content'], 'throttled')
        self.assertEqual(proxy(request())['content'], 'throttled')
        self.assertEqual(clock.waits, [60])
        self.assertEqual(proxy.telemetry[1]['waitSeconds'], 60)
        self.assertEqual(proxy.dead_letters, [])

    def test_provider_specific_rate_and_independent_proxies(self):
        proxy, clock = self.proxy(lambda payload: response(), rate={'test': 2})
        other, other_clock = self.proxy(lambda payload: response(), rate=2)
        proxy(request())
        proxy(request())
        other(request())
        self.assertEqual(clock.waits, [])
        self.assertEqual(other_clock.waits, [])

    def test_nonretryable_http_and_invalid_results_are_provider_failures(self):
        cases = [lambda payload: (_ for _ in ()).throw(ProviderHTTPError(401, 'denied')),
                 lambda payload: (_ for _ in ()).throw(RuntimeError('network failed')),
                 lambda payload: response(3),
                 lambda payload: response(prompt_tokens=True),
                 lambda payload: response(modelVersion='different'),
                 lambda payload: {'answer': 'old envelope'},
                 lambda payload: {'content': 'ok', 'finishReason': 'stop'}]
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
        proxy, _ = self.proxy(lambda payload: response('wrong', prompt_tokens=99,
                                                     modelVersion='unpinned'))
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
            value['messages'][1]['content'] = 'credential ' + key
            value['tierSegments'][1]['content'] = 'credential ' + key
            with self.assertRaises(ProviderFailure) as raised:
                proxy(value)
            archive = json.dumps([proxy.telemetry, proxy.dead_letters])
            self.assertNotIn(key, archive)
            self.assertNotIn(key, str(raised.exception))
            self.assertIn('[REDACTED]', archive)
            self.assertEqual(proxy.dead_letters[0]['request']['tierSegments'][0]['content'], 'fixed')

    def test_abort_prevents_new_calls_and_poisoned_inflight_answer(self):
        entered = threading.Event()
        release = threading.Event()
        errors = []
        def provider(payload):
            entered.set()
            release.wait(2)
            return response('late')
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
            return response(payload['request']['requestId'], prompt_tokens=13)
        proxy, _ = self.proxy(provider)
        answers = []
        def worker(index):
            value = request()
            value['requestId'] = str(index)
            try:
                answers.append(proxy(value)['content'])
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

    def test_recovery_replay_reuses_responses_without_provider_or_quota(self):
        calls = []
        def provider(payload):
            calls.append(payload)
            return response('stored answer')
        proxy, clock = self.proxy(provider, rate=2)
        proxy(request())
        records = copy.deepcopy(proxy.telemetry)
        proxy.begin_run()
        proxy.begin_replay(records)
        self.assertEqual(proxy(request()), response('stored answer'))
        proxy.end_replay()
        self.assertEqual(len(calls), 1)
        self.assertEqual(proxy.telemetry, records)
        self.assertEqual(clock.waits, [])
        proxy.begin_run()
        self.assertEqual(proxy(request()), response('stored answer'))
        self.assertEqual(len(calls), 2)
        self.assertEqual(clock.waits, [])

    def test_replay_divergence_and_incomplete_consumption_are_adapter_failures(self):
        proxy, _ = self.proxy(lambda payload: response('stored'))
        proxy(request())
        records = copy.deepcopy(proxy.telemetry)
        proxy.begin_run()
        proxy.begin_replay(records)
        with self.assertRaisesRegex(AdapterFailure, 'consume'):
            proxy.end_replay()
        changed = request()
        changed['messages'][1]['content'] = 'divergent input'
        changed['tierSegments'][1]['content'] = 'divergent input'
        with self.assertRaisesRegex(AdapterFailure, 'differs'):
            proxy(changed)
        proxy.begin_run()
        self.assertEqual(proxy(request()), response('stored'))

    def test_replay_rejects_old_or_invalid_envelopes(self):
        proxy, _ = self.proxy(lambda payload: response('stored'))
        for records in ([{'status': 'error', 'response': response('stored')}],
                        [{'status': 'ok'}], [{'status': 'ok', 'answer': 'legacy'}],
                        [{'status': 'ok', 'response': {'content': 'invalid'}}]):
            with self.subTest(records=records), self.assertRaises(AdapterFailure):
                proxy.begin_replay(records)

    def test_live_requests_cannot_spoof_archive_metadata(self):
        calls = []
        proxy, _ = self.proxy(lambda payload: calls.append(payload))
        for key, value in (('response', response()), ('answer', 'forged'),
                           ('providerTokens', 99), ('status', 'ok')):
            native = request()
            native[key] = value
            with self.subTest(key=key), self.assertRaises(AdapterFailure):
                proxy(native)
        self.assertEqual(calls, [])

    def test_tool_only_response_archives_and_replays_without_flattening_or_aliasing(self):
        native = response(None, finishReason='tool_calls', toolCalls=[
            {'id': 'call-1', 'name': 'lookup', 'arguments': {'query': '雪', 'limit': 2}}])
        calls = []
        def provider(payload):
            calls.append(payload)
            return native
        proxy, clock = self.proxy(provider)
        first = proxy(request())
        self.assertEqual(first, native)
        self.assertEqual(proxy.telemetry[0]['response'], native)
        self.assertNotIn('answer', proxy.telemetry[0])
        first['toolCalls'][0]['arguments']['query'] = 'mutated'
        self.assertEqual(proxy.telemetry[0]['response']['toolCalls'][0]['arguments']['query'], '雪')
        records = copy.deepcopy(proxy.telemetry)
        proxy.begin_run()
        proxy.begin_replay(records)
        records[0]['response']['toolCalls'][0]['arguments']['query'] = 'outside mutation'
        replayed = proxy(request())
        proxy.end_replay()
        self.assertEqual(replayed, native)
        replayed['toolCalls'][0]['arguments']['query'] = 'replay mutation'
        self.assertEqual(proxy.telemetry[0]['response'], native)
        self.assertEqual(len(calls), 1)
        self.assertEqual(clock.waits, [])

    def test_replay_rejects_removed_native_request_fields(self):
        proxy, _ = self.proxy(lambda payload: response())
        original = request()
        original['tools'] = [{
            'name': 'lookup', 'description': 'Find a value',
            'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}}}]
        original['tierSegments'].append({
            'tier': 'unstable', 'content': json.dumps(original['tools'][0]),
            'tokenCount': 1, 'toolNames': ['lookup']})
        proxy(original)
        records = copy.deepcopy(proxy.telemetry)
        proxy.begin_run()
        proxy.begin_replay(records)
        with self.assertRaisesRegex(AdapterFailure, 'differs'):
            proxy(request())

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
            proxy, _ = self.proxy(lambda payload: response('answer with ' + key))
            proxy(request())
            self.assertEqual(proxy.telemetry[0]['response']['content'], 'answer with [REDACTED]')


class HTTPProviderTests(unittest.TestCase):
    def payload(self):
        return {'model': manifest()['model'], 'request': request()}

    def test_real_transport_preserves_roles_content_and_pinned_model(self):
        def opener(req, timeout):
            body = json.loads(req.data)
            self.assertEqual(body['model'], 'model-2026')
            self.assertEqual(body['messages'], request()['messages'])
            self.assertEqual(body['temperature'], 0)
            self.assertEqual(body['seed'], 7)
            self.assertEqual(req.data.decode('utf-8'), canonical_wire(request()))
            self.assertEqual(req.get_header('Authorization'), 'Bearer environment-only')
            return io.BytesIO(json.dumps({
                'choices': [{'message': {'content': 'answer'}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 14, 'completion_tokens': 3},
                'model': 'model-2026'}).encode())
        with patch.dict('os.environ', {'TEST_API_KEY': 'environment-only'}):
            provider = OpenAICompatibleProvider(api_key_env='TEST_API_KEY', opener=opener)
            self.assertEqual(provider(self.payload()), response(
                'answer', prompt_tokens=14, modelVersion='model-2026'))

    def test_transport_preserves_tool_schema_structured_content_and_multiturn_history(self):
        tool = {'name': 'lookup', 'description': 'Look up Unicode text',
                'parameters': {'type': 'object', 'properties': {
                    'query': {'type': 'string'}}, 'required': ['query'],
                    'additionalProperties': False}}
        call = {'id': 'call-1', 'name': 'lookup', 'arguments': {'query': '雪'}}
        payload = self.payload()
        native = payload['request']
        native['tools'] = [tool]
        native['toolCalls'] = [call]
        native['toolResults'] = [{'toolCallId': 'call-1', 'result': {'value': 7}}]
        native['messages'] = [
            {'role': 'system', 'content': 'fixed'},
            {'role': 'user', 'content': [{'type': 'text', 'text': '雪'}, {
                'type': 'image_url', 'image_url': {'url': 'https://example.test/image.png'}}]},
            {'role': 'assistant', 'content': 'Looking it up', 'toolCalls': [call]},
            {'role': 'tool', 'content': '{"value":7}', 'toolCallId': 'call-1'},
            {'role': 'assistant', 'content': 'Found seven'},
            {'role': 'user', 'content': 'Explain the result'}]
        def opener(req, timeout):
            body = json.loads(req.data)
            self.assertEqual(body['tools'], [{'type': 'function', 'function': tool}])
            self.assertEqual(body['messages'][1], native['messages'][1])
            self.assertEqual(body['messages'][2], {
                'role': 'assistant', 'content': 'Looking it up', 'tool_calls': [{
                    'id': 'call-1', 'type': 'function', 'function': {
                        'name': 'lookup', 'arguments': '{"query":"雪"}'}}]})
            self.assertEqual(body['messages'][3], {
                'role': 'tool', 'content': '{"value":7}', 'tool_call_id': 'call-1'})
            self.assertEqual(body['messages'][4:], native['messages'][4:])
            self.assertEqual(req.data.decode('utf-8'), canonical_wire(native))
            return io.BytesIO(json.dumps({
                'choices': [{'message': {'content': 'Seven was found', 'tool_calls': [{
                    'id': 'call-2', 'type': 'function', 'function': {
                        'name': 'lookup', 'arguments': '{"query":"more","limit":2}'}}]},
                    'finish_reason': 'tool_calls'}],
                'usage': {'prompt_tokens': 44, 'completion_tokens': 8},
                'model': 'model-2026'}).encode())
        with patch.dict('os.environ', {'TEST_API_KEY': 'environment-only'}):
            provider = OpenAICompatibleProvider(api_key_env='TEST_API_KEY', opener=opener)
            result = provider(payload)
        self.assertEqual(result, {
            'content': 'Seven was found', 'finishReason': 'tool_calls',
            'toolCalls': [{'id': 'call-2', 'name': 'lookup',
                           'arguments': {'query': 'more', 'limit': 2}}],
            'usage': {'promptTokens': 44, 'completionTokens': 8},
            'modelVersion': 'model-2026'})

    def test_transport_preserves_tool_only_assistant_history_and_native_results(self):
        payload = self.payload()
        native = payload['request']
        call = {'id': 'call-1', 'name': 'lookup', 'arguments': {'query': 'snow'}}
        native['messages'].extend([
            {'role': 'assistant', 'content': None, 'toolCalls': [call]},
            {'role': 'tool', 'content': 'found snow', 'toolCallId': 'call-1'}])
        native['toolCalls'] = [call]
        native['toolResults'] = [{'toolCallId': 'call-1', 'result': 'found snow'}]
        native['tierSegments'][1]['messageIndices'] = [1, 2, 3]
        def opener(req, timeout):
            messages = json.loads(req.data)['messages']
            self.assertIsNone(messages[2]['content'])
            self.assertEqual(messages[2]['tool_calls'][0]['function'], {
                'name': 'lookup', 'arguments': '{"query":"snow"}'})
            self.assertEqual(messages[3], {
                'role': 'tool', 'content': 'found snow', 'tool_call_id': 'call-1'})
            self.assertEqual(len(messages), 4)
            return io.BytesIO(json.dumps({
                'choices': [{'message': {'content': 'The lookup found snow'},
                             'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 30, 'completion_tokens': 5}}).encode())
        with patch.dict('os.environ', {'TEST_API_KEY': 'environment-only'}):
            result = OpenAICompatibleProvider(
                api_key_env='TEST_API_KEY', opener=opener)(payload)
        self.assertEqual(result['content'], 'The lookup found snow')

    def test_transport_accepts_tool_only_and_structured_response_content(self):
        calls = [{'id': 'call-1', 'type': 'function',
                  'function': {'name': 'lookup', 'arguments': '{"query":"雪"}'}}]
        for content in (None, [{'type': 'text', 'text': 'Searching'}]):
            with self.subTest(content=content):
                def opener(req, timeout):
                    return io.BytesIO(json.dumps({
                        'choices': [{'message': {'content': content, 'tool_calls': calls},
                                     'finish_reason': 'tool_calls'}],
                        'usage': {'prompt_tokens': 20, 'completion_tokens': 5}}).encode())
                with patch.dict('os.environ', {'TEST_API_KEY': 'environment-only'}):
                    result = OpenAICompatibleProvider(
                        api_key_env='TEST_API_KEY', opener=opener)(self.payload())
                self.assertEqual(result['content'], content)
                self.assertEqual(result['toolCalls'], [{
                    'id': 'call-1', 'name': 'lookup', 'arguments': {'query': '雪'}}])
                self.assertEqual(result['finishReason'], 'tool_calls')
                self.assertEqual(result['usage'], {'promptTokens': 20, 'completionTokens': 5})

    def test_transport_rejects_invalid_usage_and_unstructured_tool_arguments(self):
        valid = {
            'choices': [{'message': {'content': 'answer'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 20, 'completion_tokens': 5}}
        malformed = []
        for usage in (None, {'prompt_tokens': 20}, {
                'prompt_tokens': True, 'completion_tokens': 5}):
            value = copy.deepcopy(valid)
            value['usage'] = usage
            malformed.append(value)
        for arguments in ('not-json', '"string"', '[1,2]'):
            value = copy.deepcopy(valid)
            value['choices'][0]['message']['tool_calls'] = [{
                'id': 'call-1', 'type': 'function', 'function': {
                    'name': 'lookup', 'arguments': arguments}}]
            malformed.append(value)
        for value in malformed:
            with self.subTest(value=value), patch.dict('os.environ', {'TEST_API_KEY': 'key'}):
                provider = OpenAICompatibleProvider(
                    api_key_env='TEST_API_KEY',
                    opener=lambda req, timeout: io.BytesIO(json.dumps(value).encode()))
                with self.assertRaises(ProviderFailure):
                    provider(self.payload())

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
