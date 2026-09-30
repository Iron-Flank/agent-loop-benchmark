import copy
import unittest

from almm_adapter.contract import validate_manifest, validate_response
from almm_adapter.conformance import ConformanceError, check_adapter, sample_manifest
from almm_adapter.reference import ReferenceAdapter, smoke_model


class AdapterTests(unittest.TestCase):
    def test_reference_conforms(self):
        checks = check_adapter(ReferenceAdapter(smoke_model, len))
        self.assertIn('state reset', checks)
        self.assertIn('native exact replay', checks)

    def test_missing_tier_identifies_segment(self):
        class MissingTier(ReferenceAdapter):
            def handleTurn(self, turn):
                response = super().handleTurn(turn)
                del response['requests'][0]['tierSegments'][0]['tier']
                return response
        with self.assertRaisesRegex(ConformanceError, r'tierSegments\[0\].tier'):
            check_adapter(MissingTier(smoke_model, len))

    def test_reset_removes_history_and_telemetry(self):
        requests = []
        def model(request):
            requests.append(copy.deepcopy(request))
            return {'content': 'response', 'finishReason': 'stop',
                    'usage': {'promptTokens': 1, 'completionTokens': 1}}
        adapter = ReferenceAdapter(model, len)
        adapter.initialize(sample_manifest())
        adapter.handleTurn({'turnId': 'old-turn', 'sessionId': 'old-session',
                            'role': 'user', 'text': 'secret old fact'})
        adapter.answerProbe({'probeId': 'old-probe', 'question': 'old question'})
        adapter.initialize(sample_manifest('second'))
        self.assertEqual(adapter.getRequestTelemetry(), [])
        adapter.answerProbe({'probeId': 'new-probe', 'question': 'new question'})
        self.assertNotIn('secret old fact', str(requests[-1]))
        self.assertNotIn('old-session', str(requests[-1]))
        self.assertNotIn('old-turn', str(requests[-1]))
        self.assertNotIn('old question', str(requests[-1]))
        self.assertEqual(requests[-1]['requestId'], 'second:1')

    def test_proxy_receives_complete_context_and_source_ids(self):
        requests = []
        def model(request):
            requests.append(copy.deepcopy(request))
            return {'content': 'model response', 'finishReason': 'stop',
                    'usage': {'promptTokens': 1, 'completionTokens': 1}}
        adapter = ReferenceAdapter(model, len)
        adapter.initialize(sample_manifest())
        result = adapter.handleTurn({'turnId': 't1', 'role': 'user', 'text': 'a fact'})
        probe = adapter.answerProbe({'probeId': 'p1', 'question': 'recall?'})
        self.assertEqual(result['response'], 'model response')
        self.assertEqual(probe['answer'], 'model response')
        self.assertEqual(probe['requests'][0], requests[-1])
        self.assertIn({'role': 'user', 'content': 'a fact'}, requests[-1]['messages'])
        self.assertIn({'role': 'assistant', 'content': 'model response'}, requests[-1]['messages'])
        self.assertTrue(any(segment.get('sourceIds') == ['t1']
                            for segment in requests[-1]['tierSegments']))
        self.assertEqual(adapter.getRequestTelemetry(), requests)

    def test_manifest_missing_or_incompatible_version_is_rejected(self):
        for version in (None, '1.0'):
            manifest = sample_manifest()
            if version is None:
                del manifest['adapter']['contractVersion']
            else:
                manifest['adapter']['contractVersion'] = version
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, 'contractVersion'):
                validate_manifest(manifest)

    def test_uninitialized_calls_fail(self):
        adapter = ReferenceAdapter(smoke_model, len)
        with self.assertRaisesRegex(ValueError, 'initialize'):
            adapter.answerProbe({'probeId': 'p1', 'question': 'question'})
        with self.assertRaisesRegex(ValueError, 'initialize'):
            adapter.getRequestTelemetry()

    def test_response_rejects_malformed_segment_fields(self):
        adapter = ReferenceAdapter(smoke_model, len)
        adapter.initialize(sample_manifest())
        result = adapter.handleTurn({'turnId': 't1', 'role': 'user', 'text': 'hello'})
        for field, value in [('tier', 'invalid'), ('content', 3), ('tokenCount', -1),
                             ('tokenCount', True), ('sourceIds', 't1')]:
            bad = copy.deepcopy(result)
            bad['requests'][0]['tierSegments'][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field):
                validate_response(bad, 'response')

    def test_gold_records_rejected_at_adapter_boundary(self):
        adapter = ReferenceAdapter(smoke_model, len)
        adapter.initialize(sample_manifest())
        with self.assertRaisesRegex(ValueError, 'expected'):
            adapter.answerProbe({'probeId': 'p1', 'question': '?',
                                 'expected': {'acceptedAnswers': ['secret']}})

    def test_conformance_detects_leaking_state(self):
        class LeakingAdapter(ReferenceAdapter):
            def initialize(self, manifest):
                old = getattr(self, '_history', [])
                super().initialize(manifest)
                self._history = old
        with self.assertRaisesRegex(ConformanceError, 'prior-run'):
            check_adapter(LeakingAdapter(smoke_model, len))

    def test_tool_only_and_structured_assistant_history_survive_without_emulation(self):
        requests = []
        responses = [
            {'content': None, 'toolCalls': [
                {'id': 'unknown-call', 'name': 'external_memory',
                 'arguments': {'fact': 'blue'}}],
             'finishReason': 'tool_calls', 'usage': {'promptTokens': 1, 'completionTokens': 1}},
            {'content': [{'type': 'text', 'text': 'visible text'},
                         {'type': 'image_url', 'image_url': {'url': 'https://example.invalid/image'}}],
             'finishReason': 'stop', 'usage': {'promptTokens': 1, 'completionTokens': 1}},
            {'content': 'done', 'finishReason': 'stop',
             'usage': {'promptTokens': 1, 'completionTokens': 1}}]
        def model(request):
            requests.append(copy.deepcopy(request))
            return copy.deepcopy(responses[len(requests) - 1])
        adapter = ReferenceAdapter(model, len)
        adapter.initialize(sample_manifest())
        tool_only = adapter.handleTurn({'turnId': 't1', 'role': 'user', 'text': 'remember blue'})
        self.assertEqual(tool_only['response'], '')
        structured = adapter.handleTurn({'turnId': 't2', 'role': 'user', 'text': 'show me'})
        self.assertEqual(structured['response'], 'visible text')
        adapter.answerProbe({'probeId': 'p', 'question': 'continue'})
        assistants = [message for message in requests[-1]['messages'] if message['role'] == 'assistant']
        self.assertEqual(assistants, [
            {'role': 'assistant', 'content': None, 'toolCalls': responses[0]['toolCalls']},
            {'role': 'assistant', 'content': responses[1]['content']}])
        self.assertFalse(any(message['role'] == 'tool' for message in requests[-1]['messages']))


if __name__ == '__main__':
    unittest.main()
