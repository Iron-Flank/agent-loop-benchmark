import copy
import unittest

from almm_adapter.contract import validate_native_response
from almm_adapter.native import canonical_wire, wire_request, response_text
from almm_harness.budget import BudgetVerifier
from almm_harness.errors import BudgetFailure, AdapterFailure


def manifest():
    return {'nativeModelEnvelopeVersion': '1.0',
            'model': {'provider': 'local', 'name': 'model', 'version': 'pinned',
                      'decoding': {'temperature': 0}, 'seed': 42},
            'tokenizer': {'name': 'characters-smoke', 'version': '1'},
            'stablePrefix': ['contract']}


def request(text='hello'):
    return {'requestId': 'r1', 'model': copy.deepcopy(manifest()['model']),
            'decodingSettings': {'temperature': 0}, 'seed': 42,
            'messages': [{'role': 'system', 'content': 'contract'},
                         {'role': 'user', 'content': text}],
            'tierSegments': [
                {'tier': 'stable', 'content': 'contract', 'tokenCount': 0,
                 'messageIndices': [0]},
                {'tier': 'unstable', 'content': text, 'tokenCount': 0,
                 'messageIndices': [1], 'sourceIds': ['t1']}]}


def sized_request(size):
    value = request('')
    value['messages'][1]['content'] = 'x' * (size - len(canonical_wire(value)))
    value['tierSegments'][1]['content'] = value['messages'][1]['content']
    return value


class BudgetTests(unittest.TestCase):
    def test_exact_ceiling_and_forged_counts(self):
        verifier = BudgetVerifier(manifest(), len)
        value = sized_request(25000)
        result = verifier.verify(value)
        self.assertEqual(result['totalTokens'], 25000)
        self.assertGreater(result['tierTokens']['unstable'], 0)
        self.assertEqual(result['tierSegments'][-1]['sourceIds'], ['t1'])
        self.assertEqual(value['tierSegments'][0]['tokenCount'], 0)
        with self.assertRaises(BudgetFailure):
            verifier.verify(sized_request(25001))

    def test_forged_tier_cannot_hide_changed_system_prefix(self):
        value = request()
        value['messages'][0]['content'] = 'changed!'
        with self.assertRaises(AdapterFailure):
            BudgetVerifier(manifest(), len).verify(value)

    def test_prefix_must_really_be_first_system_message(self):
        value = request()
        value['messages'][0]['role'] = 'user'
        with self.assertRaises(AdapterFailure):
            BudgetVerifier(manifest(), len).verify(value)

    def test_pinned_fields_cannot_change(self):
        for field, replacement in [('model', {}), ('decodingSettings', {'temperature': 1}),
                                   ('decodingSettings', {'temperature': False}), ('seed', 7)]:
            with self.subTest(field=field):
                value = request()
                value[field] = replacement
                with self.assertRaises(AdapterFailure):
                    BudgetVerifier(manifest(), len).verify(value)

    def test_each_message_and_tool_requires_unique_attribution(self):
        for indices in [[], [0, 1], [1, 1], [2]]:
            with self.subTest(indices=indices):
                value = request()
                value['tierSegments'][1]['messageIndices'] = indices
                with self.assertRaises(AdapterFailure):
                    BudgetVerifier(manifest(), len).verify(value)
        value = request()
        value['tools'] = [{'name': 'remember', 'description': '', 'parameters': {}}]
        with self.assertRaises(AdapterFailure):
            BudgetVerifier(manifest(), len).verify(value)

    def test_full_wire_tokenization_and_overhead_control_ceiling(self):
        config = manifest()
        config['tokenizer']['requestOverheadTokens'] = 7
        verifier = BudgetVerifier(config, len)
        self.assertEqual(verifier.verify(sized_request(24993))['totalTokens'], 25000)
        with self.assertRaises(BudgetFailure):
            verifier.verify(sized_request(24994))
        wire = canonical_wire(request())
        with self.assertRaises(BudgetFailure):
            BudgetVerifier(manifest(), lambda text: 25001 if text == wire else 0).verify(request())

    def test_schema_and_native_tool_history_count_without_emulation(self):
        value = request()
        value['tools'] = [{'name': 'remember', 'description': 'Store',
                           'parameters': {'type': 'object', 'description': 'x' * 25000}}]
        value['messages'] += [
            {'role': 'assistant', 'content': None, 'toolCalls': [
                {'id': 'c1', 'name': 'remember', 'arguments': {'fact': 'blue'}}]},
            {'role': 'tool', 'toolCallId': 'c1', 'content': 'stored'}]
        value['tierSegments'] += [
            {'tier': 'unstable', 'content': '', 'tokenCount': 0,
             'toolNames': ['remember']},
            {'tier': 'unstable', 'content': '', 'tokenCount': 0,
             'messageIndices': [2, 3]}]
        with self.assertRaises(BudgetFailure):
            BudgetVerifier(manifest(), len).verify(value)
        wire = wire_request(value)
        self.assertIsNone(wire['messages'][2]['content'])
        self.assertEqual(wire['messages'][2]['tool_calls'][0]['function']['arguments'],
                         '{"fact":"blue"}')
        self.assertEqual(wire['messages'][3]['tool_call_id'], 'c1')

    def test_unattached_or_conflicting_top_level_payload_rejected(self):
        value = request()
        value['toolCalls'] = [{'id': 'c1', 'name': 'remember', 'arguments': {}}]
        with self.assertRaises(AdapterFailure):
            BudgetVerifier(manifest(), len).verify(value)
        value = request()
        value['toolResults'] = [{'toolCallId': 'c1', 'result': 'stored'}]
        with self.assertRaises(AdapterFailure):
            BudgetVerifier(manifest(), len).verify(value)

    def test_linked_top_level_history_is_not_duplicated_and_conflicts_fail(self):
        value = request()
        call = {'id': 'c1', 'name': 'remember', 'arguments': {'fact': 'blue'}}
        value['messages'] += [
            {'role': 'assistant', 'content': None, 'toolCalls': [call]},
            {'role': 'tool', 'toolCallId': 'c1', 'content': '{"value": 7}'}]
        value['tierSegments'].append(
            {'tier': 'unstable', 'content': '', 'tokenCount': 0,
             'messageIndices': [2, 3]})
        original_wire = canonical_wire(value)
        value['toolCalls'] = [copy.deepcopy(call)]
        value['toolResults'] = [{'toolCallId': 'c1', 'result': {'value': 7}}]
        self.assertEqual(canonical_wire(value), original_wire)
        result = BudgetVerifier(manifest(), len).verify(value)
        self.assertEqual(result['toolResults'], value['toolResults'])
        value['toolResults'][0]['result']['value'] = 8
        with self.assertRaises(AdapterFailure):
            BudgetVerifier(manifest(), len).verify(value)

    def test_ambiguous_calls_and_out_of_order_results_fail(self):
        call = {'id': 'c1', 'name': 'remember', 'arguments': {}}
        for history in [
            [{'role': 'assistant', 'content': None, 'toolCalls': [call, call]}],
            [{'role': 'tool', 'toolCallId': 'c1', 'content': 'stored'},
             {'role': 'assistant', 'content': None, 'toolCalls': [call]}]]:
            with self.subTest(history=history):
                value = request()
                value['messages'] += history
                value['tierSegments'].append(
                    {'tier': 'unstable', 'content': '', 'tokenCount': 0,
                     'messageIndices': list(range(2, len(value['messages'])))})
                with self.assertRaises(AdapterFailure):
                    BudgetVerifier(manifest(), len).verify(value)

    def test_native_json_and_roles_are_strict(self):
        mutations = [
            lambda value: value['messages'][1].update(role='developer'),
            lambda value: value['messages'][1].update(content=None),
            lambda value: value['messages'][1].update(content=[{'type': 'text', 'text': 3}]),
            lambda value: value['decodingSettings'].update(temperature=float('nan')),
            lambda value: value.update(seed=True),
            lambda value: value['messages'][1].update(toolCallId='unattached')]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                value = request()
                mutate(value)
                with self.assertRaises(AdapterFailure):
                    BudgetVerifier(manifest(), len).verify(value)

    def test_structured_content_preserved_and_its_payload_controls_budget(self):
        value = request()
        parts = [{'type': 'text', 'text': 'hello'},
                 {'type': 'image_url', 'image_url': {'url': 'x' * 25000}}]
        value['messages'][1]['content'] = parts
        with self.assertRaises(BudgetFailure) as caught:
            BudgetVerifier(manifest(), len).verify(value)
        self.assertEqual(caught.exception.request['messages'][1]['content'], parts)
        self.assertEqual(wire_request(value)['messages'][1]['content'], parts)

    def test_invalid_tokenizer_output_rejected(self):
        with self.assertRaises(ValueError):
            BudgetVerifier(manifest(), lambda text: -1).verify(request())

    def test_response_tool_calls_and_structured_text_remain_separate(self):
        response = {'content': None, 'toolCalls': [
            {'id': 'c1', 'name': 'remember', 'arguments': {'fact': 'blue'}}],
            'finishReason': 'tool_calls',
            'usage': {'promptTokens': 12, 'completionTokens': 3}}
        validate_native_response(response)
        self.assertEqual(response_text(response), '')
        response['content'] = [{'type': 'text', 'text': 'Stored.'},
                               {'type': 'image_url', 'image_url': {'url': 'native'}}]
        self.assertEqual(response_text(response), 'Stored.')
        self.assertEqual(response['toolCalls'][0]['arguments'], {'fact': 'blue'})
        del response['usage']
        with self.assertRaises(ValueError):
            validate_native_response(response)

    def test_response_requires_valid_usage_and_structured_arguments(self):
        response = {'content': 'hello', 'finishReason': 'stop',
                    'usage': {'promptTokens': 12, 'completionTokens': 3}}
        for count in [-1, True, 1.5]:
            with self.subTest(count=count):
                invalid = copy.deepcopy(response)
                invalid['usage']['promptTokens'] = count
                with self.assertRaises(ValueError):
                    validate_native_response(invalid)
        response['toolCalls'] = [{'id': 'c1', 'name': 'remember', 'arguments': '{}'}]
        with self.assertRaises(ValueError):
            validate_native_response(response)
