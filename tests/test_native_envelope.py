"""Native memory operations survive the model boundary and replay."""
import copy
import unittest

from almm_harness.budget import BudgetVerifier
from almm_harness.proxy import ModelProxy


def manifest():
    return {'runId': 'native', 'nativeModelEnvelopeVersion': '1.0',
            'stablePrefix': ['fixed'],
            'tokenizer': {'name': 'characters-smoke', 'version': '1'},
            'model': {'provider': 'local', 'name': 'model', 'version': 'pinned',
                      'decoding': {'temperature': 0}, 'seed': 42}}


def request():
    return {'requestId': 'native:1', 'messages': [
        {'role': 'system', 'content': 'fixed'},
        {'role': 'user', 'content': 'Remember my choice.'}],
        'tools': [{'name': 'remember', 'description': 'Store a fact',
                   'parameters': {'type': 'object', 'properties': {
                       'fact': {'type': 'string'}}}}],
        'tierSegments': [
            {'tier': 'stable', 'content': 'fixed', 'tokenCount': 0,
             'messageIndices': [0]},
            {'tier': 'unstable', 'content': 'Remember my choice.', 'tokenCount': 0,
             'messageIndices': [1]},
            {'tier': 'unstable', 'content': '', 'tokenCount': 0,
             'toolNames': ['remember']}],
        'model': manifest()['model'], 'decodingSettings': {'temperature': 0}, 'seed': 42}


class NativeEnvelopeTests(unittest.TestCase):
    def test_tool_only_response_is_archived_and_replayed_losslessly(self):
        response = {'content': None, 'toolCalls': [
            {'id': 'call-1', 'name': 'remember', 'arguments': {'fact': 'blue'}}],
            'finishReason': 'tool_calls',
            'usage': {'promptTokens': 31, 'completionTokens': 7}}
        config = manifest()
        proxy = ModelProxy(config, BudgetVerifier(config, len),
                           lambda payload: copy.deepcopy(response))
        original = proxy(request())
        self.assertEqual(original, response)
        archive = copy.deepcopy(proxy.telemetry)
        self.assertEqual(archive[0]['response'], response)
        proxy.begin_run()
        proxy.begin_replay(archive)
        self.assertEqual(proxy(request()), response)
        proxy.end_replay()
