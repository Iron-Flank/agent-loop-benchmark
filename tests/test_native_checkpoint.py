import copy
import json
from pathlib import Path
import tempfile
import unittest

from almm_fixture.engine import generate
from test_harness import CompactAdapter, make_runner, records


class MemoryAdapter(CompactAdapter):
    """Small native memory chain, retained into the next session's request."""
    def initialize(self, manifest):
        super().initialize(manifest)
        self.previous_chain = []
        self.requests = []

    def respond(self, text, source_id, field):
        self.sequence += 1
        initial = self.assemble(text, source_id)
        for message in self.previous_chain:
            index = len(initial['messages'])
            initial['messages'].append(copy.deepcopy(message))
            initial['tierSegments'].append({
                'tier': 'semi-stable', 'content': message['content'] or '',
                'tokenCount': 0, 'messageIndices': [index]})
        initial['tools'] = [{'name': 'remember', 'description': 'Store a fact',
                             'parameters': {'type': 'object'}}]
        initial['tierSegments'].append({'tier': 'unstable', 'content': '',
                                       'tokenCount': 0, 'toolNames': ['remember']})
        first = self.proxy(initial)
        call = first['toolCalls'][0]
        assistant = {'role': 'assistant', 'content': first['content'],
                     'toolCalls': first['toolCalls']}
        tool = {'role': 'tool', 'toolCallId': call['id'], 'content': 'stored'}
        self.sequence += 1
        followup = copy.deepcopy(initial)
        followup['requestId'] = f"{self.manifest['runId']}:{self.sequence}"
        for message in (assistant, tool):
            index = len(followup['messages'])
            followup['messages'].append(message)
            followup['tierSegments'].append({'tier': 'unstable',
                                            'content': message['content'] or '',
                                            'tokenCount': 0, 'messageIndices': [index]})
        response = self.proxy(followup)
        self.previous_chain = copy.deepcopy([assistant, tool])
        self.requests.extend([initial, followup])
        return {field: response['content'], 'requests': [initial, followup]}


def memory_provider(payload):
    request = payload['request']
    if int(request['requestId'].rsplit(':', 1)[1]) % 2:
        return {'content': None, 'toolCalls': [{
            'id': 'call-' + request['requestId'], 'name': 'remember',
            'arguments': {'fact': request['messages'][1]['content']}}],
            'finishReason': 'tool_calls',
            'usage': {'promptTokens': 43, 'completionTokens': 9}}
    return {'content': 'stored', 'finishReason': 'stop',
            'usage': {'promptTokens': 62, 'completionTokens': 2}}


class NativeCheckpointTests(unittest.TestCase):
    def test_interruption_reconstructs_tool_history_without_provider_calls(self):
        fixture = generate(42, 10)
        with tempfile.TemporaryDirectory() as root:
            def interrupted(payload):
                if payload['request']['requestId'].endswith(':21'):
                    raise KeyboardInterrupt('session two interrupted')
                return memory_provider(payload)
            runner, _, _ = make_runner(root, provider=interrupted, adapter_class=MemoryAdapter)
            with self.assertRaises(KeyboardInterrupt):
                runner.run(fixture)
            directory = next(Path(root).iterdir())
            checkpoint = json.loads((directory / 'checkpoint.json').read_text())
            self.assertEqual(checkpoint['lastCompletedSession'], 1)
            prefix = records(directory, 'requests.jsonl')[:20]
            provider_ids = []
            def live(payload):
                provider_ids.append(payload['request']['requestId'])
                return memory_provider(payload)
            resumed, adapter, _ = make_runner(root, provider=live, adapter_class=MemoryAdapter)
            result = resumed.run(fixture, resume=True)
            self.assertFalse(result['manifest']['recovery'].get('restarted', False))
            self.assertEqual(result['results']['answeredCount'], 5)
            self.assertEqual(provider_ids[0], 'harness-test:21')
            archived = records(result['artifactDir'], 'requests.jsonl')
            self.assertEqual(archived[:20], prefix)
            self.assertEqual(archived[20]['messages'][-2:], archived[19]['messages'][-2:])
            self.assertEqual(archived[0]['response']['finishReason'], 'tool_calls')
            self.assertEqual(archived[0]['response']['usage'],
                             {'promptTokens': 43, 'completionTokens': 9})
            self.assertEqual(adapter.requests[1]['messages'][-1]['toolCallId'],
                             archived[0]['response']['toolCalls'][0]['id'])
