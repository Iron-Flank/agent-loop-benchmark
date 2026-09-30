"""Trusted full-history example; deliberately not a scalable memory baseline."""
from copy import deepcopy
import json
from typing import Callable

from .contract import (NativeModelRequest, NativeModelResponse, RuntimeAdapter,
                       validate_input, validate_manifest, validate_native_response,
                       validate_response, validate_telemetry)
from .native import response_text


class ReferenceAdapter(RuntimeAdapter):
    def __init__(self, model_proxy: Callable[[NativeModelRequest], NativeModelResponse],
                 count_tokens: Callable[[str], int]):
        self._model_proxy = model_proxy
        self._count_tokens = count_tokens
        self._manifest = None
        self._history = []
        self._requests = []

    def initialize(self, runManifest):
        validate_manifest(runManifest)
        self._manifest = deepcopy(runManifest)
        self._history = []
        self._requests = []

    def _require_initialized(self):
        if self._manifest is None:
            raise ValueError('call initialize(runManifest) before using the adapter')

    def _segment(self, tier, content, index, source_ids=None):
        segment = {'tier': tier, 'content': content,
                   'tokenCount': self._count_tokens(content), 'messageIndices': [index]}
        if source_ids is not None:
            segment['sourceIds'] = source_ids
        return segment

    def _respond(self, text, source_id, field):
        self._require_initialized()
        messages = [{'role': 'system', 'content': content}
                    for content in self._manifest['stablePrefix']]
        segments = [self._segment('stable', message['content'], index)
                    for index, message in enumerate(messages)]
        current = {'message': {'role': 'user', 'content': text}, 'sourceIds': [source_id]}
        for entry in [*self._history, current]:
            index = len(messages)
            messages.append(deepcopy(entry['message']))
            content = entry['message']['content']
            # Attribution text is separate from the lossless native message.
            text_content = (content if isinstance(content, str) else
                            '' if content is None else json.dumps(content, ensure_ascii=False))
            segments.append(self._segment('unstable', text_content, index, entry['sourceIds']))
        request = {'requestId': f"{self._manifest['runId']}:{len(self._requests) + 1}",
                   'messages': messages, 'tierSegments': segments,
                   'model': deepcopy(self._manifest['model']),
                   'decodingSettings': deepcopy(self._manifest['model']['decoding'])}
        if 'seed' in self._manifest['model']:
            request['seed'] = self._manifest['model']['seed']
        validate_telemetry([request])
        # Archive every attempted assembled request, including proxy failures.
        self._requests.append(deepcopy(request))
        response = self._model_proxy(deepcopy(request))
        validate_native_response(response)
        result = {field: response_text(response), 'requests': [request]}
        validate_response(result, field)
        assistant = {'role': 'assistant', 'content': deepcopy(response['content'])}
        if response.get('toolCalls'):
            assistant['toolCalls'] = deepcopy(response['toolCalls'])
        # This example preserves calls, but cannot execute a runtime's tools.
        self._history.extend([current, {'message': assistant, 'sourceIds': [source_id]}])
        return result

    def handleTurn(self, turn):
        validate_input(turn, 'turn')
        return self._respond(turn['text'], turn['turnId'], 'response')

    def answerProbe(self, probe):
        validate_input(probe, 'probe')
        return self._respond(probe['question'], probe['probeId'], 'answer')

    def getRequestTelemetry(self):
        self._require_initialized()
        return deepcopy(self._requests)


def smoke_model(request):
    """Offline smoke-only proxy. Never use for scored benchmark runs."""
    message = request['messages'][-1]
    content = message['content'] if isinstance(message['content'], str) else ''
    return {'content': 'Smoke response: ' + content, 'finishReason': 'stop',
            'usage': {'promptTokens': 0, 'completionTokens': 0}}


def create_adapter():
    # Character counts are deterministic smoke telemetry, not model-token counts.
    return ReferenceAdapter(smoke_model, len)
