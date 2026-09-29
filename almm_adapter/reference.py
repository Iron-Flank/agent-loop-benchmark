"""Trusted full-history example; deliberately not a scalable memory baseline."""
from copy import deepcopy
from typing import Callable

from .contract import (ModelRequest, RuntimeAdapter, validate_input,
                       validate_manifest, validate_response, validate_telemetry)


class ReferenceAdapter(RuntimeAdapter):
    def __init__(self, model_proxy: Callable[[ModelRequest], str],
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

    def _segment(self, tier, content, source_ids=None):
        segment = {'tier': tier, 'content': content,
                   'tokenCount': self._count_tokens(content)}
        if source_ids is not None:
            segment['sourceIds'] = source_ids
        return segment

    def _respond(self, text, source_id, field):
        self._require_initialized()
        segments = [self._segment('stable', content)
                    for content in self._manifest['stablePrefix']]
        segments.extend(deepcopy(self._history))
        current = self._segment('unstable', f'user: {text}', [source_id])
        segments.append(current)
        request = {'requestId': f"{self._manifest['runId']}:{len(self._requests) + 1}",
                   'segments': segments}
        validate_telemetry([request])
        # Archive every attempted assembled request, including proxy failures.
        self._requests.append(deepcopy(request))
        answer = self._model_proxy(deepcopy(request))
        result = {field: answer, 'requests': [request]}
        validate_response(result, field)
        self._history.append(current)
        self._history.append(self._segment('unstable', f'assistant: {answer}', [source_id]))
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
    return 'Smoke response: ' + request['segments'][-1]['content']


def create_adapter():
    # Character counts are deterministic smoke telemetry, not model-token counts.
    return ReferenceAdapter(smoke_model, len)
