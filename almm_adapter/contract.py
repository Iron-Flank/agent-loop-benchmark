"""RuntimeAdapter 2.0 and native model envelope 1.0 JSON contracts."""
import json
import math
from typing import Literal, NotRequired, Protocol, TypedDict

CONTRACT_VERSION = '2.0'
NATIVE_MODEL_ENVELOPE_VERSION = '1.0'
TIERS = ('stable', 'semi-stable', 'unstable')


class Segment(TypedDict):
    tier: Literal['stable', 'semi-stable', 'unstable']
    content: str
    tokenCount: int
    sourceIds: NotRequired[list[str]]
    messageIndices: NotRequired[list[int]]
    toolNames: NotRequired[list[str]]


class ToolCall(TypedDict):
    id: str
    name: str
    arguments: dict


class NativeMessage(TypedDict):
    role: Literal['system', 'user', 'assistant', 'tool']
    content: str | list[dict] | None
    toolCallId: NotRequired[str]
    toolCalls: NotRequired[list[ToolCall]]


class NativeTool(TypedDict):
    name: str
    description: str
    parameters: dict


class ToolResult(TypedDict):
    toolCallId: str
    result: object


class NativeModelRequest(TypedDict):
    requestId: str
    messages: list[NativeMessage]
    tools: NotRequired[list[NativeTool]]
    toolCalls: NotRequired[list[ToolCall]]
    toolResults: NotRequired[list[ToolResult]]
    tierSegments: list[Segment]
    model: dict
    decodingSettings: dict
    seed: NotRequired[int]


class NativeUsage(TypedDict):
    promptTokens: int
    completionTokens: int


class NativeModelResponse(TypedDict):
    content: str | list[dict] | None
    toolCalls: NotRequired[list[ToolCall]]
    finishReason: str
    usage: NativeUsage
    modelVersion: NotRequired[str]


class AdapterIdentity(TypedDict):
    name: str
    revision: str
    contractVersion: str


class RunManifest(TypedDict):
    runId: str
    adapter: AdapterIdentity
    model: dict
    tokenizer: dict
    stablePrefix: list[str]
    nativeModelEnvelopeVersion: str


class Turn(TypedDict):
    turnId: str
    sessionId: NotRequired[str]
    role: Literal['user']
    text: str


class Probe(TypedDict):
    probeId: str
    question: str


class TurnResult(TypedDict):
    response: str
    requests: list[NativeModelRequest]


class ProbeResult(TypedDict):
    answer: str
    requests: list[NativeModelRequest]


class RuntimeAdapter(Protocol):
    def initialize(self, runManifest: RunManifest) -> None: ...
    def handleTurn(self, turn: Turn) -> TurnResult: ...
    def answerProbe(self, probe: Probe) -> ProbeResult: ...
    def getRequestTelemetry(self) -> list[NativeModelRequest]: ...


def _object(value, path):
    if not isinstance(value, dict):
        raise ValueError(f'{path}: expected object')


def _string(value, path):
    if not isinstance(value, str) or not value:
        raise ValueError(f'{path}: expected nonempty string')


def validate_manifest(manifest):
    _object(manifest, 'runManifest')
    _string(manifest.get('runId'), 'runManifest.runId')
    identity = manifest.get('adapter')
    _object(identity, 'runManifest.adapter')
    if identity.get('contractVersion') != CONTRACT_VERSION:
        raise ValueError(f'runManifest.adapter.contractVersion: required {CONTRACT_VERSION}')
    for key in ('name', 'revision'):
        _string(identity.get(key), f'runManifest.adapter.{key}')
    for key in ('model', 'tokenizer'):
        _object(manifest.get(key), f'runManifest.{key}')
    prefix = manifest.get('stablePrefix')
    if not isinstance(prefix, list) or not prefix or not all(isinstance(s, str) for s in prefix):
        raise ValueError('runManifest.stablePrefix: expected nonempty list of strings')
    if manifest.get('nativeModelEnvelopeVersion') != NATIVE_MODEL_ENVELOPE_VERSION:
        raise ValueError('runManifest.nativeModelEnvelopeVersion: required 1.0')
    _string(manifest['model'].get('version'), 'runManifest.model.version')
    _object(manifest['model'].get('decoding'), 'runManifest.model.decoding')
    _json(manifest['model'], 'runManifest.model')


def validate_input(value, kind):
    """Only adapter-visible fields cross this boundary; scorer records are rejected."""
    _object(value, kind)
    keys = {'turn': {'turnId', 'sessionId', 'role', 'text'},
            'probe': {'probeId', 'question'}}[kind]
    unexpected = value.keys() - keys
    if unexpected:
        raise ValueError(f'{kind}: unexpected fields {sorted(unexpected)}; strip scorer data')
    for key in (('turnId', 'text') if kind == 'turn' else ('probeId', 'question')):
        _string(value.get(key), f'{kind}.{key}')
    if kind == 'turn':
        if value.get('role') != 'user':
            raise ValueError('turn.role: required user')
        if 'sessionId' in value:
            _string(value['sessionId'], 'turn.sessionId')


def validate_telemetry(requests, path='requests', allow_empty=False):
    if not isinstance(requests, list) or (not requests and not allow_empty):
        raise ValueError(f'{path}: expected request list')
    ids = set()
    for i, request in enumerate(requests):
        rp = f'{path}[{i}]'
        validate_native_request(request, rp)
        if request['requestId'] in ids:
            raise ValueError(f'{rp}.requestId: duplicate')
        ids.add(request['requestId'])


def validate_response(response, field):
    _object(response, 'result')
    if not isinstance(response.get(field), str):
        raise ValueError(f'result.{field}: expected string')
    validate_telemetry(response.get('requests'))


def _json(value, path):
    """Reject non-JSON values rather than coercing them during serialization."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            _json(item, path)
        return
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        for item in value.values():
            _json(item, path)
        return
    raise ValueError(f'{path}: expected finite JSON-compatible value')


def _keys(value, allowed, path):
    _object(value, path)
    unexpected = value.keys() - allowed
    if unexpected:
        raise ValueError(f'{path}: unexpected fields {sorted(unexpected)}')


def _content(value, path, allow_none=False):
    if isinstance(value, str) or (value is None and allow_none):
        return
    if not isinstance(value, list) or not value:
        raise ValueError(f'{path}: expected string or structured content parts')
    for part in value:
        _object(part, path)
        _string(part.get('type'), f'{path}.type')
        _json(part, path)
        if part['type'] == 'text' and not isinstance(part.get('text'), str):
            raise ValueError(f'{path}.text: expected string')


def _calls(calls, path):
    if not isinstance(calls, list):
        raise ValueError(f'{path}: expected tool call list')
    ids = set()
    for call in calls:
        _keys(call, {'id', 'name', 'arguments'}, path)
        for key in ('id', 'name'):
            _string(call.get(key), f'{path}.{key}')
        _object(call.get('arguments'), f'{path}.arguments')
        _json(call['arguments'], f'{path}.arguments')
        if call['id'] in ids:
            raise ValueError(f'{path}: duplicate tool call id')
        ids.add(call['id'])


def validate_native_request(request, path='request'):
    _keys(request, {'requestId', 'messages', 'tools', 'toolCalls', 'toolResults',
                    'tierSegments', 'model', 'decodingSettings', 'seed',
                    'tierTokens', 'totalTokens', 'requestOverheadTokens',
                    'stablePrefixHash'}, path)
    _string(request.get('requestId'), f'{path}.requestId')
    _object(request.get('model'), f'{path}.model')
    _string(request['model'].get('version'), f'{path}.model.version')
    _object(request.get('decodingSettings'), f'{path}.decodingSettings')
    _json(request['model'], f'{path}.model')
    _json(request['decodingSettings'], f'{path}.decodingSettings')
    if 'seed' in request and type(request['seed']) is not int:
        raise ValueError(f'{path}.seed: expected integer')
    messages = request.get('messages')
    if not isinstance(messages, list) or not messages:
        raise ValueError(f'{path}.messages: expected nonempty list')
    calls, results = {}, {}
    for i, message in enumerate(messages):
        mp = f'{path}.messages[{i}]'
        _keys(message, {'role', 'content', 'toolCalls', 'toolCallId'}, mp)
        role = message.get('role')
        if role not in ('system', 'user', 'assistant', 'tool'):
            raise ValueError(f'{mp}.role: invalid native role')
        if 'content' not in message:
            raise ValueError(f'{mp}.content: required')
        _content(message['content'], f'{mp}.content',
                 allow_none=role == 'assistant' and bool(message.get('toolCalls')))
        if 'toolCalls' in message:
            if role != 'assistant':
                raise ValueError(f'{mp}.toolCalls: assistant role required')
            _calls(message['toolCalls'], f'{mp}.toolCalls')
            for call in message['toolCalls']:
                if call['id'] in calls:
                    raise ValueError(f'{mp}.toolCalls: ambiguous duplicate call id')
                calls[call['id']] = call
        if role == 'tool':
            _string(message.get('toolCallId'), f'{mp}.toolCallId')
            call_id = message['toolCallId']
            if call_id not in calls or call_id in results:
                raise ValueError(f'{mp}.toolCallId: missing or ambiguous preceding call')
            results[call_id] = message['content']
        elif 'toolCallId' in message:
            raise ValueError(f'{mp}.toolCallId: tool role required')
    if 'toolCalls' in request:
        _calls(request['toolCalls'], f'{path}.toolCalls')
        for call in request['toolCalls']:
            if call['id'] not in calls or not _same_json(calls[call['id']], call):
                raise ValueError(f'{path}.toolCalls: missing or conflicting message linkage')
    if 'toolResults' in request:
        entries = request['toolResults']
        if not isinstance(entries, list):
            raise ValueError(f'{path}.toolResults: expected list')
        seen = set()
        for entry in entries:
            _keys(entry, {'toolCallId', 'result'}, f'{path}.toolResults')
            _string(entry.get('toolCallId'), f'{path}.toolResults.toolCallId')
            if 'result' not in entry:
                raise ValueError(f'{path}.toolResults.result: required')
            _json(entry['result'], f'{path}.toolResults.result')
            call_id = entry['toolCallId']
            result = entry['result']
            linked = results.get(call_id)
            if not isinstance(result, str) and isinstance(linked, str):
                try:
                    linked = json.loads(linked)
                except (ValueError, TypeError):
                    pass
            if call_id in seen or call_id not in results or not _same_json(linked, result):
                raise ValueError(f'{path}.toolResults: missing, ambiguous or conflicting message linkage')
            seen.add(call_id)
    tools = request.get('tools', [])
    if not isinstance(tools, list):
        raise ValueError(f'{path}.tools: expected list')
    names = set()
    for tool in tools:
        _keys(tool, {'name', 'description', 'parameters'}, f'{path}.tools')
        _string(tool.get('name'), f'{path}.tools.name')
        if not isinstance(tool.get('description'), str):
            raise ValueError(f'{path}.tools.description: expected string')
        _object(tool.get('parameters'), f'{path}.tools.parameters')
        _json(tool['parameters'], f'{path}.tools.parameters')
        if tool['name'] in names:
            raise ValueError(f'{path}.tools: duplicate tool name')
        names.add(tool['name'])
    segments = request.get('tierSegments')
    if not isinstance(segments, list) or not segments:
        raise ValueError(f'{path}.tierSegments: expected nonempty list')
    for i, segment in enumerate(segments):
        sp = f'{path}.tierSegments[{i}]'
        _keys(segment, {'tier', 'content', 'tokenCount', 'sourceIds',
                        'messageIndices', 'toolNames'}, sp)
        if segment.get('tier') not in TIERS:
            raise ValueError(f'{sp}.tier: missing or invalid')
        if not isinstance(segment.get('content'), str):
            raise ValueError(f'{sp}.content: expected string')
        if type(segment.get('tokenCount')) is not int or segment['tokenCount'] < 0:
            raise ValueError(f'{sp}.tokenCount: expected nonnegative integer')
        sources = segment.get('sourceIds', [])
        if not isinstance(sources, list) or not all(isinstance(s, str) and s for s in sources):
            raise ValueError(f'{sp}.sourceIds: expected string list')
        indices, tool_names = segment.get('messageIndices', []), segment.get('toolNames', [])
        if (not isinstance(indices, list) or
                not all(type(i) is int and 0 <= i < len(messages) for i in indices) or
                not isinstance(tool_names, list) or
                not all(isinstance(n, str) and n in names for n in tool_names) or
                not (indices or tool_names)):
            raise ValueError(f'{sp}: invalid or missing native attribution')


def validate_native_response(response):
    _keys(response, {'content', 'toolCalls', 'finishReason', 'usage', 'modelVersion'}, 'response')
    if 'content' not in response:
        raise ValueError('response.content: required')
    _content(response['content'], 'response.content', allow_none=True)
    if 'toolCalls' in response:
        _calls(response['toolCalls'], 'response.toolCalls')
    _string(response.get('finishReason'), 'response.finishReason')
    _keys(response.get('usage'), {'promptTokens', 'completionTokens'}, 'response.usage')
    for key in ('promptTokens', 'completionTokens'):
        if type(response['usage'].get(key)) is not int or response['usage'][key] < 0:
            raise ValueError(f'response.usage.{key}: expected nonnegative integer')
    if 'modelVersion' in response:
        _string(response['modelVersion'], 'response.modelVersion')


def _same_json(left, right):
    return json.dumps(left, sort_keys=True, ensure_ascii=False, separators=(',', ':')) == json.dumps(
        right, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
