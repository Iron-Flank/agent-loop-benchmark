"""Lossless native envelope to provider wire conversion, shared by counting and I/O."""
from copy import deepcopy
import json

ARCHIVE_FIELDS = frozenset({
    'status', 'latencyMs', 'attempts', 'retryHistory', 'waitSeconds',
    'providerTokens', 'modelVersion', 'response', 'answer', 'category', 'error',
    'sessionId', 'turnId', 'probeId',
})


def archived_request(record):
    """Remove archive diagnostics, retaining every native and budget field."""
    return {key: value for key, value in record.items() if key not in ARCHIVE_FIELDS}


def _serialize(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def _wire_call(call):
    return {'id': call['id'], 'type': 'function',
            'function': {'name': call['name'], 'arguments': _serialize(call['arguments'])}}


def wire_request(request):
    """Convert native fields once; linked top-level history is corroboration only."""
    from .contract import validate_native_request
    validate_native_request(request)
    body = {'model': request['model']['version'], 'messages': []}
    reserved = {'model', 'messages', 'tools', 'tool_calls', 'seed', 'stream'}
    if reserved.intersection(request['decodingSettings']):
        raise ValueError('decodingSettings: cannot override native provider fields')
    body.update(deepcopy(request['decodingSettings']))
    if 'seed' in request:
        body['seed'] = request['seed']
    for message in request['messages']:
        wire = {'role': message['role'], 'content': deepcopy(message['content'])}
        if 'toolCalls' in message:
            wire['tool_calls'] = [_wire_call(call) for call in message['toolCalls']]
        if 'toolCallId' in message:
            wire['tool_call_id'] = message['toolCallId']
        body['messages'].append(wire)
    if 'tools' in request:
        body['tools'] = [{'type': 'function', 'function': deepcopy(tool)}
                         for tool in request['tools']]
    return body


def canonical_wire(request):
    """Serialize the exact provider body used for both budget and transport."""
    return _serialize(wire_request(request))


def response_text(response):
    """Extract conversational text without turning native tool calls into prose."""
    from .contract import validate_native_response
    validate_native_response(response)
    content = response['content']
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join(part['text'] for part in content if part['type'] == 'text')
    return ''
