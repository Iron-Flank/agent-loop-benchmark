"""Black-box conformance checks, identical for local and remote adapters."""
from copy import deepcopy
import inspect
import json

from .contract import (CONTRACT_VERSION, validate_response,
                       validate_telemetry)


class ConformanceError(AssertionError):
    pass


def sample_manifest(run_id='conformance-1'):
    return {'runId': run_id,
            'adapter': {'name': 'conformance-target', 'revision': 'local',
                        'contractVersion': CONTRACT_VERSION},
            'nativeModelEnvelopeVersion': '1.0',
            'model': {'provider': 'offline-smoke', 'name': 'deterministic',
                      'version': 'deterministic-v1', 'decoding': {'temperature': 0},
                      'seed': 42},
            'tokenizer': {'name': 'smoke-character-count', 'version': '1'},
            'stablePrefix': ['ALMM: conversation and built-in memory only.',
                             'SOUL: helpful assistant.', 'Agent: memory assistant.',
                             'User: conformance user.']}


def _stable(request):
    return [s['content'] for s in request['tierSegments'] if s['tier'] == 'stable']


def check_adapter(adapter):
    """Raise an actionable diagnostic on the first failed requirement.

    This tests observable isolation, not inaccessible private storage. Adapter
    authors must additionally audit their storage reset behavior.
    """
    stage = 'methods'
    try:
        methods = {'initialize': sample_manifest(),
                   'handleTurn': {'turnId': 't', 'role': 'user', 'text': 'hello'},
                   'answerProbe': {'probeId': 'p', 'question': 'hello'}}
        for name in (*methods, 'getRequestTelemetry'):
            method = getattr(adapter, name, None)
            if not callable(method):
                raise ValueError(f'implement {name}()')
            args = () if name == 'getRequestTelemetry' else (methods[name],)
            inspect.signature(method).bind(*args)

        stage = 'contract version'
        for version in (None, 'incompatible'):
            manifest = sample_manifest()
            if version is None:
                del manifest['adapter']['contractVersion']
            else:
                manifest['adapter']['contractVersion'] = version
            try:
                adapter.initialize(manifest)
            except ValueError:
                pass
            else:
                raise ValueError('initialize must reject missing/incompatible contractVersion')

        stage = 'lifecycle'
        if adapter.initialize(sample_manifest()) is not None:
            raise ValueError('initialize must return None (JSON null)')
        if adapter.getRequestTelemetry() != []:
            raise ValueError('initialize must clear request telemetry')
        sentinel = 'PRIOR_RUN_FACT_71cdd56d'
        turn = adapter.handleTurn({'turnId': 'prior-turn', 'sessionId': 'prior-session',
                                   'role': 'user', 'text': f'My access phrase is {sentinel}.'})
        validate_response(turn, 'response')
        probe = adapter.answerProbe({'probeId': 'prior-probe',
                                     'question': 'What is my access phrase?'})
        validate_response(probe, 'answer')

        stage = 'telemetry'
        all_requests = adapter.getRequestTelemetry()
        validate_telemetry(all_requests)
        if all_requests != turn['requests'] + probe['requests']:
            raise ValueError('getRequestTelemetry must expose every request in chronological order')
        stable = _stable(all_requests[0])
        if not stable:
            raise ValueError('stable prefix must include run-constant instructions')
        if any(_stable(r) != stable for r in all_requests):
            raise ValueError('stable prefix changed within run')
        # Returned snapshots must not give callers ownership of internal history.
        snapshot = deepcopy(all_requests)
        all_requests.clear()
        if adapter.getRequestTelemetry() != snapshot:
            raise ValueError('request telemetry must be an independent snapshot')

        stage = 'state reset'
        adapter.initialize(sample_manifest('conformance-2'))
        if adapter.getRequestTelemetry() != []:
            raise ValueError('prior-run telemetry persists after initialize')
        fresh = adapter.answerProbe({'probeId': 'fresh-probe',
                                     'question': 'What is my access phrase?'})
        validate_response(fresh, 'answer')
        fresh_requests = adapter.getRequestTelemetry()
        validate_telemetry(fresh_requests)
        if fresh_requests != fresh['requests']:
            raise ValueError('prior-run requests persist after initialize')
        visible = json.dumps(fresh)
        for marker in (sentinel, 'prior-turn', 'prior-session', 'prior-probe'):
            if marker in visible:
                raise ValueError(f'prior-run state leaked: {marker}')
        stage = 'native harness'
        return ['methods', 'contract version', 'lifecycle', 'telemetry', 'state reset',
                *check_native_harness()]
    except Exception as exc:
        raise ConformanceError(f'{stage}: {exc}') from exc


def _native_request(manifest, request_id, history, *, calls=None, results=None):
    """Assemble native fixtures using the same envelope as runtime adapters."""
    messages = [{'role': 'system', 'content': text} for text in manifest['stablePrefix']]
    messages.extend(deepcopy(history))
    segments = [{'tier': 'stable' if index < len(manifest['stablePrefix']) else 'unstable',
                 'content': message['content'] if isinstance(message['content'], str) else '',
                 'tokenCount': 0, 'messageIndices': [index]}
                for index, message in enumerate(messages)]
    tools = [{'name': 'remember', 'description': 'Persist a memory fact',
              'parameters': {'type': 'object', 'properties': {'fact': {'type': 'string'}},
                             'required': ['fact'], 'additionalProperties': False}}]
    segments.append({'tier': 'semi-stable', 'content': '', 'tokenCount': 0,
                     'toolNames': ['remember']})
    request = {'requestId': request_id, 'messages': messages, 'tools': tools,
               'tierSegments': segments, 'model': deepcopy(manifest['model']),
               'decodingSettings': deepcopy(manifest['model']['decoding']),
               'seed': manifest['model']['seed']}
    if calls is not None:
        request['toolCalls'] = deepcopy(calls)
    if results is not None:
        request['toolResults'] = deepcopy(results)
    return request


class _NativeMemoryProvider:
    """Offline functional provider: choose a memory operation, then read its result.

    This exercises the real budget/proxy path without network credentials. It is
    not an adapter mock: malformed or stripped native tools/history fail here.
    """
    def __init__(self):
        self.calls = 0

    def __call__(self, payload):
        from .native import wire_request
        self.calls += 1
        request = payload['request']
        body = wire_request(request)
        schemas = body.get('tools', [])
        if not any(tool['function']['name'] == 'remember' and
                   tool['function']['parameters'].get('required') == ['fact']
                   for tool in schemas):
            raise ValueError('native memory schema was stripped')
        messages = body['messages']
        last = messages[-1]
        if last['role'] == 'tool':
            previous = messages[-2]
            calls = previous.get('tool_calls', [])
            if len(calls) != 1 or calls[0]['id'] != last.get('tool_call_id'):
                raise ValueError('native memory call linkage was stripped')
            arguments = json.loads(calls[0]['function']['arguments'])
            result = json.loads(last['content'])
            if result != {'stored': arguments['fact']}:
                raise ValueError('native memory result was stripped or changed')
            content, tool_calls, finish = 'Remembered: ' + result['stored'], [], 'stop'
        elif last['role'] == 'user':
            command, separator, fact = last['content'].partition(': ')
            if not separator or command not in ('remember-only', 'remember-and-say'):
                raise ValueError('native memory tool result was stripped')
            content = None if command == 'remember-only' else 'I will remember that.'
            tool_calls = [{'id': 'memory-call', 'name': 'remember', 'arguments': {'fact': fact}}]
            finish = 'tool_calls'
        else:
            raise ValueError('native memory tool result was stripped')
        response = {'content': content, 'finishReason': finish,
                    'usage': {'promptTokens': request['totalTokens'], 'completionTokens': 7},
                    'modelVersion': body['model']}
        if tool_calls:
            response['toolCalls'] = tool_calls
        return response


def check_native_harness():
    """Five executable native-memory patterns against ModelProxy/BudgetVerifier."""
    from almm_harness.budget import BudgetVerifier
    from almm_harness.errors import AdapterFailure, BudgetFailure, HarnessFailure
    from almm_harness.proxy import ModelProxy
    from .contract import validate_native_response
    from .native import canonical_wire, wire_request

    manifest = sample_manifest('native-conformance')
    manifest['tokenizer']['requestOverheadTokens'] = 11
    provider = _NativeMemoryProvider()
    budget = BudgetVerifier(manifest, len)
    proxy = ModelProxy(manifest, budget, provider)
    passed = []
    stage = 'native tool-only'
    try:
        tool_only = _native_request(manifest, 'native:1', [
            {'role': 'user', 'content': 'remember-only: blue'}])
        first = proxy(tool_only)
        validate_native_response(first)
        if first['content'] is not None or first.get('toolCalls') != [
                {'id': 'memory-call', 'name': 'remember', 'arguments': {'fact': 'blue'}}]:
            raise ValueError('tool-only response lost its native call')
        passed.append(stage)

        stage = 'native text+tool'
        text_tool = _native_request(manifest, 'native:2', [
            {'role': 'user', 'content': 'remember-and-say: green'}])
        second = proxy(text_tool)
        if second['content'] != 'I will remember that.' or second['toolCalls'][0]['arguments'] != {'fact': 'green'}:
            raise ValueError('text plus tool response was flattened')
        passed.append(stage)

        stage = 'native tool-result follow-up'
        # Execute the declared memory operation, not a textual substitute.
        memory = {}
        call = first['toolCalls'][0]
        memory['fact'] = call['arguments']['fact']
        result = {'stored': memory['fact']}
        followup = _native_request(manifest, 'native:3', [
            {'role': 'user', 'content': 'remember-only: blue'},
            {'role': 'assistant', 'content': None, 'toolCalls': first['toolCalls']},
            {'role': 'tool', 'toolCallId': call['id'], 'content': json.dumps(result)}],
            calls=first['toolCalls'], results=[{'toolCallId': call['id'], 'result': result}])
        third = proxy(followup)
        if third['content'] != 'Remembered: blue' or third.get('toolCalls'):
            raise ValueError('native result did not drive the follow-up answer')
        foreground = proxy.telemetry[0]
        maintenance = proxy.telemetry[-1]
        if (foreground['stablePrefixHash'] != maintenance['stablePrefixHash'] or
                _stable(foreground) != _stable(maintenance) or
                foreground['messages'][:len(manifest['stablePrefix'])] !=
                maintenance['messages'][:len(manifest['stablePrefix'])]):
            raise ValueError('foreground/memory-maintenance stable prefix differs')
        passed.append(stage)

        stage = 'native schemas/calls/results budget'
        wire = wire_request(followup)
        if wire['messages'][-2]['tool_calls'][0]['function']['name'] != 'remember':
            raise ValueError('wire lost the native assistant call')
        if wire['messages'][-1]['tool_call_id'] != call['id']:
            raise ValueError('wire lost the native tool result')
        normalized = budget.verify(followup)
        if normalized['totalTokens'] != len(canonical_wire(followup)) + 11:
            raise ValueError('budget does not cover exact serialized native body')
        # Both valid-but-stripped schema/history and oversized native fields
        # must be caught at the real harness/provider boundary.
        for field in ('schemas', 'calls', 'results'):
            stripped = deepcopy(followup)
            stripped['requestId'] = 'stripped:' + field
            if field == 'schemas':
                del stripped['tools']
                stripped['tierSegments'] = [s for s in stripped['tierSegments'] if 'toolNames' not in s]
            elif field == 'calls':
                del stripped['toolCalls']
                stripped['messages'][-2] = {'role': 'assistant', 'content': ''}
            else:
                del stripped['toolResults']
                stripped['messages'].pop()
                index = len(stripped['messages'])
                stripped['tierSegments'] = [s for s in stripped['tierSegments']
                                            if s.get('messageIndices') != [index]]
            try:
                proxy(stripped)
            except (ValueError, HarnessFailure):
                pass
            else:
                raise ValueError(f'stripped native {field} was accepted')

            oversized = deepcopy(followup)
            oversized['requestId'] = 'oversized:' + field
            if field == 'schemas':
                oversized['tools'][0]['description'] = 'x' * 25_001
            elif field == 'calls':
                oversized['toolCalls'][0]['arguments']['fact'] = 'x' * 25_001
                oversized['messages'][-2]['toolCalls'] = deepcopy(oversized['toolCalls'])
            else:
                oversized['toolResults'][0]['result']['stored'] = 'x' * 25_001
                oversized['messages'][-1]['content'] = json.dumps(oversized['toolResults'][0]['result'])
            before = provider.calls
            try:
                proxy(oversized)
            except BudgetFailure:
                pass
            else:
                raise ValueError(f'oversized native {field} bypassed budget')
            if provider.calls != before:
                raise ValueError('budget rejection reached the provider')
        changed = deepcopy(followup)
        changed['messages'][0]['content'] += ' changed'
        try:
            budget.verify(changed)
        except AdapterFailure:
            pass
        else:
            raise ValueError('memory-maintenance changed the stable system prefix')
        passed.append(stage)

        stage = 'native exact replay'
        archive = deepcopy([row for row in proxy.telemetry if row.get('status') == 'ok'])
        before = provider.calls
        proxy.begin_run()
        proxy.begin_replay(archive)
        for request, response in ((tool_only, first), (text_tool, second), (followup, third)):
            replayed = proxy(request)
            if replayed != response:
                raise ValueError('replay lost native content/calls/usage/metadata')
            replayed['usage']['completionTokens'] = 999
            if (archive[len(proxy.telemetry) - 1]['response']['usage']['completionTokens'] != 7 or
                    proxy.telemetry[-1]['response']['usage']['completionTokens'] != 7):
                raise ValueError('replay response aliases the archive')
        proxy.end_replay()
        if provider.calls != before:
            raise ValueError('replay contacted the provider')
        passed.append(stage)
        return passed
    except Exception as exc:
        raise ConformanceError(f'{stage}: {exc}') from exc
