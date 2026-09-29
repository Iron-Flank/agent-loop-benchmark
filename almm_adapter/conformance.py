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
            'model': {'provider': 'offline-smoke', 'name': 'deterministic',
                      'temperature': 0, 'seed': 42},
            'tokenizer': {'name': 'smoke-character-count', 'version': '1'},
            'stablePrefix': ['ALMM: conversation and built-in memory only.',
                             'SOUL: helpful assistant.', 'Agent: memory assistant.',
                             'User: conformance user.']}


def _stable(request):
    return [s['content'] for s in request['segments'] if s['tier'] == 'stable']


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
        return ['methods', 'contract version', 'lifecycle', 'telemetry', 'state reset']
    except Exception as exc:
        raise ConformanceError(f'{stage}: {exc}') from exc
