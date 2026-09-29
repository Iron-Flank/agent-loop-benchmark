import argparse
import importlib
import json
import sys

from .conformance import ConformanceError, check_adapter, sample_manifest
from .http import HttpAdapter, create_server
from .reference import create_adapter


def load_adapter(factory):
    module, separator, name = factory.partition(':')
    if not separator:
        raise ValueError('factory must be module:function')
    return getattr(importlib.import_module(module), name)()


def smoke(adapter):
    """Ten sequential sessions with accumulated cross-session references."""
    adapter.initialize(sample_manifest('smoke'))
    for session in range(1, 11):
        adapter.handleTurn({'turnId': f't-{session}', 'sessionId': f's-{session}',
                            'role': 'user',
                            'text': ('My project code is ALMM-42.' if session == 1 else
                                     f'Session {session}: keep using the project code from session 1.')})
        result = adapter.answerProbe({'probeId': f'p-{session}',
                                      'question': 'What project code did I name in session 1?'})
        if not isinstance(result['answer'], str):
            raise ValueError('smoke: missing probe answer')
    requests = adapter.getRequestTelemetry()
    if len(requests) != 20:
        raise ValueError('smoke: expected 20 requests across 10 sessions')
    if not any('ALMM-42' in s['content'] for s in requests[-1]['segments']):
        raise ValueError('smoke: reference adapter lost session-1 context')
    return {'sessions': 10, 'requests': len(requests), 'crossSessionContext': 'preserved',
            'model': 'offline-smoke', 'tokenizer': 'character-count (not model tokens)',
            'scored': False}


def main():
    parser = argparse.ArgumentParser(description='ALMM adapter scaffold')
    parser.add_argument('--smoke', action='store_true', help='offline reference adapter smoke')
    parser.add_argument('--url', help='separate-process HTTP adapter for smoke')
    commands = parser.add_subparsers(dest='command')
    serve = commands.add_parser('serve', help='serve a trusted local factory over loopback HTTP')
    serve.add_argument('--factory', default='almm_adapter.reference:create_adapter')
    serve.add_argument('--port', type=int, default=8765)
    conform = commands.add_parser('conformance', help='check a local factory or HTTP adapter')
    target = conform.add_mutually_exclusive_group()
    target.add_argument('--factory', default='almm_adapter.reference:create_adapter')
    target.add_argument('--url')
    args = parser.parse_args()
    try:
        if args.smoke:
            if args.command:
                parser.error('--smoke cannot be combined with a subcommand')
            adapter = HttpAdapter(args.url) if args.url else create_adapter()
            print(json.dumps(smoke(adapter)))
        elif args.command == 'serve':
            with create_server(load_adapter(args.factory), args.port) as server:
                print(json.dumps({'url': f'http://127.0.0.1:{server.server_port}'}), flush=True)
                server.serve_forever()
        elif args.command == 'conformance':
            adapter = HttpAdapter(args.url) if args.url else load_adapter(args.factory)
            print(json.dumps({'passed': check_adapter(adapter)}))
        else:
            parser.error('choose --smoke, conformance, or serve')
    except (ConformanceError, ValueError, RuntimeError, OSError) as exc:
        print(f'FAIL: {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == '__main__':
    sys.exit(main())
