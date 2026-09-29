"""Loopback HTTP JSON transport. Adapter code runs only in the server process."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from socketserver import TCPServer
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .contract import (validate_input, validate_manifest, validate_response,
                       validate_telemetry)

MAX_BODY_BYTES = 16 * 1024 * 1024
METHODS = {'initialize', 'handleTurn', 'answerProbe', 'getRequestTelemetry'}


class HttpAdapter:
    def __init__(self, url, timeout=30):
        parsed = urlsplit(url)
        if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
            raise ValueError('HTTP adapters must use a loopback http URL')
        if parsed.path not in ('', '/') or parsed.query or parsed.fragment or parsed.username:
            raise ValueError('HTTP adapter URL must be an origin without credentials')
        self._url = url.rstrip('/')
        self._timeout = timeout

    def _call(self, method, argument=None):
        request = Request(self._url + '/v1/' + method,
                          data=json.dumps(argument).encode(),
                          headers={'Content-Type': 'application/json'})
        try:
            with urlopen(request, timeout=self._timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            with exc:
                error = json.load(exc)
            exception = ValueError if exc.code == 400 else RuntimeError
            raise exception(error.get('error', 'adapter HTTP failure')) from exc
        return result

    def initialize(self, runManifest):
        result = self._call('initialize', runManifest)
        if result is not None:
            raise ValueError('initialize must return null')

    def handleTurn(self, turn):
        validate_input(turn, 'turn')
        result = self._call('handleTurn', turn)
        validate_response(result, 'response')
        return result

    def answerProbe(self, probe):
        validate_input(probe, 'probe')
        result = self._call('answerProbe', probe)
        validate_response(result, 'answer')
        return result

    def getRequestTelemetry(self):
        result = self._call('getRequestTelemetry')
        validate_telemetry(result, allow_empty=True)
        return result


def create_server(adapter, port=0):
    """One sequential adapter per loopback server, matching fixture ordering."""
    class LoopbackServer(HTTPServer):
        def server_bind(self):
            # Fixed loopback transport needs no reverse DNS (which may block).
            TCPServer.server_bind(self)
            self.server_name, self.server_port = self.server_address

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Request payloads can contain private conversations; don't log them.
            pass

        def _send(self, status, value):
            payload = json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            method = self.path.removeprefix('/v1/')
            if self.path != '/v1/' + method or method not in METHODS:
                self._send(404, {'error': 'unknown adapter method'})
                return
            try:
                self.connection.settimeout(30)
                length = int(self.headers.get('Content-Length', '0'))
                if length <= 0 or length > MAX_BODY_BYTES:
                    raise ValueError('Content-Length must be between 1 and 16777216')
                if self.headers.get('Content-Type') != 'application/json':
                    raise ValueError('Content-Type must be application/json')
                value = json.loads(self.rfile.read(length))
                if method == 'initialize':
                    validate_manifest(value)
                elif method == 'handleTurn':
                    validate_input(value, 'turn')
                elif method == 'answerProbe':
                    validate_input(value, 'probe')
                elif value is not None:
                    raise ValueError('getRequestTelemetry body must be null')
                function = getattr(adapter, method)
                result = function() if method == 'getRequestTelemetry' else function(value)
                if method == 'initialize' and result is not None:
                    raise ValueError('initialize must return null')
                if method in ('handleTurn', 'answerProbe'):
                    validate_response(result, 'response' if method == 'handleTurn' else 'answer')
                elif method == 'getRequestTelemetry':
                    validate_telemetry(result, allow_empty=True)
            except (ValueError, UnicodeError) as exc:
                self._send(400, {'error': str(exc)})
                return
            except NotImplementedError as exc:
                self._send(501, {'error': str(exc)})
                return
            except Exception:
                self._send(500, {'error': 'adapter failure; inspect adapter process diagnostics'})
                return
            self._send(200, result)

    return LoopbackServer(('127.0.0.1', port), Handler)
