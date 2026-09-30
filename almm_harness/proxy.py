"""Budget-checked model calls, independent provider quotas, and redacted archives."""
from copy import deepcopy
import json
import math
import os
import re
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from almm_adapter.contract import validate_native_response
from almm_adapter.native import ARCHIVE_FIELDS, archived_request, canonical_wire, response_text

from .errors import AdapterFailure, HarnessFailure, ProviderFailure, TimeoutFailure

MAX_RETRIES = 3
MAX_WAIT_SECONDS = 30
_SECRET_FIELDS = {'apikey', 'xapikey', 'authorization', 'proxyauthorization',
                  'accesstoken', 'refreshtoken', 'password', 'secret', 'clientsecret',
                  'credential', 'credentials', 'key', 'token', 'auth', 'privatekey',
                  'secretkey', 'apisecret'}
_AUTH_HEADER = re.compile(
    r'''(?i)(\b(?:proxy[-_ ]?)?authorization["']?\s*[:=]\s*)'''
    r'''(?:"[^"\r\n]*"|'[^'\r\n]*'|[^\r\n,;]+)''')
_API_KEY_FIELD = re.compile(
    r'''(?i)(\b(?:x[-_ ]?)?api[-_ ]?key["']?\s*[:=]\s*)'''
    r'''(?:"[^"\r\n]*"|'[^'\r\n]*'|[^\s,;"']+)''')


def _secret_field(name):
    return re.sub(r'[^a-z]', '', str(name).lower()) in _SECRET_FIELDS


def _secrets():
    return sorted({value for name, value in os.environ.items() if value and
                   re.search(r'(?:^|_)(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIALS?)(?:_|$)',
                             name.upper())}, key=len, reverse=True)


def redact(value, secrets=None):
    """Return a JSON-safe copy with credentials masked; never mutate the input."""
    secrets = _secrets() if secrets is None else secrets
    if isinstance(value, dict):
        return {str(key): '[REDACTED]' if _secret_field(key) else redact(item, secrets)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) == 2 and isinstance(value[0], str) and _secret_field(value[0]):
            return [value[0], '[REDACTED]']
        return [redact(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, '[REDACTED]')
        value = _AUTH_HEADER.sub(lambda match: match[1] + '[REDACTED]', value)
        value = _API_KEY_FIELD.sub(lambda match: match[1] + '[REDACTED]', value)
        return re.sub(r'(?i)\bBearer\s+[^\s,;"\']+', 'Bearer [REDACTED]', value)
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return redact(str(value), secrets)


def _has_credentials(value):
    if isinstance(value, dict):
        return any(_secret_field(key) or _has_credentials(item)
                   for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_has_credentials(item) for item in value)
    return False


class ProviderHTTPError(ProviderFailure):
    """Transport HTTP error; only status 429 and 503 are eligible for retries."""
    def __init__(self, status, message):
        super().__init__(redact(str(message)))
        self.status = status


class ModelProxy:
    """Callable model boundary; each instance owns a thread-safe token bucket.

    Provider payload is ``{'model': pinned_model_config, 'request': normalized}``.
    A provider returns a NativeModelResponse, including structured calls and usage.
    Rate limits allow an initial burst of RPM tokens, refilling at RPM / 60.
    First-attempt throttling is bounded by the runner's wall timeout, not retry
    policy. Retries back off 1/2/4 seconds, with 30 seconds of total retry waits.
    ``begin_run`` resets archives, never quota, and requires no active calls.
    ``abort_run`` poisons an interrupted run; Python cannot cancel in-flight I/O.
    """
    def __init__(self, manifest, budget, provider, *, clock=time.monotonic, sleep=time.sleep):
        if _has_credentials(manifest):
            raise ProviderFailure('provider credentials must come from environment variables')
        self.manifest = deepcopy(manifest)
        self.budget = budget
        self.provider = provider
        self.clock = clock
        self.sleep = sleep
        model = self.manifest.get('model', {})
        if not all(isinstance(model.get(key), str) and model[key]
                   for key in ('provider', 'name', 'version')):
            raise ProviderFailure('model provider, name and version must be pinned')
        if not isinstance(model.get('decoding'), dict):
            raise ProviderFailure('model decoding must be a pinned object')
        rate = self.manifest.get('rateLimitRpm', 60)
        if isinstance(rate, dict):
            rate = rate.get(model['provider'], 60)
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            raise ProviderFailure('rateLimitRpm must be positive and finite')
        self._capacity = max(1, rate)
        self._refill_rate = rate / 60
        self._tokens = self._capacity
        self._refilled_at = clock()
        self._quota_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._active = 0
        self._aborted = False
        self.telemetry = []
        self.dead_letters = []
        self._replay_records = None
        self._replay_index = 0

    def _redaction_secrets(self):
        secrets = _secrets()
        key_env = getattr(self.provider, 'api_key_env', None)
        if key_env and os.environ.get(key_env):
            secrets.append(os.environ[key_env])
        return secrets

    def redact(self, value):
        return redact(value, self._redaction_secrets())

    def begin_replay(self, records):
        """Reconstruct adapter memory from successful archived calls, not scores."""
        with self._state_lock:
            if self._active or self._aborted:
                raise AdapterFailure('cannot replay with active calls or an aborted run')
            if not isinstance(records, list):
                raise AdapterFailure('replay requires successful native request/response archives')
            for row in records:
                try:
                    if not isinstance(row, dict) or row.get('status') != 'ok':
                        raise ValueError('archive must contain a successful request')
                    validate_native_response(row.get('response'))
                    canonical_wire(archived_request(row))
                    if ('modelVersion' in row['response'] and
                            row['response']['modelVersion'] != self.manifest['model']['version']):
                        raise ValueError('archived response modelVersion does not match pinned version')
                except (ValueError, TypeError, KeyError) as exc:
                    raise AdapterFailure('invalid native replay archive: ' + self.redact(str(exc))) from None
            self._replay_records = deepcopy(records)
            self._replay_index = 0

    def end_replay(self):
        with self._state_lock:
            if self._active:
                raise AdapterFailure('cannot finish replay with active calls')
            if self._replay_records is None:
                raise AdapterFailure('replay has not been started')
            if self._replay_index != len(self._replay_records):
                raise AdapterFailure('replay did not consume every archived request')
            self._replay_records = None
            self._replay_index = 0

    def begin_run(self):
        with self._state_lock:
            if self._active:
                raise ProviderFailure('cannot begin a run with active provider calls')
            self._aborted = False
            self.telemetry.clear()
            self.dead_letters.clear()
            self._replay_records = None
            self._replay_index = 0

    def abort_run(self):
        with self._state_lock:
            self._aborted = True

    def _check_running(self):
        with self._state_lock:
            if self._aborted:
                raise TimeoutFailure('model proxy run was aborted')

    def _token_delay(self):
        with self._quota_lock:
            now = self.clock()
            self._tokens = min(self._capacity, self._tokens +
                               max(0, now - self._refilled_at) * self._refill_rate)
            self._refilled_at = now
            if self._tokens + 1e-12 >= 1:
                self._tokens = max(0, self._tokens - 1)
                return 0
            return (1 - self._tokens) / self._refill_rate

    def __call__(self, request):
        started = self.clock()
        secrets = self._redaction_secrets()
        normalized = request
        history = []
        reported = {}
        waited = 0
        retry_waited = 0
        attempts = 0
        with self._state_lock:
            self._active += 1

        def wait_for(seconds, reason):
            nonlocal waited, retry_waited
            self._check_running()
            if attempts and retry_waited + seconds > MAX_WAIT_SECONDS + 1e-9:
                raise ProviderFailure('provider total retry wait limit of 30 seconds exceeded')
            self.sleep(seconds)
            waited += seconds
            if attempts:
                retry_waited += seconds
            if history:
                history[-1][reason] = history[-1].get(reason, 0) + seconds
            self._check_running()

        try:
            self._check_running()
            # Verifier may keep stable-prefix state, so serialize only verification.
            with self._state_lock:
                try:
                    if isinstance(request, dict) and request.keys() & ARCHIVE_FIELDS:
                        raise ValueError('native model request cannot contain archive metadata')
                    normalized = self.budget.verify(request)
                except ValueError as exc:
                    raise AdapterFailure(redact(str(exc), secrets)) from None
                if self._replay_records is not None:
                    if self._aborted:
                        raise TimeoutFailure('model proxy run was aborted')
                    if self._replay_index >= len(self._replay_records):
                        raise AdapterFailure('replay produced an extra model request')
                    recorded = self._replay_records[self._replay_index]
                    if archived_request(recorded) != normalized:
                        raise AdapterFailure('replayed model request differs from archived request')
                    self._replay_index += 1
                    self.telemetry.append(redact(recorded, secrets))
                    return deepcopy(recorded['response'])
            for retry in range(MAX_RETRIES + 1):
                self._check_running()
                delay = self._token_delay()
                while delay:
                    wait_for(delay, 'rateWaitSeconds')
                    delay = self._token_delay()
                self._check_running()
                attempts += 1
                item = {'attempt': attempts}
                history.append(item)
                try:
                    result = self.provider({'model': deepcopy(self.manifest['model']),
                                            'request': deepcopy(normalized)})
                except ProviderHTTPError as exc:
                    item.update(status=exc.status, error=redact(str(exc), secrets))
                    if exc.status not in (429, 503) or retry == MAX_RETRIES:
                        raise
                    wait_for(2 ** retry, 'backoffSeconds')
                    continue
                except Exception as exc:
                    item.update(status='error', error=redact(str(exc), secrets))
                    raise
                self._check_running()
                try:
                    validate_native_response(result)
                except ValueError as exc:
                    raise ProviderFailure('invalid native provider response: ' +
                                          redact(str(exc), secrets)) from None
                reported['providerTokens'] = result['usage']['promptTokens']
                reported['response'] = deepcopy(result)
                if 'modelVersion' in result:
                    reported['modelVersion'] = result['modelVersion']
                    if result['modelVersion'] != self.manifest['model']['version']:
                        raise ProviderFailure('provider modelVersion does not match pinned model version')
                item['status'] = 'ok'
                text = response_text(result)
                if text:
                    reported['answer'] = text
                self._archive(normalized, started, waited, attempts, history, reported, secrets)
                return deepcopy(result)
        except Exception as exc:
            if not isinstance(exc, HarnessFailure):
                category = TimeoutFailure if isinstance(exc, TimeoutError) else ProviderFailure
                exc = category(redact(str(exc), secrets))
            else:
                exc.args = (redact(str(exc), secrets),)
            normalized = getattr(exc, 'request', normalized)
            if hasattr(exc, 'request'):
                exc.request = redact(normalized, secrets)
            if history and 'status' not in history[-1]:
                history[-1].update(status='error', error=redact(str(exc), secrets))
            self._archive(normalized, started, waited, attempts, history, reported, secrets, exc)
            raise exc from None
        finally:
            with self._state_lock:
                self._active -= 1

    def _archive(self, request, started, waited, attempts, history, reported, secrets, error=None):
        latency = max(0, self.clock() - started) * 1000
        row = deepcopy(request) if isinstance(request, dict) else {'request': request}
        row.update(status='error' if error else 'ok', latencyMs=latency,
                   attempts=attempts, retryHistory=history, waitSeconds=waited, **reported)
        letter = None
        if error:
            details = {'type': type(error).__name__, 'category': error.category,
                       'message': str(error)}
            if hasattr(error, 'status'):
                details['status'] = error.status
            row.update(category=error.category, error=details)
            letter = {'runId': self.manifest.get('runId'), 'manifest': self.manifest,
                      'request': request, 'error': details, 'retryHistory': history,
                      'latencyMs': latency, 'waitSeconds': waited, **reported}
        with self._state_lock:
            self.telemetry.append(redact(row, secrets))
            if letter is not None:
                self.dead_letters.append(redact(letter, secrets))


class _RejectRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward Authorization to a server-selected URL.
        return None


class OpenAICompatibleProvider:
    """Dependency-free POST /chat/completions transport, with no fake fallback.

    ``endpoint`` is the full HTTPS URL (loopback HTTP allowed for local serving).
    ``api_key_env`` names the environment variable, never the credential itself.
    ``timeout`` bounds each HTTP attempt. An optional urllib-compatible ``opener``
    accepts (Request, timeout=seconds), useful for boundary tests.

    Sends the canonical native envelope wire body without flattening messages,
    tool schemas, calls, or results. Response usage and model are preserved in
    the native response; arguments are decoded into structured objects.
    Redirects are rejected to protect credentials.
    """
    def __init__(self, endpoint='https://api.openai.com/v1/chat/completions',
                 api_key_env='OPENAI_API_KEY', timeout=60, *, opener=None):
        parsed = urlsplit(endpoint)
        loopback = parsed.hostname in ('localhost', '127.0.0.1', '::1')
        if (parsed.scheme not in ('https', 'http') or not parsed.hostname or
                (parsed.scheme == 'http' and not loopback) or
                parsed.username is not None or parsed.password is not None or
                parsed.query or parsed.fragment):
            raise ProviderFailure('provider endpoint must be HTTPS without credentials or query')
        if not isinstance(api_key_env, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', api_key_env):
            raise ProviderFailure('api_key_env must name an environment variable')
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ProviderFailure('provider timeout must be positive and finite')
        self.endpoint = endpoint
        self.api_key_env = api_key_env
        self.timeout = timeout
        self._open = opener or build_opener(_RejectRedirect()).open

    def __call__(self, payload):
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ProviderFailure(f'provider environment variable {self.api_key_env} is unset')
        try:
            if payload['model'] != payload['request']['model']:
                raise ValueError('provider model differs from native request model')
            body = canonical_wire(payload['request'])
        except (ValueError, TypeError, KeyError) as exc:
            raise ProviderFailure('invalid native provider request: ' + redact(str(exc))) from None
        req = Request(self.endpoint, data=body.encode('utf-8'),
                      headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
        secrets = _secrets() + [key]
        try:
            with self._open(req, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            with exc:
                message = exc.read().decode('utf-8', errors='replace')
            raise ProviderHTTPError(exc.code, redact(message or str(exc), secrets)) from None
        except (TimeoutError, URLError) as exc:
            error = TimeoutFailure if isinstance(exc, TimeoutError) or isinstance(getattr(exc, 'reason', None), TimeoutError) else ProviderFailure
            raise error(redact(str(exc), secrets)) from None
        except Exception as exc:
            raise ProviderFailure(redact(str(exc), secrets)) from None
        try:
            choice = result['choices'][0]
            message = choice['message']
            usage = result['usage']
            normalized = {
                'content': deepcopy(message['content']),
                'finishReason': choice['finish_reason'],
                'usage': {'promptTokens': usage['prompt_tokens'],
                          'completionTokens': usage['completion_tokens']}}
            if 'tool_calls' in message:
                calls = message['tool_calls']
                if not isinstance(calls, list):
                    raise ValueError('provider tool_calls must be an array')
                normalized['toolCalls'] = []
                for call in calls:
                    if call['type'] != 'function':
                        raise ValueError('provider tool call must be a function')
                    function = call['function']
                    arguments = function['arguments']
                    if not isinstance(arguments, str):
                        raise ValueError('provider function arguments must be JSON text')
                    normalized['toolCalls'].append({
                        'id': call['id'], 'name': function['name'],
                        'arguments': json.loads(arguments)})
            if 'model' in result:
                normalized['modelVersion'] = result['model']
            validate_native_response(normalized)
            return normalized
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderFailure('invalid provider response: ' + redact(str(exc), secrets)) from None
