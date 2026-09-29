"""RuntimeAdapter 1.0: JSON-compatible types shared by local and HTTP adapters."""
from typing import Literal, NotRequired, Protocol, TypedDict

CONTRACT_VERSION = '1.0'
TIERS = ('stable', 'semi-stable', 'unstable')


class Segment(TypedDict):
    tier: Literal['stable', 'semi-stable', 'unstable']
    content: str
    tokenCount: int
    sourceIds: NotRequired[list[str]]


class ModelRequest(TypedDict):
    requestId: str
    segments: list[Segment]


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
    requests: list[ModelRequest]


class ProbeResult(TypedDict):
    answer: str
    requests: list[ModelRequest]


class RuntimeAdapter(Protocol):
    def initialize(self, runManifest: RunManifest) -> None: ...
    def handleTurn(self, turn: Turn) -> TurnResult: ...
    def answerProbe(self, probe: Probe) -> ProbeResult: ...
    def getRequestTelemetry(self) -> list[ModelRequest]: ...


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
        _object(request, rp)
        _string(request.get('requestId'), f'{rp}.requestId')
        if request['requestId'] in ids:
            raise ValueError(f'{rp}.requestId: duplicate')
        ids.add(request['requestId'])
        segments = request.get('segments')
        if not isinstance(segments, list) or not segments:
            raise ValueError(f'{rp}.segments: expected nonempty list')
        for j, segment in enumerate(segments):
            sp = f'{rp}.segments[{j}]'
            _object(segment, sp)
            if segment.get('tier') not in TIERS:
                raise ValueError(f'{sp}.tier: missing or invalid; expected {TIERS}')
            if not isinstance(segment.get('content'), str):
                raise ValueError(f'{sp}.content: expected string')
            count = segment.get('tokenCount')
            if type(count) is not int or count < 0:
                raise ValueError(f'{sp}.tokenCount: expected nonnegative integer')
            sources = segment.get('sourceIds', [])
            if not isinstance(sources, list) or not all(isinstance(s, str) and s for s in sources):
                raise ValueError(f'{sp}.sourceIds: expected list of nonempty strings')


def validate_response(response, field):
    _object(response, 'result')
    if not isinstance(response.get(field), str):
        raise ValueError(f'result.{field}: expected string')
    validate_telemetry(response.get('requests'))
