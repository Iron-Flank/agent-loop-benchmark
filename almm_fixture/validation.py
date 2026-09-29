"""Validate fixture structure and the temporal evidence behind gold answers."""

import json
from datetime import datetime


ABILITIES = frozenset({
    'information-extraction', 'cross-session-reasoning', 'temporal-reasoning',
    'knowledge-update', 'abstention',
})


def _object(value, path, fields):
    if not isinstance(value, dict):
        raise ValueError(f'{path}: expected an object')
    for field in fields:
        if field not in value:
            raise ValueError(f'{path}.{field}: required field is missing')


def _string(value, path):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{path}: expected a nonempty string')


def _list(value, path):
    if not isinstance(value, list):
        raise ValueError(f'{path}: expected a list')


def _strings(value, path):
    _list(value, path)
    for index, item in enumerate(value):
        _string(item, f'{path}[{index}]')
    if len(set(value)) != len(value):
        raise ValueError(f'{path}: duplicate entries are not allowed')


def _choice(value, choices, path):
    _string(value, path)
    if value not in choices:
        raise ValueError(f'{path}: expected one of {", ".join(sorted(choices))}')


class SchemaValidator:
    """Reject malformed fixtures and gold answers unsupported at their checkpoint."""

    @staticmethod
    def validate(fixture):
        _object(fixture, 'fixture', (
            'schemaVersion', 'generatorVersion', 'fixtureId', 'seed',
            'sessions', 'facts', 'probes',
        ))
        _choice(fixture['schemaVersion'], {'1.0'}, 'fixture.schemaVersion')
        _choice(fixture['generatorVersion'], {'1.0.0'}, 'fixture.generatorVersion')
        if type(fixture['seed']) is not int:
            raise ValueError('fixture.seed: expected an integer, not a boolean')
        identities = set()

        def register(identity, path):
            _string(identity, path)
            if identity in identities:
                raise ValueError(f'{path}: duplicate globally unique ID {identity!r}')
            identities.add(identity)

        register(fixture['fixtureId'], 'fixture.fixtureId')
        if 'contentHash' in fixture:
            digest = fixture['contentHash']
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(character not in '0123456789abcdef' for character in digest)):
                raise ValueError('fixture.contentHash: expected a lowercase SHA-256 digest')

        sessions = fixture['sessions']
        _list(sessions, 'fixture.sessions')
        if not sessions:
            raise ValueError('fixture.sessions: at least one session is required')
        session_positions = {}
        turn_positions = {}
        turns = {}
        introductions = {}
        previous_timestamp = None
        for session_index, session in enumerate(sessions):
            path = f'fixture.sessions[{session_index}]'
            _object(session, path, ('sessionId', 'timestamp', 'turns'))
            register(session['sessionId'], f'{path}.sessionId')
            if session['sessionId'] != f's-{session_index + 1:04d}':
                raise ValueError(f'{path}.sessionId: sessions must be ordered s-0001, s-0002, ...')
            session_positions[session['sessionId']] = session_index
            _string(session['timestamp'], f'{path}.timestamp')
            try:
                timestamp = datetime.fromisoformat(session['timestamp'].replace('Z', '+00:00'))
            except ValueError as error:
                raise ValueError(f'{path}.timestamp: expected an ISO timestamp') from error
            if timestamp.tzinfo is None:
                raise ValueError(f'{path}.timestamp: timestamp must include a timezone')
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise ValueError(f'{path}.timestamp: timestamps must increase in session order')
            previous_timestamp = timestamp
            _list(session['turns'], f'{path}.turns')
            if len(session['turns']) != 10:
                raise ValueError(f'{path}.turns: expected exactly 10 user turns')
            for turn_index, turn in enumerate(session['turns']):
                turn_path = f'{path}.turns[{turn_index}]'
                _object(turn, turn_path, (
                    'turnId', 'role', 'text', 'introducedFactIds', 'referencedFactIds', 'topicId',
                ))
                register(turn['turnId'], f'{turn_path}.turnId')
                _choice(turn['role'], {'user'}, f'{turn_path}.role')
                _string(turn['text'], f'{turn_path}.text')
                _string(turn['topicId'], f'{turn_path}.topicId')
                _strings(turn['introducedFactIds'], f'{turn_path}.introducedFactIds')
                _strings(turn['referencedFactIds'], f'{turn_path}.referencedFactIds')
                position = (session_index, turn_index)
                turn_positions[turn['turnId']] = position
                turns[turn['turnId']] = (turn, turn_path)
                for fact_id in turn['introducedFactIds']:
                    if fact_id in introductions:
                        raise ValueError(f'{turn_path}.introducedFactIds: {fact_id!r} introduced twice')
                    introductions[fact_id] = turn['turnId']

        _list(fixture['facts'], 'fixture.facts')
        facts = {}
        fact_paths = {}
        for index, fact in enumerate(fixture['facts']):
            path = f'fixture.facts[{index}]'
            _object(fact, path, (
                'factId', 'value', 'state', 'introducedAt', 'supersedes', 'invalidatedAt', 'topicId',
            ))
            register(fact['factId'], f'{path}.factId')
            _string(fact['value'], f'{path}.value')
            _string(fact['topicId'], f'{path}.topicId')
            _choice(fact['state'], {'active', 'superseded', 'expired'}, f'{path}.state')
            _string(fact['introducedAt'], f'{path}.introducedAt')
            if fact['introducedAt'] not in turns:
                raise ValueError(f'{path}.introducedAt: unknown turn {fact["introducedAt"]!r}')
            if introductions.get(fact['factId']) != fact['introducedAt']:
                raise ValueError(f'{path}.introducedAt: fact must be introduced exactly once at this turn')
            introduced_turn = turns[fact['introducedAt']][0]
            if introduced_turn['topicId'] != fact['topicId']:
                raise ValueError(f'{path}.topicId: must match the introducing turn topicId')
            for field in ('supersedes', 'invalidatedAt'):
                if fact[field] is not None:
                    _string(fact[field], f'{path}.{field}')
            invalidation = fact['invalidatedAt']
            if invalidation is not None:
                if invalidation not in turns:
                    raise ValueError(f'{path}.invalidatedAt: unknown invalidation turn')
                if turn_positions[invalidation] <= turn_positions[fact['introducedAt']]:
                    raise ValueError(f'{path}.invalidatedAt: must occur after introduction')
            if (fact['state'] == 'active') != (invalidation is None):
                raise ValueError(f'{path}.invalidatedAt: active facts need null; invalidated facts need a turn')
            facts[fact['factId']] = fact
            fact_paths[fact['factId']] = path
        for fact_id in introductions:
            if fact_id not in facts:
                raise ValueError(f'introducedFactIds: unknown fact {fact_id!r}')

        replacements = {}
        for fact_id, fact in facts.items():
            old_id = fact['supersedes']
            if old_id is None:
                continue
            path = fact_paths[fact_id]
            if old_id not in facts:
                raise ValueError(f'{path}.supersedes: unknown fact {old_id!r}')
            old = facts[old_id]
            if turn_positions[old['introducedAt']] >= turn_positions[fact['introducedAt']]:
                raise ValueError(f'{path}.supersedes: must identify an earlier fact, never self or forward')
            if old_id in replacements:
                raise ValueError(f'{path}.supersedes: an old fact cannot have multiple replacements')
            if old['topicId'] != fact['topicId']:
                raise ValueError(f'{path}.supersedes: replacement must retain the old topicId')
            if old['invalidatedAt'] != fact['introducedAt']:
                raise ValueError(f'{path}.invalidatedAt: old fact invalidation must equal replacement introduction')
            if old['state'] != 'superseded':
                raise ValueError(f'{path}.supersedes: replaced fact must have superseded state')
            replacements[old_id] = fact_id
        for fact_id, fact in facts.items():
            if fact['state'] == 'superseded' and fact_id not in replacements:
                raise ValueError(f'{fact_paths[fact_id]}.supersedes: superseded fact has no replacement')
        for turn_id, (turn, path) in turns.items():
            for fact_id in turn['referencedFactIds']:
                if fact_id not in facts:
                    raise ValueError(f'{path}.referencedFactIds: unknown fact {fact_id!r}')
                if turn_positions[facts[fact_id]['introducedAt']][0] >= turn_positions[turn_id][0]:
                    raise ValueError(f'{path}.referencedFactIds: evidence must come from a prior session')

        _list(fixture['probes'], 'fixture.probes')
        for index, probe in enumerate(fixture['probes']):
            path = f'fixture.probes[{index}]'
            _object(probe, path, (
                'probeId', 'afterSessionId', 'question', 'ability', 'answerability',
                'answerAsOf', 'sessionDistance', 'expected',
            ))
            register(probe['probeId'], f'{path}.probeId')
            _string(probe['afterSessionId'], f'{path}.afterSessionId')
            if probe['afterSessionId'] not in session_positions:
                raise ValueError(f'{path}.afterSessionId: unknown probe checkpoint')
            checkpoint_session = session_positions[probe['afterSessionId']]
            checkpoint = (checkpoint_session, 9)
            _string(probe['question'], f'{path}.question')
            _choice(probe['ability'], ABILITIES, f'{path}.ability')
            _choice(probe['answerAsOf'], {'current', 'historical'}, f'{path}.answerAsOf')
            if type(probe['answerability']) is not bool:
                raise ValueError(f'{path}.answerability: expected a boolean')
            if type(probe['sessionDistance']) is not int or probe['sessionDistance'] < 0:
                raise ValueError(f'{path}.sessionDistance: expected a nonnegative integer')
            expected = probe['expected']
            expected_path = f'{path}.expected'
            _object(expected, expected_path, (
                'matchType', 'acceptedAnswers', 'requiredFactIds', 'forbiddenFactIds',
            ))
            _choice(expected['matchType'], {'exact', 'ordered-list', 'abstain'},
                    f'{expected_path}.matchType')
            for field in ('acceptedAnswers', 'requiredFactIds', 'forbiddenFactIds'):
                _strings(expected[field], f'{expected_path}.{field}')
            required = expected['requiredFactIds']
            forbidden = expected['forbiddenFactIds']
            answers = expected['acceptedAnswers']
            abstains = expected['matchType'] == 'abstain'
            if abstains or not probe['answerability'] or probe['ability'] == 'abstention':
                if (not abstains or probe['answerability'] or probe['ability'] != 'abstention'
                        or required or forbidden or answers != ['I do not know.']):
                    raise ValueError(f'{expected_path}: abstain requires false answerability, '
                                     'abstention ability, no evidence, and only I do not know.')
                if probe['sessionDistance'] != 0:
                    raise ValueError(f'{path}.sessionDistance: abstain without evidence must use 0')
                continue
            if not required:
                raise ValueError(f'{expected_path}.requiredFactIds: answerable probes require evidence')
            if set(required) & set(forbidden):
                raise ValueError(f'{expected_path}.forbiddenFactIds: cannot overlap required evidence')
            for field, fact_ids in (('requiredFactIds', required), ('forbiddenFactIds', forbidden)):
                for fact_id in fact_ids:
                    if fact_id not in facts:
                        raise ValueError(f'{expected_path}.{field}: unknown fact {fact_id!r}')
                    introduction = turn_positions[facts[fact_id]['introducedAt']]
                    if introduction[0] >= checkpoint_session:
                        raise ValueError(f'{expected_path}.{field}: evidence must come from a prior session')
            earliest = min(turn_positions[facts[fact_id]['introducedAt']][0] for fact_id in required)
            if probe['sessionDistance'] != checkpoint_session - earliest:
                raise ValueError(f'{path}.sessionDistance: must equal checkpoint minus earliest evidence session')
            if probe['answerAsOf'] == 'current':
                for fact_id in required:
                    invalidation = facts[fact_id]['invalidatedAt']
                    if invalidation is not None and turn_positions[invalidation] <= checkpoint:
                        raise ValueError(f'{expected_path}.requiredFactIds: {fact_id!r} is not active at checkpoint')
            for fact_id in forbidden:
                invalidation = facts[fact_id]['invalidatedAt']
                if invalidation is None or turn_positions[invalidation] > checkpoint:
                    raise ValueError(f'{expected_path}.forbiddenFactIds: forbidden evidence is still active')
            if probe['ability'] == 'knowledge-update':
                ancestors = set()
                for fact_id in required:
                    old_id = facts[fact_id]['supersedes']
                    while old_id is not None:
                        ancestors.add(old_id)
                        old_id = facts[old_id]['supersedes']
                if not ancestors or set(forbidden) != ancestors:
                    raise ValueError(f'{expected_path}.forbiddenFactIds: knowledge-update needs exactly '
                                     'the old supersession chain')
            if (probe['ability'] == 'cross-session-reasoning'
                    and expected['matchType'] != 'ordered-list'):
                raise ValueError(f'{expected_path}.matchType: cross-session-reasoning requires ordered-list')
            if expected['matchType'] == 'exact':
                if len(required) != 1 or answers != [facts[required[0]]['value']]:
                    raise ValueError(f'{expected_path}.acceptedAnswers: exact answer must equal its single fact value')
            else:
                values = [facts[fact_id]['value'] for fact_id in sorted(required)]
                if not answers:
                    raise ValueError(f'{expected_path}.acceptedAnswers: an ordered-list answer is required')
                for answer in answers:
                    try:
                        decoded = json.loads(answer)
                    except (ValueError, TypeError) as error:
                        raise ValueError(f'{expected_path}.acceptedAnswers: expected a JSON array') from error
                    if decoded != values:
                        raise ValueError(f'{expected_path}.acceptedAnswers: array must contain exact fact values in ID order')
