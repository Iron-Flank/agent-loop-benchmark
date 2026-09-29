"""Seed-driven fixture planning, fact lifecycle, and checkpoint-specific answers."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import random

from .topics import TopicPool, TurnRenderer
from .validation import SchemaValidator

GENERATOR_VERSION = '1.0.0'
SCALES = (10, 100, 500, 1000)
# The actual evaluation seeds remain private; only the reserved range is public.
EVALUATION_SEED_START = 1_000_000_000


def canonical_json(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


class FactGraph:
    def __init__(self):
        self.facts = []
        self.by_id = {}
        self.current = {}
        self.turn_sessions = {}

    def plant(self, topic, value, turn_id, session_index):
        previous = self.current.get(topic.topic_id)
        fact_id = f'f-{len(self.facts) + 1:06d}'
        fact = {'factId': fact_id, 'value': value, 'state': 'active',
                'introducedAt': turn_id, 'supersedes': previous['factId'] if previous else None,
                'invalidatedAt': None, 'topicId': topic.topic_id}
        if previous:
            if self.turn_sessions[previous['introducedAt']] >= session_index:
                raise ValueError('supersession requires an earlier session')
            previous.update(state='superseded', invalidatedAt=turn_id)
        self.facts.append(fact)
        self.by_id[fact_id] = fact
        self.current[topic.topic_id] = fact
        self.turn_sessions[turn_id] = session_index
        return fact, previous

    def expire(self, topic_id, turn_id, session_index):
        fact = self.current.pop(topic_id)
        fact.update(state='expired', invalidatedAt=turn_id)
        self.turn_sessions[turn_id] = session_index
        return fact

    def active_at(self, fact, checkpoint):
        return (self.turn_sessions[fact['introducedAt']] <= checkpoint and
                (fact['invalidatedAt'] is None or
                 self.turn_sessions[fact['invalidatedAt']] > checkpoint))


class ProbePlacer:
    def __init__(self, graph, pool):
        self.graph = graph
        self.pool = pool

    def place(self, scale, anchors):
        probes = []
        checkpoints = sorted({scale} | {1 + d for d in (10, 100, 500, 999) if 1 + d <= scale})
        for checkpoint in checkpoints:
            first, old, expired, other, updated = anchors
            topic = self.pool.by_id[first['topicId']]
            update_topic = self.pool.by_id[updated['topicId']]
            expired_topic = self.pool.by_id[expired['topicId']]
            definitions = [
                ('information-extraction', f'What is the {topic.slot_name} for {topic.label}?',
                 [first], [], 'current', 'exact'),
                ('cross-session-reasoning',
                 f'List the values for {topic.label} and {self.pool.by_id[other["topicId"]].label}, in that order.',
                 [first, other], [], 'current', 'ordered-list'),
                ('temporal-reasoning',
                 f'Before it expired, what was the {expired_topic.slot_name} for {expired_topic.label}?',
                 [expired], [], 'historical', 'exact'),
                ('knowledge-update',
                 f'What is the current {update_topic.slot_name} for {update_topic.label}, after the change?',
                 [updated], [old], 'current', 'exact'),
                ('abstention', 'What is my emergency verification phrase? I have not shared it.',
                 [], [], 'current', 'abstain'),
            ]
            for ability, question, required, forbidden, as_of, match_type in definitions:
                if any(self.graph.turn_sessions[f['introducedAt']] >= checkpoint for f in required):
                    raise ValueError('probe requires evidence from a prior session')
                if as_of == 'current' and any(not self.graph.active_at(f, checkpoint) for f in required):
                    raise ValueError('current probe cannot use invalidated facts')
                values = [f['value'] for f in required]
                answers = ([json.dumps(values, ensure_ascii=False)] if match_type == 'ordered-list'
                           else values if values else ['I do not know.'])
                distance = checkpoint - min(self.graph.turn_sessions[f['introducedAt']]
                                            for f in required) if required else 0
                probes.append({'probeId': f'p-{len(probes) + 1:05d}',
                               'afterSessionId': f's-{checkpoint:04d}',
                               'question': question, 'ability': ability,
                               'answerability': ability != 'abstention', 'answerAsOf': as_of,
                               'sessionDistance': distance,
                               'expected': {'matchType': match_type, 'acceptedAnswers': answers,
                                            'requiredFactIds': [f['factId'] for f in required],
                                            'forbiddenFactIds': [f['factId'] for f in forbidden]}})
        return probes


def generate(seed, scale, *, held_out=False):
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if held_out != (seed >= EVALUATION_SEED_START):
        raise ValueError('development and reserved held-out seed ranges must be separate')
    if type(scale) is not int or scale not in SCALES:
        raise ValueError(f'sessions must be one of {SCALES}')
    rng = random.Random(seed)
    pool = TopicPool()
    renderer = TurnRenderer(rng)
    topics = list(pool.topics)
    rng.shuffle(topics)
    fixed = topics[:6]
    # At the smallest scale, repeat enough topics rather than spending every
    # remaining slot on a new topic. Larger fixtures cover the entire pool.
    rotating = topics[6:6 + min(len(topics) - 6, max(16, scale))]
    graph = FactGraph()
    sessions = []
    originals = {}
    updated = None
    other = None
    epoch = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    for index in range(1, scale + 1):
        selected = fixed + [rotating[((index - 1) * 4 + offset) % len(rotating)]
                            for offset in range(4)]
        turns = []
        for offset, topic in enumerate(selected):
            turn_id = f't-{index:04d}-{offset + 1:02d}'
            introduced, referenced = [], []
            if index == 2 and offset == 2:
                fact = graph.expire(topic.topic_id, turn_id, index)
                text = renderer.expire(topic, fact['value'])
                referenced = [fact['factId']]
            elif index > 2 and offset == 2:
                fact = originals[2]
                text = f"We no longer have a current {topic.slot_name} for {topic.label}; the old value expired."
                referenced = [fact['factId']]
            elif (topic.topic_id not in graph.current or (index == 2 and offset == 1)
                  or (index == 3 and offset == 3)):
                value = topic.value(rng)
                previous = graph.current.get(topic.topic_id)
                if previous and value == previous['value']:
                    # Deterministic finite vocabulary must still produce a real update.
                    for _ in range(100):
                        value = topic.value(rng)
                        if value != previous['value']:
                            break
                    else:
                        raise ValueError('topic value generator cannot produce an update')
                fact, previous = graph.plant(topic, value, turn_id, index)
                text = renderer.introduce(topic, value, previous['value'] if previous else None)
                introduced = [fact['factId']]
                referenced = [previous['factId']] if previous else []
                if index == 1 and offset < 6:
                    originals[offset] = fact
                if index == 2 and offset == 1:
                    updated = fact
                if index == 3 and offset == 3:
                    other = fact
            else:
                fact = graph.current[topic.topic_id]
                text = renderer.reference(topic, fact['value'])
                referenced = [fact['factId']]
            turns.append({'turnId': turn_id, 'role': 'user', 'text': text,
                          'introducedFactIds': introduced, 'referencedFactIds': referenced,
                          'topicId': topic.topic_id})
        sessions.append({'sessionId': f's-{index:04d}',
                         'timestamp': (epoch + timedelta(days=index - 1)).isoformat().replace('+00:00', 'Z'),
                         'turns': turns})
    probes = ProbePlacer(graph, pool).place(scale, [originals[0], originals[1], originals[2],
                                                 other, updated])
    fixture = {'schemaVersion': '1.0', 'generatorVersion': GENERATOR_VERSION,
               'fixtureId': f'almm-v{GENERATOR_VERSION}-{scale}-{seed}', 'seed': seed,
               'sessions': sessions, 'facts': graph.facts, 'probes': probes}
    SchemaValidator.validate(fixture)
    fixture['contentHash'] = hashlib.sha256(canonical_json(fixture)).hexdigest()
    return fixture


def adapter_view(fixture):
    """Allowlist, never removal-based: new scorer fields cannot leak by default."""
    return {'sessions': [{'sessionId': s['sessionId'], 'timestamp': s['timestamp'],
                          'turns': [{key: t[key] for key in ('turnId', 'role', 'text')}
                                    for t in s['turns']]} for s in fixture['sessions']],
            'probes': [{key: p[key] for key in ('probeId', 'afterSessionId', 'question')}
                       for p in fixture['probes']]}
