import copy
import json
import unittest

from almm_fixture.validation import SchemaValidator


def valid_fixture():
    sessions = []
    for number in (1, 2):
        sessions.append({
            'sessionId': f's-{number:04d}',
            'timestamp': f'2025-01-{number:02d}T00:00:00Z',
            'turns': [{
                'turnId': f't-{number:04d}-{turn:02d}', 'role': 'user',
                'text': 'An ordinary conversation turn.',
                'introducedFactIds': [], 'referencedFactIds': [], 'topicId': 'travel',
            } for turn in range(1, 11)],
        })
    sessions[0]['turns'][0]['introducedFactIds'] = ['f-0001']
    sessions[1]['turns'][0]['referencedFactIds'] = ['f-0001']
    return {
        'schemaVersion': '1.0', 'generatorVersion': '1.0.0',
        'fixtureId': 'fixture-test', 'seed': 7, 'sessions': sessions,
        'facts': [{
            'factId': 'f-0001', 'value': 'Oslo', 'state': 'active',
            'introducedAt': 't-0001-01', 'supersedes': None,
            'invalidatedAt': None, 'topicId': 'travel',
        }],
        'probes': [{
            'probeId': 'p-0001', 'afterSessionId': 's-0002',
            'question': 'Which city?', 'ability': 'information-extraction',
            'answerability': True, 'answerAsOf': 'current', 'sessionDistance': 1,
            'expected': {'matchType': 'exact', 'acceptedAnswers': ['Oslo'],
                         'requiredFactIds': ['f-0001'], 'forbiddenFactIds': []},
        }],
    }


class FixtureValidationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = valid_fixture()

    def reject(self, diagnostic):
        with self.assertRaisesRegex(ValueError, diagnostic):
            SchemaValidator.validate(self.fixture)

    def add_update(self):
        old = self.fixture['facts'][0]
        old['state'] = 'superseded'
        old['invalidatedAt'] = 't-0001-02'
        self.fixture['sessions'][0]['turns'][1]['introducedFactIds'] = ['f-0002']
        self.fixture['facts'].append({
            'factId': 'f-0002', 'value': 'Bergen', 'state': 'active',
            'introducedAt': 't-0001-02', 'supersedes': 'f-0001',
            'invalidatedAt': None, 'topicId': 'travel',
        })

    def test_valid_fixture_without_hash_is_accepted(self):
        SchemaValidator.validate(self.fixture)

    def test_rejects_duplicate_ids_across_entity_kinds(self):
        self.fixture['probes'][0]['probeId'] = 'f-0001'
        self.reject('duplicate')

    def test_rejects_missing_schema_fields_and_wrong_types(self):
        for path, value in [('seed', True), ('facts', {}), ('schemaVersion', '2.0')]:
            with self.subTest(path=path):
                self.fixture = valid_fixture()
                self.fixture[path] = value
                self.reject(path)
        self.fixture = valid_fixture()
        del self.fixture['sessions'][0]['turns'][0]['topicId']
        self.reject('topicId')

    def test_rejects_wrong_turn_count_and_non_user_role(self):
        self.fixture['sessions'][0]['turns'].pop()
        self.reject('10')
        self.fixture = valid_fixture()
        self.fixture['sessions'][0]['turns'][0]['role'] = 'assistant'
        self.reject('role')

    def test_rejects_out_of_order_sessions_and_timestamps(self):
        self.fixture['sessions'].reverse()
        self.reject('sessionId')
        self.fixture = valid_fixture()
        self.fixture['sessions'][1]['timestamp'] = '2024-01-01T00:00:00Z'
        self.reject('timestamp')

    def test_rejects_unknown_reference_and_same_session_reference(self):
        self.fixture['sessions'][1]['turns'][0]['referencedFactIds'] = ['missing']
        self.reject('referencedFactIds')
        self.fixture = valid_fixture()
        self.fixture['sessions'][0]['turns'][1]['referencedFactIds'] = ['f-0001']
        self.reject('prior session')

    def test_facts_must_be_introduced_once_at_declared_turn(self):
        self.fixture['sessions'][0]['turns'][1]['introducedFactIds'] = ['f-0001']
        self.reject('introduced')
        self.fixture = valid_fixture()
        self.fixture['facts'][0]['introducedAt'] = 't-0001-02'
        self.reject('introducedAt')

    def test_rejects_unresolved_and_forward_supersession(self):
        self.add_update()
        self.fixture['facts'][1]['supersedes'] = 'missing'
        self.reject('supersedes')
        self.fixture = valid_fixture()
        self.add_update()
        self.fixture['facts'][0]['supersedes'] = 'f-0002'
        self.reject('supersedes')

    def test_rejects_invalidation_before_introduction_or_wrong_update_event(self):
        self.add_update()
        self.fixture['facts'][0]['invalidatedAt'] = 't-0001-01'
        self.reject('invalidatedAt')
        self.fixture = valid_fixture()
        self.add_update()
        self.fixture['facts'][0]['invalidatedAt'] = 't-0001-03'
        self.reject('invalidatedAt')

    def test_rejects_active_fact_with_invalidation(self):
        self.fixture['facts'][0]['invalidatedAt'] = 't-0002-01'
        self.reject('invalidatedAt')

    def test_rejects_unplaced_probe_and_non_prior_evidence(self):
        self.fixture['probes'][0]['afterSessionId'] = 'missing'
        self.reject('afterSessionId')
        self.fixture = valid_fixture()
        self.fixture['probes'][0]['afterSessionId'] = 's-0001'
        self.reject('prior session')

    def test_rejects_wrong_distance_and_answer(self):
        self.fixture['probes'][0]['sessionDistance'] = 2
        self.reject('sessionDistance')
        self.fixture = valid_fixture()
        self.fixture['probes'][0]['expected']['acceptedAnswers'] = ['Paris']
        self.reject('acceptedAnswers')

    def test_current_cannot_use_superseded_or_expired_evidence(self):
        self.add_update()
        self.reject('active')
        self.fixture = valid_fixture()
        self.fixture['facts'][0].update(state='expired', invalidatedAt='t-0002-01')
        self.reject('active')

    def test_historical_can_use_superseded_and_expired_evidence(self):
        self.add_update()
        self.fixture['probes'][0]['answerAsOf'] = 'historical'
        SchemaValidator.validate(self.fixture)
        self.fixture = valid_fixture()
        self.fixture['facts'][0].update(state='expired', invalidatedAt='t-0002-01')
        self.fixture['probes'][0]['answerAsOf'] = 'historical'
        SchemaValidator.validate(self.fixture)

    def test_current_before_later_invalidation_is_valid(self):
        self.fixture['facts'][0].update(state='expired', invalidatedAt='t-0002-10')
        # A third session makes a later event than the second-session checkpoint.
        third = copy.deepcopy(self.fixture['sessions'][1])
        third['sessionId'] = 's-0003'
        third['timestamp'] = '2025-01-03T00:00:00Z'
        for index, turn in enumerate(third['turns'], 1):
            turn['turnId'] = f't-0003-{index:02d}'
            turn['referencedFactIds'] = []
        self.fixture['sessions'].append(third)
        self.fixture['facts'][0]['invalidatedAt'] = 't-0003-01'
        SchemaValidator.validate(self.fixture)

    def test_knowledge_update_requires_old_chain_in_forbidden_ids(self):
        self.add_update()
        probe = self.fixture['probes'][0]
        probe['ability'] = 'knowledge-update'
        probe['expected'].update(requiredFactIds=['f-0002'], acceptedAnswers=['Bergen'])
        self.reject('forbiddenFactIds')
        probe['expected']['forbiddenFactIds'] = ['f-0001']
        SchemaValidator.validate(self.fixture)

    def test_ordered_list_requires_exact_values_in_fact_id_order(self):
        second = copy.deepcopy(self.fixture['facts'][0])
        second.update(factId='f-0002', value='Monday', introducedAt='t-0001-02')
        self.fixture['facts'].append(second)
        self.fixture['sessions'][0]['turns'][1]['introducedFactIds'] = ['f-0002']
        expected = self.fixture['probes'][0]['expected']
        self.fixture['probes'][0]['ability'] = 'cross-session-reasoning'
        expected.update(matchType='ordered-list', requiredFactIds=['f-0001', 'f-0002'],
                        acceptedAnswers=[json.dumps(['Oslo', 'Monday'])])
        SchemaValidator.validate(self.fixture)
        expected['acceptedAnswers'] = [json.dumps(['Monday', 'Oslo'])]
        self.reject('acceptedAnswers')

    def test_abstention_cannot_hide_evidence_or_factual_answer(self):
        probe = self.fixture['probes'][0]
        probe.update(ability='abstention', answerability=False, sessionDistance=0)
        probe['expected'].update(matchType='abstain', requiredFactIds=[],
                                 acceptedAnswers=['I do not know.'])
        SchemaValidator.validate(self.fixture)
        probe['expected']['forbiddenFactIds'] = ['f-0001']
        self.reject('abstain')
        probe['expected']['forbiddenFactIds'] = []
        probe['expected']['acceptedAnswers'] = ['Oslo']
        self.reject('abstain')

    def test_rejects_unknown_ability_and_non_boolean_answerability(self):
        self.fixture['probes'][0]['ability'] = 'invented'
        self.reject('ability')
        self.fixture = valid_fixture()
        self.fixture['probes'][0]['answerability'] = 1
        self.reject('answerability')

    def test_rejects_unknown_probe_evidence_and_empty_answers(self):
        self.fixture['probes'][0]['expected']['requiredFactIds'] = ['missing']
        self.reject('requiredFactIds')
        self.fixture = valid_fixture()
        self.fixture['probes'][0]['expected']['acceptedAnswers'] = []
        self.reject('acceptedAnswers')

    def test_rejects_unintroduced_facts_and_missing_replacement(self):
        self.fixture['sessions'][0]['turns'][0]['introducedFactIds'] = []
        self.reject('introducedAt')
        self.fixture = valid_fixture()
        self.fixture['facts'][0].update(state='superseded', invalidatedAt='t-0002-01')
        self.reject('replacement')

    def test_rejects_cross_session_reasoning_with_scalar_answer(self):
        self.fixture['probes'][0]['ability'] = 'cross-session-reasoning'
        self.reject('matchType')

    def test_rejects_malformed_nested_shapes(self):
        for field, value in [('acceptedAnswers', 'Oslo'), ('requiredFactIds', [{}]),
                             ('forbiddenFactIds', ['f-0001', 'f-0001'])]:
            with self.subTest(field=field):
                self.fixture = valid_fixture()
                self.fixture['probes'][0]['expected'][field] = value
                self.reject(field)

    def test_rejects_non_iso_or_timezone_free_timestamp(self):
        for value in ('not-a-date', '2025-01-01T00:00:00'):
            with self.subTest(value=value):
                self.fixture = valid_fixture()
                self.fixture['sessions'][0]['timestamp'] = value
                self.reject('timestamp')
