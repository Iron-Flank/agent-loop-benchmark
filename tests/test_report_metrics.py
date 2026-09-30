"""Consumer-visible reporting metrics and undefined-value boundaries."""
from copy import deepcopy
import math
import unittest

from almm_report.metrics import build_report, distribution


def inputs(count=2):
    fixture = {
        'fixtureId': 'fixture-a',
        'sessions': [
            {'sessionId': 's1', 'turns': [
                {'turnId': 't1', 'introducedFactIds': ['f1', 'f2']}]},
            {'sessionId': 's2', 'turns': [
                {'turnId': 't2', 'introducedFactIds': ['f3']}]},
            {'sessionId': 's3', 'turns': []},
        ],
        'facts': [{'factId': 'f1', 'introducedAt': 't1'},
                  {'factId': 'f2', 'introducedAt': 't1'},
                  {'factId': 'f3', 'introducedAt': 't2'}],
        'probes': [],
    }
    probes, requests, scores = [], [], []
    for i in range(count):
        pid, rid = f'p{i}', f'r{i}'
        fixture['probes'].append({
            'probeId': pid, 'afterSessionId': 's3', 'ability': 'recall',
            'answerability': True,
            'expected': {'requiredFactIds': ['f1'], 'matchType': 'exact'},
        })
        request = {
            'requestId': rid, 'probeId': pid, 'status': 'ok',
            'totalTokens': 100, 'tierTokens': {
                'stable': 20, 'semi-stable': 30, 'unstable': 45},
            'requestOverheadTokens': 5, 'latencyMs': 10 + i,
            'segments': [{'tier': 'unstable', 'content': 'evidence'}],
        }
        requests.append(request)
        probes.append({'probeId': pid, 'eligibleForAccuracy': True,
                       'status': 'ok', 'answer': 'answer', 'latencyMs': 20 + i,
                       'requests': [deepcopy(request)], 'sourceIds': []})
        scores.append({'probeId': pid, 'eligibleForAccuracy': True,
                       'normalizedResult': {'correct': True, 'judgment': 'pass'}})
    manifest = {'adapter': {'name': 'runtime', 'revision': 'rev'},
                'sessionCount': 3}
    return fixture, manifest, probes, requests, scores


def fail(values, index, category):
    values[2][index].update(eligibleForAccuracy=False, status='error', category=category)
    values[4][index].update(eligibleForAccuracy=False, failureCategory=category,
                            normalizedResult={'correct': False, 'judgment': 'incomplete'})


class ReportMetricsTests(unittest.TestCase):
    def test_accuracy_and_completion_have_distinct_denominators(self):
        values = inputs(4)
        values[4][1]['normalizedResult'].update(correct=False, judgment='fail')
        fail(values, 2, 'provider')
        values[2].pop()
        values[3].pop()
        values[4][3].update(eligibleForAccuracy=False, failureCategory='unattempted',
                            normalizedResult={'correct': False, 'judgment': 'incomplete'})
        report = build_report(*values)
        self.assertEqual(report['accuracy'], {
            'correct': 1, 'answered': 2, 'total': 4,
            'accuracy': 0.5, 'completionRate': 0.5})
        self.assertEqual(report['operationalTelemetry']['unattemptedCount'], 1)
        self.assertEqual(report['breakdown']['ability']['recall'], report['accuracy'])
        self.assertEqual(report['breakdown']['sessionDistance']['2']['total'], 4)
        self.assertEqual(report['breakdown']['evidenceCardinality']['1']['answered'], 2)
        self.assertEqual(report['breakdown']['fixtureId']['fixture-a']['correct'], 1)

    def test_investigation_threshold_is_strict_and_budget_is_separate(self):
        values = inputs(20)
        fail(values, 0, 'timeout')
        fail(values, 1, 'budget')
        report = build_report(*values)
        operations = report['operationalTelemetry']
        self.assertEqual(operations['incompleteProbeRate'], 0.05)
        self.assertFalse(operations['investigationRequired'])
        self.assertEqual(operations['overBudgetFailures'], 1)
        fail(values, 2, 'adapter')
        self.assertTrue(build_report(*values)['operationalTelemetry']['investigationRequired'])

    def test_empty_and_no_answer_denominators_are_null(self):
        report = build_report(*inputs(0))
        self.assertIsNone(report['accuracy']['accuracy'])
        self.assertIsNone(report['accuracy']['completionRate'])
        self.assertIsNone(report['operationalTelemetry']['incompleteProbeRate'])
        self.assertIsNone(report['tokenEfficiency']['accuracyPer1000InputTokens'])
        self.assertIsNone(report['abstention']['falseAbstentionRate'])
        values = inputs(1)
        fail(values, 0, 'budget')
        report = build_report(*values)
        self.assertIsNone(report['accuracy']['accuracy'])
        self.assertEqual(report['accuracy']['completionRate'], 0)

    def test_nearest_rank_p95_and_finite_value_validation(self):
        result = distribution(range(1, 21))
        self.assertEqual(result, {'count': 20, 'mean': 10.5, 'p50': 10,
                                  'p95': 19, 'min': 1, 'max': 20})
        self.assertEqual(distribution([]), {'count': 0, 'mean': None, 'p50': None,
                                            'p95': None, 'min': None, 'max': None})
        for value in (math.inf, math.nan, True, '1'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                distribution([value])

    def test_all_requests_contribute_tokens_and_latency(self):
        values = inputs(2)
        values[3].append({'requestId': 'turn-request', 'turnId': 't1',
                          'status': 'error', 'category': 'budget', 'totalTokens': 300,
                          'tierTokens': {'stable': 50, 'semi-stable': 100, 'unstable': 140},
                          'requestOverheadTokens': 10, 'latencyMs': 100,
                          'segments': []})
        report = build_report(*values)
        tokens = report['tokenEfficiency']
        self.assertAlmostEqual(tokens['assembledContextTokens']['mean'], 500 / 3)
        self.assertEqual(tokens['assembledContextTokens']['p95'], 300)
        self.assertEqual(tokens['tierTokens'], {'stable': 90, 'semi-stable': 160, 'unstable': 230})
        self.assertEqual(tokens['requestOverheadTokens'], 20)
        self.assertAlmostEqual(tokens['accuracyPer1000InputTokens'], 6)
        self.assertEqual(report['operationalTelemetry']['requestLatencyMs']['p95'], 100)
        self.assertEqual(report['operationalTelemetry']['probeLatencyMs']['p95'], 21)

    def test_provenance_normalizes_turns_and_deduplicates_facts(self):
        values = inputs(2)
        values[0]['probes'][0]['expected']['requiredFactIds'] = ['f1', 'f3']
        values[3][0]['segments'][0]['sourceIds'] = ['t1', 'f1', 'unknown']
        values[3][1]['segments'][0]['sourceIds'] = []
        report = build_report(*values)['contextRelevance']
        self.assertEqual(report['status'], 'reported')
        self.assertEqual(report['reportedCount'], 2)
        first, second = report['perProbe']
        self.assertAlmostEqual(first['precision'], 1 / 3)
        self.assertEqual(first['recall'], 0.5)
        self.assertEqual(second['precision'], 0)
        self.assertEqual(second['recall'], 0)

    def test_reference_turn_retrieval_keeps_original_session_distance(self):
        values = inputs(1)
        values[0]['sessions'][1]['turns'].append({
            'turnId': 'reference-turn', 'introducedFactIds': [], 'referencedFactIds': ['f1']})
        values[3][0]['segments'][0]['sourceIds'] = ['reference-turn']
        report = build_report(*values)
        self.assertEqual(report['contextRelevance']['precision'], 1)
        self.assertEqual(report['contextRelevance']['recall'], 1)
        self.assertEqual(report['breakdown']['sessionDistance']['2']['correct'], 1)

    def test_fact_ledger_supplies_provenance_without_turn_annotations(self):
        values = inputs(1)
        values[0]['sessions'][0]['turns'][0].pop('introducedFactIds')
        values[3][0]['segments'][0]['sourceIds'] = ['t1']
        report = build_report(*values)
        self.assertEqual(report['contextRelevance']['precision'], 0.5)
        self.assertEqual(report['contextRelevance']['recall'], 1)
        self.assertEqual(report['breakdown']['sessionDistance']['2']['correct'], 1)

    def test_factless_retrieved_turn_is_false_positive(self):
        values = inputs(1)
        values[0]['sessions'][1]['turns'].append({
            'turnId': 'unrelated-turn', 'introducedFactIds': [], 'referencedFactIds': []})
        values[3][0]['segments'][0]['sourceIds'] = ['f1', 'unrelated-turn']
        relevance = build_report(*values)['contextRelevance']
        self.assertEqual(relevance['precision'], 0.5)
        self.assertEqual(relevance['recall'], 1)

    def test_flattened_empty_sources_do_not_imply_provenance(self):
        values = inputs(2)
        report = build_report(*values)['contextRelevance']
        self.assertEqual(report['status'], 'not-reported')
        self.assertEqual(report['notReportedCount'], 2)
        self.assertIsNone(report['precision'])
        self.assertIsNone(report['recall'])
        values[3][0]['segments'][0]['sourceIds'] = ['f1']
        report = build_report(*values)['contextRelevance']
        self.assertEqual(report['status'], 'partial')
        self.assertEqual((report['reportedCount'], report['notReportedCount']), (1, 1))
        self.assertIsNone(report['precision'])
        self.assertIsNone(report['recall'])
        self.assertEqual(report['reportedPrecision'], 1)
        self.assertEqual(report['reportedRecall'], 1)

    def test_empty_gold_relevance_is_undefined_not_perfect(self):
        values = inputs(1)
        values[0]['probes'][0]['expected']['requiredFactIds'] = []
        values[3][0]['segments'][0]['sourceIds'] = []
        result = build_report(*values)['contextRelevance']['perProbe'][0]
        self.assertIsNone(result['recall'])
        self.assertIsNone(result['precision'])

    def test_semantic_abstention_is_not_synonymous_with_correctness(self):
        values = inputs(4)
        values[0]['probes'][0]['answerability'] = False
        values[0]['probes'][1]['answerability'] = False
        values[4][0]['normalizedResult'] = {'correct': True, 'judgment': 'abstain'}
        values[4][1]['normalizedResult'] = {'correct': False, 'judgment': 'fail'}
        values[4][2]['normalizedResult'] = {'correct': False, 'judgment': 'abstain'}
        values[4][3]['normalizedResult'] = {'correct': False, 'judgment': 'fail'}
        result = build_report(*values)['abstention']
        self.assertEqual(result['correctAbstentions'], 1)
        self.assertEqual(result['unsupportedAnswers'], 1)
        self.assertEqual(result['falseAbstentions'], 1)
        self.assertEqual(result['correctAbstentionRate'], 0.5)
        self.assertEqual(result['unsupportedAnswerRate'], 0.5)
        self.assertEqual(result['falseAbstentionRate'], 0.5)
        values[4][0]['normalizedResult']['correct'] = False
        result = build_report(*values)['abstention']
        self.assertEqual(result['correctAbstentions'], 0)
        self.assertEqual(result['unanswerableAbstentions'], 1)
        self.assertEqual(result['unsupportedAnswers'], 1)

    def test_distance_label_overrides_derived_oldest_fact_distance(self):
        values = inputs(1)
        values[0]['probes'][0]['expected']['requiredFactIds'] = ['f1', 'f3']
        self.assertEqual(build_report(*values)['breakdown']['sessionDistance']['2']['total'], 1)
        self.assertEqual(build_report(*values)['breakdown']['evidenceCardinality']['2']['answered'], 1)
        values[0]['probes'][0]['sessionDistance'] = 99
        self.assertEqual(build_report(*values)['breakdown']['sessionDistance']['99']['total'], 1)

    def test_invalid_scores_cannot_silently_change_accuracy(self):
        mutations = [
            lambda v: v[4].append(deepcopy(v[4][0])),
            lambda v: v[4].pop(),
            lambda v: v[4][0].update(probeId='unknown'),
            lambda v: v[4][0].update(eligibleForAccuracy=False),
            lambda v: v[4][0]['normalizedResult'].update(correct=1),
            lambda v: v[3][0].update(totalTokens=math.inf),
            lambda v: v[2][0].update(latencyMs=-1),
            lambda v: v[3][0].update(latencyMs=math.nan),
            lambda v: v[3][0].update(requestOverheadTokens=-1),
            lambda v: v[3][0]['tierTokens'].update(stable=True),
        ]
        for mutate in mutations:
            values = inputs(2)
            mutate(values)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                build_report(*values)


if __name__ == '__main__':
    unittest.main()
