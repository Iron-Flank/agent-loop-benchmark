import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from almm_scorer.calibration import CalibrationGate


REAL_CORPUS = Path(__file__).resolve().parents[1] / 'calibration' / 'scorer-v1.0.0'
ABILITIES = ('information-extraction', 'cross-session-reasoning',
             'temporal-reasoning', 'knowledge-update', 'abstention')
MATCHES = ('semantic',) * 4 + ('exact',) * 2 + ('numeric',) * 2 + ('ordered-list', 'abstain')


def synthetic_corpus():
    """Independently authored test data; provenance simulates humans, not real review."""
    rows = []
    for index in range(100):
        kind = MATCHES[index % 10]
        fact_id = f'synthetic-f-{index}'
        expected = {
            'matchType': kind, 'acceptedAnswers': ['blue'],
            'requiredFactIds': [fact_id], 'forbiddenFactIds': [],
        }
        candidate = 'blue' if index % 3 else 'red'
        gold = 'pass' if candidate == 'blue' else 'fail'
        if kind == 'semantic':
            expected.update(rubric='All required claims, no contradictions.',
                            requiredClaims=['The color is blue.'],
                            disallowedContradictions=['The color is red.'])
        elif kind == 'numeric':
            expected.update(targetNumber='10', tolerance='0.5',
                            toleranceMode='absolute-inclusive', acceptedAnswers=['10'])
            candidate = '10' if index % 3 else '11'
        elif kind == 'ordered-list':
            expected.update(items=['blue', 'green'], acceptedAnswers=['["blue", "green"]'])
            candidate = '["blue", "green"]' if index % 3 else '["green", "blue"]'
        elif kind == 'abstain':
            expected.update(acceptedAnswers=['I do not know.'], requiredFactIds=[])
            candidate = 'I do not know.' if index % 3 else 'blue'
            gold = 'abstain' if index % 3 else 'fail'
        # A false abstention must be compared as abstain, even though incorrect.
        if index == 0:
            candidate, gold = 'I do not know.', 'abstain'
        rows.append({
            'probeId': f'synthetic-{index:03d}', 'ability': ABILITIES[index // 20],
            'question': f'What is the recorded value for synthetic entry {index}?',
            'candidateAnswer': candidate, 'expected': expected,
            'goldJudgment': gold, 'justification': f'Independent test label {index}: {gold}.',
            'edgeCases': ['synthetic'], 'answerability': kind != 'abstain',
            'answerAsOf': 'current', 'afterSessionIndex': 10,
            'evidence': [{'factId': fact_id, 'sessionIndex': 1,
                          'text': 'The recorded value is blue.', 'state': 'active'}],
        })
    probes = {
        'schemaVersion': 'calibration-1.0', 'calibrationSetId': 'synthetic-test-only',
        'calibrationSetVersion': 'test-1', 'frozen': True,
        'labelingMethod': 'Synthetic fixture simulating independent human labels; not real annotations.',
        'labelProvenance': {'kind': 'human', 'labelerId': 'synthetic-labeler'},
        'probes': rows,
    }
    review = {
        'calibrationSetId': probes['calibrationSetId'], 'reviewedAt': '2026-01-01T00:00:00Z',
        'reviewMethod': 'Synthetic fixture simulating independent human review.',
        'reviewProvenance': {'kind': 'human', 'reviewerId': 'synthetic-reviewer'},
        'result': 'passed',
        'records': [{'probeId': row['probeId'], 'confirmedGoldJudgment': row['goldJudgment'],
                     'justificationConsistent': True, 'evidenceConsistent': True,
                     'reviewNote': row['justification']} for row in rows],
    }
    return probes, review


def freeze(directory, probes, review):
    """Freeze independently created temporary fixtures, never repository labels."""
    def save(name, value):
        content = (json.dumps(value, indent=2) + '\n').encode()
        (directory / name).write_bytes(content)
        return hashlib.sha256(content).hexdigest()

    probes_hash = save('probes.json', probes)
    review['probesSha256'] = probes_hash
    review_hash = save('review.json', review)
    manifest = {
        'schemaVersion': 'calibration-manifest-1.0',
        'calibrationSetId': probes['calibrationSetId'],
        'calibrationSetVersion': probes['calibrationSetVersion'],
        'retainedWithScorerVersion': '1.0.0', 'frozenAt': '2026-01-01T00:00:00Z',
        'files': {'probes.json': probes_hash, 'review.json': review_hash},
        'requiredAgreement': 0.95,
    }
    save('manifest.json', manifest)


def behavior_scorer(mismatches=()):
    def score(probe, answer):
        kind = probe['expected']['matchType']
        if answer == 'I do not know.':
            judgment = 'abstain'
        elif kind == 'numeric':
            judgment = 'pass' if abs(float(answer) - 10) <= 0.5 else 'fail'
        elif kind == 'ordered-list':
            judgment = 'pass' if json.loads(answer) == ['blue', 'green'] else 'fail'
        else:
            judgment = 'pass' if kind != 'abstain' and answer == 'blue' else 'fail'
        if probe['probeId'] in mismatches:
            judgment = 'fail' if judgment != 'fail' else 'pass'
        return {'judgment': judgment, 'correct': judgment == 'pass' or (
                    judgment == 'abstain' and not probe['answerability']),
                'normalizedAnswer': answer, 'matchingMethod': kind, 'judgeOutput': None}
    return score


class CalibrationGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.probes, self.review = synthetic_corpus()
        freeze(self.directory, self.probes, self.review)

    def evaluate(self, score=None, **kwargs):
        return CalibrationGate(self.directory).evaluate(
            score or behavior_scorer(), scorer_version='1.0.0', **kwargs)

    def reject(self, diagnostic, **kwargs):
        with self.assertRaisesRegex(ValueError, diagnostic) as error:
            self.evaluate(**kwargs)
        return error.exception

    def test_95_percent_is_inclusive_and_reports_all_five_mismatches(self):
        wrong = [row['probeId'] for row in self.probes['probes'][:5]]
        report = self.evaluate(behavior_scorer(wrong))
        self.assertTrue(report['approved'])
        self.assertTrue(report['canonical'])
        self.assertEqual((report['matched'], report['total'], report['agreement']), (95, 100, 0.95))
        self.assertEqual({row['probeId'] for row in report['mismatches']}, set(wrong))
        self.assertEqual(len(report['judgments']), 100)

    def test_94_percent_rejects_and_attaches_full_report(self):
        wrong = [row['probeId'] for row in self.probes['probes'][:6]]
        error = self.reject('94/100.*95%', score=behavior_scorer(wrong))
        self.assertFalse(error.report['approved'])
        self.assertEqual(error.report['matched'], 94)
        self.assertEqual({row['probeId'] for row in error.report['mismatches']}, set(wrong))

    def test_agreement_uses_judgment_not_abstention_correctness(self):
        report = self.evaluate()
        self.assertEqual(report['agreement'], 1.0)
        first = report['judgments'][0]
        self.assertEqual(first['goldJudgment'], 'abstain')
        self.assertEqual(first['judgment'], 'abstain')
        self.assertFalse(first['correct'])

    def test_scorer_does_not_receive_labels_or_mutate_frozen_gold(self):
        score = behavior_scorer()
        original = (self.directory / 'probes.json').read_bytes()
        def mutate(probe, answer):
            self.assertNotIn('goldJudgment', probe)
            self.assertNotIn('justification', probe)
            result = score(probe, answer)
            probe['expected']['acceptedAnswers'].clear()
            return result
        self.assertEqual(self.evaluate(mutate)['agreement'], 1.0)
        self.assertEqual((self.directory / 'probes.json').read_bytes(), original)

    def test_report_pins_full_judge_settings_scorer_and_hashes(self):
        config = {'provider': 'test', 'model': 'pinned-model-2026-01',
                  'temperature': 0, 'maxTokens': 300,
                  'rubricVersion': '1.0', 'settings': {'seed': 12}}
        report = self.evaluate(judge_config=config)
        config['settings']['seed'] = 99
        self.assertEqual(report['judgeConfig']['settings']['seed'], 12)
        self.assertEqual(report['scorerVersion'], '1.0.0')
        for name in ('manifest.json', 'probes.json', 'review.json'):
            self.assertEqual(report['calibrationHashes'][name],
                             hashlib.sha256((self.directory / name).read_bytes()).hexdigest())
        self.assertEqual(report['labelProvenance']['kind'], 'human')

    def test_evaluation_reruns_current_scorer_instead_of_reusing_approval(self):
        gate = CalibrationGate(self.directory)
        self.assertTrue(gate.evaluate(behavior_scorer(), '1.0.0')['approved'])
        with self.assertRaises(ValueError) as error:
            gate.evaluate(behavior_scorer([row['probeId'] for row in self.probes['probes']]), '1.1.0')
        self.assertEqual(error.exception.report['scorerVersion'], '1.1.0')
        self.assertEqual(error.exception.report['matched'], 0)

    def test_tampering_with_either_frozen_file_rejects(self):
        for name in ('probes.json', 'review.json'):
            with self.subTest(name=name):
                freeze(self.directory, self.probes, self.review)
                with (self.directory / name).open('ab') as stream:
                    stream.write(b' ')
                self.reject('hash|SHA-256')

    def test_refreezing_modified_official_corpus_cannot_claim_human_labels(self):
        probes = json.loads((REAL_CORPUS / 'probes.json').read_text())
        review = json.loads((REAL_CORPUS / 'review.json').read_text())
        probes['labelProvenance'] = {'kind': 'human', 'labelerId': 'pretend-human'}
        review['reviewProvenance'] = {'kind': 'human', 'reviewerId': 'pretend-reviewer'}
        freeze(self.directory, probes, review)
        self.reject('frozen|hash|immutable')

    def test_renaming_agent_rows_cannot_rebrand_them_as_human(self):
        probes = json.loads((REAL_CORPUS / 'probes.json').read_text())
        review = json.loads((REAL_CORPUS / 'review.json').read_text())
        probes['calibrationSetId'] = review['calibrationSetId'] = 'renamed-agent-corpus'
        probes['labelProvenance'] = {'kind': 'human', 'labelerId': 'pretend-human'}
        review['reviewProvenance'] = {'kind': 'human', 'reviewerId': 'pretend-reviewer'}
        probes['labelingMethod'] = 'Claimed human authorship.'
        review['reviewMethod'] = 'Claimed human review.'
        freeze(self.directory, probes, review)
        labels = {row['probeId']: row['goldJudgment'] for row in probes['probes']}
        def oracle(probe, answer):
            return {'judgment': labels[probe['probeId']], 'correct': False,
                    'normalizedAnswer': answer, 'matchingMethod': probe['expected']['matchType'],
                    'judgeOutput': None}
        error = self.reject('human', score=oracle)
        self.assertEqual(error.report['labelProvenance']['kind'], 'agent')

    def test_schema_stratification_and_review_coverage_are_enforced(self):
        cases = (
            ('duplicate', lambda p, r: p['probes'][1].update(probeId=p['probes'][0]['probeId'])),
            ('justification', lambda p, r: p['probes'][0].update(justification='')),
            ('goldJudgment', lambda p, r: p['probes'][0].update(goldJudgment='unknown')),
            ('90.*110', lambda p, r: p.update(probes=p['probes'][:89])),
            ('ability|abilities', lambda p, r: [row.update(ability='information-extraction') for row in p['probes']]),
            ('match|semantic', lambda p, r: [row['expected'].update(matchType='exact') for row in p['probes'] if row['expected']['matchType'] == 'semantic']),
            ('ability|abilities', lambda p, r: [row.update(ability='information-extraction') for row in p['probes'][80:91]]),
            ('semantic', lambda p, r: [row['expected'].update(matchType='exact') for row in p['probes'][:30] if row['expected']['matchType'] == 'semantic']),
            ('match', lambda p, r: [row['expected'].update(matchType='exact') for row in p['probes'] if row['expected']['matchType'] == 'ordered-list']),
            ('review', lambda p, r: r['records'].pop()),
            ('review|confirmed', lambda p, r: r['records'][0].update(confirmedGoldJudgment='pass')),
            ('review', lambda p, r: r['records'][0].update(evidenceConsistent=False)),
            ('requiredClaims', lambda p, r: p['probes'][0]['expected'].update(requiredClaims=[])),
        )
        for diagnostic, change in cases:
            with self.subTest(diagnostic=diagnostic):
                probes, review = synthetic_corpus()
                change(probes, review)
                freeze(self.directory, probes, review)
                self.reject(diagnostic)

    def test_missing_or_agent_provenance_cannot_approve_canonical(self):
        for provenance in (None, {'kind': 'agent', 'labelerId': 'agent'}):
            with self.subTest(provenance=provenance):
                probes, review = synthetic_corpus()
                probes['labelProvenance'] = provenance
                freeze(self.directory, probes, review)
                error = self.reject('human')
                self.assertFalse(error.report['approved'])
                self.assertEqual(error.report['total'], 100)

    def test_invalid_scorer_identity_and_non_json_config_reject(self):
        for version in ('', None):
            with self.subTest(version=version):
                with self.assertRaisesRegex(ValueError, 'scorer_version|scorer version'):
                    CalibrationGate(self.directory).evaluate(behavior_scorer(), version)
        self.reject('judge_config|judge configuration', judge_config={'temperature': float('nan')})

    def test_invalid_scorer_judgment_rejects_instead_of_counting_agreement(self):
        score = behavior_scorer()
        def invalid(probe, answer):
            result = score(probe, answer)
            if probe['probeId'] == 'synthetic-000':
                result['judgment'] = 'unknown'
            return result
        self.reject('judgment', score=invalid)

    def test_real_corpus_is_intact_but_explicitly_nonhuman(self):
        probes = json.loads((REAL_CORPUS / 'probes.json').read_text())
        labels = {row['probeId']: row['goldJudgment'] for row in probes['probes']}
        # Oracle isolates integrity/provenance validation, not scorer capability.
        def gold_oracle(probe, answer):
            judgment = labels[probe['probeId']]
            return {'judgment': judgment, 'correct': judgment == 'pass',
                    'normalizedAnswer': answer, 'matchingMethod': probe['expected']['matchType'],
                    'judgeOutput': None}
        gate = CalibrationGate(REAL_CORPUS)
        with self.assertRaisesRegex(ValueError, 'human') as error:
            gate.evaluate(gold_oracle, '1.0.0')
        self.assertEqual(error.exception.report['matched'], 100)
        self.assertEqual(error.exception.report['labelProvenance']['kind'], 'agent')
        report = gate.evaluate(gold_oracle, '1.0.0', require_human=False)
        self.assertTrue(report['approved'])
        self.assertFalse(report['canonical'])
        self.assertEqual(report['labelProvenance']['labelingMethod'], probes['labelingMethod'])
        self.assertEqual(report['calibrationHashes']['probes.json'],
                         'b881d3a250edcfbba967015bd4d2ff36a284b5d7a1ec41b569e584d6b2f14430')


if __name__ == '__main__':
    unittest.main()
