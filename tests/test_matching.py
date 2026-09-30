import json
from pathlib import Path
import unittest

from almm_scorer.matcher import match_atomic, normalize


def probe(match_type, answers, **settings):
    return {'answerability': match_type != 'abstain',
            'expected': {'matchType': match_type, 'acceptedAnswers': answers, **settings}}


class AtomicMatchingTests(unittest.TestCase):
    def test_unicode_case_and_whitespace_normalize_without_substrings(self):
        gold = probe('exact', ['Straße A-12'])
        self.assertEqual(normalize('  ＳＴＲＡＳＳＥ\tA-12  '), 'strasse a-12')
        self.assertTrue(match_atomic(gold, '  ＳＴＲＡＳＳＥ\tA-12  ')['correct'])
        for answer in ('The code is Straße A-12', 'Straße A-123', 'Straße A12'):
            with self.subTest(answer=answer):
                self.assertEqual(match_atomic(gold, answer)['judgment'], 'fail')

    def test_boolean_values_are_not_integer_aliases(self):
        self.assertTrue(match_atomic(probe('exact', [True]), ' TRUE ')['correct'])
        self.assertFalse(match_atomic(probe('exact', [True]), 1)['correct'])
        self.assertFalse(match_atomic(probe('exact', [False]), 0)['correct'])
        self.assertTrue(match_atomic(probe('exact', [23]), '23')['correct'])

    def test_ordered_lists_normalize_elements_but_preserve_order_and_count(self):
        gold = probe('ordered-list', ['["pharmacy", "bakery", "post office"]'])
        self.assertTrue(match_atomic(gold, '["PHARMACY", " bakery ", "post  office"]')['correct'])
        for answer in ('["bakery", "pharmacy", "post office"]',
                       '["pharmacy", "bakery"]',
                       '["pharmacy", "bakery", "post office", "bank"]',
                       'pharmacy, bakery, post office', '{"0": "pharmacy"}'):
            with self.subTest(answer=answer):
                self.assertFalse(match_atomic(gold, answer)['correct'])
        self.assertEqual(match_atomic(gold, ['pharmacy', 'bakery', 'post office'])[
            'normalizedAnswer'], ['pharmacy', 'bakery', 'post office'])

    def test_decimal_tolerance_is_explicit_inclusive_and_not_float_rounded(self):
        gold = probe('numeric', ['0.3'], targetNumber='0.3', tolerance='0.1')
        for answer in ('0.2', '0.4', '4e-1'):
            with self.subTest(answer=answer):
                self.assertTrue(match_atomic(gold, answer)['correct'])
        self.assertFalse(match_atomic(gold, '0.4000000000000000000000000001')['correct'])
        exact = probe('numeric', ['2.0'])
        self.assertTrue(match_atomic(exact, '2')['correct'])
        self.assertFalse(match_atomic(exact, '2.00000001')['correct'])
        for tolerance in ('-0.01', 'NaN', 'Infinity', True):
            with self.subTest(tolerance=tolerance), self.assertRaisesRegex(ValueError, 'tolerance'):
                match_atomic(probe('numeric', ['2'], tolerance=tolerance), '2')

    def test_numeric_rejects_nonfinite_and_partial_parses(self):
        gold = probe('numeric', ['2'], tolerance='0.5')
        for answer in ('NaN', 'Infinity', '-Infinity', '2 liters', 'answer: 2',
                       '1,999', '2_0', True, None):
            with self.subTest(answer=answer):
                self.assertEqual(match_atomic(gold, answer)['judgment'], 'fail')

    def test_correct_abstention_requires_exact_accepted_form(self):
        gold = probe('abstain', ['I do not know.', 'That information was not provided.'])
        result = match_atomic(gold, ' THAT information was NOT provided. ')
        self.assertEqual((result['judgment'], result['correct']), ('abstain', True))
        for answer in ('I do not know. But probably 4931.', '4931', '', None, 'I cannot answer.'):
            with self.subTest(answer=answer):
                self.assertEqual((match_atomic(gold, answer)['judgment'],
                                  match_atomic(gold, answer)['correct']), ('fail', False))

    def test_false_abstention_only_when_no_substantive_claim(self):
        gold = probe('exact', ['ASH-8042'])
        for answer in ('I do not know.', "I don't know", 'I cannot answer.',
                       'There is not enough information to say.',
                       'That information was not provided.'):
            with self.subTest(answer=answer):
                result = match_atomic(gold, answer)
                self.assertEqual((result['judgment'], result['correct']), ('abstain', False))
        for answer in ('I do not know, but the code is ASH-8042.',
                       'I cannot answer. The code is 4931.', '', None):
            with self.subTest(answer=answer):
                self.assertEqual(match_atomic(gold, answer)['judgment'], 'fail')

    def test_frozen_atomic_calibration_labels(self):
        corpus = json.loads(Path('calibration/scorer-v1.0.0/probes.json').read_text())
        rows = corpus['probes'] if isinstance(corpus, dict) else corpus
        for row in rows:
            if row['expected']['matchType'] == 'semantic':
                continue
            with self.subTest(probe_id=row['probeId']):
                result = match_atomic(row, row['candidateAnswer'])
                self.assertEqual(result['judgment'], row['goldJudgment'])


if __name__ == '__main__':
    unittest.main()
