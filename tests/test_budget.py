import unittest

from almm_harness.budget import BudgetVerifier
from almm_harness.errors import BudgetFailure, AdapterFailure


def manifest():
    return {'tokenizer': {'name': 'characters-smoke', 'version': '1'},
            'stablePrefix': ['contract']}


def request(size):
    return {'requestId': 'r1', 'segments': [
        {'tier': 'stable', 'content': 'contract', 'tokenCount': 0},
        {'tier': 'semi-stable', 'content': 's' * 10, 'tokenCount': 0},
        {'tier': 'unstable', 'content': 'x' * (size - 18), 'tokenCount': 0,
         'sourceIds': ['t1']}]}


class BudgetTests(unittest.TestCase):
    def test_exact_ceiling_and_forged_counts(self):
        verifier = BudgetVerifier(manifest(), len)
        result = verifier.verify(request(25000))
        self.assertEqual(result['totalTokens'], 25000)
        self.assertEqual(result['tierTokens'], {'stable': 8, 'semi-stable': 10, 'unstable': 24982})
        self.assertEqual(result['segments'][-1]['sourceIds'], ['t1'])
        with self.assertRaises(BudgetFailure):
            verifier.verify(request(25001))

    def test_no_per_tier_quota_and_stable_prefix_guard(self):
        verifier = BudgetVerifier(manifest(), len)
        verifier.verify(request(24999))
        bad = request(100)
        bad['segments'][0]['content'] = 'changed!'
        with self.assertRaises(AdapterFailure):
            verifier.verify(bad)

    def test_complete_text_not_adapter_claim_controls_budget(self):
        verifier = BudgetVerifier(manifest(), lambda text: len(text) * 2)
        with self.assertRaises(BudgetFailure):
            verifier.verify(request(12501))

    def test_invalid_tokenizer_output_rejected(self):
        with self.assertRaises(ValueError):
            BudgetVerifier(manifest(), lambda text: -1).verify(request(30))

    def test_content_order_and_tokenizer_framing(self):
        seen = []
        def tokenizer(text):
            seen.append(text)
            return len(text)
        BudgetVerifier(manifest(), tokenizer).verify(request(30))
        self.assertEqual(seen[-1], 'contract' + 's' * 10 + 'x' * 12)
