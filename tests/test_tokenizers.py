import unittest
from almm_harness.tokenizers import load_tokenizer


class TokenizerTests(unittest.TestCase):
    def test_smoke_tokenizer_is_explicit_and_versioned(self):
        count = load_tokenizer({'name': 'characters-smoke', 'version': '1'})
        self.assertEqual(count('abc'), 3)
        with self.assertRaises(ValueError):
            load_tokenizer({'name': 'characters-smoke', 'version': '2'})
        with self.assertRaises(ValueError):
            load_tokenizer({'name': 'unknown', 'version': '1'})

    def test_tiktoken_pin_and_unicode(self):
        from importlib.metadata import version
        count = load_tokenizer({'name': 'tiktoken:cl100k_base', 'version': version('tiktoken')})
        self.assertEqual(count('hello world'), 2)
        self.assertGreater(count('你好'), 0)
        with self.assertRaises(ValueError):
            load_tokenizer({'name': 'tiktoken:cl100k_base', 'version': 'wrong'})
