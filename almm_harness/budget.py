"""Retokenize the full wire content; never trust adapter token claims."""
from copy import deepcopy
import hashlib
import json

from almm_adapter.contract import TIERS, validate_telemetry
from .errors import AdapterFailure, BudgetFailure

TOKEN_BUDGET = 25_000


class BudgetVerifier:
    def __init__(self, manifest, count_tokens):
        tokenizer = manifest.get('tokenizer', {})
        if not all(isinstance(tokenizer.get(k), str) and tokenizer[k]
                   for k in ('name', 'version')):
            raise ValueError('tokenizer requires pinned name and version')
        self.count_tokens = count_tokens
        self.prefix = list(manifest['stablePrefix'])
        self.stable_hash = hashlib.sha256(json.dumps(
            self.prefix, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        self.overhead = tokenizer.get('requestOverheadTokens', 0)
        if type(self.overhead) is not int or self.overhead < 0:
            raise ValueError('requestOverheadTokens must be a nonnegative integer')

    def _count(self, text):
        count = self.count_tokens(text)
        if type(count) is not int or count < 0:
            raise ValueError('declared tokenizer returned invalid token count')
        return count

    def verify(self, request):
        try:
            validate_telemetry([request])
        except ValueError as error:
            raise AdapterFailure(str(error)) from error
        normalized = deepcopy(request)
        segments = normalized['segments']
        stable = [s['content'] for s in segments if s['tier'] == 'stable']
        if stable != self.prefix or [s['tier'] for s in segments[:len(stable)]] != ['stable'] * len(stable):
            raise AdapterFailure('stable prefix changed or is not the request prefix')
        tiers = dict.fromkeys(TIERS, 0)
        for segment in segments:
            segment['tokenCount'] = self._count(segment['content'])
            tiers[segment['tier']] += segment['tokenCount']
        # Tokenization across segment boundaries can differ from tokenizing each
        # segment. The complete wire text, plus declared protocol framing, wins.
        total = self._count(''.join(s['content'] for s in segments)) + self.overhead
        normalized.update(tierTokens=tiers, totalTokens=total,
                          requestOverheadTokens=self.overhead,
                          stablePrefixHash=self.stable_hash)
        if total > TOKEN_BUDGET:
            error = BudgetFailure(f'assembled request has {total} tokens; limit {TOKEN_BUDGET}')
            error.request = normalized
            raise error
        return normalized
