"""Retokenize the full native provider body; never trust adapter token claims."""
from copy import deepcopy
import hashlib
import json

from almm_adapter.contract import TIERS, validate_native_request
from almm_adapter.native import canonical_wire, wire_request
from .errors import AdapterFailure, BudgetFailure

TOKEN_BUDGET = 25_000


class BudgetVerifier:
    def __init__(self, manifest, count_tokens):
        tokenizer = manifest.get('tokenizer', {})
        if not all(isinstance(tokenizer.get(k), str) and tokenizer[k]
                   for k in ('name', 'version')):
            raise ValueError('tokenizer requires pinned name and version')
        if manifest.get('nativeModelEnvelopeVersion') != '1.0':
            raise ValueError('nativeModelEnvelopeVersion must be 1.0')
        self.model = deepcopy(manifest['model'])
        self.pinned_model = json.dumps(self.model, sort_keys=True, allow_nan=False)
        self.pinned_decoding = json.dumps(self.model['decoding'], sort_keys=True, allow_nan=False)
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
            validate_native_request(request)
            if (json.dumps(request['model'], sort_keys=True) != self.pinned_model or
                    json.dumps(request['decodingSettings'], sort_keys=True) != self.pinned_decoding or
                    ('seed' in request) != ('seed' in self.model) or
                    request.get('seed') != self.model.get('seed')):
                raise ValueError('native request changed pinned model, decodingSettings or seed')
            wire = wire_request(request)
            serialized = canonical_wire(request)
            self._validate_attribution(request)
        except ValueError as error:
            raise AdapterFailure(str(error)) from error
        normalized = deepcopy(request)
        tiers = dict.fromkeys(TIERS, 0)
        tool_wires = {tool['function']['name']: tool for tool in wire.get('tools', [])}
        for segment in normalized['tierSegments']:
            attributed = [wire['messages'][i] for i in segment.get('messageIndices', [])]
            attributed.extend(tool_wires[name] for name in segment.get('toolNames', []))
            segment['tokenCount'] = self._count(json.dumps(
                attributed, ensure_ascii=False, separators=(',', ':')))
            tiers[segment['tier']] += segment['tokenCount']
        total = self._count(serialized) + self.overhead
        normalized.update(tierTokens=tiers, totalTokens=total,
                          requestOverheadTokens=self.overhead,
                          stablePrefixHash=self.stable_hash)
        if total > TOKEN_BUDGET:
            error = BudgetFailure(f'assembled request has {total} tokens; limit {TOKEN_BUDGET}')
            error.request = normalized
            raise error
        return normalized

    def _validate_attribution(self, request):
        segments, messages = request['tierSegments'], request['messages']
        indices = [i for segment in segments for i in segment.get('messageIndices', [])]
        names = [name for segment in segments for name in segment.get('toolNames', [])]
        if sorted(indices) != list(range(len(messages))):
            raise ValueError('tierSegments: every message must be attributed exactly once')
        if sorted(names) != sorted(tool['name'] for tool in request.get('tools', [])):
            raise ValueError('tierSegments: every tool must be attributed exactly once')
        stable = [segment for segment in segments if segment['tier'] == 'stable']
        if ([segment['content'] for segment in stable] != self.prefix or
                segments[:len(stable)] != stable):
            raise ValueError('stable prefix changed or is not the request prefix')
        for i, (segment, prefix) in enumerate(zip(stable, self.prefix)):
            if (segment.get('messageIndices') != [i] or segment.get('toolNames') or
                    i >= len(messages) or messages[i]['role'] != 'system' or
                    messages[i]['content'] != prefix):
                raise ValueError('stable prefix must match the actual first system messages')
