"""Whole-answer deterministic matching; no fuzzy or substring acceptance."""
from decimal import Decimal, InvalidOperation, localcontext
import json
import re
import unicodedata


_NUMBER = re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z')
_DECLINES = {
    'i do not know', "i don't know", 'i cannot answer', "i can't answer",
    'i cannot answer this question', "i can't answer this question",
    'i cannot determine the answer', 'i am unable to answer',
    'there is not enough information to say', 'there is insufficient information',
    'not enough information', 'insufficient information',
    'that information was not provided', 'the information was not provided',
    'the requested information was not provided', 'unknown',
}


def normalize(value):
    """NFKC/casefold text and collapse whitespace; booleans are not integers."""
    if value is None:
        return ''
    text = str(value).lower() if isinstance(value, bool) else str(value)
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def _number(value):
    if isinstance(value, bool) or value is None or isinstance(value, (dict, list, tuple)):
        return None
    text = normalize(value)
    if not _NUMBER.fullmatch(text):
        return None
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def _ordered(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return None
    if not isinstance(value, list) or any(not isinstance(item, (str, int, float, bool))
                                          for item in value):
        return None
    return [normalize(item) for item in value]


def _within(candidate, target, tolerance):
    if candidate == target:
        return True
    if tolerance == 0:
        return False
    # Keep every decimal place for inclusive boundaries, independent of ambient
    # Decimal precision (whose default would round a near-boundary miss).
    values = (candidate, target, tolerance)
    precision = max(number.adjusted() for number in values) - min(
        number.as_tuple().exponent for number in values) + 3
    with localcontext() as context:
        context.prec = max(28, precision)
        context.Emax = max(context.Emax, max(number.adjusted() for number in values) + 2)
        context.Emin = min(context.Emin, min(number.adjusted() for number in values) - 2)
        return abs(candidate - target) <= tolerance


def match_atomic(probe, answer):
    expected = probe['expected']
    method = expected['matchType']
    if method not in ('exact', 'ordered-list', 'numeric', 'abstain'):
        raise ValueError('unsupported atomic matchType')
    accepted = expected.get('acceptedAnswers', [])
    if not isinstance(accepted, list) or not accepted:
        raise ValueError('acceptedAnswers must be a nonempty list')
    normalized = normalize(answer)
    correct = False
    if method == 'ordered-list':
        candidate = _ordered(answer)
        targets = [_ordered(value) for value in accepted]
        if any(target is None for target in targets):
            raise ValueError('ordered-list acceptedAnswers must be JSON arrays of scalar items')
        normalized = candidate if candidate is not None else normalized
        correct = candidate is not None and candidate in targets
    elif method == 'numeric':
        tolerance = _number(expected.get('tolerance', 0))
        if tolerance is None or tolerance < 0:
            raise ValueError('numeric tolerance must be finite and nonnegative')
        if expected.get('toleranceMode', 'absolute-inclusive') != 'absolute-inclusive':
            raise ValueError('unsupported numeric toleranceMode')
        targets = [_number(expected['targetNumber'])] if 'targetNumber' in expected else [
            _number(value) for value in accepted]
        if any(target is None for target in targets):
            raise ValueError('numeric target must be a finite whole-string number')
        candidate = _number(answer)
        correct = candidate is not None and any(_within(candidate, target, tolerance)
                                                for target in targets)
    else:
        correct = (isinstance(answer, (str, int, float, bool)) and
                   normalized in [normalize(value) for value in accepted])

    if method == 'abstain':
        judgment = 'abstain' if correct else 'fail'
        correct = correct and probe.get('answerability', False) is False
    elif correct:
        judgment = 'pass'
    elif isinstance(answer, str) and normalize(answer).rstrip('.!?') in _DECLINES:
        judgment = 'abstain'
    else:
        judgment = 'fail'
    return {'judgment': judgment, 'correct': bool(correct), 'normalizedAnswer': normalized,
            'matchingMethod': method, 'judgeOutput': None}
