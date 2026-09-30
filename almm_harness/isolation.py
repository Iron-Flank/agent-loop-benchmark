"""Validate the actual adapter input before a run can produce artifacts."""
import json
from pathlib import Path
import re
import stat

from almm_fixture.__main__ import _outside_checkout, _private_parent
from almm_fixture.engine import EVALUATION_SEED_START, adapter_view, canonical_json


_GOLD_FIELDS = {
    'expected', 'expectedanswer', 'expectedanswers', 'acceptedanswers',
    'requiredfactids', 'forbiddenfactids', 'goldevidenceids', 'evidenceids',
    'matchtype', 'facts', 'introducedfactids', 'referencedfactids',
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('adapter input JSON contains duplicate object fields')
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError('fixture JSON must not contain non-finite numbers')


def _read_json(path):
    if not path.is_file():
        raise ValueError('fixture input must be a specific JSON file, not a directory')
    try:
        return json.loads(path.read_text(), object_pairs_hook=_unique_object,
                          parse_constant=_invalid_constant)
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError('fixture input must contain valid JSON with unique fields') from error


def _reject_gold(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if re.sub(r'[^a-z]', '', key.lower()) in _GOLD_FIELDS:
                raise ValueError('adapter input contains scorer-only expected-answer or evidence data')
            _reject_gold(item)
    elif isinstance(value, list):
        for item in value:
            _reject_gold(item)


def _validate_shape(value, projection):
    # The allowlisted adapter_view is the schema; never maintain a second set of
    # public fields. Matching primitive types also rejects bool/int equivalence.
    if type(value) is not type(projection):
        raise ValueError('adapter input does not match the adapter-view schema')
    if isinstance(projection, dict):
        if value.keys() != projection.keys():
            raise ValueError('adapter input contains missing or non-adapter fields')
        for key in projection:
            _validate_shape(value[key], projection[key])
    elif isinstance(projection, list):
        if len(value) != len(projection):
            raise ValueError('adapter input does not match the scorer fixture projection')
        for item, expected in zip(value, projection):
            _validate_shape(item, expected)


def _private_file(path):
    resolved = _outside_checkout(path)
    _private_parent(resolved)
    if not resolved.is_file() or stat.S_IMODE(resolved.stat().st_mode) & 0o077:
        raise ValueError('held-out fixture file must have private permissions (0600)')
    return resolved


def validate_adapter_input(fixture, adapter_input_path=None):
    """Return only the clean projection, rejecting invalid inputs without writes.

    A supplied path names one JSON file. Reserved held-out seeds use the private
    generator layout: adapter.json with a sibling scorer.json matching fixture.
    Pure in-memory callers retain the existing allowlisted projection behavior.
    """
    projection = adapter_view(fixture)
    if adapter_input_path is None:
        view = projection
    else:
        path = Path(adapter_input_path).resolve()
        if fixture['seed'] >= EVALUATION_SEED_START:
            path = _private_file(path)
            if path.name != 'adapter.json':
                raise ValueError('held-out adapter input must use the generator adapter.json layout')
            scorer_path = _private_file(path.parent / 'scorer.json')
            if canonical_json(_read_json(scorer_path)) != canonical_json(fixture):
                raise ValueError('held-out scorer file does not match the supplied fixture')
        view = _read_json(path)
    _reject_gold(view)
    _validate_shape(view, projection)
    if canonical_json(view) != canonical_json(projection):
        raise ValueError('adapter input does not match the scorer fixture projection')
    return view
