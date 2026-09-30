"""Frozen-label calibration and provenance gate for every scoring run."""

from collections import Counter
import copy
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path

from almm_fixture.validation import ABILITIES


MATCH_TYPES = frozenset({'exact', 'ordered-list', 'numeric', 'semantic', 'abstain'})
JUDGMENTS = frozenset({'pass', 'fail', 'abstain'})
FROZEN_SET_ID = 'almm-calibration-1.0.0'
FROZEN_HASHES = {
    'manifest.json': '65d41c90e1fa8bf1c0c08043ceaaef28755685a9e56d96d4e966883238a16f03',
    'probes.json': 'b881d3a250edcfbba967015bd4d2ff36a284b5d7a1ec41b569e584d6b2f14430',
    'review.json': 'd2940dae4ff15c9ce275d4bf5ee622db9b1c58933b0ab033cc83ba9441ac5499',
}
FROZEN_ROWS_HASH = '0303fca868dcd59339232d72b9596688bf742697c2f232a0557d11444e9c2a28'


def _object(value, path, fields=()):
    if not isinstance(value, dict):
        raise ValueError(f'{path}: expected an object')
    for field in fields:
        if field not in value:
            raise ValueError(f'{path}.{field}: required field missing')


def _string(value, path):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{path}: expected a nonempty string')


def _strings(value, path, nonempty=False):
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f'{path}: expected {"a nonempty" if nonempty else "a"} list')
    for index, item in enumerate(value):
        _string(item, f'{path}[{index}]')


def _choice(value, choices, path):
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f'{path}: expected one of {sorted(choices)}')


def _unique(values, path):
    if len(values) != len(set(values)):
        raise ValueError(f'{path}: duplicate IDs')


def _json_snapshot(value, path):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f'{path}: expected finite JSON-compatible data') from error


def _json_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f'duplicate JSON field: {key}')
        value[key] = item
    return value


def _validate_probe(probe, index):
    path = f'probes[{index}]'
    _object(probe, path, ('probeId', 'ability', 'question', 'candidateAnswer', 'expected',
                          'goldJudgment', 'justification', 'answerability', 'answerAsOf',
                          'afterSessionIndex', 'evidence', 'edgeCases'))
    for field in ('probeId', 'question', 'candidateAnswer', 'justification'):
        _string(probe[field], f'{path}.{field}')
    _choice(probe['ability'], ABILITIES, f'{path}.ability')
    _choice(probe['goldJudgment'], JUDGMENTS, f'{path}.goldJudgment')
    _choice(probe['answerAsOf'], {'current', 'historical'}, f'{path}.answerAsOf')
    if type(probe['answerability']) is not bool:
        raise ValueError(f'{path}.answerability: expected a boolean')
    after = probe['afterSessionIndex']
    if type(after) is not int or after < 1:
        raise ValueError(f'{path}.afterSessionIndex: expected a positive integer')
    _strings(probe['edgeCases'], f'{path}.edgeCases')
    evidence = probe['evidence']
    if not isinstance(evidence, list):
        raise ValueError(f'{path}.evidence: expected a list')
    facts = {}
    for fact_index, fact in enumerate(evidence):
        fact_path = f'{path}.evidence[{fact_index}]'
        _object(fact, fact_path, ('factId', 'sessionIndex', 'text', 'state'))
        _string(fact['factId'], f'{fact_path}.factId')
        _string(fact['text'], f'{fact_path}.text')
        _choice(fact['state'], {'active', 'superseded', 'expired'}, f'{fact_path}.state')
        if type(fact['sessionIndex']) is not int or not 0 < fact['sessionIndex'] < after:
            raise ValueError(f'{fact_path}.sessionIndex: evidence must precede the probe')
        if fact['factId'] in facts:
            raise ValueError(f'{fact_path}.factId: duplicate ID')
        facts[fact['factId']] = fact
    expected = probe['expected']
    expected_path = f'{path}.expected'
    _object(expected, expected_path, ('matchType', 'acceptedAnswers',
                                     'requiredFactIds', 'forbiddenFactIds'))
    _choice(expected['matchType'], MATCH_TYPES, f'{expected_path}.matchType')
    _strings(expected['acceptedAnswers'], f'{expected_path}.acceptedAnswers', nonempty=True)
    for field in ('requiredFactIds', 'forbiddenFactIds'):
        ids = expected[field]
        _strings(ids, f'{expected_path}.{field}')
        _unique(ids, f'{expected_path}.{field}')
        if any(fact_id not in facts for fact_id in ids):
            raise ValueError(f'{expected_path}.{field}: unknown evidence ID')
    if set(expected['requiredFactIds']) & set(expected['forbiddenFactIds']):
        raise ValueError(f'{expected_path}: required and forbidden facts overlap')
    if probe['answerAsOf'] == 'current' and any(
            facts[fact_id]['state'] != 'active' for fact_id in expected['requiredFactIds']):
        raise ValueError(f'{expected_path}.requiredFactIds: current answers require active facts')
    kind = expected['matchType']
    if kind == 'semantic':
        _object(expected, expected_path, ('rubric', 'requiredClaims', 'disallowedContradictions'))
        _string(expected['rubric'], f'{expected_path}.rubric')
        _strings(expected['requiredClaims'], f'{expected_path}.requiredClaims', nonempty=True)
        _strings(expected['disallowedContradictions'], f'{expected_path}.disallowedContradictions')
    elif kind == 'numeric':
        _object(expected, expected_path, ('targetNumber', 'tolerance', 'toleranceMode'))
        _choice(expected['toleranceMode'], {'absolute-inclusive'}, f'{expected_path}.toleranceMode')
        for field in ('targetNumber', 'tolerance'):
            _string(expected[field], f'{expected_path}.{field}')
            try:
                number = Decimal(expected[field])
            except InvalidOperation as error:
                raise ValueError(f'{expected_path}.{field}: expected a decimal') from error
            if not number.is_finite() or (field == 'tolerance' and number < 0):
                raise ValueError(f'{expected_path}.{field}: expected a finite nonnegative tolerance or target')
    elif kind == 'ordered-list':
        _object(expected, expected_path, ('items',))
        _strings(expected['items'], f'{expected_path}.items', nonempty=True)
        try:
            decoded = [json.loads(answer) for answer in expected['acceptedAnswers']]
        except ValueError as error:
            raise ValueError(f'{expected_path}.acceptedAnswers: expected JSON arrays') from error
        if any(items != expected['items'] for items in decoded):
            raise ValueError(f'{expected_path}.acceptedAnswers: must agree with ordered items')
    elif kind == 'abstain':
        if probe['answerability'] or expected['requiredFactIds'] or expected['forbiddenFactIds']:
            raise ValueError(f'{expected_path}: abstain expected records must be unanswerable')


def _provenance(probes, review):
    labels = probes.get('labelProvenance')
    reviewed = review.get('reviewProvenance')
    human = (
        isinstance(labels, dict) and labels.get('kind') == 'human'
        and isinstance(labels.get('labelerId'), str) and bool(labels['labelerId'].strip())
        and isinstance(reviewed, dict) and reviewed.get('kind') == 'human'
        and isinstance(reviewed.get('reviewerId'), str) and bool(reviewed['reviewerId'].strip())
    )
    # Renaming a set or replacing provenance metadata cannot turn the shipped
    # agent-authored labels into independently human-authored labels.
    rows_hash = hashlib.sha256(json.dumps(
        probes['probes'], sort_keys=True, separators=(',', ':'), allow_nan=False,
    ).encode()).hexdigest()
    agent_declared = any('agent' in method.casefold() for method in
                         (probes['labelingMethod'], review['reviewMethod']))
    if (probes['calibrationSetId'] == FROZEN_SET_ID
            or rows_hash == FROZEN_ROWS_HASH or agent_declared):
        kind = 'agent'
    elif human:
        kind = 'human'
    elif isinstance(labels, dict) and labels.get('kind') == 'agent':
        kind = 'agent'
    else:
        kind = 'unknown'
    return {
        'kind': kind, 'labelingMethod': probes['labelingMethod'],
        'reviewMethod': review['reviewMethod'],
        'labels': copy.deepcopy(labels), 'review': copy.deepcopy(reviewed),
    }


class CalibrationGate:
    """Revalidate bytes and score every immutable label on each evaluation.

    Rejections of agreement or provenance attach the complete report to the
    ValueError's ``report`` attribute. Development approvals are noncanonical.
    New human-reviewed sets declare labelProvenance.kind/labelerId and
    reviewProvenance.kind/reviewerId; prose alone never establishes human labels.
    """

    def __init__(self, directory):
        self.directory = Path(directory)

    def _load(self):
        documents = {}
        hashes = {}
        for name in ('manifest.json', 'probes.json', 'review.json'):
            try:
                content = (self.directory / name).read_bytes()
                documents[name] = json.loads(content, object_pairs_hook=_json_object)
            except (OSError, UnicodeError, ValueError) as error:
                raise ValueError(f'calibration {name}: {error}') from error
            hashes[name] = hashlib.sha256(content).hexdigest()
            _object(documents[name], name)
        manifest, probes, review = (documents[name] for name in
                                    ('manifest.json', 'probes.json', 'review.json'))
        _object(manifest, 'manifest', ('schemaVersion', 'calibrationSetId',
                                     'calibrationSetVersion', 'retainedWithScorerVersion',
                                     'frozenAt', 'files', 'requiredAgreement'))
        _object(probes, 'probes', ('schemaVersion', 'calibrationSetId', 'calibrationSetVersion',
                                 'frozen', 'labelingMethod', 'probes'))
        _object(review, 'review', ('calibrationSetId', 'probesSha256', 'reviewedAt',
                                 'reviewMethod', 'result', 'records'))
        if manifest['schemaVersion'] != 'calibration-manifest-1.0':
            raise ValueError('manifest.schemaVersion: unsupported calibration schema')
        if probes['schemaVersion'] != 'calibration-1.0' or probes['frozen'] is not True:
            raise ValueError('probes: unsupported schemaVersion or corpus is not frozen')
        for document, name in ((manifest, 'manifest'), (probes, 'probes'), (review, 'review')):
            _string(document['calibrationSetId'], f'{name}.calibrationSetId')
        if len({manifest['calibrationSetId'], probes['calibrationSetId'], review['calibrationSetId']}) != 1:
            raise ValueError('calibrationSetId: manifest, probes and review must agree')
        for document, name in ((manifest, 'manifest'), (probes, 'probes')):
            _string(document['calibrationSetVersion'], f'{name}.calibrationSetVersion')
        if probes['calibrationSetVersion'] != manifest['calibrationSetVersion']:
            raise ValueError('calibrationSetVersion: manifest and probes must agree')
        for field in ('retainedWithScorerVersion', 'frozenAt'):
            _string(manifest[field], f'manifest.{field}')
        for document, name, field in ((probes, 'probes', 'labelingMethod'),
                                      (review, 'review', 'reviewMethod'),
                                      (review, 'review', 'reviewedAt')):
            _string(document[field], f'{name}.{field}')
        if manifest['requiredAgreement'] != 0.95:
            raise ValueError('manifest.requiredAgreement: gate requires 95% agreement')
        _object(manifest['files'], 'manifest.files', ('probes.json', 'review.json'))
        if set(manifest['files']) != {'probes.json', 'review.json'}:
            raise ValueError('manifest.files: expected exactly probes.json and review.json')
        for name, digest in manifest['files'].items():
            if not isinstance(digest, str) or hashes[name] != digest:
                raise ValueError(f'{name}: frozen SHA-256 hash mismatch')
        if probes['calibrationSetId'] == FROZEN_SET_ID and hashes != FROZEN_HASHES:
            raise ValueError('immutable calibration set: committed frozen hashes do not match')
        if review['probesSha256'] != hashes['probes.json']:
            raise ValueError('review.probesSha256: frozen probe hash mismatch')
        rows = probes['probes']
        if not isinstance(rows, list) or not 90 <= len(rows) <= 110:
            raise ValueError('calibration probes: require 90..110 rows')
        for index, row in enumerate(rows):
            _validate_probe(row, index)
        _unique([row['probeId'] for row in rows], 'probes.probeId')
        ability_counts = Counter(row['ability'] for row in rows)
        if set(ability_counts) != ABILITIES or min(ability_counts.values()) < 10:
            raise ValueError('calibration abilities: each of the five abilities requires at least 10 rows')
        match_counts = Counter(row['expected']['matchType'] for row in rows)
        if set(match_counts) != MATCH_TYPES:
            raise ValueError('calibration match types: all five match types must be present')
        if match_counts['semantic'] * 10 < len(rows) * 3:
            raise ValueError('calibration semantic proportion: requires at least 30%')
        records = review['records']
        if review['result'] != 'passed' or not isinstance(records, list) or len(records) != len(rows):
            raise ValueError('review: every frozen label must have a passed review record')
        reviewed = {}
        for index, record in enumerate(records):
            path = f'review.records[{index}]'
            _object(record, path, ('probeId', 'confirmedGoldJudgment', 'justificationConsistent',
                                   'evidenceConsistent', 'reviewNote'))
            _string(record['probeId'], f'{path}.probeId')
            _string(record['reviewNote'], f'{path}.reviewNote')
            _choice(record['confirmedGoldJudgment'], JUDGMENTS, f'{path}.confirmedGoldJudgment')
            if record['probeId'] in reviewed:
                raise ValueError(f'{path}.probeId: duplicate review ID')
            if record['justificationConsistent'] is not True or record['evidenceConsistent'] is not True:
                raise ValueError(f'{path}: review must confirm justification and evidence')
            reviewed[record['probeId']] = record
        if set(reviewed) != {row['probeId'] for row in rows}:
            raise ValueError('review: probe IDs must exactly cover all frozen labels')
        for row in rows:
            if reviewed[row['probeId']]['confirmedGoldJudgment'] != row['goldJudgment']:
                raise ValueError(f'review {row["probeId"]}: confirmed judgment differs from frozen goldJudgment')
        counts = review.get('counts')
        if counts is not None:
            _object(counts, 'review.counts', ('abilities', 'matchTypes', 'goldJudgments', 'edgeCases'))
            actual = {
                'abilities': dict(ability_counts), 'matchTypes': dict(match_counts),
                'goldJudgments': dict(Counter(row['goldJudgment'] for row in rows)),
                'edgeCases': dict(Counter(tag for row in rows for tag in row['edgeCases'])),
            }
            if counts != actual:
                raise ValueError('review.counts: counts do not match frozen probes')
        return manifest, probes, review, hashes

    def evaluate(self, score, scorer_version, judge_config=None, require_human=True):
        _string(scorer_version, 'scorer_version')
        if not callable(score):
            raise ValueError('score: expected a callable scorer')
        if type(require_human) is not bool:
            raise ValueError('require_human: expected a boolean')
        if judge_config is not None:
            _object(judge_config, 'judge_config')
        config = _json_snapshot(judge_config, 'judge_config')
        manifest, probes, review, hashes = self._load()
        provenance = _provenance(probes, review)
        judgments = []
        mismatches = []
        for row in probes['probes']:
            # Calibration annotations are never judge input, and scorer mutation
            # cannot alter gold labels used for agreement or subsequent runs.
            probe = copy.deepcopy({key: value for key, value in row.items()
                                   if key not in {'goldJudgment', 'justification',
                                                  'candidateAnswer', 'edgeCases'}})
            result = score(probe, row['candidateAnswer'])
            _object(result, f'scorer {row["probeId"]}', ('judgment', 'correct', 'normalizedAnswer',
                                                       'matchingMethod', 'judgeOutput'))
            _choice(result['judgment'], JUDGMENTS, f'scorer {row["probeId"]}.judgment')
            if type(result['correct']) is not bool:
                raise ValueError(f'scorer {row["probeId"]}.correct: expected a boolean')
            _choice(result['matchingMethod'], MATCH_TYPES, f'scorer {row["probeId"]}.matchingMethod')
            record = _json_snapshot(result, f'scorer {row["probeId"]}')
            record.update(probeId=row['probeId'], goldJudgment=row['goldJudgment'],
                          agrees=result['judgment'] == row['goldJudgment'])
            judgments.append(record)
            if not record['agrees']:
                mismatches.append(copy.deepcopy(record))
        total = len(judgments)
        matched = total - len(mismatches)
        reasons = []
        if matched * 100 < total * 95:
            reasons.append(f'agreement {matched}/{total} is below required 95%')
        if require_human and provenance['kind'] != 'human':
            reasons.append(f'human label and review provenance required; actual provenance is {provenance["kind"]}')
        report = {
            'approved': not reasons, 'canonical': require_human and not reasons,
            'agreement': matched / total, 'matched': matched, 'total': total,
            'requiredAgreement': 0.95, 'scorerVersion': scorer_version, 'judgeConfig': config,
            'calibrationSetId': probes['calibrationSetId'],
            'calibrationSetVersion': probes['calibrationSetVersion'],
            'retainedWithScorerVersion': manifest['retainedWithScorerVersion'],
            'calibrationHashes': hashes, 'labelProvenance': provenance,
            'judgments': judgments, 'mismatches': mismatches, 'rejectionReasons': reasons,
        }
        if reasons:
            error = ValueError('calibration rejected: ' + '; '.join(reasons))
            error.report = report
            raise error
        return report
