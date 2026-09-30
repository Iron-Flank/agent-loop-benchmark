"""Independent accuracy, completion, retrieval, token and operational metrics."""
import math
from collections import Counter, defaultdict

from almm_adapter.contract import TIERS


_FAILURES = ('budget', 'provider', 'adapter', 'timeout')


def _number(value, name, *, nonnegative=False, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (nonnegative and value < 0)
            or (integer and type(value) is not int)):
        raise ValueError(f'{name}: expected finite'
                         f'{" nonnegative" if nonnegative else ""}'
                         f'{" integer" if integer else " number"}')
    return value


def distribution(values):
    """Finite numeric observations; nearest-rank percentiles, null if empty."""
    ordered = sorted(_number(value, 'distribution observation') for value in values)
    count = len(ordered)
    if not count:
        return dict.fromkeys(('mean', 'p50', 'p95', 'min', 'max'), None) | {'count': 0}
    return {'count': count, 'mean': math.fsum(ordered) / count,
            'p50': ordered[math.ceil(count * 0.5) - 1],
            'p95': ordered[math.ceil(count * 0.95) - 1],
            'min': ordered[0], 'max': ordered[-1]}


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _accuracy(correct, answered, total):
    return {'correct': correct, 'answered': answered, 'total': total,
            'accuracy': _ratio(correct, answered),
            'completionRate': _ratio(answered, total)}


def _indexed(rows, key, name):
    indexed = {}
    for row in rows:
        identity = row.get(key)
        if not isinstance(identity, str) or not identity or identity in indexed:
            raise ValueError(f'{name}: missing or duplicate {key}')
        indexed[identity] = row
    return indexed


def _fixture_sources(fixture):
    sessions, turns, introductions = {}, {}, {}
    for index, session in enumerate(fixture['sessions']):
        sessions[session['sessionId']] = index
        for turn in session['turns']:
            facts = turn.get('introducedFactIds', [])
            turns[turn['turnId']] = set(facts) | set(turn.get('referencedFactIds', []))
            for fact in facts:
                introductions[fact] = index
    # The fact graph pins the original introduction even when turn annotations
    # are unavailable in an imported fixture.
    turn_sessions = {turn['turnId']: sessions[session['sessionId']]
                     for session in fixture['sessions'] for turn in session['turns']}
    for fact in fixture.get('facts', []):
        if fact.get('introducedAt') in turn_sessions:
            introductions[fact['factId']] = turn_sessions[fact['introducedAt']]
            turns[fact['introducedAt']].add(fact['factId'])
    return sessions, turns, introductions


def _distance(probe, sessions, introductions):
    label = probe.get('sessionDistance')
    if label is None and isinstance(probe.get('labels'), dict):
        label = probe['labels'].get('sessionDistance')
    if label is not None:
        return _number(label, 'probe.sessionDistance', nonnegative=True, integer=True)
    required = probe['expected']['requiredFactIds']
    if not required:
        return 0
    if probe.get('afterSessionId') not in sessions or any(fact not in introductions for fact in required):
        raise ValueError('cannot derive session distance from checkpoint/fact introductions')
    checkpoint = sessions[probe['afterSessionId']]
    if any(introductions[fact] > checkpoint for fact in required):
        raise ValueError('required evidence introduced after probe checkpoint')
    return max(checkpoint - introductions[fact] for fact in required)


def _retrieval(probe, requests, turns):
    sources, reported = set(), False
    for request in requests:
        for segment in request.get('tierSegments', []):
            if 'sourceIds' not in segment:
                continue
            reported = True
            identifiers = segment['sourceIds']
            if not isinstance(identifiers, list) or any(
                    not isinstance(source, str) or not source for source in identifiers):
                raise ValueError('segment.sourceIds: expected nonempty string identifiers')
            for source in identifiers:
                # Unknown IDs remain false positives; dropping them would inflate
                # precision. Factless turns also remain false positives.
                sources.update((turns[source] or [source]) if source in turns else [source])
    required = set(probe['expected']['requiredFactIds'])
    matched = len(sources & required)
    return {'probeId': probe['probeId'], 'status': 'reported' if reported else 'not-reported',
            'retrievedCount': len(sources) if reported else None,
            'requiredCount': len(required), 'matchedCount': matched if reported else None,
            'precision': (matched / len(sources) if sources else 0 if required else None)
                         if reported else None,
            'recall': _ratio(matched, len(required)) if reported else None}


def _relevance(rows):
    reported = [row for row in rows if row['status'] == 'reported']
    not_reported = len(rows) - len(reported)
    precision = [row['precision'] for row in reported if row['precision'] is not None]
    recall = [row['recall'] for row in reported if row['recall'] is not None]
    mean_precision = distribution(precision)['mean']
    mean_recall = distribution(recall)['mean']
    status = 'not-reported' if not reported else 'partial' if not_reported else 'reported'
    return {'status': status, 'reportedCount': len(reported), 'notReportedCount': not_reported,
            'precision': mean_precision if not not_reported else None,
            'recall': mean_recall if not not_reported else None,
            'reportedPrecision': mean_precision, 'reportedRecall': mean_recall,
            'precisionDefinedCount': len(precision), 'recallDefinedCount': len(recall),
            'aggregation': 'macro mean over probes with defined precision/recall; '
                           'full-run means require provenance for every probe',
            'perProbe': rows}


def build_report(fixture, manifest, probes, requests, scores):
    """Build JSON metrics from archived records without mutating any input.

    A completed abstention is answered for completion and eligibility purposes;
    a failed or unattempted probe never enters the accuracy denominator.
    """
    gold = _indexed(fixture['probes'], 'probeId', 'fixture probes')
    captured = _indexed(probes, 'probeId', 'captured probes')
    scored = _indexed(scores, 'probeId', 'scores')
    by_request = _indexed(requests, 'requestId', 'requests')
    if set(scored) != set(gold):
        raise ValueError('scorer probe IDs must exactly match fixture probe IDs')
    if not set(captured) <= set(gold):
        raise ValueError('captured probe IDs not present in fixture')

    request_groups = defaultdict(list)
    token_values, request_latencies = [], []
    tier_tokens = dict.fromkeys(TIERS, 0)
    overhead, budget_request_failures = 0, 0
    for request in by_request.values():
        pid = request.get('probeId')
        if pid is not None:
            if pid not in captured:
                raise ValueError('request references unknown or uncaptured probe')
            request_groups[pid].append(request)
        token_values.append(_number(request.get('totalTokens'), 'request.totalTokens',
                                    nonnegative=True, integer=True))
        tiers = request.get('tierTokens')
        if not isinstance(tiers, dict) or set(tiers) != set(TIERS):
            raise ValueError('request.tierTokens: all three tiers required')
        for tier in TIERS:
            tier_tokens[tier] += _number(tiers[tier], f'request.tierTokens.{tier}',
                                        nonnegative=True, integer=True)
        overhead += _number(request.get('requestOverheadTokens'), 'request.requestOverheadTokens',
                            nonnegative=True, integer=True)
        if 'latencyMs' in request:
            request_latencies.append(_number(request['latencyMs'], 'request.latencyMs', nonnegative=True))
        budget_request_failures += request.get('status') == 'error' and request.get('category') == 'budget'

    sessions, turns, introductions = _fixture_sources(fixture)
    breakdown = {name: defaultdict(lambda: [0, 0, 0]) for name in
                 ('ability', 'sessionDistance', 'evidenceCardinality', 'fixtureId')}
    correct, answered, answerable, unanswerable = 0, 0, 0, 0
    correct_abstentions, unsupported_answers, false_abstentions = 0, 0, 0
    unanswerable_abstentions = 0
    failures, probe_latencies, retrieval = Counter(), [], []
    for pid, probe in gold.items():
        record, score = captured.get(pid), scored[pid]
        eligible = record.get('eligibleForAccuracy') if record else False
        if type(eligible) is not bool or score.get('eligibleForAccuracy') is not eligible:
            raise ValueError('scorer eligibility differs from captured probe')
        if record:
            if record.get('status') != ('ok' if eligible else 'error'):
                raise ValueError('probe eligibility/status mismatch')
            if not eligible:
                category = record.get('category')
                if category not in _FAILURES:
                    raise ValueError('failed probe requires known failure category')
                failures[category] += 1
            if 'latencyMs' in record:
                probe_latencies.append(_number(record['latencyMs'], 'probe.latencyMs', nonnegative=True))
        result = score.get('normalizedResult')
        if (not isinstance(result, dict) or type(result.get('correct')) is not bool
                or result.get('judgment') not in {'pass', 'fail', 'abstain', 'incomplete'}):
            raise ValueError('scorer normalizedResult requires boolean correct and known judgment')
        is_correct = result['correct']
        judgment = result['judgment']
        if (not eligible and (is_correct or judgment != 'incomplete')) or (eligible and judgment == 'incomplete'):
            raise ValueError('scorer result contradicts probe eligibility')
        if 'failureCategory' in score:
            expected_failure = record.get('category') if record else 'unattempted'
            if score['failureCategory'] != expected_failure:
                raise ValueError('scorer failure category differs from captured probe')
        answered += eligible
        correct += is_correct
        if eligible:
            answerability = probe.get('answerability', True)
            if type(answerability) is not bool:
                raise ValueError('probe.answerability must be boolean')
            abstains = judgment == 'abstain'
            if answerability:
                answerable += 1
                false_abstentions += abstains
            else:
                unanswerable += 1
                unanswerable_abstentions += abstains
                correct_abstentions += abstains and is_correct
                unsupported_answers += not abstains
        labels = {'ability': probe['ability'],
                  'sessionDistance': _distance(probe, sessions, introductions),
                  'evidenceCardinality': len(probe['expected']['requiredFactIds']),
                  'fixtureId': fixture['fixtureId']}
        for dimension, label in labels.items():
            counts = breakdown[dimension][str(label)]
            counts[0] += is_correct
            counts[1] += eligible
            counts[2] += 1
        retrieval.append(_retrieval(probe, request_groups[pid], turns))

    total, total_tokens = len(gold), sum(token_values)
    accuracy = _accuracy(correct, answered, total)
    incomplete = sum(failures[category] for category in ('provider', 'adapter', 'timeout'))
    incomplete_rate = _ratio(incomplete, total)
    tier_total = sum(tier_tokens.values())
    return {
        'schemaVersion': 'report-1.0',
        'runtime': {'name': manifest['adapter']['name'], 'revision': manifest['adapter']['revision']},
        'scale': len(fixture['sessions']),
        'accuracy': accuracy,
        'breakdown': {dimension: {label: _accuracy(*counts) for label, counts in sorted(groups.items())}
                      for dimension, groups in breakdown.items()},
        'abstention': {
            'unanswerableAnswered': unanswerable, 'answerableAnswered': answerable,
            'unanswerableAbstentions': unanswerable_abstentions,
            'correctAbstentions': correct_abstentions,
            'correctAbstentionRate': _ratio(correct_abstentions, unanswerable),
            'unsupportedAnswers': unsupported_answers,
            'unsupportedAnswerRate': _ratio(unsupported_answers, unanswerable),
            'falseAbstentions': false_abstentions,
            'falseAbstentionRate': _ratio(false_abstentions, answerable),
            'definition': 'rates use eligible answered probes of the corresponding answerability; '
                          'abstention is judgment=abstain, not correctness',
        },
        'tokenEfficiency': {
            'assembledContextTokens': distribution(token_values), 'totalInputTokens': total_tokens,
            'tierTokens': tier_tokens,
            'tierTokenDistribution': {tier: _ratio(value, tier_total) for tier, value in tier_tokens.items()},
            'requestOverheadTokens': overhead,
            'accuracyPer1000InputTokens': accuracy['accuracy'] * len(token_values) / total_tokens * 1000
                                         if accuracy['accuracy'] is not None and total_tokens else None,
            'definition': 'accuracy / mean assembled-context input tokens per request * 1000; '
                          'all requests included, including turns and failures; '
                          'assembled total includes request overhead; tier proportions exclude overhead',
        },
        'contextRelevance': _relevance(retrieval),
        'operationalTelemetry': {
            'requestCount': len(by_request), 'overBudgetFailures': failures['budget'],
            'budgetRequestFailures': budget_request_failures,
            'incompleteCount': incomplete, 'incompleteProbeRate': incomplete_rate,
            'investigationRequired': incomplete_rate is not None and incomplete_rate > 0.05,
            'unattemptedCount': total - len(captured),
            'failureCounts': {category: failures[category] for category in _FAILURES},
            'requestLatencyMs': distribution(request_latencies),
            'probeLatencyMs': distribution(probe_latencies),
            'definition': 'incomplete probes include only provider, adapter and timeout failures; '
                          'budget failures and unattempted probes are separate; '
                          'overBudgetFailures counts probes, budgetRequestFailures counts requests',
        },
    }
