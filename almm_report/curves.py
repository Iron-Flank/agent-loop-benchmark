"""Dual-line curves preserve individual observations; repeats are never averaged away."""
from collections import defaultdict
import hashlib
from html import escape
import statistics

from almm_fixture.engine import canonical_json

SCALES = (10, 100, 500, 1000)


def _variance(values):
    values = [value for value in values if value is not None]
    return {'values': values, 'count': len(values),
            'populationVariance': statistics.pvariance(values) if len(values) > 1 else None}


def build_curves(entries):
    groups = {}
    repeats = defaultdict(list)
    seen = set()
    for entry in entries:
        manifest, report = entry['manifest'], entry['report']
        identity = entry['manifestHash']
        if identity in seen:
            raise ValueError('duplicate artifact is not a repeated run')
        seen.add(identity)
        scale = report['scale']
        if scale not in SCALES:
            raise ValueError('curve scale must be 10, 100, 500, or 1000')
        config = {key: manifest.get(key) for key in (
            'adapter', 'model', 'tokenizer', 'scorerVersion', 'scorerHash', 'judge',
            'calibrationHashes', 'harnessHash', 'seed', 'fixtureSeed', 'generatorVersion', 'concurrency',
            'canonical')}
        key = hashlib.sha256(canonical_json(config)).hexdigest()
        series = groups.setdefault(key, {'seriesId': key, 'runtime': manifest['adapter'],
                                         'configuration': config, 'points': []})
        accuracy = report['accuracy']
        point = {'scale': scale, 'manifestHash': identity,
                 'fixtureHash': manifest.get('fixtureHash'), 'fixtureId': manifest.get('fixtureId'),
                 **{field: accuracy[field] for field in ('correct', 'answered', 'total', 'accuracy', 'completionRate')}}
        point['operationalTelemetry'] = report.get('operationalTelemetry')
        point['tokenEfficiency'] = report.get('tokenEfficiency')
        point['contextRelevance'] = report.get('contextRelevance')
        series['points'].append(point)
        repeats[(key, scale, manifest.get('fixtureHash'))].append(point)
    variances = []
    for (key, scale, fixture_hash), points in repeats.items():
        model = groups[key]['configuration']['model']
        if model.get('deterministic') is not True:
            variances.append({'seriesId': key, 'scale': scale, 'fixtureHash': fixture_hash,
                              'status': 'reported' if len(points) > 1 else 'not-reported',
                              'reason': None if len(points) > 1 else 'requires at least two independent runs',
                              'accuracy': _variance([p['accuracy'] for p in points]),
                              'completionRate': _variance([p['completionRate'] for p in points])})
    for series in groups.values():
        series['points'].sort(key=lambda point: point['scale'])
        series['missingScales'] = [scale for scale in SCALES if scale not in {p['scale'] for p in series['points']}]
    return {'schemaVersion': 'degradation-1.0', 'scales': list(SCALES),
            'series': list(groups.values()), 'repeatedRunVariance': variances}


def render_svg(curves):
    """Render one panel per compatible runtime/configuration, ordinal scale axis."""
    panels = curves['series']
    height = max(1, len(panels)) * 330
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="860" height="{height}" viewBox="0 0 860 {height}">',
                '<title>ALMM accuracy and completion degradation curves</title>',
                '<rect width="100%" height="100%" fill="white"/>']
    for index, series in enumerate(panels):
        top = index * 330
        runtime = series['runtime']
        title = escape(f'{runtime["name"]} @ {runtime["revision"]} — {series["seriesId"][:12]}')
        elements.append(f'<text x="70" y="{top + 24}" font-family="sans-serif" font-size="16">{title}</text>')
        for tick in (0, .25, .5, .75, 1):
            y = top + 245 - tick * 180
            elements.extend([f'<path d="M70 {y} H790" stroke="#ddd"/>',
                             f'<text x="30" y="{y + 5}" font-family="sans-serif" font-size="12">{tick:g}</text>'])
        for offset, scale in enumerate(SCALES):
            x = 100 + offset * 220
            elements.append(f'<text x="{x - 15}" y="{top + 266}" font-family="sans-serif" font-size="12">{scale}</text>')
        elements.append(f'<text x="330" y="{top + 286}" font-family="sans-serif" font-size="12">Conversation sessions</text>')
        points_by_scale = defaultdict(list)
        for point in series['points']:
            points_by_scale[point['scale']].append(point)
        # A line per repeat index retains each observation, including null gaps.
        for repeat in range(max((len(p) for p in points_by_scale.values()), default=0)):
            for field, color, dashed in (('accuracy', '#1864ab', ''), ('completionRate', '#c05621', ' stroke-dasharray="6 4"')):
                path = []
                connected = False
                for offset, scale in enumerate(SCALES):
                    observed = points_by_scale[scale]
                    value = observed[repeat][field] if repeat < len(observed) else None
                    if value is None:
                        connected = False
                        continue
                    if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                        raise ValueError('curve values must be finite ratios in 0..1')
                    x, y = 100 + offset * 220, top + 245 - value * 180
                    path.append(f'{"L" if connected else "M"}{x} {y}')
                    connected = True
                    elements.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"><title>{field}: {value:g}; scale {scale}; repeat {repeat + 1}</title></circle>')
                elements.append(f'<path d="{" ".join(path)}" fill="none" stroke="{color}" stroke-width="2"{dashed}/>')
        elements.extend([f'<text x="70" y="{top + 310}" fill="#1864ab" font-family="sans-serif" font-size="12">Accuracy (correct / answered)</text>',
                         f'<text x="390" y="{top + 310}" fill="#c05621" font-family="sans-serif" font-size="12">Completion (answered / total) — dashed</text>'])
    elements.append('</svg>')
    return '\n'.join(elements) + '\n'
