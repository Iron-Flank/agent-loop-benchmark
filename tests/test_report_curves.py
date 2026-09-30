import unittest
from almm_report.curves import build_curves, render_svg


class CurveTests(unittest.TestCase):
    def entries(self):
        entries = []
        for scale, answered, correct in ((10, 10, 10), (100, 5, 5), (500, 10, 5), (1000, 0, 0)):
            entries.append({'manifestHash': str(scale), 'manifest': {
                'adapter': {'name': 'runtime', 'revision': '1'}, 'model': {'name': 'model', 'deterministic': False},
                'tokenizer': {'name': 'tokens'}, 'scorerVersion': '1', 'harnessHash': 'h',
                'seed': 42, 'fixtureSeed': 42, 'generatorVersion': '1'}, 'report': {
                'scale': scale, 'accuracy': {'correct': correct, 'answered': answered, 'total': 10,
                    'accuracy': correct / answered if answered else None, 'completionRate': answered / 10}}})
        return entries

    def test_accuracy_and_completion_remain_independent_with_missing_answers(self):
        curves = build_curves(self.entries())
        points = curves['series'][0]['points']
        self.assertEqual([p['accuracy'] for p in points], [1, 1, .5, None])
        self.assertEqual([p['completionRate'] for p in points], [1, .5, 1, 0])
        svg = render_svg(curves)
        self.assertIn('stroke-dasharray', svg)
        self.assertIn('Accuracy (correct / answered)', svg)
        self.assertIn('Completion (answered / total)', svg)

    def test_repeats_preserve_points_and_variance_is_separate(self):
        entries = self.entries()
        repeat = dict(entries[0], manifestHash='repeat', report={'scale': 10, 'accuracy': {
            'correct': 0, 'answered': 10, 'total': 10, 'accuracy': 0, 'completionRate': 1}})
        curves = build_curves(entries + [repeat])
        self.assertEqual(len(curves['series'][0]['points']), 5)
        variance = curves['repeatedRunVariance'][0]
        self.assertEqual(variance['accuracy']['populationVariance'], .25)
        self.assertEqual(variance['accuracy']['values'], [1, 0])

    def test_changed_model_is_not_collapsed_into_runtime_curve(self):
        entries = self.entries()
        entries[0]['manifest']['model']['name'] = 'different'
        self.assertEqual(len(build_curves(entries)['series']), 2)


if __name__ == '__main__':
    unittest.main()
