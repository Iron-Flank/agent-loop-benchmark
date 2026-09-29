import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from almm_fixture.engine import generate, canonical_json, adapter_view
from almm_fixture.validation import SchemaValidator
from almm_fixture.topics import TopicPool


class FixtureTests(unittest.TestCase):
    def test_all_scales_accumulate_topics_and_abilities(self):
        for scale in (10, 100, 500, 1000):
            with self.subTest(scale=scale):
                fixture = generate(42, scale)
                SchemaValidator.validate(fixture)
                self.assertEqual(len(fixture['sessions']), scale)
                self.assertTrue(all(len(s['turns']) == 10 for s in fixture['sessions']))
                categories = {p['ability'] for p in fixture['probes']}
                self.assertEqual(categories, {'information-extraction', 'cross-session-reasoning',
                                             'temporal-reasoning', 'knowledge-update', 'abstention'})
                visits = {}
                for session in fixture['sessions']:
                    self.assertEqual(len({t['topicId'] for t in session['turns']}), 10)
                    for turn in session['turns']:
                        visits.setdefault(turn['topicId'], set()).add(session['sessionId'])
                self.assertGreaterEqual(sum(len(v) > 1 for v in visits.values()) / len(visits), .2)
                self.assertEqual({f['state'] for f in fixture['facts']},
                                 {'active', 'superseded', 'expired'})

    def test_distances_are_honest_and_evidence_is_prior_session(self):
        fixture = generate(91, 1000)
        sessions = {s['sessionId']: i for i, s in enumerate(fixture['sessions'])}
        turns = {t['turnId']: i for i, s in enumerate(fixture['sessions']) for t in s['turns']}
        facts = {f['factId']: f for f in fixture['facts']}
        distances = set()
        for probe in fixture['probes']:
            required = probe['expected']['requiredFactIds']
            if required:
                distance = sessions[probe['afterSessionId']] - min(
                    turns[facts[f]['introducedAt']] for f in required)
                self.assertEqual(probe['sessionDistance'], distance)
                self.assertGreater(distance, 0)
                distances.add(distance)
        self.assertTrue({10, 100, 500, 999} <= distances)
        self.assertNotIn(1000, distances)  # mathematical bound, not a passed ac-2 claim

    def test_determinism_and_hash(self):
        first = generate(42, 10)
        second = generate(42, 10)
        self.assertEqual(canonical_json(first), canonical_json(second))
        unhashed = dict(first)
        digest = unhashed.pop('contentHash')
        self.assertEqual(digest, hashlib.sha256(canonical_json(unhashed)).hexdigest())
        self.assertNotEqual(canonical_json(first), canonical_json(generate(43, 10)))

    def test_topic_pool_has_diverse_templates(self):
        pool = TopicPool()
        self.assertGreaterEqual(len(pool.topics), 100)
        self.assertEqual(len(pool.by_id), len(pool.topics))
        self.assertEqual({t.category for t in pool.topics},
                         {'people', 'places', 'preferences', 'events', 'decisions', 'quantities'})

    def test_adapter_projection_uses_allowlist(self):
        fixture = generate(42, 10)
        fixture['sessions'][0]['turns'][0]['futureSecret'] = 'hidden'
        view = adapter_view(fixture)
        self.assertEqual(set(view), {'sessions', 'probes'})
        self.assertTrue(all(set(t) == {'turnId', 'role', 'text'}
                            for s in view['sessions'] for t in s['turns']))
        self.assertTrue(all(set(p) == {'probeId', 'afterSessionId', 'question'} for p in view['probes']))
        for key in ('expected', 'requiredFactIds', 'forbiddenFactIds', 'matchType', 'introducedFactIds'):
            self.assertNotIn('"' + key + '"', json.dumps(view))

    def test_validator_rejects_corrupted_generator_output(self):
        fixture = generate(42, 10)
        mutations = [
            lambda f: f['sessions'][1]['turns'][0].update(turnId=f['sessions'][0]['turns'][0]['turnId']),
            lambda f: f['facts'][0].update(introducedAt='missing'),
            lambda f: f['facts'][0].update(supersedes='missing'),
            lambda f: f['probes'][0].update(afterSessionId='s-0001'),
            lambda f: f['probes'][0]['expected'].update(acceptedAnswers=['incorrect']),
        ]
        for mutation in mutations:
            bad = copy.deepcopy(fixture)
            mutation(bad)
            with self.assertRaises(ValueError):
                SchemaValidator.validate(bad)

    def test_cli_validation_failure_emits_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'invalid.json'
            source.write_text('{"schemaVersion":"invalid"}')
            result = subprocess.run([sys.executable, '-m', 'almm_fixture', 'validate', str(source)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('error', result.stderr.lower())
            self.assertEqual(result.stdout, '')

    def test_held_out_requires_private_storage_and_distinct_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            seeds = root / 'seed.json'
            seeds.write_text('{"seed": 1000000042}')
            seeds.chmod(0o600)
            command = [sys.executable, '-m', 'almm_fixture', 'generate', '--sessions', '10',
                       '--held-out-seed-file', str(seeds), '--output', str(root / 'output')]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = root / 'output'
            adapter = json.loads((output / 'adapter.json').read_text())
            scorer = json.loads((output / 'scorer.json').read_text())
            self.assertNotEqual(scorer['seed'], 42)
            self.assertEqual(adapter, adapter_view(scorer))
            self.assertEqual((output.stat().st_mode & 0o777), 0o700)
            self.assertEqual(((output / 'scorer.json').stat().st_mode & 0o777), 0o600)
            rerun = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rerun.returncode, 0)  # immutable, never overwrite fixture versions

    def test_public_seed_range_and_invalid_scale(self):
        for seed, scale in ((-1, 10), (1000000000, 10), (True, 10), (42, 11)):
            with self.subTest(seed=seed, scale=scale), self.assertRaises(ValueError):
                generate(seed, scale)


if __name__ == '__main__':
    unittest.main()
