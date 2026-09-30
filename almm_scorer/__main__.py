"""Scoring configuration lives in a manifest; only secrets use environment variables."""
import argparse
import json
from pathlib import Path
import sys

from almm_harness.proxy import redact
from .pipeline import ScoringPipeline
from .semantic import SemanticJudge


def main():
    parser = argparse.ArgumentParser(description='Calibrated ALMM archive scorer')
    parser.add_argument('--smoke', action='store_true', help='offline noncanonical full-pipeline smoke')
    parser.add_argument('--fixture', type=Path, help='scorer-only full fixture JSON')
    parser.add_argument('--run', type=Path, help='finalized FixtureRunner artifact directory')
    parser.add_argument('--manifest', type=Path, help='scoring configuration JSON')
    parser.add_argument('--output', type=Path, help='immutable scoring artifact root')
    args = parser.parse_args()
    try:
        if args.smoke:
            if any((args.fixture, args.run, args.manifest, args.output)):
                parser.error('--smoke cannot be combined with scoring paths')
            from .smoke import smoke
            result = smoke()
        else:
            if not all((args.fixture, args.run, args.manifest, args.output)):
                parser.error('require --fixture, --run, --manifest, and --output')
            config = json.loads(args.manifest.read_text())
            if not isinstance(config, dict) or redact(config) != config:
                raise ValueError('scoring manifest must be an object without credentials')
            canonical = config.get('canonical', True)
            if type(canonical) is not bool:
                raise ValueError('canonical must be a boolean')
            judge = SemanticJudge(config['judge'], runtime_key_env=config.get('runtimeKeyEnv', 'OPENAI_API_KEY'))
            pipeline = ScoringPipeline(judge, calibration_dir=config['calibrationDirectory'],
                                       version=config['scorerVersion'])
            result = pipeline.run(json.loads(args.fixture.read_text()), args.run, args.output, canonical=canonical)
            result = {'artifactDir': result['artifactDir'], 'probeCount': result['probeCount'],
                      'canonical': canonical, 'scorerVersion': result['manifest']['scorerVersion'],
                      'calibrationAgreement': result['manifest']['calibration']['agreement']}
        print(json.dumps(redact(result), allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print(f'FAIL: {redact(str(error))}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
