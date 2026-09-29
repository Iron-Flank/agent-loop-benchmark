"""Generate immutable public fixtures or private, split held-out fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import stat
import sys
import time

from .engine import adapter_view, canonical_json, generate
from .validation import SchemaValidator


def _outside_checkout(path):
    resolved = path.resolve()
    for start in (Path.cwd(), Path(__file__).resolve().parent, resolved.parent):
        for parent in (start, *start.parents):
            if (parent / '.git').exists() and resolved.is_relative_to(parent):
                raise ValueError('held-out paths must be outside the public repository')
    return resolved


def _private_parent(path):
    parent = path.parent
    if not parent.is_dir() or stat.S_IMODE(parent.stat().st_mode) & 0o077:
        raise ValueError('held-out parent directory must exist with private permissions (0700)')


def _write_exclusive(path, value):
    # O_EXCL prevents overwrites and following a pre-existing symlink.
    import os
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(canonical_json(value))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    generator = commands.add_parser('generate')
    generator.add_argument('--sessions', type=int, choices=(10, 100, 500, 1000), required=True)
    seed_args = generator.add_mutually_exclusive_group()
    seed_args.add_argument('--seed', type=int, default=None)
    seed_args.add_argument('--held-out-seed-file', type=Path)
    generator.add_argument('--output', type=Path, required=True)
    validator = commands.add_parser('validate')
    validator.add_argument('fixture', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'validate':
            fixture = json.loads(args.fixture.read_text())
            SchemaValidator.validate(fixture)
            unhashed = dict(fixture)
            digest = unhashed.pop('contentHash', None)
            if digest is not None and digest != hashlib.sha256(canonical_json(unhashed)).hexdigest():
                raise ValueError('contentHash does not match canonical fixture contents')
            print('valid fixture')
            return 0
        start = time.perf_counter()
        held_out = args.held_out_seed_file is not None
        if held_out:
            source = _outside_checkout(args.held_out_seed_file)
            _private_parent(source)
            if stat.S_IMODE(source.stat().st_mode) & 0o077:
                raise ValueError('held-out seed file must have private permissions (0600)')
            seed = json.loads(source.read_text())['seed']
            output = _outside_checkout(args.output)
            _private_parent(output)
        else:
            seed = args.seed if args.seed is not None else 42
            output = args.output
        fixture = generate(seed, args.sessions, held_out=held_out)
        # No path is created until full validation and serialization succeed.
        if held_out:
            output.mkdir(mode=0o700)
            _write_exclusive(output / 'scorer.json', fixture)
            _write_exclusive(output / 'adapter.json', adapter_view(fixture))
        else:
            _write_exclusive(output, fixture)
        print(json.dumps({'sessions': args.sessions, 'heldOut': held_out,
                          'elapsedSeconds': round(time.perf_counter() - start, 6)}, sort_keys=True))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        # Held-out values/seeds never appear in diagnostics.
        message = 'held-out generation failed; check private paths, seed range and output existence' if (
            args.command == 'generate' and args.held_out_seed_file is not None) else str(error)
        print(f'error: {message}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
