"""Publish a report from archived execution/scoring, or plot archived reports."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

from almm_fixture.engine import canonical_json
from almm_harness.proxy import redact
from .curves import build_curves, render_svg
from .writer import ReportWriter, publish_directory, validate_report_manifest


def load_report(directory):
    directory = Path(directory)
    content = (directory / 'manifest.json').read_bytes()
    identity = hashlib.sha256(content).hexdigest()
    if directory.name != identity:
        raise ValueError('report directory does not match manifest hash')
    manifest = json.loads(content)
    validate_report_manifest(manifest)
    size = len(content)
    for name, expected in manifest['artifactHashes'].items():
        path = directory / name
        if Path(name).name != name or name == 'manifest.json':
            raise ValueError('invalid report artifact filename')
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                size += len(chunk)
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f'report artifact hash mismatch: {name}')
    if size != manifest['artifactSizeBytes']:
        raise ValueError('report artifact size mismatch')
    return {'manifestHash': identity, 'manifest': manifest,
            'report': json.loads((directory / 'report.json').read_text())}


def publish_curves(directories, output):
    curves = build_curves([load_report(path) for path in directories])
    content = canonical_json(curves)
    identity = hashlib.sha256(content).hexdigest()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / identity
    files = {'curves.json': content, 'curves.svg': render_svg(curves).encode()}
    temporary = Path(tempfile.mkdtemp(prefix='.curves-', dir=output))
    try:
        for name, data in files.items():
            path = temporary / name
            path.write_bytes(data)
            path.chmod(0o444)
        try:
            publish_directory(temporary, destination)
        except FileExistsError:
            if {path.name for path in destination.iterdir()} != set(files) or any(
                    (destination / name).read_bytes() != data for name, data in files.items()):
                raise FileExistsError('immutable curve artifacts differ') from None
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return {'artifactDir': str(destination), 'seriesCount': len(curves['series']),
            'repeatedRunVariance': curves['repeatedRunVariance']}


def main():
    parser = argparse.ArgumentParser(description='Immutable ALMM ReportWriter')
    parser.add_argument('--smoke', action='store_true', help='offline noncanonical four-scale full pipeline')
    parser.add_argument('--fixture', type=Path, help='scorer-only full fixture JSON')
    parser.add_argument('--run', type=Path, help='finalized runner archive')
    parser.add_argument('--scores', type=Path, help='scorer archive for that run')
    parser.add_argument('--reports', type=Path, nargs='+', help='report directories to plot')
    parser.add_argument('--output', type=Path, help='immutable publication root')
    args = parser.parse_args()
    try:
        if args.smoke:
            if any((args.fixture, args.run, args.scores, args.reports, args.output)):
                parser.error('--smoke cannot be combined with paths')
            from .smoke import smoke
            result = smoke()
        elif args.reports:
            if not args.output or any((args.fixture, args.run, args.scores)):
                parser.error('--reports requires --output and excludes run/scorer paths')
            result = publish_curves(args.reports, args.output)
        else:
            if not all((args.fixture, args.run, args.scores, args.output)):
                parser.error('require --fixture, --run, --scores, --output (or --reports/--smoke)')
            published = ReportWriter().write(json.loads(args.fixture.read_text()), args.run, args.scores, args.output)
            result = {'artifactDir': published['artifactDir'],
                      'artifactSizeBytes': published['manifest']['artifactSizeBytes'],
                      'warnings': published['manifest']['warnings'], 'accuracy': published['report']['accuracy']}
        print(json.dumps(redact(result), allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        print(f'FAIL: {redact(str(error))}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
