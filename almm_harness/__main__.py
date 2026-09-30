"""Run the persistent offline CI pipeline without external model credentials."""
import argparse
import json
from pathlib import Path
import sys

from .proxy import redact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', required=True,
                        help='offline noncanonical 10-session fixture/run/score/report pipeline')
    parser.add_argument('--output', type=Path, default=Path('artifacts/smoke'),
                        help='persistent artifact root (default: artifacts/smoke)')
    args = parser.parse_args(argv)
    try:
        from .smoke import smoke
        result = smoke(args.output)
        print(json.dumps(redact(result), sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        print(f'FAIL: {redact(str(error))}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
