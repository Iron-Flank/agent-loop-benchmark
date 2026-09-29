"""Run after implementing adapter.py; deliberate stubs fail with guidance."""
import argparse
import sys

from almm_adapter.conformance import ConformanceError, check_adapter
from almm_adapter.http import HttpAdapter
from adapter import create_adapter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', help='check an already running separate-process adapter')
    args = parser.parse_args()
    try:
        target = HttpAdapter(args.url) if args.url else create_adapter()
        print('PASSED:', ', '.join(check_adapter(target)))
    except (ConformanceError, ValueError, RuntimeError, OSError) as exc:
        print(f'FAIL: {exc}', file=sys.stderr)
        sys.exit(1)
