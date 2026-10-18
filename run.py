"""Run a repository script with shared modules and repository-relative paths."""
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print('Usage: python run.py <script-name or repository-relative script> [arguments]')
        print('Examples: python run.py predict_best --help; python run.py rank_models')
        return
    name = sys.argv[1]
    target = ROOT / name
    if not target.is_file():
        target = ROOT / 'code' / (name if name.endswith('.py') else name + '.py')
    target = target.resolve()
    if not target.is_relative_to(ROOT) or target.suffix != '.py' or not target.is_file() or target == Path(__file__).resolve():
        raise SystemExit(f'Unknown repository script: {name}')
    os.chdir(ROOT)
    paths = [str(target.parent), str(ROOT / 'code'), str(ROOT)]
    sys.path[:0] = paths
    os.environ['PYTHONPATH'] = os.pathsep.join(paths + ([os.environ['PYTHONPATH']] if os.environ.get('PYTHONPATH') else []))
    sys.argv = [str(target), *sys.argv[2:]]
    runpy.run_path(str(target), run_name='__main__')


if __name__ == '__main__':
    main()
