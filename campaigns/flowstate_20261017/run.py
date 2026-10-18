"""Durable Nightwatch command: feature inference, gated training, audit."""
import os
import subprocess
import sys
from pathlib import Path

os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
here = Path(__file__).resolve().parent
for args in [('experiment.py', 'features', '--start', '28', '--end', '333'),
             ('experiment.py', 'smoke'), ('experiment.py', 'train'),
             ('experiment.py', 'features', '--start', '333', '--end', '363'),
             ('experiment.py', 'score'), ('audit.py',)]:
    subprocess.run([sys.executable, str(here/args[0]), *args[1:]], check=True)
