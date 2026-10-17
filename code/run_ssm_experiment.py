"""Execution-host Nightwatch child; success requires independent final audit."""
import os,subprocess,sys
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
for args in [('code/ssm_experiment.py','prepare'),('code/ssm_experiment.py','smoke'),('code/ssm_experiment.py','train'),('code/ssm_experiment.py','rolling'),('code/audit_ssm_experiment.py',)]:
    print('STAGE',*args,flush=True);subprocess.run([sys.executable,*args],check=True)
