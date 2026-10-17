import os, subprocess, sys
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
for args in [('joint_zone_cnn.py','--prepare'),('joint_zone_cnn.py','--smoke'),('joint_zone_cnn.py',),('audit_joint_zone_cnn.py',)]:
    subprocess.run([sys.executable,*args],check=True)
