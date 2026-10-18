"""Fixed small architecture; three seeds for each history condition."""
import os, subprocess, sys
os.environ.update(JOINT_CNN_SIMPLE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
for args in [('code/joint_zone_cnn.py','--prepare'),('code/joint_zone_cnn.py','--smoke'),('code/joint_zone_cnn.py',),('code/audit_joint_zone_cnn.py',)]:
    subprocess.run([sys.executable,*args],check=True)
