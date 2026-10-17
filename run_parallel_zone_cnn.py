import os, subprocess, sys
os.environ.update(JOINT_CNN_PARALLEL='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
for args in [('joint_zone_cnn.py','--prepare'),('joint_zone_cnn.py','--smoke'),('joint_zone_cnn.py',),('audit_joint_zone_cnn.py',)]:
    subprocess.run([sys.executable,*args],check=True)
from joint_zone_cnn import ROOT,JointCNN
from parallel_zone_cnn import export_filters
export_filters(ROOT,JointCNN)
