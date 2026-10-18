import os,subprocess,sys
os.environ.update(JOINT_CNN_PARALLEL='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
for args in [('code/tune_parallel_cnn.py','--prepare'),('code/tune_parallel_cnn.py','--smoke'),('code/tune_parallel_cnn.py',),('code/audit_tuned_parallel_cnn.py',)]:
    subprocess.run([sys.executable,*args],check=True)
