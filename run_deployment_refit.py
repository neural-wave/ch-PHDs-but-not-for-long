import os,subprocess,sys
env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONPATH='/home/mnogales/Projects/HACKATON/catboost_vendor')
for args in [['--smoke'],[]]:subprocess.run([sys.executable,'revin_campaign/refit_deployment.py']+args,env=env,check=True)
