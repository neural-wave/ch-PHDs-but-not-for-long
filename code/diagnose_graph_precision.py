"""Measure CPU/GPU convolution precision on a fixed selected graph component."""
import json,os,pickle,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'catboost_vendor'))
import numpy as np
import torch
from reconstruction import read_targets,rich_features
from graph_models import GraphNet

assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
torch.set_num_threads(4)
directory=Path('results/monthly_graph_gpu1')
with (directory/'model.pkl').open('rb') as f:bundle=pickle.load(f)
item=max((i for i in bundle['models'] if i['name'].startswith('graph_')),key=lambda i:sum(i['weight']))
cfg=item['config'];dates,y=read_targets('train_prices_15min.csv');h=y.mean(-1)
x=rich_features(dates,h,dates,y)
xx=np.clip((x-bundle['mu'])/bundle['sd'],-15,15).reshape(-1,4,24,x.shape[1])
cpu=GraphNet(x.shape[1],cfg['width'],cfg['blocks'],cfg['dropout']);cpu.load_state_dict(item['artifact']);cpu.eval()
with torch.no_grad():reference=cpu(torch.tensor(xx[333:])).numpy()*20
gpu=GraphNet(x.shape[1],cfg['width'],cfg['blocks'],cfg['dropout']).cuda();gpu.load_state_dict(item['artifact']);gpu.eval()
xt=torch.tensor(xx,device='cuda');results={}
for flag in [True,False]:
    torch.backends.cudnn.allow_tf32=flag
    with torch.no_grad():pred=gpu(xt).cpu().numpy()[333:]*20
    results['cudnn_tf32_'+str(flag)]=dict(max_cpu_difference=float(np.abs(pred-reference).max()),mean_cpu_difference=float(np.abs(pred-reference).mean()))
result=dict(component=item['name'],matmul_tf32=torch.backends.cuda.matmul.allow_tf32,device=torch.cuda.get_device_name(0),results=results)
(directory/'precision_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
