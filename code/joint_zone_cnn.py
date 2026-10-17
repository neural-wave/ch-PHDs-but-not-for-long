"""Joint four-zone reconstruction, with matched causal seven-day ablation."""
import argparse, concurrent.futures as cf, datetime as dt, hashlib, itertools, json, multiprocessing as mp, time, os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from reconstruction import read_targets, baseline

SIMPLE = os.environ.get('JOINT_CNN_SIMPLE') == '1'
PARALLEL = os.environ.get('JOINT_CNN_PARALLEL') == '1'
ROOT = Path('results/parallel_2d_zone_cnn_20261017' if PARALLEL else 'results/simple_joint_zone_cnn_20261017' if SIMPLE else 'results/joint_zone_cnn_20261017')
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
SHA = 'f71ae6dbcca5f04e76e35840e29d064e69ad292f26d8ec7bf013bff2d904f65c'

def inputs(dates, hourly, history_dates, history_y, history):
    """Channels are fixed zone identities; time axis is 96 quarters."""
    if PARALLEL:
        from parallel_zone_cnn import build_inputs
        return build_inputs(dates,hourly,history_dates,history_y,history)
    hist = {str(d): y for d, y in zip(history_dates, history_y)}
    rows = []
    for date, h in zip(dates, hourly):
        day = dt.date.fromisoformat(str(date))
        q = np.arange(96)
        channels = [np.repeat(h, 4, axis=-1)/100, baseline(h[None])[0].reshape(4,96)/100,
                    np.stack([np.sin(2*np.pi*q/96), np.cos(2*np.pi*q/96),
                              np.sin(2*np.pi*(q%4)/4), np.cos(2*np.pi*(q%4)/4)]),
                    np.broadcast_to(np.array([np.sin(2*np.pi*day.timetuple().tm_yday/365.25),
                                              np.cos(2*np.pi*day.timetuple().tm_yday/365.25),
                                              np.sin(2*np.pi*day.weekday()/7),
                                              np.cos(2*np.pi*day.weekday()/7)])[:,None], (4,96))]
        if SIMPLE:
            channels = [np.repeat(h,4,axis=-1)/100, np.eye(4,dtype=np.float32)[q%4].T]
        if history:
            for lag in range(1,8):
                key = str(day-dt.timedelta(days=lag))
                assert key < str(date)
                yy = hist.get(key)
                yy = np.zeros((4,24,4)) if yy is None else yy
                channels += [yy.reshape(4,96)/100, np.repeat(yy.mean(-1),4,axis=-1)/100,
                             np.full((1,96),float(key in hist))]
        rows.append(np.concatenate(channels))
    return np.asarray(rows,dtype=np.float32)

class JointCNN(nn.Module):
    def __init__(self, n, cfg):
        super().__init__()
        if PARALLEL:
            from parallel_zone_cnn import ParallelCNN
            self.net=ParallelCNN(n,cfg)
            return
        layers=[]
        for i in range(cfg['depth']):
            # Dilations 1/2/4/8 spread context across the daily curve.
            dilation=1 if SIMPLE else 2**(i%4); k=cfg['kernel']
            layers += [nn.Conv1d(n,cfg['width'],k,padding=dilation*(k//2),dilation=dilation,padding_mode='replicate'),nn.ReLU() if SIMPLE else nn.SiLU()]
            n=cfg['width']
        layers += [nn.Conv1d(n,4,1)]
        self.net=nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):
        y=self.net(x).reshape(-1,4,24,4)
        return y-y.mean(-1,keepdim=True)

def configs():
    if PARALLEL:
        return [dict(id=i,history=h,width=8,depth=1,kernel=[2,8],lr=.001,seed=s)
                for i,(h,s) in enumerate(itertools.product([False,True],[17,43,71]))]
    if SIMPLE:
        return [dict(id=i,history=h,width=16,depth=1,kernel=5,lr=.001,seed=s)
                for i,(h,s) in enumerate(itertools.product([False,True],[17,43,71]))]
    return [dict(id=i,history=h,width=w,depth=d,kernel=k,lr=lr,seed=s)
            for i,(h,w,d,k,lr,s) in enumerate(itertools.product([False,True],[32,64,128],[2,4,6],[3,5],[.001,.003],[17,43]))]

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    source=Path('train_prices_15min.csv');assert hashlib.sha256(source.read_bytes()).hexdigest()==SHA
    dates,y=read_targets(source);h=y.mean(-1);b=baseline(h)
    assert len(dates)==363 and dates[302]=='2026-08-01' and dates[333]=='2026-09-01'
    np.savez(ROOT/'data.npz',dates=dates.astype('U10'),y=y,h=h,b=b)
    for history in [False,True]:
        x=inputs(dates,h,dates,y,history)
        changed=y.copy();changed[333:]+=np.arange(4)*12345
        assert np.array_equal(x[333:334],inputs(dates[333:334],h[333:334],dates,changed,history))
        assert np.array_equal(x[333:334],inputs(dates[333:334],h[333:334],dates[:333],y[:333],history))
        mu=x[:302].mean((0,2),keepdims=True);sd=x[:302].std((0,2),keepdims=True).clip(.05)
        np.savez(ROOT/f'features_{int(history)}.npz',x=np.clip((x-mu)/sd,-15,15),raw=x,mu=mu,sd=sd)
    (ROOT/'manifest.json').write_text(json.dumps(dict(sha256=SHA,train=302,validation=31,test=30,configs=configs(),
        selection='August only; September is previously inspected development data',
        architecture='Parallel Q/H rows: Conv2d(5,8,(2,8)), ReLU, late information fusion8, output4' if PARALLEL else 'Conv1d(input,16,5), ReLU, Conv1d(16,4,1)' if SIMPLE else '96 time steps, all zones mixed as channels, output 4 zones jointly',
        history='calendar lags 1..7; raw quarter prices and repeated hourly means, missing-date masks'),indent=2))

def trial(cfg,epochs=300,save=True):
    torch.set_num_threads(1);torch.manual_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=time.perf_counter();d=np.load(ROOT/'data.npz');f=np.load(ROOT/f"features_{int(cfg['history'])}.npz")
    x=torch.tensor(f['x'][:333],device='cuda');target=torch.tensor((d['y'][:333]-d['b'][:333])/20,dtype=torch.float32,device='cuda')
    net=JointCNN(x.shape[1],cfg).cuda();opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=.01)
    best=float('inf');best_epoch=0;state=None
    for epoch in range(epochs):
        net.train()
        for ids in torch.randperm(302,device='cuda').split(32):
            loss=(net(x[ids])-target[ids]).abs().mean();opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
        net.eval()
        with torch.no_grad():val=float((net(x[302:333])-target[302:333]).abs().mean())*20
        assert np.isfinite(val)
        if val<best:
            best=val;best_epoch=epoch+1;state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        if epoch+1-best_epoch>=30:break
    meta=dict(config=cfg,validation_mae=best,best_epoch=best_epoch,epochs=epoch+1,
              parameters=sum(p.numel() for p in net.parameters()),seconds=time.perf_counter()-start)
    if save:
        folder=ROOT/'trials'/str(cfg['id']);folder.mkdir(parents=True,exist_ok=True)
        torch.save(dict(config=cfg,state=state,mu=f['mu'],sd=f['sd']),folder/'model.pt')
        (folder/'metrics.json').write_text(json.dumps(meta,indent=2))
    return meta

def pool(jobs,workers,epochs,save):
    with cf.ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as ex:
        futures=[ex.submit(trial,c,epochs,save) for c in jobs];rows=[]
        for f in cf.as_completed(futures):
            rows.append(f.result())
            if save:
                (ROOT/'progress.json').write_text(json.dumps(dict(completed=len(rows),total=len(jobs),latest=rows[-1]),indent=2))
                print('FIT',len(rows),len(jobs),rows[-1],flush=True)
    return rows

def finalize(rows):
    # Persist choices before evaluating any September predictions.
    winners={str(int(h)):min([r for r in rows if r['config']['history']==h],key=lambda r:r['validation_mae']) for h in [False,True]}
    (ROOT/'selection.json').write_text(json.dumps(winners,indent=2))
    d=np.load(ROOT/'data.npz');reports={}
    for key,r in winners.items():
        f=np.load(ROOT/f'features_{key}.npz');c=r['config'];folder=ROOT/'trials'/str(c['id'])
        checkpoint=torch.load(folder/'model.pt',weights_only=False,map_location='cpu')
        net=JointCNN(f['x'].shape[1],c).cuda().eval();net.load_state_dict(checkpoint['state'])
        with torch.no_grad():raw=net(torch.tensor(f['x'][302:],device='cuda')).cpu().numpy().astype('float64')
        pred=d['b'][302:]+20*(raw-raw.mean(-1,keepdims=True))
        net2=JointCNN(f['x'].shape[1],c).cuda().eval();net2.load_state_dict(checkpoint['state'])
        with torch.no_grad():replay=net2(torch.tensor(f['x'][302:],device='cuda')).cpu().numpy()
        assert np.max(np.abs(raw-replay))<1e-7
        np.save(folder/'predictions.npy',pred)
        error=np.abs(pred[31:]-d['y'][333:]);conservation=float(np.abs(pred.mean(-1)-d['h'][302:]).max())
        assert conservation<1e-8 and np.isfinite(pred).all()
        reports[key]=dict(**r,september_mae=float(error.mean()),by_zone=error.mean((0,2,3)).tolist(),
                          hourly_mean_error=conservation,replay_max_error=float(np.max(np.abs(raw-replay))))
    report=dict(by_history=reports,interpolation_september_mae=float(np.abs(d['b'][333:]-d['y'][333:]).mean()),
                caveat='August selection only; September developmental; no deployment refit',candidates=rows)
    (ROOT/'metrics.json').write_text(json.dumps(report,indent=2));print('RESULT',reports,flush=True)

def smoke():
    d=np.load(ROOT/'data.npz');dates=d['dates'];y=d['y'];h=d['h']
    if PARALLEL:
        from parallel_zone_cnn import smoke as smoke_parallel
        smoke_parallel(d,configs(),ROOT,model=JointCNN)
        pool(configs(),2,5,False)
        (ROOT/'smoke.json').write_text(json.dumps(dict(passed=True,lag_values=True,missing_masks=True,
            cross_zone_gradient=True,parallel_rows=True,kernel=[2,8],profile=[dict(workers=2,criterion='fixed small six-fit campaign')]),indent=2))
        return
    x=inputs(dates[333:334],h[333:334],dates,y,True)
    hist={str(date):yy for date,yy in zip(dates,y)}
    for lag in range(1,8):
        key=str(dt.date.fromisoformat(str(dates[333]))-dt.timedelta(days=lag));start=(8 if SIMPLE else 16)+(lag-1)*9
        np.testing.assert_allclose(x[0,start:start+4],hist[key].reshape(4,96)/100,rtol=1e-6)
        np.testing.assert_allclose(x[0,start+4:start+8],np.repeat(hist[key].mean(-1),4,axis=-1)/100,rtol=1e-6)
    empty=inputs(dates[:1],h[:1],[],[],True);assert not empty[:,8 if SIMPLE else 16:].any()
    # Cross-zone dependency and nonzero gradients after opening zero-initialized head.
    for history in [False,True]:
        f=np.load(ROOT/f'features_{int(history)}.npz');cfg=next(c for c in configs() if c['history']==history)
        net=JointCNN(f['x'].shape[1],cfg).cuda()
        torch.nn.init.normal_(net.net[-1].weight,std=.01)
        xx=torch.tensor(f['x'][:2],device='cuda',requires_grad=True);out=net(xx)
        out[:,0,:,0].sum().backward();assert xx.grad[:,1].abs().sum()>0
        assert out.mean(-1).abs().max()<1e-6
    profile=[]
    jobs=configs() if SIMPLE else [configs()[i] for i in [0,16,32,64,72,88,104,136]]
    for workers in ([2] if SIMPLE else [4,8]):
        start=time.perf_counter();pool(jobs,workers,5,False)
        profile.append(dict(workers=workers,seconds=time.perf_counter()-start))
    (ROOT/'smoke.json').write_text(json.dumps(dict(passed=True,lag_values=True,missing_masks=True,cross_zone_gradient=True,profile=profile),indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    elif a.smoke:smoke()
    else:
        profile=json.loads((ROOT/'smoke.json').read_text())['profile'];workers=min(profile,key=lambda r:r.get('seconds',0))['workers']
        print('WORKERS',workers,flush=True);finalize(pool(configs(),workers,300,True))

if __name__=='__main__':main()
