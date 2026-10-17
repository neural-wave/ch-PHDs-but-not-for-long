"""Focused, August-selected improvements to the small parallel 2D CNN."""
import argparse, concurrent.futures as cf, datetime as dt, hashlib, itertools, json, multiprocessing as mp, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from parallel_zone_cnn import build_inputs, AUX
from reconstruction import read_targets, baseline

ROOT=Path('results/parallel_2d_tuning_20261017')
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
ARMS={
    'control':dict(width=8,clock=False,residual=False,analog=0,context=1),
    'width16':dict(width=16,clock=False,residual=False,analog=0,context=1),
    'clock':dict(width=16,clock=True,residual=False,analog=0,context=1),
    'residual':dict(width=16,clock=True,residual=True,analog=0,context=1),
    'analog7':dict(width=16,clock=True,residual=True,analog=7,context=1),
    'analog90':dict(width=16,clock=True,residual=True,analog=90,context=1),
    'context':dict(width=16,clock=True,residual=True,analog=90,context=5),
}

def configs():
    return [dict(id=i,arm=arm,lr=lr,seed=seed,**ARMS[arm]) for i,(arm,lr,seed) in enumerate(itertools.product(ARMS,[.001,.003],[17,43,71]))]

def make_features(dates,h,history_dates,history_y,arm):
    cfg=ARMS[arm];base=build_inputs(dates,h,history_dates,history_y,True)
    hist={str(d):yy for d,yy in zip(history_dates,history_y)}
    hist_h={d:yy.mean(-1) for d,yy in hist.items()}
    errors={d:yy-baseline(hist_h[d][None])[0] for d,yy in hist.items()}
    allx=[]
    for i,(date,hh) in enumerate(zip(dates,h)):
        day=dt.date.fromisoformat(str(date));grid=base[i,:-AUX].reshape(5,16,96);aux=base[i,-AUX:]
        if cfg['residual']:
            residual=[]
            for lag in range(7,-1,-1):
                key=str(day-dt.timedelta(days=lag));available=lag>0 and key in hist
                residual.extend([errors[key].reshape(4,96)/20 if available else np.zeros((4,96)),
                    np.repeat(hist_h[key]-hh,4,axis=-1)/100 if available else np.zeros((4,96))])
            grid=np.concatenate([grid,np.stack(residual,axis=1)],axis=0)
        if cfg['clock']:
            q=np.arange(96);year=day.timetuple().tm_yday
            clock=np.stack([np.sin(2*np.pi*q/96),np.cos(2*np.pi*q/96),
                            np.full(96,np.sin(2*np.pi*year/365.25)),np.full(96,np.cos(2*np.pi*year/365.25))])
            aux=np.concatenate([aux,clock])
        if cfg['analog']:
            earliest=str(day-dt.timedelta(days=cfg['analog']));prior=sorted(d for d in hist if earliest<=d<str(date))
            analog=np.zeros((4,24,4))
            if prior:
                ph=np.array([hist_h[d] for d in prior]);pe=np.array([errors[d] for d in prior])
                for z in range(4):
                    for hour in range(24):
                        ix=np.clip(hour+np.arange(-2,3),0,23)
                        shape=hh[z,ix]-hh[z,hour]
                        dist=np.mean((ph[:,z,ix]-ph[:,z,hour,None]-shape)**2,axis=1)+.1*(ph[:,z,hour]-hh[z,hour])**2
                        nearest=np.argsort(dist)[:10];weights=1/(25+dist[nearest]);weights/=weights.sum()
                        analog[z,hour]=(pe[nearest,z,hour]*weights[:,None]).sum(0)
            aux=np.concatenate([aux,analog.reshape(4,96)/20])
        allx.append(np.concatenate([grid.reshape(-1,96),aux]))
    return np.asarray(allx,dtype=np.float32)

def dimensions(cfg):
    return (9 if cfg['residual'] else 5),23+4*int(cfg['clock'])+4*int(bool(cfg['analog']))

class Small2D(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.channels,self.aux=dimensions(cfg);w=cfg['width'];k=cfg['context']
        self.conv=nn.Conv2d(self.channels,w,(2,8),stride=(2,1))
        self.fusion=nn.Conv1d(w*8+self.aux,w,k,padding=k//2,padding_mode='replicate')
        self.output=nn.Conv1d(w,4,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,x):
        grid=x[:,:-self.aux].reshape(-1,self.channels,16,96)
        encoded=F.relu(self.conv(F.pad(grid,(3,4,0,0),mode='replicate'))).flatten(1,2)
        y=self.output(F.relu(self.fusion(torch.cat([encoded,x[:,-self.aux:]],1)))).reshape(-1,4,24,4)
        return y-y.mean(-1,keepdim=True)

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True);source=Path('train_prices_15min.csv')
    sha=hashlib.sha256(source.read_bytes()).hexdigest();assert sha=='f71ae6dbcca5f04e76e35840e29d064e69ad292f26d8ec7bf013bff2d904f65c'
    dates,y=read_targets(source);h=y.mean(-1);b=baseline(h)
    assert len(y)==363 and dates[302]=='2026-08-01' and dates[333]=='2026-09-01'
    np.savez(ROOT/'data.npz',dates=dates.astype('U10'),y=y,h=h,b=b)
    changed=y.copy();changed[333:]+=np.arange(4)*4321
    for arm in ARMS:
        raw=make_features(dates,h,dates,y,arm)
        probe=make_features(dates[333:334],h[333:334],dates,changed,arm)
        np.testing.assert_array_equal(raw[333:334],probe)
        np.testing.assert_array_equal(probe,make_features(dates[333:334],h[333:334],dates[:333],y[:333],arm))
        mu=raw[:302].mean((0,2),keepdims=True);sd=raw[:302].std((0,2),keepdims=True).clip(.05)
        np.savez(ROOT/f'{arm}.npz',x=np.clip((raw-mu)/sd,-15,15),mu=mu,sd=sd)
        print('PREPARED',arm,raw.shape,flush=True)
    (ROOT/'manifest.json').write_text(json.dumps(dict(sha256=sha,arms=ARMS,configs=configs(),train=302,validation=31,test=30,
        note='August only selection; September developmental, no target-based stopping. Grid always 7 days; analog90 uses up to90 strictly prior days.'),indent=2))

def trial(cfg,epochs=600,save=True):
    torch.set_num_threads(1);torch.manual_seed(cfg['seed']);start=time.perf_counter()
    d=np.load(ROOT/'data.npz');f=np.load(ROOT/f"{cfg['arm']}.npz");x=torch.tensor(f['x'][:333],device='cuda')
    target=torch.tensor((d['y'][:333]-d['b'][:333])/20,dtype=torch.float32,device='cuda')
    net=Small2D(cfg).cuda();opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=.01)
    best=float('inf');best_epoch=0;state=None
    for epoch in range(epochs):
        net.train()
        for ids in torch.randperm(302,device='cuda').split(32):
            loss=(net(x[ids])-target[ids]).abs().mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
        net.eval()
        with torch.no_grad():val=float((net(x[302:])-target[302:]).abs().mean())*20
        assert np.isfinite(val)
        if val<best:
            best=val;best_epoch=epoch+1;state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        if epoch+1-best_epoch>=60:break
    meta=dict(config=cfg,validation_mae=best,best_epoch=best_epoch,epochs=epoch+1,
              parameters=sum(p.numel() for p in net.parameters()),seconds=time.perf_counter()-start)
    if save:
        folder=ROOT/'trials'/str(cfg['id']);folder.mkdir(parents=True,exist_ok=True)
        torch.save(dict(config=cfg,state=state,mu=f['mu'],sd=f['sd']),folder/'model.pt')
        (folder/'metrics.json').write_text(json.dumps(meta,indent=2))
    return meta

def pool(jobs,epochs,save):
    with cf.ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as ex:
        rows=[]
        for future in cf.as_completed([ex.submit(trial,c,epochs,save) for c in jobs]):
            rows.append(future.result())
            if save:
                (ROOT/'progress.json').write_text(json.dumps(dict(completed=len(rows),total=len(jobs),latest=rows[-1]),indent=2))
                print('FIT',len(rows),len(jobs),rows[-1],flush=True)
    return rows

def finalize(rows):
    winners={arm:min([r for r in rows if r['config']['arm']==arm],key=lambda r:r['validation_mae']) for arm in ARMS}
    selection=dict(selected=min(rows,key=lambda r:r['validation_mae']),by_arm=winners)
    (ROOT/'selection.json').write_text(json.dumps(selection,indent=2))
    d=np.load(ROOT/'data.npz');results={}
    for arm,r in winners.items():
        folder=ROOT/'trials'/str(r['config']['id']);checkpoint=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
        net=Small2D(r['config']).cuda().eval();net.load_state_dict(checkpoint['state']);f=np.load(ROOT/f'{arm}.npz')
        with torch.no_grad():raw=net(torch.tensor(f['x'][302:],device='cuda')).cpu().numpy().astype('float64')
        pred=d['b'][302:]+20*(raw-raw.mean(-1,keepdims=True));np.save(folder/'predictions.npy',pred)
        assert np.isfinite(pred).all() and np.abs(pred.mean(-1)-d['h'][302:]).max()<1e-8
        results[arm]=dict(**r,september_mae=float(np.abs(pred[31:]-d['y'][333:]).mean()),
                         by_zone=np.abs(pred[31:]-d['y'][333:]).mean((0,2,3)).tolist())
        np.savez(folder/'learned_filters.npz',conv2d=net.conv.weight.detach().cpu().numpy(),fusion=net.fusion.weight.detach().cpu().numpy(),output=net.output.weight.detach().cpu().numpy(),mu=f['mu'],sd=f['sd'])
    report=dict(selected_arm=selection['selected']['config']['arm'],by_arm=results,candidates=rows,
                note='All choices frozen using August before September inference. September developmental; target5.1-5.2 not guaranteed.')
    (ROOT/'metrics.json').write_text(json.dumps(report,indent=2));print('RESULT',results,flush=True)

def smoke():
    d=np.load(ROOT/'data.npz')
    # Baseline reproduces original data/model, isolating each change.
    old=np.load('results/parallel_2d_zone_cnn_20261017/features_1.npz');new=np.load(ROOT/'control.npz')
    np.testing.assert_array_equal(old['x'],new['x'])
    from joint_zone_cnn import JointCNN
    cfg=next(c for c in configs() if c['arm']=='control');oldnet=JointCNN(new['x'].shape[1],cfg).cuda();net=Small2D(cfg).cuda()
    nn.init.normal_(oldnet.net.output.weight,std=.01)  # Make parity nontrivial despite zero-initialized output.
    net.conv.load_state_dict(oldnet.net.conv.state_dict());net.fusion.load_state_dict(oldnet.net.fusion.state_dict());net.output.load_state_dict(oldnet.net.output.state_dict())
    xx=torch.tensor(new['x'][302:303],device='cuda')
    with torch.no_grad():np.testing.assert_array_equal(oldnet(xx).cpu().numpy(),net(xx).cpu().numpy())
    for arm in ARMS:
        cfg=next(c for c in configs() if c['arm']==arm);f=np.load(ROOT/f'{arm}.npz');net=Small2D(cfg).cuda()
        nn.init.normal_(net.output.weight,std=.01);xx=torch.tensor(f['x'][:2],device='cuda',requires_grad=True)
        out=net(xx);out[:,0,:,0].sum().backward();assert xx.grad[:,31].abs().sum()>0
        assert out.mean(-1).abs().max()<1e-6
        raw=make_features(d['dates'][333:334],d['h'][333:334],[],[],arm)
        ch,aux=dimensions(cfg);grid=raw[:,:-aux].reshape(1,ch,16,96)
        assert not grid[:,:4,:14].any() and not grid[:,4,:14].any()
        if cfg['residual']:assert not grid[:,5:].any()
    pool([next(c for c in configs() if c['arm']==arm) for arm in ARMS],5,False)
    (ROOT/'smoke.json').write_text(json.dumps(dict(passed=True,control_parity=True,causality=True,missing_masks=True,cross_zone_gradient=True),indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    elif a.smoke:smoke()
    else:finalize(pool(configs(),600,True))

if __name__=='__main__':main()
