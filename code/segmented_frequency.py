"""Eight history-predicted coefficients with constant or seasonal segment length."""
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
import datetime as dt
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from frequency_experiment import project, oracle_l1, score, make_bases

ROOT=Path('results/segmented_frequency_20261017')
WIDTH=.5  # sigmoid transition scale in hours, fixed before any evaluation


def seasonal_coordinates(dates):
    doy=np.array([dt.date.fromisoformat(str(d)).timetuple().tm_yday-1 for d in dates])
    return np.stack([np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25)],-1)


def numpy_basis(boundaries, labels):
    """Independent double-precision segment-time Fourier construction."""
    t=(np.arange(96)+.5)/4
    b=np.asarray(boundaries)[:,None]
    gate=1/(1+np.exp((t-b)/WIDTH))
    rows=[]
    for side in range(2):
        u=t/b if side==0 else (t-b)/(24-b)
        g=gate if side==0 else 1-gate
        for k,phase in labels[side]:
            v=project(g*getattr(np,phase)(2*np.pi*k*u))
            rows.append(v/np.linalg.norm(v,axis=-1,keepdims=True).clip(1e-12))
    return np.stack(rows,axis=1)


def select_modes(residual):
    """Training-only conditional SSE selection, four distinct modes per side."""
    candidates=[(k,p) for k in range(1,13) for p in ['sin','cos']]
    labels=[]
    for side in range(2):
        picked=[]; q=[]; used=set()
        for _ in range(4):
            options=[]
            for k,p in candidates:
                if k in used: continue
                # Same candidate on both sides just to reuse the public constructor.
                v=numpy_basis([12], [[(k,p)],[(k,p)]])[0,side].copy()
                if q:
                    qq=np.stack(q); v-=(v@qq.T)@qq
                norm=np.linalg.norm(v)
                if norm>1e-8:
                    v/=norm; options.append((float(np.square(residual@v).sum()),k,p,v))
            _,k,p,v=max(options,key=lambda row:row[0])
            picked.append((k,p));q.append(v);used.add(k)
        labels.append(picked)
    return labels


class Segmented(nn.Module):
    def __init__(self,nin,cfg,labels,global_basis):
        super().__init__();self.cfg=cfg;self.labels=labels
        self.register_buffer('time',(torch.arange(96,dtype=torch.float32)+.5)/4)
        self.register_buffer('global_basis',torch.tensor(global_basis,dtype=torch.float32))
        family=cfg['family']
        self.net=(nn.Linear(nin,8) if family=='linear' else nn.Sequential(nn.Linear(nin,64),nn.SiLU(),nn.Linear(64,8)))
        last=self.net if family=='linear' else self.net[-1]
        nn.init.zeros_(last.weight);nn.init.zeros_(last.bias)
        # b0 is restricted to 9..15h; seasonal amplitude cannot leave 8..16h.
        initial=cfg['initial_hour']
        raw0=np.arctanh(np.clip((initial-12)/3,-.999,.999))
        self.boundary=nn.Parameter(torch.tensor([raw0,0.,0.],dtype=torch.float32),requires_grad=cfg['mode'] in ['constant','seasonal'])

    def boundary_parameters(self):
        if self.cfg['mode'] in ['noon','global8']:
            z=self.boundary[0]*0
            return 12+z,z,z
        b0=12+3*torch.tanh(self.boundary[0])
        if self.cfg['mode']=='constant':
            z=b0*0;return b0,z,z
        v=self.boundary[1:];norm=torch.sqrt(v.square().sum()+1e-12)
        margin=4-torch.abs(b0-12)
        cs=v*(margin*torch.tanh(norm)/norm)
        return b0,cs[0],cs[1]

    def basis(self,calendar):
        b0,bs,bc=self.boundary_parameters()
        b=b0+bs*calendar[:,0]+bc*calendar[:,1]
        if self.cfg['mode']=='global8':
            return self.global_basis[None].expand(len(calendar),-1,-1),b
        t=self.time[None]; bb=b[:,None]; gate=torch.sigmoid((bb-t)/WIDTH)
        rows=[]
        for side in range(2):
            u=t/bb if side==0 else (t-bb)/(24-bb)
            g=gate if side==0 else 1-gate
            for k,phase in self.labels[side]:
                v=g*getattr(torch,phase)(2*np.pi*k*u)
                q=v.reshape(-1,24,4);v=(q-q.mean(-1,keepdim=True)).reshape(-1,96)
                rows.append(v/v.norm(dim=-1,keepdim=True).clamp_min(1e-12))
        return torch.stack(rows,dim=1),b

    def forward(self,x,calendar):
        basis,b=self.basis(calendar); coef=self.net(x)
        return torch.einsum('dzk,dkt->dzt',coef,basis),coef,basis,b


def train(cfg,x,r,cal,labels,global_basis,epochs):
    torch.manual_seed(cfg['seed']);net=Segmented(x.shape[-1],cfg,labels,global_basis).cuda()
    groups=[dict(params=net.net.parameters(),lr=cfg['lr'])]
    if net.boundary.requires_grad:groups.append(dict(params=[net.boundary],lr=.02,weight_decay=0.))
    opt=torch.optim.AdamW(groups,weight_decay=.01)
    best=float('inf');best_epoch=0;best_state=None;boundary_gradient=0.
    for epoch in range(epochs):
        net.train()
        for ids in torch.randperm(302,device='cuda').split(32):
            pred,_,_,_=net(x[ids],cal[ids]);loss=(pred-r[ids]).abs().mean()
            opt.zero_grad();loss.backward()
            if net.boundary.grad is not None:boundary_gradient=max(boundary_gradient,float(net.boundary.grad.abs().max()))
            nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
        net.eval()
        with torch.no_grad(): val=float((net(x[302:333],cal[302:333])[0]-r[302:333]).abs().mean())*20
        assert np.isfinite(val)
        if val<best:
            best=val;best_epoch=epoch+1;best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        if epoch+1-best_epoch>=30:break
    assert best_state is not None
    if net.boundary.requires_grad: assert boundary_gradient>0
    net.load_state_dict(best_state);net.eval()
    with torch.no_grad():
        _,coef,_,boundary=net(x,cal)
        coef=coef.cpu().numpy().astype('float64')*20
        boundary=boundary.cpu().numpy().astype('float64')
        pars=[float(v) for v in net.boundary_parameters()]
    # Deployment reconstruction uses double precision and explicit conservation.
    b=pars[0]+pars[1]*cal.cpu().numpy()[:,0]+pars[2]*cal.cpu().numpy()[:,1]
    # Use double-precision calendar below for saved/replayed inference.
    meta=dict(config=cfg,validation_mae=best,best_epoch=best_epoch,epochs_run=epoch+1,
              predictor_parameters=sum(p.numel() for p in net.net.parameters()),
              shared_boundary_parameters=3 if cfg['mode']=='seasonal' else 1 if cfg['mode']=='constant' else 0,
              b0=pars[0],bs=pars[1],bc=pars[2],boundary_gradient_max=boundary_gradient)
    return meta,best_state,coef


def main():
    from reconstruction import read_targets,baseline,rich_features
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=ROOT/'smoke' if a.smoke else ROOT;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    source=Path('train_prices_15min.csv');sha=hashlib.sha256(source.read_bytes()).hexdigest()
    assert sha=='f71ae6dbcca5f04e76e35840e29d064e69ad292f26d8ec7bf013bff2d904f65c'
    dates,y=read_targets(source);h=y.mean(-1);anchor=baseline(h);r=(y-anchor).reshape(363,4,96)
    assert dates[302]=='2026-08-01' and dates[333]=='2026-09-01'
    f=rich_features(dates,h,dates,y).reshape(363,4,-1)
    changed=y.copy();changed[333:]+=np.arange(4)*1234
    assert np.array_equal(f[333:334],rich_features(dates[333:334],h[333:334],dates,changed).reshape(1,4,-1))
    mu=f[:302].mean((0,1));sd=f[:302].std((0,1)).clip(.05)
    x=np.concatenate([np.clip((f-mu)/sd,-15,15),np.broadcast_to(np.eye(4),(363,4,4))],-1).astype('float32')
    cal=seasonal_coordinates(dates);labels=select_modes(r[:302].reshape(-1,96))
    global_basis=make_bases(r[:302].reshape(-1,96))[0]['fourier4_freephase']
    assert global_basis.shape==(8,96)
    np.savez(out/'normalization.npz',mu=mu,sd=sd,global_basis=global_basis)
    manifest=dict(source_sha256=sha,train_days=302,validation_days=31,test_days=30,labels=labels,
                  split='training through July; August selection; September developmental evaluation only',
                  daily_outputs=8,segment_time='quarter midpoint divided by segment length',
                  transition_scale_hours=WIDTH,boundary_range_hours=[8,16],mode_selection='train-only greedy SSE at noon; frozen for all segment variants',
                  seasonal='b0+bs*sin(2*pi*doy/365.25)+bc*cos(2*pi*doy/365.25); shared across zones',
                  optimization='joint fit on training MAE, August stopping/selection; no September gradients',
                  caveat='one annual cycle; seasonal extrapolation not established; smooth overlap rather than strictly disjoint segments')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    xt=torch.tensor(x,device='cuda');rt=torch.tensor(r/20,dtype=torch.float32,device='cuda');ct=torch.tensor(cal,dtype=torch.float32,device='cuda')
    all_results=[];selected={}
    for mode in ['global8','noon','constant','seasonal']:
        settings=([('mlp',.001,17,12)] if a.smoke else itertools.product(['linear','mlp'],[.001,.003],[17,43],[12] if mode in ['global8','noon'] else [10,12,14]))
        winner=None
        for family,lr,seed,initial in settings:
            cfg=dict(mode=mode,family=family,lr=lr,seed=seed,initial_hour=initial)
            meta,state,coef=train(cfg,xt,rt,ct,labels,global_basis,3 if a.smoke else 300)
            b=meta['b0']+meta['bs']*cal[:,0]+meta['bc']*cal[:,1]
            assert b.min()>=8-1e-6 and b.max()<=16+1e-6
            basis=np.broadcast_to(global_basis,(363,8,96)).copy() if mode=='global8' else numpy_basis(b,labels)
            assert np.min(np.linalg.svd(basis,compute_uv=False))>1e-6
            pred=anchor.reshape(363,4,96)+project(np.einsum('dzk,dkt->dzt',coef,basis))
            meta['validation_mae']=score(pred[302:333].reshape(-1,96),y[302:333].reshape(-1,96),h[302:333])['mae']
            all_results.append(meta)
            if winner is None or meta['validation_mae']<winner[0]['validation_mae']:
                winner=(meta,state,coef,b,basis,pred)
            print('TRIAL',meta,flush=True)
            (out/'progress.json').write_text(json.dumps(dict(completed=len(all_results),latest=meta),indent=2))
        meta,state,coef,b,basis,pred=winner
        folder=out/mode;folder.mkdir(exist_ok=True)
        torch.save(dict(state=state,meta=meta,labels=labels),folder/'model.pt')
        np.savez(folder/'outputs.npz',predictions=pred[302:],coefficients=coef[302:],basis=basis[302:],boundaries=b,calendar=cal)
        replay=Segmented(x.shape[-1],meta['config'],labels,global_basis).cuda().eval()
        replay.load_state_dict(torch.load(folder/'model.pt',weights_only=False)['state'])
        with torch.no_grad(): rr=replay(xt,ct)[1].cpu().numpy().astype('float64')*20
        assert np.array_equal(rr,coef)
        # Check actual gradient against finite differences on a train-only synthetic
        # weighted reconstruction, including seasonal stretch and overlap.
        if mode=='seasonal':
            check=Segmented(x.shape[-1],meta['config'],labels,global_basis).double().cpu()
            check.load_state_dict(state)
            cc=torch.tensor(cal[:4],dtype=torch.float64);weights=torch.linspace(-1,1,96,dtype=torch.float64)
            value=(check.basis(cc)[0]*weights).sum();value.backward()
            analytical=check.boundary.grad.detach().numpy().copy(); numerical=[]
            for j in range(3):
                with torch.no_grad():check.boundary[j]+=1e-5
                plus=float((check.basis(cc)[0]*weights).sum().detach())
                with torch.no_grad():check.boundary[j]-=2e-5
                minus=float((check.basis(cc)[0]*weights).sum().detach())
                with torch.no_grad():check.boundary[j]+=1e-5
                numerical.append((plus-minus)/2e-5)
            assert np.allclose(analytical,numerical,rtol=1e-3,atol=1e-4),(analytical,numerical)
        selected[mode]=meta
    (out/'selection.json').write_text(json.dumps(selected,indent=2))
    (out/'trials.json').write_text(json.dumps(all_results,indent=2))
    if a.smoke:
        (out/'passed.json').write_text(json.dumps({'passed':True},indent=2));print('SMOKE PASSED',flush=True);return
    metrics={}
    # Only now calculate September predictor scores and target-derived capacity.
    for mode,meta in selected.items():
        d=np.load(out/mode/'outputs.npz');pred=d['predictions'];basis=d['basis']
        oracle=np.stack([oracle_l1(row,bb) for row,bb in zip(r[302:],basis)])
        op=anchor[302:].reshape(61,4,96)+oracle
        np.save(out/mode/'oracle.npy',op)
        metrics[mode]=dict(selected=meta)
        for sl,key,hh,truth in [(slice(0,31),'august',h[302:333],y[302:333]),(slice(31,None),'september',h[333:],y[333:])]:
            metrics[mode][key]=score(pred[sl].reshape(-1,96),truth.reshape(-1,96),hh)
            metrics[mode]['oracle_'+key]=score(op[sl].reshape(-1,96),truth.reshape(-1,96),hh)
            assert metrics[mode]['oracle_'+key]['mae']<=metrics[mode][key]['mae']+1e-6
    metrics['interpolation']={key:score(anchor[sl].reshape(-1,96),y[sl].reshape(-1,96),h[sl]) for sl,key in [(slice(302,333),'august'),(slice(333,None),'september')]}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    lines=['# Two-segment Fourier reconstruction — 2026-10-17','',
           'Eight predicted coefficients per day/zone. Training through July; August selection; September developmental.',
           'Shared seasonal boundary fitted jointly on training MAE; three seasonal parameters beyond daily coefficients.',
           'Smooth overlap scale 0.5h; segment-relative time stretches basis functions; exact hourly-mean projection.',
           '', '| Model | August MAE | September MAE | September oracle MAE | b0 (h) | sin (h) | cos (h) |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for mode,m in metrics.items():
        v=m.get('selected',{})
        lines.append(f"| {mode} | {m['august']['mae']:.6f} | {m['september']['mae']:.6f} | {m.get('oracle_september',{}).get('mae',float('nan')):.6f} | {v.get('b0',float('nan')):.3f} | {v.get('bs',0):.3f} | {v.get('bc',0):.3f} |")
    lines+=['','Modes per segment: '+json.dumps(labels),'Oracle weights use true targets solely to diagnose capacity. No deployment or refit.',
            'One annual cycle and August-only stopping limit seasonal evidence. Sine shape is a hypothesis, not an established physical boundary.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('DONE',json.dumps({k:v['september']['mae'] for k,v in metrics.items()}),flush=True)


if __name__=='__main__':main()
