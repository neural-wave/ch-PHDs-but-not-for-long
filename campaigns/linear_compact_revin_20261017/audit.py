"""Independent replay, frozen-recipe, chronology, physical loss and CQR checks."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
import train as t

def close(a,b,tol=1e-8):np.testing.assert_allclose(a,b,rtol=0,atol=tol)

def edge_checks():
    t.e.p.setup();h=np.stack([np.full((4,24),-50.),np.stack([np.arange(24)*v-100 for v in [0,1,3,10]])]);y=np.repeat(h[...,None],4,-1);dates=np.array(['2025-10-01','2025-10-02'])
    outputs=[]
    for i in range(2):outputs.append(t.query_features(dates[i:i+1],h[i:i+1],dates[:i],y[:i]))
    data={k:np.clip(np.concatenate([f[k] for f in outputs]),-15,15).astype('float32') for k in ['rich','compact','cnn','curve']}
    data['compact']=np.concatenate([data['compact'],np.broadcast_to(np.eye(4)[None,:,None,:],(2,4,24,4))],-1).astype('float32')
    for k in ['rich','compact']:data[k+'_knots']=np.quantile(data[k].reshape(-1,data[k].shape[-1]),[.25,.5,.75],axis=0).astype('float32')
    for family in t.e.p.FAMILIES:
        cfg=dict(family=family);net=t.Quantiles(cfg,data);net.initialize(np.zeros((2,4,24,4)))
        with torch.no_grad():net.log_gamma.fill_(.3);net.beta.fill_(.2);net.head.weight.normal_(std=.001)
        net.eval();x=torch.tensor(data[net.feature]);curve=torch.tensor(data['curve']);ax,ac,g=net.affine(x,curve);actual=net(x,curve)
        ref=t.e.Quantiles.forward(net,ax,ac)/g;torch.testing.assert_close(actual,ref);torch.testing.assert_close(actual[:1],net(x[:1],curve[:1]))
        assert torch.isfinite(actual).all() and torch.diff(actual,dim=-1).min()>=0
        actual.square().mean().backward();assert net.log_gamma.grad is not None and torch.isfinite(net.log_gamma.grad);assert net.beta.grad is not None and torch.isfinite(net.beta.grad)
        scale=np.sqrt(h.var(-1)+1);phys=t.physical(actual.detach().numpy(),np.repeat(h[...,None],4,-1),scale);close(phys[...,5].mean(-1),h)
    # Original-currency objective is identical after reversing any scale.
    p=torch.linspace(-2,2,33).reshape(3,1,1,1,11).requires_grad_();scale=torch.tensor([1.,20.,100.]).reshape(3,1,1,1,1);truth=torch.tensor([3.,-7.,17.]).reshape(3,1,1,1,1);tau=torch.tensor(t.TAUS,dtype=torch.float32)
    a=truth/20-p*scale/20;b=truth-p*scale
    l1=a[...,5].abs().mean()+torch.maximum(tau*a,(tau-1)*a).mean();l2=(b[...,5].abs().mean()+torch.maximum(tau*b,(tau-1)*b).mean())/20;torch.testing.assert_close(l1,l2)
    print('EDGE CHECKS PASSED',flush=True)

def audit(smoke=False):
    t.e.p.setup();edge_checks();manifest=json.loads((t.ROOT/'manifest.json').read_text())
    for p,s in manifest['code_hashes'].items():assert t.sha(t.PROJECT/p)==s,p
    assert t.sha(t.ROOT/'raw.npz')==manifest['feature_source_sha256']
    raw=np.load(t.ROOT/'raw.npz');source=np.load(t.SOURCE/'raw.npz');reg=t.registry();folds=reg['folds'];folds=[folds[0],folds[-1]] if smoke else folds;count=0;ratios={};errors={};max_replay=0.;max_mean=0.
    close(raw['scale'],np.sqrt(source['h'].var(-1)+1))
    for k in ['dates','y','h','b']:np.testing.assert_array_equal(raw[k],source[k])
    # Recompute selected day-level normalized features independently of cache.
    for i in [0,91,150,302,333]:
        f=t.query_features(raw['dates'][i:i+1],source['h'][i:i+1],raw['dates'][:i].astype(object),source['y'][:i],{k:source[k][i:i+1] for k in ['rich','compact','cnn','curve','analog']})
        for k in f:close(f[k],raw[k][i:i+1],2e-6)
    for fold in folds:
        folder=t.ROOT/('smoke' if smoke else 'folds')/fold['test_month'];d=np.load(folder/'data.npz');nt,ns,nc,ne=(int(d[k]) for k in ['nt','ns','nc','ne']);start=nt+ns+nc;dates=d['dates'];truth=d['y'][start:];base=d['b'][start:];den=float(abs(truth-base).mean());assert ns==15
        assert all(str(x)[:7]<fold['selection_month'] for x in dates[:nt]);assert all(str(x)[:7]==fold['selection_month'] and int(str(x)[8:])<=15 for x in dates[nt:nt+ns]);assert all(str(x)[:7]==fold['selection_month'] and int(str(x)[8:])>15 for x in dates[nt+ns:start]);assert all(str(x)[:7]==fold['test_month'] for x in dates[start:])
        for k in ['rich','compact','cnn','curve']:
            expected=np.clip(raw[k][:len(dates)],-15,15).astype('float32')
            if k=='compact':expected=np.concatenate([expected,np.broadcast_to(np.eye(4)[None,:,None,:],(len(dates),4,24,4))],-1).astype('float32')
            np.testing.assert_array_equal(expected,d[k])
            if k in ['rich','compact']:np.testing.assert_array_equal(d[k+'_knots'],np.quantile(expected[:nt].reshape(-1,expected.shape[-1]),[.25,.5,.75],axis=0).astype('float32'))
        months=np.array([str(x)[:7] for x in dates[:nt]]);weights=np.zeros(nt)
        for month in set(months):
            ids=months==month;weights[ids]=1/(ids.sum()*abs(d['y'][:nt][ids]-d['b'][:nt][ids]).mean())
        weights/=weights.mean();close(weights,d['weights_ratio'],2e-7)
        sel=json.loads((folder/'selection.json').read_text());rows=sel['candidates'];byid={str(v['config']['id']):v for v in rows};pred={}
        leaves=fold['leaves']
        if smoke:
            seen=set();leaves=[v for v in leaves if not(v['config']['family'] in seen or seen.add(v['config']['family']))]
        assert set(byid)=={str(v['config']['id']) for v in leaves}
        for leaf in leaves:
            cid=str(leaf['config']['id']);row=byid[cid];assert row['config']==t.config(leaf);dest=folder/'trials'/cid;ck=torch.load(dest/'model.pt',weights_only=False,map_location='cpu');assert ck['config']==row['config'];net=t.Quantiles(row['config'],d).cuda().eval();net.load_state_dict(ck['state']);q=t.predict(net,d,nt);saved=np.load(dest/'quantiles.npy');delta=float(abs(q-saved).max());assert delta<1e-6;max_replay=max(max_replay,delta);pred[cid]=saved
            assert saved.shape==d['y'][nt:].shape+(11,) and np.isfinite(saved).all() and np.diff(saved,axis=-1).min()>=-1e-8
            violation=float(abs(saved[...,5].mean(-1)-d['h'][nt:]).max());assert violation<1e-8;max_mean=max(max_mean,violation)
            close(abs(saved[:ns,...,5]-d['y'][nt:nt+ns]).mean(),row['validation_mae']);curve=np.loadtxt(dest/'curves.csv',delimiter=',',skiprows=1);assert row['best_epoch']==int(curve[curve[:,1].argmin(),0]);close(curve[:,1].min(),row['selection_ratio'],2e-6);count+=1
        expected={str(v['config']['id']):v['weight']/sum(x['weight'] for x in leaves) for v in leaves};assert sel['frozen_weights']==expected
        close(sum(pred[k]*v for k,v in expected.items()),np.load(folder/'trials/linear_compact_revin/quantiles.npy'))
        original=t.original_predictions(folder.name);np.testing.assert_array_equal(original,np.load(folder/'trials/original_linear_compact/quantiles.npy'))
        report=json.loads((folder/'metrics.json').read_text())
        for recipe in sel['by_recipe']:
            vals=np.load(folder/'trials'/str(sel['by_recipe'][recipe]['config']['id'])/'quantiles.npy');cal=vals[ns:ns+nc];test=vals[ns+nc:];z=np.load(folder/'selected'/recipe/'predictions.npz');np.testing.assert_array_equal(test,z['quantiles']);np.testing.assert_array_equal(z['observed'],truth);np.testing.assert_array_equal(z['dates'],dates[start:]);err=abs(test[...,5]-truth);mae=float(err.mean());close(mae,report['results'][recipe]['mae']);close(mae/den,report['results'][recipe]['ratio']);close(err.mean((0,2,3)),report['results'][recipe]['by_zone'])
            pb=truth[...,None]-test;close(np.maximum(t.TAUS*pb,(t.TAUS-1)*pb).mean((0,1,2,3)),report['results'][recipe]['pinball_by_quantile']);prevlo=prevhi=None
            for nominal,lo,hi in t.e.INTERVALS:
                level=round(nominal*100);scores=np.maximum.reduce([cal[...,lo]-d['y'][nt+ns:start],d['y'][nt+ns:start]-cal[...,hi],np.zeros_like(cal[...,lo])]);v=scores.transpose(1,3,0,2).reshape(4,4,-1);rank=math.ceil((v.shape[-1]+1)*nominal);c=np.sort(v,axis=-1)[...,rank-1][None,:,None,:];lower=test[...,lo]-c;upper=test[...,hi]+c
                if prevlo is not None:lower=np.minimum(lower,prevlo);upper=np.maximum(upper,prevhi)
                close(lower,z[f'lower_{level}']);close(upper,z[f'upper_{level}']);metric=next(v for v in report['results'][recipe]['intervals'] if v['kind']=='calibrated' and v['nominal']==nominal);close(((truth>=lower)&(truth<=upper)).mean(),metric['coverage']);close((upper-lower).mean(),metric['width']);prevlo,prevhi=lower,upper
            ratios.setdefault(recipe,[]).append(mae/den);errors.setdefault(recipe,[]).append(err.ravel())
    assert count==(2 if smoke else 15)
    out=t.ROOT/'smoke' if smoke else t.ROOT;summary=json.loads((out/'metrics.json').read_text())
    for row in summary['summary']:close(np.mean(ratios[row['recipe']]),row['mean_monthly_ratio']);close(np.concatenate(errors[row['recipe']]).mean(),row['pooled_mae'])
    if not smoke:close(np.mean(ratios['original_linear_compact']),.7245525295146528)
    t.e.p.write(out/'audit.json',dict(passed=True,folds=len(folds),checkpoints=count,max_replay_error=max_replay,max_hourly_mean_error=max_mean,frozen_champion_weights=True,causality=True,original_linear_compact_identity=True,calibration_recomputed=True))
    print('AUDIT PASSED',count,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--edge-only',action='store_true');a=p.parse_args();edge_checks() if a.edge_only else audit(a.smoke)
