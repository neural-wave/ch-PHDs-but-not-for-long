"""Independent normalization equations, artifact replay and evaluation audit."""
import argparse,hashlib,json,math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import train as t
import numpy as np
import pandas as pd
import torch


def reference_inputs(raw,nt,mode):
    q=raw['q'].copy();r=raw['r'].copy();h=raw['h']
    scale=np.full(h.shape[:2],20.)
    if mode.startswith('day_'):
        mean=np.mean(h,axis=2);s=np.sqrt(np.mean((h-mean[...,None])**2,axis=2)+1)
        for zone in range(4):
            for feature in range(27):
                channel=zone*27+feature
                if feature==0:q[:,:,channel]=(q[:,:,channel]*100-mean[:,zone,None])/s[:,zone,None]
                elif feature in [1,2,5,6,7,8] or feature>=18:q[:,:,channel]=q[:,:,channel]*20/s[:,zone,None]
            for feature in range(10):
                channel=zone*10+feature
                if feature<8:r[:,:,channel]=r[:,:,channel]*20/s[:,zone,None]
                elif feature==9:r[:,:,channel]=(r[:,:,channel]*100-mean[:,zone,None])/s[:,zone,None]
                r[:,:,channel]*=raw['r'][:,:,zone*10+8]
        if mode=='day_reversible':scale=s
    values={}
    for key,array in [('q',q),('r',r)]:
        mean=np.zeros((1,1,array.shape[-1])) if mode in ['fixed','day_reversible'] else np.mean(array[:nt],axis=(0,1),keepdims=True)
        sd=np.ones_like(mean) if mode in ['fixed','day_reversible'] else np.maximum(np.std(array[:nt],axis=(0,1),keepdims=True),.05)
        x=np.clip((array-mean)/sd,-15,15).astype('float32')
        if key=='r' and mode not in ['fixed','day_reversible']:
            for zone in range(4):
                mask=array[:,:,10*zone+8]
                x[:,:,10*zone:10*zone+10]*=mask[:,:,None];x[:,:,10*zone+8]=mask
        values.update({key:x,'mu'+key:mean,'sd'+key:sd})
    values['scale']=scale
    return values


def edge_checks():
    # Include flat negative prices, a varied negative curve and unequal zone scales.
    h=np.stack([np.full((4,24),-50.),np.stack([-100+np.arange(24)*v for v in [0,1,3,10]])])
    y=np.repeat(h[...,None],4,-1);dates=np.array(['2025-10-01','2025-10-02'])
    q,r=t.old.features(dates,h,dates,y);raw=dict(q=q,r=r,h=h)
    for mode in t.MODES:
        actual=t.transform(q,r,h,1,mode);expected=reference_inputs(raw,1,mode)
        for key in actual:np.testing.assert_allclose(actual[key],expected[key],atol=2e-6,rtol=1e-6)
        assert all(np.isfinite(v).all() for v in actual.values())
    cfg=t.configs()[4]  # first learned-affine reversible candidate
    assert cfg['mode']=='day_reversible'
    t.setup(17)
    net=t.QuantileSSM(cfg,108,43)
    target=np.zeros((2,4,24,4));net.initialize(target)
    with torch.no_grad():
        net.affine_log_scale.copy_(torch.tensor([.2,-.3,.4,-.1]))
        net.affine_bias.copy_(torch.tensor([.1,-.2,.3,.4]))
    v=t.transform(q,r,h,1,'day_reversible')
    query=torch.tensor(v['q']);hist=torch.tensor(v['r'])
    aq,ar,gamma=net.affine_inputs(query,hist)
    shaped=query.reshape(2,24,4,27)
    torch.testing.assert_close(aq.reshape(2,24,4,27)[...,0],shaped[...,0]*gamma[None,None,:]+net.affine_bias[None,None,:])
    torch.testing.assert_close(aq.reshape(2,24,4,27)[...,3:5],shaped[...,3:5])
    net.eval();actual=net(query,hist)
    ref=t.old.QuantileSSM.forward(net,aq,ar)/gamma[None,:,None,None,None]
    torch.testing.assert_close(actual,ref)
    torch.testing.assert_close(actual[:1],net(query[:1],hist[:1]))
    assert torch.isfinite(actual).all() and torch.diff(actual,dim=-1).min()>=0
    actual.square().mean().backward()
    assert net.affine_log_scale.grad is not None and torch.isfinite(net.affine_log_scale.grad).all()
    assert net.affine_bias.grad is not None and torch.isfinite(net.affine_bias.grad).all()
    # Output reversal must not change the original-currency objective's weighting.
    s=torch.tensor([1.,20.,100.]).reshape(3,1,1,1,1)
    pred=torch.linspace(-2,2,33).reshape(3,1,1,1,11).requires_grad_()
    truth=torch.tensor([3.,-7.,17.]).reshape(3,1,1,1,1)
    tau=torch.tensor(t.TAUS,dtype=torch.float32)
    err1=truth/20-pred*s/20;err2=truth-pred*s
    loss1=err1[...,5].abs().mean()+torch.maximum(tau*err1,(tau-1)*err1).mean()
    loss2=(err2[...,5].abs().mean()+torch.maximum(tau*err2,(tau-1)*err2).mean())/20
    torch.testing.assert_close(loss1,loss2);loss1.backward();assert torch.isfinite(pred.grad).all()


def main(smoke=False):
    t.setup(17);edge_checks();root=t.ROOT/('smoke' if smoke else 'folds')
    raw=np.load(t.ROOT/'raw.npz');manifest=json.loads((t.ROOT/'manifest.json').read_text())
    assert hashlib.sha256((t.PROJECT/'train_prices_15min.csv').read_bytes()).hexdigest()==t.SHA
    for path,sha in manifest['code_hashes'].items():assert hashlib.sha256((t.PROJECT/path).read_bytes()).hexdigest()==sha,path
    jobs=t.folds(raw['dates']);jobs=[jobs[0],jobs[-1]] if smoke else jobs
    ratios={};pooled={};counts={};total=0;candidate_rows=[];exports=[];max_replay=0.
    for fold in jobs:
        folder=root/fold['test'];d=np.load(folder/'data.npz');nt,ns,nc=(int(d[k]) for k in ['nt','ns','nc']);n=len(d['y'])
        assert n==nt+ns+nc+fold['test_days']
        dates=d['dates'];assert all(x[:7]<fold['validation'] for x in dates[:nt])
        assert all(x[:7]==fold['validation'] and int(x[8:10])<=15 for x in dates[nt:nt+ns])
        assert all(x[:7]==fold['validation'] and int(x[8:10])>=16 for x in dates[nt+ns:nt+ns+nc])
        assert all(x[:7]==fold['test'] for x in dates[nt+ns+nc:])
        for key in ['y','h','b','dates']:np.testing.assert_array_equal(d[key],raw[key][:n])
        partial={k:raw[k][:n] for k in ['q','r','h']}
        for mode in t.MODES:
            ref=reference_inputs(partial,nt,mode)
            for key,value in ref.items():np.testing.assert_allclose(d[mode+'_'+key],value,atol=2e-6,rtol=1e-6)
            changed={k:v.copy() for k,v in partial.items()}
            for k in changed:changed[k][nt:]+=11111
            other=t.transform(changed['q'],changed['r'],changed['h'],nt,mode)
            for key in ['muq','sdq','mur','sdr']:np.testing.assert_allclose(other[key],ref[key],atol=1e-10,rtol=1e-10)
            for key in ['q','r','scale']:np.testing.assert_allclose(other[key][:nt],ref[key][:nt],atol=2e-6,rtol=1e-6)
        s=json.loads((folder/'selection.json').read_text());m=json.loads((folder/'metrics.json').read_text())
        expected=[c for c in t.configs() if not smoke or (c['seed']==17 and c['lr']==.001)]
        assert sorted([r['config'] for r in s['candidates']],key=lambda r:r['id'])==expected
        truth=d['y'][nt+ns+nc:];base=d['b'][nt+ns+nc:];den=float(np.abs(base-truth).mean());assert den>0
        for row in s['candidates']:
            cfg=row['config'];mode=cfg['mode'];dest=folder/'trials'/str(cfg['id'])
            ck=torch.load(dest/'model.pt',weights_only=False,map_location='cpu');assert ck['config']==cfg
            net=t.QuantileSSM(cfg,d[mode+'_q'].shape[-1],d[mode+'_r'].shape[-1]).cuda().eval();net.load_state_dict(ck['state']);out=[]
            with torch.no_grad():
                for start in range(nt,n,32):
                    out.append(net(torch.tensor(d[mode+'_q'][start:start+32],device='cuda'),torch.tensor(d[mode+'_r'][start:start+32],device='cuda')).cpu().numpy())
            normalized=np.concatenate(out).astype('float64')
            normalized-=normalized[...,5].mean(-1,keepdims=True)[...,None]
            replay=d['b'][nt:,...,None]+normalized*d[mode+'_scale'][nt:,:,None,None,None]
            saved=np.load(dest/'quantiles.npy');delta=float(np.abs(replay-saved).max());assert delta<1e-5;max_replay=max(max_replay,delta)
            assert np.isfinite(saved).all() and np.diff(saved,axis=-1).min()>=-1e-8
            assert np.abs(saved[...,5].mean(-1)-d['h'][nt:]).max()<1e-8
            val=float(np.abs(saved[:ns,...,5]-d['y'][nt:nt+ns]).mean());assert abs(val-row['validation_mae'])<1e-8
            assert abs(val/float(d['selection_interpolation_mae'])-row['selection_ratio'])<1e-10
            curve=np.loadtxt(dest/'curves.csv',delimiter=',',skiprows=1);assert int(curve[curve[:,1].argmin(),0])==row['best_epoch']
            assert abs(curve[:,1].min()-row['best_curve_ratio'])<1e-10
            parameters=sum(p.numel() for p in net.parameters());assert parameters==row['parameters']
            assert cfg['family'] not in counts or counts[cfg['family']]==parameters;counts[cfg['family']]=parameters
            mae=float(np.abs(saved[ns+nc:,...,5]-truth).mean())
            candidate_rows.append(dict(month=fold['test'],family=cfg['family'],mode=mode,lr=cfg['lr'],seed=cfg['seed'],mae=mae,ratio=mae/den));total+=1
        for recipe in t.RECIPES:
            expected=min([r for r in s['candidates'] if r['config']['family']+'_'+r['config']['mode']==recipe],key=lambda r:(r['selection_ratio'],r['config']['id']))
            assert s['by_recipe'][recipe]==expected
        winner=min(t.RECIPES,key=lambda k:(s['by_recipe'][k]['selection_ratio'],k))
        assert s['by_recipe']['validation_selector']['chosen_recipe']==winner
        for recipe,row in s['by_recipe'].items():
            q=np.load(folder/'trials'/str(row['config']['id'])/'quantiles.npy');cal=q[ns:ns+nc];test=q[ns+nc:]
            saved=np.load(folder/'selected'/recipe/'predictions.npz');np.testing.assert_array_equal(saved['quantiles'],test)
            np.testing.assert_array_equal(saved['observed'],truth);np.testing.assert_array_equal(saved['dates'],dates[nt+ns+nc:])
            stats=m['results'][recipe];error=np.abs(test[...,5]-truth);mae=float(error.mean());ratio=mae/den
            assert abs(mae-stats['mae'])<1e-10 and abs(ratio-stats['ratio'])<1e-10
            np.testing.assert_allclose(error.mean((0,2,3)),stats['by_zone'],rtol=0,atol=1e-10)
            np.testing.assert_allclose(error.mean((0,2,3))/np.abs(base-truth).mean((0,2,3)),stats['zone_ratio'],rtol=0,atol=1e-10)
            pin=truth[...,None]-test;pb=np.maximum(t.TAUS*pin,(t.TAUS-1)*pin).mean((0,1,2,3))
            np.testing.assert_allclose(pb,stats['pinball_by_quantile'],rtol=0,atol=1e-10)
            previous_lo=previous_hi=None
            for coverage,lo,hi in t.evaluation.INTERVALS:
                level=round(coverage*100);scores=np.maximum.reduce([cal[...,lo]-d['y'][nt+ns:nt+ns+nc],d['y'][nt+ns:nt+ns+nc]-cal[...,hi],np.zeros_like(cal[...,lo])])
                group=scores.transpose(1,3,0,2).reshape(4,4,-1);rank=math.ceil((group.shape[-1]+1)*coverage);corr=np.sort(group,axis=-1)[...,rank-1][None,:,None,:]
                lower=test[...,lo]-corr;upper=test[...,hi]+corr
                if previous_lo is not None:lower=np.minimum(lower,previous_lo);upper=np.maximum(upper,previous_hi)
                np.testing.assert_allclose(lower,saved[f'lower_{level}'],rtol=0,atol=1e-10);np.testing.assert_allclose(upper,saved[f'upper_{level}'],rtol=0,atol=1e-10)
                reported=next(a for a in stats['intervals'] if a['kind']=='calibrated' and a['nominal']==coverage)
                assert abs(float(((truth>=lower)&(truth<=upper)).mean())-reported['coverage'])<1e-12
                assert abs(float((upper-lower).mean())-reported['width'])<1e-10
                previous_lo,previous_hi=lower,upper
            ratios.setdefault(recipe,[]).append(ratio);pooled.setdefault(recipe,[]).append(error.ravel())
        export=pd.read_csv(folder/'predictions.csv.gz');selector=np.load(folder/'selected/validation_selector/predictions.npz')
        for k,tau in enumerate(t.TAUS):np.testing.assert_allclose(export[f'q{tau:g}'],selector['quantiles'][...,k].ravel(),rtol=0,atol=1e-7)
        exports.append(export)
    out=t.ROOT/'smoke' if smoke else t.ROOT;summary=json.loads((out/'metrics.json').read_text())
    for row in summary['summary']:
        assert abs(row['mean_monthly_ratio']-np.mean(ratios[row['recipe']]))<1e-10
        assert abs(row['pooled_mae']-np.concatenate(pooled[row['recipe']]).mean())<1e-10
    assert total==(4 if smoke else 64)
    actual=pd.DataFrame(candidate_rows).sort_values(['month','family','mode','lr','seed']).reset_index(drop=True)
    reported=pd.read_csv(out/'all_candidate_scores.csv').sort_values(['month','family','mode','lr','seed']).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual,reported,check_exact=False,atol=1e-10,rtol=0)
    if not smoke:
        export=pd.read_csv(t.ROOT/'predictions.csv.gz');assert len(export)==92544 and not export.duplicated(['delivery_date','market_id','hour_index','quarter_index']).any()
        pd.testing.assert_frame_equal(export,pd.concat(exports,ignore_index=True),check_exact=False,atol=1e-8,rtol=0)
        pairs=actual.pivot(index=['month','family','lr','seed'],columns='mode',values='ratio')
        for mode in t.MODES:
            pairs[mode+'_minus_fixed']=pairs[mode]-pairs['fixed']
        pairs.to_csv(out/'paired_candidate_ratios.csv')
        selected=pd.read_csv(out/'monthly_scores.csv');selected=selected[selected.recipe.isin(t.RECIPES)].copy()
        selected['family']=selected.recipe.map(lambda v:next(f for f in t.FAMILIES if v.startswith(f+'_')))
        selected['mode']=selected.apply(lambda v:v.recipe[len(v.family)+1:],axis=1)
        sel=selected.pivot(index=['month','family'],columns='mode',values='ratio')
        for mode in t.MODES:sel[mode+'_minus_fixed']=sel[mode]-sel['fixed']
        sel.to_csv(out/'paired_selected_ratios.csv')
        lines=['# RevIN quantile matched comparison','',
            'Eight developmental expanding folds. Compact CNN with fixed physical units versus input-day reversible normalization with learned positive affine parameters. Both arms fit from scratch with identical lr/seed budgets and original-currency loss. Lower mean monthly MAE/interpolation MAE is better.','',
            '| Recipe | Mean monthly ratio | Pooled MAE | Calibrated90 coverage | Width |',
            '|---|---:|---:|---:|---:|']
        for row in summary['summary']:
            lines.append(f"| {row['recipe']} | {row['mean_monthly_ratio']:.6f} | {row['pooled_mae']:.6f} | {row['coverage90']:.2%} | {row['width90']:.3f} |")
        lines+=['','This is RevIN adapted to constrained interpolation residuals, not a claim of exact published architecture replication. Learned affine input bias is not added to output deviations; interpolation restores the known hourly level. Scale remains positive and all quantiles use the same inverse affine. No deployment promotion. See UNCERTAINTY.md for original and normalized bands, interval coverage, paired April26 ZONE_3 and March14 ZONE_2 examples, and error associations.']
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    t.write(out/'audit.json',dict(passed=True,folds=len(jobs),checkpoints=total,max_replay_error=max_replay,parameters_by_family=counts,
        independent_normalization_equations=True,heldout_preprocessing_invariance=True,flat_negative_inputs=True,physical_loss_equivalence=True,
        chronology=True,all_candidate_replay=True,calibration_rank_recomputed=True,quantile_order_and_hourly_conservation=True,mean_monthly_ratio_recomputed=True))
    print('AUDIT PASSED',smoke,total,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.smoke)
