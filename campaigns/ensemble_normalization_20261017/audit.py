"""Independent source/control, preprocessing, checkpoint, blend and interval audit."""
import argparse,hashlib,itertools,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import train as t
import numpy as np
import pandas as pd
import torch


def close(a,b,tol=1e-8):np.testing.assert_allclose(a,b,atol=tol,rtol=0)


def audit(smoke=False):
    t.e.p.setup();root=t.ROOT/'smoke' if smoke else t.ROOT
    manifest=json.loads((t.ROOT/'manifest.json').read_text());raw=np.load(t.ROOT/'raw.npz')
    for path,sha in manifest['code_hashes'].items():assert t.sha(t.PROJECT/path)==sha,path
    assert t.sha(t.ROOT/'raw.npz')==manifest['feature_source_sha256']
    assert t.sha(t.PROJECT/'train_prices_15min.csv')==t.e.p.SHA
    jobs=manifest['folds'];jobs=[jobs[0],jobs[-1]] if smoke else jobs
    total=reused=0;maximum=0.;aggregate={};pooled={};paired=[];exports=[];parameters={}
    for fold in jobs:
        folder=(root if smoke else root/'folds')/fold['test'];d=np.load(folder/'data.npz');fd=np.load(folder/'fixed/data.npz')
        nt,ns,nc,ne=(int(d[k]) for k in ['nt','ns','nc','ne']);n=len(d['y'])
        assert n==nt+ns+nc+ne and ns==15
        dates=d['dates'];assert all(x[:7]<=fold['train_end'] for x in dates[:nt])
        assert all(x[:7]==fold['validation'] and int(x[8:10])<=15 for x in dates[nt:nt+ns])
        assert all(x[:7]==fold['validation'] and int(x[8:10])>15 for x in dates[nt+ns:nt+ns+nc])
        assert all(x[:7]==fold['test'] for x in dates[nt+ns+nc:])
        for k in ['dates','y','h','b']:
            np.testing.assert_array_equal(d[k],raw[k][:n]);np.testing.assert_array_equal(fd[k],d[k])
        for feature in ['rich','compact','cnn','curve']:
            values=raw[feature];axes=tuple(range(values.ndim-1)) if feature in ['rich','compact'] else (0,2) if feature=='cnn' else (0,)
            mu=np.mean(values[:nt],axis=axes,keepdims=True);sd=np.maximum(np.std(values[:nt],axis=axes,keepdims=True),.05)
            close(d[feature+'_mu'],mu);close(d[feature+'_sd'],sd)
            for mode,data in [('standard',d),('fixed',fd)]:
                expected=np.clip((values[:n]-mu)/sd if mode=='standard' else values[:n],-15,15).astype('float32')
                if feature=='compact':expected=np.concatenate([expected,np.broadcast_to(np.eye(4)[None,:,None,:],(n,4,24,4))],-1).astype('float32')
                np.testing.assert_array_equal(data[feature],expected)
                if mode=='fixed':close(data[feature+'_mu'],np.zeros_like(data[feature+'_mu']));close(data[feature+'_sd'],np.ones_like(data[feature+'_sd']))
                if feature in ['rich','compact']:np.testing.assert_array_equal(data[feature+'_knots'],np.quantile(expected[:nt].reshape(-1,expected.shape[-1]),[.25,.5,.75],axis=0).astype('float32'))
            np.testing.assert_array_equal(d['weights_uniform'],np.ones(nt));np.testing.assert_array_equal(fd['weights_uniform'],d['weights_uniform'])
        selection=json.loads((folder/'selection.json').read_text());rows=selection['candidates'];winners=selection['by_recipe']
        configs=[c for c in t.configs() if not smoke or (c['lr']==.001 and c['seed']==17)]
        assert sorted([r['config'] for r in rows],key=lambda c:c['id'])==sorted(configs,key=lambda c:c['id'])
        pred={};truth=d['y'][nt+ns+nc:];base=d['b'][nt+ns+nc:];den=float(np.abs(base-truth).mean())
        for row in rows:
            cfg=row['config'];data=fd if cfg['mode']=='fixed' else d;dest=folder/'trials'/str(cfg['id'])
            ck=torch.load(dest/'model.pt',weights_only=False,map_location='cpu');assert ck['config']==cfg
            net=t.e.Quantiles(cfg,data).cuda().eval();net.load_state_dict(ck['state'])
            with torch.no_grad():
                q=t.e.physical(net(torch.as_tensor(data[net.feature][nt:],device='cuda'),torch.as_tensor(data['curve'][nt:],device='cuda')).cpu().numpy(),data['b'][nt:])
            saved=np.load(dest/'quantiles.npy');delta=float(np.abs(q-saved).max());assert delta<1e-6;maximum=max(maximum,delta)
            assert np.isfinite(saved).all() and np.diff(saved,axis=-1).min()>=-1e-9
            assert np.abs(saved[...,5].mean(-1)-data['h'][nt:]).max()<1e-8
            val=float(np.abs(saved[:ns,...,5]-d['y'][nt:nt+ns]).mean());close(val,row['validation_mae']);close(val/float(d['selection_interpolation_mae']),row['selection_ratio'])
            count=sum(p.numel() for p in net.parameters());assert count==row['parameters']
            assert cfg['family'] not in parameters or count==parameters[cfg['family']];parameters[cfg['family']]=count
            if not smoke and cfg['mode']=='standard':
                source=t.SOURCE/'folds'/fold['test']/'trials'/str(cfg['reference_id']);ref=json.loads((dest/'reference.json').read_text())
                assert t.sha(source/'model.pt')==ref['model_sha256'] and t.sha(source/'quantiles.npy')==ref['prediction_sha256']
                np.testing.assert_array_equal(saved,np.load(source/'quantiles.npy'))
                original=torch.load(source/'model.pt',weights_only=False,map_location='cpu')
                for k in ck['state']:torch.testing.assert_close(ck['state'][k],original['state'][k],atol=0,rtol=0)
                assert row['reused_control'];reused+=1
            pred[str(cfg['id'])]=saved;mae=float(np.abs(saved[ns+nc:,...,5]-truth).mean())
            paired.append(dict(month=fold['test'],family=cfg['family'],mode=cfg['mode'],lr=cfg['lr'],seed=cfg['seed'],mae=mae,ratio=mae/den));total+=1
        for family,mode in itertools.product(t.FAMILIES,t.MODES):
            expected=min([r for r in rows if r['config']['family']==family and r['config']['mode']==mode],key=lambda r:(r['selection_ratio'],r['config']['id']))
            assert winners[family+'__'+mode]==expected
        for mode in ['standard','fixed','mixed']:
            name='blend__'+mode;row=winners[name]
            if mode=='mixed':
                names=[min([family+'__'+m for m in t.MODES],key=lambda k:(winners[k]['selection_ratio'],k)) for family in t.FAMILIES]
            else:names=[family+'__'+mode for family in t.FAMILIES]
            primary=next(k for k in names if winners[k]['config']['family']=='linear_rich')
            choices=[{k:1.} for k in sorted(names)]+[{primary:w,other:1-w} for other in sorted(names) if other!=primary for w in [.25,.5,.75]]
            top=sorted(names,key=lambda k:(winners[k]['selection_ratio'],k))[:3];choices.append({k:1/3 for k in top})
            assert [r['weights'] for r in row['blend_candidates']]==choices
            for candidate in row['blend_candidates']:
                q=sum(pred[str(winners[k]['config']['id'])][:ns]*w for k,w in candidate['weights'].items())
                close(float(np.abs(q[...,5]-d['y'][nt:nt+ns]).mean())/float(d['selection_interpolation_mae']),candidate['selection_ratio'])
            chosen=min(row['blend_candidates'],key=lambda v:v['selection_ratio']);assert chosen['weights']==row['weights']
            expected=sum(pred[str(winners[k]['config']['id'])]*w for k,w in row['weights'].items())
            pred[name]=np.load(folder/'trials'/name/'quantiles.npy');close(expected,pred[name])
        if not smoke:
            original=t.SOURCE/'folds'/fold['test'];old=json.loads((original/'selection.json').read_text())['by_recipe']
            for family in t.FAMILIES:assert winners[family+'__standard']['config']['reference_id']==old[family+'_uniform']['config']['id']
            close(pred['blend__standard'],np.load(original/'trials/blend_uniform/quantiles.npy'))
        chosen=min([k for k in winners if k!='validation_selector'],key=lambda k:(winners[k]['selection_ratio'],k));assert winners['validation_selector']['chosen_recipe']==chosen
        report=json.loads((folder/'metrics.json').read_text())
        for name,row in winners.items():
            q=np.load(folder/'trials'/str(row['config']['id'])/'quantiles.npy');cal=q[ns:ns+nc];test=q[ns+nc:]
            saved=np.load(folder/'selected'/name/'predictions.npz');np.testing.assert_array_equal(saved['quantiles'],test);np.testing.assert_array_equal(saved['observed'],truth)
            error=np.abs(test[...,5]-truth);stats=report['results'][name];close(error.mean(),stats['mae']);close(error.mean()/den,stats['ratio'])
            close(error.mean((0,2,3)),stats['by_zone']);close(error.mean((0,2,3))/np.abs(base-truth).mean((0,2,3)),stats['zone_ratio'])
            err=truth[...,None]-test;pb=np.maximum(t.TAUS*err,(t.TAUS-1)*err).mean((0,1,2,3));close(pb,stats['pinball_by_quantile'])
            previous_lo=previous_hi=None
            for coverage,lo,hi in t.e.INTERVALS:
                level=round(100*coverage);scores=np.maximum.reduce([cal[...,lo]-d['y'][nt+ns:nt+ns+nc],d['y'][nt+ns:nt+ns+nc]-cal[...,hi],np.zeros_like(cal[...,lo])])
                grouped=scores.transpose(1,3,0,2).reshape(4,4,-1);rank=math.ceil((grouped.shape[-1]+1)*coverage);correction=np.sort(grouped,axis=-1)[...,rank-1][None,:,None,:]
                lower=test[...,lo]-correction;upper=test[...,hi]+correction
                if previous_lo is not None:lower=np.minimum(lower,previous_lo);upper=np.maximum(upper,previous_hi)
                close(lower,saved[f'lower_{level}']);close(upper,saved[f'upper_{level}'])
                score=next(v for v in stats['intervals'] if v['nominal']==coverage and v['kind']=='calibrated')
                close(((truth>=lower)&(truth<=upper)).mean(),score['coverage']);close((upper-lower).mean(),score['width'])
                previous_lo,previous_hi=lower,upper
            aggregate.setdefault(name,[]).append(float(error.mean()/den));pooled.setdefault(name,[]).append(error.ravel())
        df=pd.read_csv(folder/'predictions.csv.gz');selected=np.load(folder/'selected/validation_selector/predictions.npz')
        assert len(df)==ne*384 and not df.duplicated(['delivery_date','market_id','hour_index','quarter_index']).any()
        for k,tau in enumerate(t.TAUS):close(df[f'q{tau:g}'],selected['quantiles'][...,k].ravel(),1e-7)
        exports.append(df)
    assert total==(24 if smoke else 384) and reused==(0 if smoke else 192)
    summary=json.loads((root/'metrics.json').read_text())['summary']
    for row in summary:
        close(row['mean_monthly_ratio'],np.mean(aggregate[row['recipe']]));close(row['pooled_mae'],np.concatenate(pooled[row['recipe']]).mean())
    if not smoke:
        close(next(r for r in summary if r['recipe']=='blend__standard')['mean_monthly_ratio'],.7204941118854624,1e-12)
        df=pd.read_csv(root/'predictions.csv.gz');assert len(df)==92544
        pd.testing.assert_frame_equal(df,pd.concat(exports,ignore_index=True),check_exact=False,atol=1e-8,rtol=0)
        pairs=pd.DataFrame(paired).pivot(index=['month','family','lr','seed'],columns='mode',values='ratio');pairs['fixed_minus_standard']=pairs.fixed-pairs.standard;pairs.to_csv(root/'paired_candidate_ratios.csv')
        lines=['# Strong six-expert ensemble: normalization comparison','', 'Eight developmental tests; original ordinary-training standardized controls reused exactly,192 new fixed-scaling fits. Lower mean monthly MAE/interpolation MAE is better. Pure standardized blend must reproduce .7204941118854624.','', '| Recipe | Mean monthly ratio | Pooled MAE | Calibrated90% coverage | Width |','|---|---:|---:|---:|---:|']
        for r in summary:lines.append(f"| {r['recipe']} | {r['mean_monthly_ratio']:.6f} | {r['pooled_mae']:.6f} | {r['coverage90']:.2%} | {r['width90']:.3f} |")
        lines+=['','Fixed retains engineered physical units and clip; standard adds fitting-only z-scores. All models and recipes predict eleven ordered quantiles. Family hyperparameters, blend menus and mixed per-family arm choices use preceding first15days only; CQR uses16–end. No outer-test selection or deployment update. The previous compact-summary CNN/S4D/S5 comparison used different models and cannot determine this ensemble result.','', 'All candidate pairings: paired_candidate_ratios.csv. Full quantiles/intervals and width/error diagnostics retained. The .718907 zone-shrinkage result belongs to a separate posthoc study and is not the baseline altered here.']
        (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    t.e.p.write(root/'audit.json',dict(passed=True,smoke=smoke,folds=len(jobs),checkpoints_replayed=total,reused_controls=reused,new_fixed_fits=total-reused if not smoke else 12,
        max_replay_error=maximum,parameters_by_family=parameters,train_only_preprocessing=True,reference_checkpoint_prediction_identity=True,
        original_blend_reproduced=not smoke,selection_blends_calibration_scores_verified=True,ordered_quantiles_hourly_conservation=True))
    print('AUDIT PASSED',smoke,total,reused,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();audit(a.smoke)
