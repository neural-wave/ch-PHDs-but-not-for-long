"""Offline 11-quantile linear/hierarchical models on the existing 16 folds."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from reconstruction import read_targets,rich_features,baseline
from train_expanding import make_folds,masks,scores
from hierarchical_models import conformal_quantile
from linear_quantile_model import LinearQuantiles,QUANTILES,joint_loss,physical_quantiles,feature_names,numpy_forward


def pinball(pred,y):
    error=y[...,None]-pred
    return np.maximum(QUANTILES*error,(QUANTILES-1)*error).mean(tuple(range(error.ndim-1)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--out',default='results/linear_quantiles_vast');a=p.parse_args()
    assert torch.cuda.is_available(),'Remote execution GPU required'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    source=Path('train_prices_15min.csv');sha=hashlib.sha256(source.read_bytes()).hexdigest()
    assert sha=='f71ae6dbcca5f04e76e35840e29d064e69ad292f26d8ec7bf013bff2d904f65c'
    dates,y=read_targets(source);h=y.mean(-1);b=baseline(h)
    x=rich_features(dates,h,dates,y).reshape(len(dates),96,-1).astype(np.float64);names=feature_names();assert x.shape[-1]==len(names)
    folds=make_folds(dates)
    if a.smoke:folds=[folds[0],folds[8]]
    configs=[dict(lr=lr,l2=l2) for lr in [.001,.003] for l2 in [.001,.01]]
    if a.smoke:configs=configs[:1]
    rows=[];intervals=[];folds_report=[]
    for fold in folds:
        train,select,cal,test=masks(dates,fold);evaluated=select|cal|test
        folder=out/(fold['plan']+'_'+str(fold['training_months'])+'m');folder.mkdir(exist_ok=True)
        probe=np.flatnonzero(test)[0];changed=y.copy();changed[probe:]+=np.arange(4)*1234
        assert np.array_equal(x[probe:probe+1].reshape(96,-1),rich_features(dates[probe:probe+1],h[probe:probe+1],dates,changed))
        mu=x[train].mean((0,1));sd=x[train].std((0,1)).clip(.05)
        xx=np.clip((x-mu)/sd,-15,15)
        xt=torch.tensor(xx.reshape(-1,len(names)),device='cuda',dtype=torch.float64)
        target=((y-b)/20).reshape(-1,4);yt=torch.tensor(target,device='cuda',dtype=torch.float64)
        ti=torch.tensor(np.flatnonzero(np.repeat(train,96)),device='cuda');si=torch.tensor(np.flatnonzero(np.repeat(select,96)),device='cuda')
        candidates=[];states=[]
        for cfg in configs:
            torch.manual_seed(17);model=LinearQuantiles(len(names)).cuda();opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'])
            # Initial nonmedian biases are training-target quantiles; no held-out fitting.
            initial=np.quantile(target[np.repeat(train,96)],QUANTILES[QUANTILES!=.5],axis=0).T.ravel()
            with torch.no_grad():model.linear.bias[3:].copy_(torch.tensor(initial,device='cuda'))
            best=float('inf');state=None;best_epoch=0;start=time.perf_counter()
            for epoch in range(2 if a.smoke else 250):
                model.train()
                for ids in ti[torch.randperm(len(ti),device='cuda')].split(2048):
                    joint,_,_=joint_loss(model(xt[ids]),yt[ids])
                    regularizer=cfg['l2']*model.linear.weight.square().sum()/43
                    opt.zero_grad();(joint+regularizer).backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                model.eval()
                with torch.no_grad():value,median,pb=joint_loss(model(xt[si]),yt[si])
                value=float(value)
                if value<best:
                    best=value;best_epoch=epoch+1;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                if epoch+1-best_epoch>=25:break
            candidates.append(dict(config=cfg,selection_joint=best*20,best_epoch=best_epoch,epochs_run=epoch+1,seconds=time.perf_counter()-start))
            states.append(state)
            print(fold['plan'],fold['training_months'],cfg,'joint',best*20,flush=True)
        selected=int(np.argmin([c['selection_joint'] for c in candidates]));cfg=configs[selected];state=states[selected]
        model=LinearQuantiles(len(names)).cuda().eval();model.load_state_dict(state)
        with torch.no_grad():raw=model(xt[np.repeat(evaluated,96)]).cpu().numpy().reshape(evaluated.sum(),4,24,4,11)
        q,unrearranged,crossing=physical_quantiles(raw,b[evaluated],h[evaluated])
        tq=q[test[evaluated]];cq=q[cal[evaluated]];tp=tq[...,5];cp=cq[...,5]
        checkpoint=dict(quantiles=QUANTILES.tolist(),feature_names=names,mu=mu.tolist(),sd=sd.tolist(),
            weight=state['linear.weight'].numpy().tolist(),bias=state['linear.bias'].numpy().tolist(),config=cfg,
            median='a+u,a-u,-a+v,-a-v correction to mean-corrected interpolation',postprocess='sort/clamp nonmedian outputs around fixed median')
        (folder/'model.json').write_text(json.dumps(checkpoint))
        # Independent NumPy replay from readable JSON, including quantile postprocessing.
        loaded=json.loads((folder/'model.json').read_text())
        replayraw=numpy_forward(np.clip((x[test]-np.array(loaded['mu']))/np.array(loaded['sd']),-15,15),loaded['weight'],loaded['bias']).reshape(test.sum(),4,24,4,11)
        replay,_,_=physical_quantiles(replayraw,b[test],h[test]);delta=float(np.max(np.abs(replay-tq)))
        assert delta<1e-8,delta
        rawpb=pinball(unrearranged[test[evaluated]],y[test]);orderedpb=pinball(tq,y[test])
        item=dict(**fold,train_days=int(train.sum()),test_days=int(test.sum()),selection_days=int(select.sum()),calibration_days=int(cal.sum()),
            candidates=candidates,selected=candidates[selected],test_metrics=scores(tp,y[test]),raw_crossing_rate=crossing,
            pinball_by_quantile=orderedpb.tolist(),raw_pinball_by_quantile=rawpb.tolist(),quantile_coverage=(y[test,...,None]<=tq).mean((0,1,2,3)).tolist(),
            serialized_replay_max_error=delta,conformal=[])
        arrays=dict(dates=dates[test].astype('U10'),observed=y[test],predicted=tp,quantiles=tq,levels=QUANTILES)
        for level in [.5,.9,.95]:
            lo=int(np.argmin(np.abs(QUANTILES-(1-level)/2)));hi=int(np.argmin(np.abs(QUANTILES-(1+level)/2)))
            rawlower,rawupper=tq[...,lo],tq[...,hi]
            errors=np.abs(cp-y[cal]).transpose(0,2,1,3).reshape(-1,4,4)
            radius=conformal_quantile(errors,1-level)[None,:,None,:]
            # Conservative CQR: nonnegative correction widens learned bounds only.
            conformal_scores=np.maximum.reduce([cq[...,lo]-y[cal],y[cal]-cq[...,hi],np.zeros_like(y[cal])])
            correction=conformal_quantile(conformal_scores.transpose(0,2,1,3).reshape(-1,4,4),1-level)[None,:,None,:]
            for method,lower,upper in [('raw_quantiles',rawlower,rawupper),('median_conformal',tp-radius,tp+radius),('quantile_conformal',rawlower-correction,rawupper+correction)]:
                assert np.isfinite(lower).all() and (lower<=upper).all()
                coverage=float(((y[test]>=lower)&(y[test]<=upper)).mean());width=float((upper-lower).mean())
                entry=dict(method=method,nominal=level,coverage=coverage,mean_width=width)
                item['conformal'].append(entry);intervals.append(dict(plan=fold['plan'],training_months=fold['training_months'],train_days=int(train.sum()),test=fold['test'],**entry))
                arrays[f'{method}_{round(level*100)}_lower']=lower;arrays[f'{method}_{round(level*100)}_upper']=upper
            # Cover the 30-minute middle stage from its median forecast, not sums of quantile endpoints.
            ph=tp.reshape(-1,4,24,2,2).mean(-1);ch=cp.reshape(-1,4,24,2,2).mean(-1)
            yh=y[test].reshape(-1,4,24,2,2).mean(-1);cyh=y[cal].reshape(-1,4,24,2,2).mean(-1)
            hr=conformal_quantile(np.abs(ch-cyh).transpose(0,2,1,3).reshape(-1,4,2),1-level)[None,:,None,:]
            item['conformal'].append(dict(method='halfhour_median_conformal',nominal=level,coverage=float((np.abs(ph-yh)<=hr).mean()),mean_width=float(2*hr.mean())))
            arrays[f'halfhour_{round(level*100)}_lower']=ph-hr;arrays[f'halfhour_{round(level*100)}_upper']=ph+hr
        np.savez_compressed(folder/'test_predictions.npz',**arrays)
        output_names=['median_half_contrast','median_first_quarter_contrast','median_second_quarter_contrast']+[f'q{tau:g}_quarter{quarter+1}' for quarter in range(4) for tau in QUANTILES if tau!=.5]
        weights=np.asarray(loaded['weight']);bias=np.asarray(loaded['bias'])
        pd.DataFrame([dict(output=label,feature=name,coefficient=float(w),eur_per_training_std=float(20*w)) for label,values in zip(output_names,weights) for name,w in zip(names,values)]).to_csv(folder/'coefficients.csv',index=False)
        (folder/'metrics.json').write_text(json.dumps(item,indent=2));folds_report.append(item)
        rows.append(dict(plan=fold['plan'],training_months=fold['training_months'],train_days=int(train.sum()),test=fold['test'],median_mae=item['test_metrics']['mae'],mean_pinball=float(orderedpb.mean()),raw_crossing_rate=crossing,**{f'mae_zone{z}':value for z,value in enumerate(item['test_metrics']['by_zone'])}))
        pd.DataFrame(rows).to_csv(out/'learning_curves.csv',index=False);pd.DataFrame(intervals).to_csv(out/'interval_curves.csv',index=False)
        (out/'progress.json').write_text(json.dumps(dict(completed_folds=len(folds_report),total_folds=len(folds),last=fold),indent=2))
        print('FOLD DONE',fold,'MAE',item['test_metrics']['mae'],flush=True)
    (out/'metrics.json').write_text(json.dumps(dict(smoke=a.smoke,quantiles=QUANTILES.tolist(),source_sha256=sha,
        objective='median MAE + mean pinball over11 quarter quantiles + L2 regularization; selection by unregularized joint validation loss',folds=folds_report,
        caveats=['Offline fitting; no online parameter updates.','Mean constraint applies to median point forecast, not each marginal quantile.','Nonmedian crossing corrected around fixed median; coefficients explain raw linear quantiles.','CQR correction is nonnegative/widen-only.','Conformal temporal coverage is empirical, not guaranteed.','No half-hour marginal quantiles inferred by summing quarter quantiles.']),indent=2))
    from plot_linear_quantiles import plot
    plot(out)
    print('DONE quantile folds, audits, coefficients and graphs',flush=True)


if __name__=='__main__':main()
