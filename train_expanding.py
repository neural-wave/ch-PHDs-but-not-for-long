"""Monthly expanding backtest and fixed-September training-history learning curves."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from reconstruction import read_targets,rich_features,baseline,ZONES
from hierarchical_models import HierarchicalNet,conformal_quantile


def make_folds(dates):
    months=sorted({str(d)[:7] for d in dates})
    assert months==[str(p) for p in pd.period_range('2025-10','2026-09',freq='M')]
    folds=[]
    for last in range(2,10):
        folds.append(dict(plan='expanding',train_start=months[0],train_end=months[last],validation=months[last+1],test=months[last+2],training_months=last+1))
    # Fix the latest training date as well as validation/test: add older months.
    for size in range(3,11):
        folds.append(dict(plan='fixed_september',train_start=months[10-size],train_end=months[9],validation=months[10],test=months[11],training_months=size))
    return folds


def masks(dates,fold):
    month=np.array([str(d)[:7] for d in dates])
    day=np.array([int(str(d)[8:10]) for d in dates])
    train=(month>=fold['train_start'])&(month<=fold['train_end'])
    validation=month==fold['validation'];test=month==fold['test']
    select=validation&(day<=15);cal=validation&(day>15)
    assert train.any() and select.any() and cal.any() and test.any()
    assert max(dates[train])<min(dates[validation]) and max(dates[validation])<min(dates[test])
    assert not (train&validation).any() and not (validation&test).any()
    return train,select,cal,test


def scores(pred,truth):
    err=np.abs(pred-truth)
    return dict(mae=float(err.mean()),by_zone=err.mean((0,2,3)).tolist(),half_hour_mae=float(np.abs((pred-truth).reshape(-1,4,24,2,2).mean(-1)).mean()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--out',default='results/expanding_vast');a=p.parse_args()
    assert torch.cuda.is_available(),'Remote GPU required'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    source=Path('train_prices_15min.csv');sha=hashlib.sha256(source.read_bytes()).hexdigest()
    assert sha=='f71ae6dbcca5f04e76e35840e29d064e69ad292f26d8ec7bf013bff2d904f65c'
    dates,y=read_targets(source);h=y.mean(-1);b=baseline(h)
    x=rich_features(dates,h,dates,y).reshape(len(dates),4,24,-1)
    folds=make_folds(dates)
    if a.smoke:folds=[folds[0],folds[8]]
    configurations=[dict(width=w,seed=s,blocks=1,dropout=.1,lr=.001,batch_size=32) for w in [32,64] for s in [17,43]]
    if a.smoke:configurations=configurations[:1]
    result=[];csv_rows=[];pairs=[]
    for number,fold in enumerate(folds):
        started=time.perf_counter();train,select,cal,test=masks(dates,fold)
        folder=out/(fold['plan']+'_'+str(fold['training_months'])+'m');folder.mkdir(exist_ok=True)
        # Verify that current/future quarter prices cannot affect the first test day's features.
        probe=np.flatnonzero(test)[0];changed=y.copy();changed[probe:]+=np.arange(4)*1234
        assert np.array_equal(x[probe:probe+1].reshape(96,-1),rich_features(dates[probe:probe+1],h[probe:probe+1],dates,changed))
        mu=x[train].mean((0,1,2));sd=x[train].std((0,1,2))+1e-5
        xx=np.clip((x-mu)/sd,-15,15).astype(np.float32)
        xt=torch.tensor(xx,device='cuda');yt=torch.tensor((y-h[...,None])/20,dtype=torch.float32,device='cuda')
        ti=torch.tensor(np.flatnonzero(train),device='cuda');si=torch.tensor(np.flatnonzero(select),device='cuda')
        evaluated=select|cal|test;ei=np.flatnonzero(evaluated)
        candidates=[];predictions=[];artifacts=[]
        for hierarchical in [False,True]:
            for cfg in configurations:
                torch.manual_seed(cfg['seed']);net=HierarchicalNet(x.shape[-1],cfg['width'],hierarchical,cfg['blocks'],cfg['dropout']).cuda()
                opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=.01)
                best=float('inf');best_epoch=0;state=None;begin=time.perf_counter()
                for epoch in range(2 if a.smoke else 160):
                    net.train()
                    for ids in ti[torch.randperm(len(ti),device='cuda')].split(cfg['batch_size']):
                        # Identical quarter-MAE loss: isolate the output-head package.
                        loss=(net(xt[ids])-yt[ids]).abs().mean();opt.zero_grad();loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
                    net.eval()
                    with torch.no_grad():value=float((net(xt[si])-yt[si]).abs().mean())
                    if value<best:
                        best=value;best_epoch=epoch+1;state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
                    if epoch+1-best_epoch>=20:break
                net.load_state_dict(state);net.eval()
                with torch.no_grad():raw=net(xt[evaluated]).cpu().numpy().astype(np.float64)
                pred=h[evaluated,...,None]+20*(raw-raw.mean(-1,keepdims=True))
                name=('hierarchical' if hierarchical else 'direct')+f"_w{cfg['width']}_s{cfg['seed']}"
                candidates.append(dict(name=name,family='hierarchical' if hierarchical else 'direct',config=cfg,
                    selection_mae=float(np.abs(pred[select[evaluated]]-y[select]).mean()),best_epoch=best_epoch,epochs_run=epoch+1,seconds=time.perf_counter()-begin))
                predictions.append(pred);artifacts.append(dict(state=state,config=cfg,hierarchical=hierarchical,mu=mu,sd=sd))
                print(fold['plan'],fold['training_months'],name,'selection',candidates[-1]['selection_mae'],flush=True)
        # Rich linear ridge baseline: coefficients fit on the exact training dates only.
        rmu=x[train].mean((0,1,2));rsd=x[train].std((0,1,2)).clip(.05)
        rx=np.clip((x-rmu)/rsd,-15,15).astype(np.float64).reshape(len(dates)*96,-1)
        rx=np.column_stack([np.ones(len(rx)),rx]);target=((y-b)/20).reshape(-1,4)
        rt=np.repeat(train,96);gram=rx[rt].T@rx[rt];cross=rx[rt].T@target[rt]
        for alpha in ([1000] if a.smoke else [1,10,100,1000,10000]):
            regularizer=np.eye(rx.shape[1])*alpha;regularizer[0,0]=0
            beta=np.linalg.solve(gram+regularizer,cross)
            raw=(rx@beta).reshape(len(dates),4,24,4)[evaluated]
            pred=b[evaluated]+20*(raw-raw.mean(-1,keepdims=True))
            candidates.append(dict(name=f'ridge_{alpha}',family='ridge',config=dict(alpha=alpha),selection_mae=float(np.abs(pred[select[evaluated]]-y[select]).mean())))
            predictions.append(pred);artifacts.append(dict(beta=beta,mu=rmu,sd=rsd))
        candidates.append(dict(name='interpolation',family='interpolation',config={},selection_mae=float(np.abs(b[select]-y[select]).mean())))
        predictions.append(b[evaluated]);artifacts.append(None)
        # Freeze every family's selection before evaluating calibration or test targets.
        winners={family:min([i for i,c in enumerate(candidates) if c['family']==family],key=lambda i:candidates[i]['selection_mae']) for family in ['direct','hierarchical','ridge','interpolation']}
        fold_report=dict(**fold,train_days=int(train.sum()),selection_days=int(select.sum()),calibration_days=int(cal.sum()),test_days=int(test.sum()),selected={},candidates=candidates)
        for family,index in winners.items():
            pred=predictions[index];testpred=pred[test[evaluated]];calpred=pred[cal[evaluated]]
            assert np.isfinite(pred).all() and np.abs(pred.mean(-1)-h[evaluated]).max()<1e-8
            item=dict(candidates[index],test=scores(testpred,y[test]),calibration=scores(calpred,y[cal]),conformal=[])
            path=folder/(family+'.pt');torch.save(dict(artifact=artifacts[index],family=family,metadata=candidates[index],fold=fold),path)
            loaded=torch.load(path,weights_only=False,map_location='cpu')['artifact']
            if family in ['hierarchical','direct']:
                cfg=loaded['config'];replay=HierarchicalNet(x.shape[-1],cfg['width'],loaded['hierarchical'],cfg['blocks'],cfg['dropout']).cuda().eval()
                replay.load_state_dict(loaded['state'])
                with torch.no_grad():raw=replay(torch.tensor(np.clip((x[test]-loaded['mu'])/loaded['sd'],-15,15),device='cuda',dtype=torch.float32)).cpu().numpy().astype(np.float64)
                pp=h[test,...,None]+20*(raw-raw.mean(-1,keepdims=True))
                assert np.max(np.abs(pp-testpred))<1e-4
            elif family=='ridge':
                raw=np.clip((x[test]-loaded['mu'])/loaded['sd'],-15,15).astype(np.float64).reshape(test.sum()*96,-1)
                pp=(np.column_stack([np.ones(len(raw)),raw])@loaded['beta']).reshape(-1,4,24,4)
                pp=b[test]+20*(pp-pp.mean(-1,keepdims=True))
                assert np.max(np.abs(pp-testpred))<1e-8
            else:pp=b[test]
            item['serialized_replay_max_error']=float(np.max(np.abs(pp-testpred)))
            err=np.abs(calpred-y[cal]);testerr=np.abs(testpred-y[test]);bands={}
            for level in [.5,.9,.95]:
                radius=conformal_quantile(err.transpose(0,2,1,3).reshape(-1,4,4),1-level)[None,:,None,:]
                bands[f'lower_{round(level*100)}']=testpred-radius;bands[f'upper_{round(level*100)}']=testpred+radius
                item['conformal'].append(dict(nominal=level,coverage=float((testerr<=radius).mean()),mean_width=float(2*radius.mean())))
            np.savez_compressed(folder/(family+'_test.npz'),dates=dates[test].astype('U10'),predicted=testpred,observed=y[test],**bands)
            daily=np.abs(testpred-y[test]).mean((1,2,3)).tolist()
            item['daily_test_mae']=daily
            fold_report['selected'][family]=item
            csv_rows.append(dict(plan=fold['plan'],training_months=fold['training_months'],train_days=int(train.sum()),train_start=fold['train_start'],train_end=fold['train_end'],validation=fold['validation'],test=fold['test'],test_days=int(test.sum()),family=family,selection_mae=item['selection_mae'],test_mae=item['test']['mae'],half_hour_mae=item['test']['half_hour_mae'],**{f'mae_{z}':v for z,v in zip(ZONES,item['test']['by_zone'])}))
            if family=='hierarchical':
                # Physical stage quantities, in EUR/MWh; these explain outputs, not feature causality.
                def contrasts(qq):return np.stack([(qq[...,0]+qq[...,1]-qq[...,2]-qq[...,3])/4,(qq[...,0]-qq[...,1])/2,(qq[...,2]-qq[...,3])/2],-1)
                np.savez_compressed(folder/'hierarchical_stages.npz',dates=dates[test].astype('U10'),predicted=contrasts(testpred),observed=contrasts(y[test]))
        for cfg in configurations:
            dd=next(i for i,c in enumerate(candidates) if c['family']=='direct' and c['config']==cfg)
            hh=next(i for i,c in enumerate(candidates) if c['family']=='hierarchical' and c['config']==cfg)
            direct=scores(predictions[dd][test[evaluated]],y[test])['mae'];hier=scores(predictions[hh][test[evaluated]],y[test])['mae']
            pairs.append(dict(**fold,config=cfg,direct_mae=direct,hierarchical_mae=hier,delta_hierarchy_minus_direct=hier-direct))
        fold_report['seconds']=time.perf_counter()-started
        (folder/'metrics.json').write_text(json.dumps(fold_report,indent=2));result.append(fold_report)
        pd.DataFrame(csv_rows).to_csv(out/'learning_curves.csv',index=False)
        (out/'progress.json').write_text(json.dumps(dict(completed_folds=len(result),total_folds=len(folds),last=fold),indent=2))
        print('FOLD DONE',fold,{f:m['test']['mae'] for f,m in fold_report['selected'].items()},flush=True)
    frame=pd.DataFrame(csv_rows);summary=[]
    for (plan,family),g in frame.groupby(['plan','family']):
        summary.append(dict(plan=plan,family=family,weighted_test_mae=float(np.average(g.test_mae,weights=g.test_days)),mean_month_mae=float(g.test_mae.mean())))
    report=dict(source_sha256=sha,smoke=a.smoke,folds=folds,summary=summary,matched_pairs=pairs,
        audits='passed: strict fold boundaries/train-only normalization/causal features/finite predictions/exact means/selected serialized replay',
        caveats=['All outcomes developmental; data have been used in previous development.','Expanding folds change test season as well as training size.','Fixed September comparison holds train end July, validation August and test September fixed; adds older training months.','Validation days1-15 select model/epoch;16-end calibrate intervals; no validation refit.','Prior-day observed history is available causally during validation/test; no online parameter fitting.','Conformal coverage empirical under dependence/drift.','Head packages have different parameter counts despite same encoder, seeds and quarter-MAE loss.'])
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    from plot_expanding import plot
    plot(out)
    print('DONE all folds and plots',flush=True)


if __name__=='__main__':main()
