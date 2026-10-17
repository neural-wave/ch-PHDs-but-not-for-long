"""Independent source/split/weight/serialized-prediction and score audit."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
import sys

ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT / 'campaigns/riccardo_comparison_20261017'
sys.path.insert(0, str(CAMPAIGN / 'upstream/scripts'))
METHODS = ['riccardo_reduced', 'riccardo_raw_hourly']


def readcsv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def csvwrite(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    import numpy as np
    import torch
    from feature_ablation_per_zone import ZonalAblationCNN
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    out = ROOT / ('results/riccardo_comparison_20261017' + ('_smoke' if args.smoke else ''))
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['smoke'] == args.smoke
    for path, sha in manifest['source_hashes'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == sha
    assert hashlib.sha256((ROOT/'train_prices_15min.csv').read_bytes()).hexdigest() == manifest['source_sha256']
    bins = {}
    for row in readcsv(ROOT/'train_prices_15min.csv'):
        key = (row['delivery_date'], row['market_id'])
        slot = (int(row['hour_index'])-1, int(row['quarter_index'])-1)
        bins.setdefault(key, {})
        assert slot not in bins[key]
        bins[key][slot] = float(row['price_eur_mwh'])
    keys = sorted(bins)
    assert len(keys) == 363*4
    assert all(len(bins[key]) == 96 for key in keys)
    y = np.array([[[bins[key][h,q] for q in range(4)] for h in range(24)] for key in keys])
    h = y.mean(-1)
    dates = np.array([k[0] for k in keys]); zones = np.array([k[1] for k in keys])
    months = np.array([d[:7] for d in dates]); days = np.array([int(d[8:]) for d in dates])
    weekday = np.array([(int(np.datetime64(d,'D').astype(int))+3)%7 for d in dates])
    monthnum = np.array([int(d[5:7])-1 for d in dates])
    b = np.array([np.interp((np.arange(96)+.5)/4, np.arange(24)+.5, hh).reshape(24,4) for hh in h])
    b += h[:,:,None] - b.mean(-1,keepdims=True)
    baselines = {r['test_month']:float(r['interpolation_mae']) for r in
                 readcsv(ROOT/'results/normalized_validation_20261017/monthly_metrics.csv')}
    for month, saved in baselines.items():
        assert abs(float(np.abs(b[months==month]-y[months==month]).mean())-saved) < 1e-9
    reports = json.loads((out/'metrics.json').read_text())['reports']
    assert len(reports) == len(manifest['folds'])*4
    assert len({(r['spec']['test'],r['zone']) for r in reports}) == len(reports)
    zone_rows, day_rows, max_replay, max_mean, weight_cases = [], [], 0., 0., 0
    configs=json.loads((CAMPAIGN/'upstream/experiments/selected_reduced_features_per_zone/run_info.json').read_text())['configs']
    for report in reports:
        spec, zone = report['spec'], report['zone']
        assert spec in manifest['folds']
        folder = out/spec['test']/zone
        tr=np.flatnonzero((zones==zone)&np.isin(months,spec['train_months']))
        se=np.flatnonzero((zones==zone)&(months==spec['validation'])&(days<=15))
        ca=np.flatnonzero((zones==zone)&(months==spec['validation'])&(days>15))
        te=np.flatnonzero((zones==zone)&(months==spec['test']))
        for ids,name in [(tr,'train'),(se,'selection'),(ca,'calibration'),(te,'test')]:
            assert dates[ids].tolist()==report[name+'_dates']
        assert dates[tr[-1]] < dates[se[0]] <= dates[se[-1]] < dates[ca[0]] <= dates[ca[-1]] < dates[te[0]]
        mean,std=float(h[tr].mean()),float(h[tr].std())
        assert mean==report['hourly_mean_train'] and std==report['hourly_std_train']
        assert [s['held_month'] for s in report['scorer']]==spec['train_months']
        for s in report['scorer']:
            inner=tr[months[tr]!=s['held_month']];held=tr[months[tr]==s['held_month']]
            assert dates[inner].tolist()==s['train_dates'] and dates[held].tolist()==s['held_dates']
            assert float(h[inner].mean())==s['hourly_mean_train']
            assert float(h[inner].std())==s['hourly_std_train']
        score=np.array([s['qweighted_mae'] for s in report['scorer']])
        assert np.isfinite(score).all()
        span=score.max()-score.min()
        raw=1+.5*(np.zeros_like(score) if span<1e-12 else (score-score.min())/span)
        counts=np.array([sum(months[tr]==m) for m in spec['train_months']])
        weights=raw/np.average(raw,weights=counts)
        assert abs(np.average(weights,weights=counts)-1)<1e-12
        assert max(abs(report['month_weights'][m]-w) for m,w in zip(spec['train_months'],weights))<1e-12
        weight_cases += len(score)
        # Independently construct the reduced profile channels from hourly inputs.
        level=(h-h.mean(1,keepdims=True))/std
        slope=np.gradient(h/std,axis=1);curvature=np.gradient(slope,axis=1)
        boundary=np.zeros_like(h);boundary[:,[0,-1]]=1;boundary[:,[1,-2]]=-1
        reduced=np.stack([level,slope,curvature,boundary],-1).astype('float32')
        for method in METHODS:
            ck=torch.load(folder/(method+'.pt'),map_location='cpu',weights_only=False)
            assert ck['config']==configs[zone] and ck['train_dates']==dates[tr].tolist()
            assert ck['spec']==spec and ck['zone']==zone and ck['method']==method
            assert ck['month_weights']==report['month_weights']
            assert ck['target_residual_scale']==float(np.abs(y[tr]-h[tr,:,None]).mean())
            best,epoch=float('inf'),0
            for item in ck['history']:
                if item['selection_mae']<best-1e-6:
                    best,epoch=item['selection_mae'],item['epoch']
            assert epoch==ck['best_epoch'] and best==ck['selection_mae']
            assert 1<=len(ck['history'])<=manifest['max_epochs']
            assert not any('month' in name for name in ck['state_dict'])
            model=ZonalAblationCNN(ck['config'],ck['profile_count'],ck['calendar']).to('cuda')
            model.load_state_dict(ck['state_dict']);model.eval()
            assert sum(p.numel() for p in model.parameters())==ck['parameter_count']
            saved=np.load(folder/(method+'.npz'))
            ids=np.r_[se,ca,te]
            assert np.array_equal(saved['ids'],ids) and np.array_equal(saved['dates'],dates[ids])
            assert np.array_equal(saved['actual'],y[ids])
            assert int(saved['selection_count'])==len(se) and int(saved['calibration_count'])==len(ca)
            x=reduced if method=='riccardo_reduced' else h[:,:,None].astype('float32')
            with torch.no_grad():
                raw=model(torch.tensor(x[ids],device='cuda'),torch.tensor(weekday[ids],device='cuda'),
                          torch.tensor(monthnum[ids],device='cuda')).cpu().numpy().astype('float64')
            raw*=ck['target_residual_scale']
            pp=h[ids,:,None]+raw-raw.mean(-1,keepdims=True)
            diff=float(np.abs(pp-saved['predicted']).max());max_replay=max(max_replay,diff)
            # Independent algebraic feature construction may differ by one float32 ULP.
            assert diff<1e-4,(method,spec['test'],zone,diff)
            pp=saved['predicted']
            assert np.isfinite(pp).all()
            meanerr=float(np.abs(pp.mean(-1)-h[ids]).max());max_mean=max(max_mean,meanerr)
            assert meanerr<.0001
            assert abs(float(np.abs(pp[:len(se)]-y[se]).mean())-ck['selection_mae'])<1e-7
            err=np.abs(pp[-len(te):]-y[te]);mae=float(err.mean())
            assert abs(mae-report['methods'][method]['test_mae'])<1e-10
            base=float(np.abs(b[te]-y[te]).mean())
            zone_rows.append(dict(method=method,test_month=spec['test'],zone=zone,train_days=len(tr),
                test_days=len(te),mae=mae,interpolation_mae=base,ratio=mae/base,
                best_epoch=ck['best_epoch'],parameters=ck['parameter_count']))
            for date,daily in zip(dates[te],err.mean((1,2))):
                day_rows.append(dict(method=method,date=date,zone=zone,mae=float(daily)))
    csvwrite(out/'per_zone.csv',zone_rows);csvwrite(out/'per_day_zone.csv',day_rows)
    monthly=[]
    for spec in manifest['folds']:
        month=spec['test']
        for method in METHODS:
            group=[r for r in zone_rows if r['method']==method and r['test_month']==month]
            assert len(group)==4 and len({r['test_days'] for r in group})==1
            mae=fmean(r['mae'] for r in group);days=group[0]['test_days']
            monthly.append(dict(method=method,test_month=month,mae=mae,test_days=days,targets=days*384,
                interpolation_mae=baselines[month],mae_over_interpolation_mae=mae/baselines[month]))
    csvwrite(out/'monthly_metrics.csv',monthly)
    summary=[]
    for method in METHODS:
        group=[r for r in monthly if r['method']==method];targets=sum(r['targets'] for r in group)
        if not args.smoke:
            assert targets==92544 and len(group)==8
        pooled=sum(r['mae']*r['targets'] for r in group)/targets
        interp=sum(r['interpolation_mae']*r['targets'] for r in group)/targets
        ratio=fmean(r['mae_over_interpolation_mae'] for r in group)
        summary.append(dict(method=method,pooled_mae=pooled,mean_monthly_ratio=ratio,
            worst_monthly_ratio=max(r['mae_over_interpolation_mae'] for r in group),
            pooled_mae_over_pooled_interpolation_mae=pooled/interp,folds=len(group)))
    csvwrite(out/'summary.csv',summary)
    audit=dict(passed=True,smoke=args.smoke,zone_folds=len(reports),serialized_replays=len(zone_rows),
        train_only_weight_scores=weight_cases,max_prediction_replay_difference=max_replay,
        max_hourly_mean_error=max_mean,official_baseline_match=True,
        checks='Official target/date identity; chronological outer splits; train-only scaler/weight ledgers and minmax; selection-only checkpoint argmin; serialized neural replay with independent feature arithmetic; finite/conserved quarters; monthly/zone/daily MAE and normalized scores',
        smoke_extra='Adapter matches original reduced-feature training; perturbing calibration/test quarters leaves fitted state unchanged' if args.smoke else None)
    (out/'independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    if not args.smoke:
        prior=readcsv(ROOT/'results/normalized_validation_20261017/summary.csv')
        combined=[{k:(r[k] if k in ['method','folds'] else float(r[k])) for k in summary[0]} for r in prior]+summary
        combined.sort(key=lambda r:r['mean_monthly_ratio']);csvwrite(out/'comparison.csv',combined)
        report=['# Riccardo branch under our monthly validation','',
            'Pinned branch `riccardo`, commit `1ed1423c5fa05586457347ab34e69b9bcfb23031`. Verified 17 October 2026.',
            '', 'Two fresh CNN pipelines use the same eight expanding folds, first-15-day epoch selection, full following-month tests, official targets and corrected-interpolation denominator as our saved nine pipelines. All learned weights and scales are fitted inside each outer training block. Days 16–month-end are reserved and unused; no new uncertainty intervals were requested.',
            '', '| Method | Pooled MAE EUR/MWh | Mean monthly ratio | Worst month ratio |', '|---|---:|---:|---:|']
        for r in combined:
            report.append(f"| {r['method']} | {r['pooled_mae']:.6f} | {r['mean_monthly_ratio']:.6f} | {r['worst_monthly_ratio']:.6f} |")
        report+=['','## Monthly results','','| Test month | Interpolation MAE | Reduced CNN MAE / ratio | Raw-hourly CNN MAE / ratio |','|---|---:|---:|---:|']
        for spec in manifest['folds']:
            m=spec['test'];g={r['method']:r for r in monthly if r['test_month']==m}
            report.append(f"| {m} | {baselines[m]:.6f} | "+' | '.join(f"{g[k]['mae']:.6f} / {g[k]['mae_over_interpolation_mae']:.6f}" for k in METHODS)+' |')
        report+=['','## What was compared','',
            'The lowest committed aggregate August–September MAE is 5.087239411 for the reduced-profile CNN using refit month weights. The four-channel input contains day-centered hourly level, slope, curvature and boundary indicator, plus learned hour and weekday embeddings. Four separate zonal two-layer CNNs have widths 64/96/96/64. The README raw-hourly model scores 5.113040454, uses one unnormalized hourly input channel and no calendar embeddings. Both predict centered quarter residuals around the known hourly means, use Q1/Q4-weighted MAE (1.5,1,1,1.5), AdamW, gradient clipping 5, up to 150 epochs and patience 16. The imported per-zone configurations and original seeds are frozen.',
            '', 'Original scores are selection results over August and September together. Their LOMO difficulty scorers train on 11 of all 12 months; October–July month scores therefore depend on August/September targets despite restricting the subsequent minmax normalization to training months. This campaign recomputes LOMO scorers strictly within each outer training block and selects final epochs only on the next month’s first 15 days. Internal LOMO scorers retain the upstream held-month early stopping and can use later months inside the outer training block; they are a train-only weight estimator, not a chronological inner forecast benchmark. The clean adaptation is distinct from evaluating the original committed checkpoints.',
            '', 'Candidate/feature configurations were historically selected with inspected August/September data. The adaptation provides fresh chronological fits, not a blind estimate or a full nested rerun of their original architecture/feature search. Our saved pipelines have different search budgets and some use causal quarter-history/cross-zone inputs; compare complete pipelines, not architectures in isolation. All results remain developmental. No deployment model was replaced. External zone-identification and October files were excluded.',
            '', 'Earlier September-only blend/analog/quantile studies remain in `results/normalized_validation_20261017/september_only.csv`; they have no matched eight-month estimates and are not pooled into this table.',
            '', 'Audit: `independent_audit.json`; per-zone and daily results: `per_zone.csv`, `per_day_zone.csv`; selected states and predictions: `<test-month>/<zone>/*.pt,*.npz`. Source provenance: `campaigns/riccardo_comparison_20261017/upstream_receipt.json`. Execution host: `root@93.91.156.85:55340`, `/workspace/HACKATON`. Reproduce with `python compare_riccardo.py` then `python audit_riccardo_comparison.py` on that CUDA host.']
        (out/'REPORT.md').write_text('\n'.join(report)+'\n')
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        old=readcsv(ROOT/'results/normalized_validation_20261017/monthly_metrics.csv')
        fig,ax=plt.subplots(figsize=(10,4.6))
        for method in ['linear','linear_quantile_median','cnn']+METHODS:
            g=sorted([r for r in (old+monthly) if r['method']==method],key=lambda r:r['test_month'])
            ax.plot([r['test_month'][5:] for r in g],[float(r['mae_over_interpolation_mae']) for r in g],'o-',label=method)
        ax.axhline(1,color='gray',linestyle='--',label='Interpolation')
        ax.set(xlabel='Test month in 2026',ylabel='MAE / same-month interpolation MAE',title='Chronological monthly backtest; lower is better')
        ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
        fig.savefig(out/'monthly_comparison.png',dpi=180);fig.savefig(out/'monthly_comparison.svg');plt.close(fig)
    print(json.dumps(dict(audit=audit,summary=summary),indent=2),flush=True)


if __name__=='__main__':
    main()
