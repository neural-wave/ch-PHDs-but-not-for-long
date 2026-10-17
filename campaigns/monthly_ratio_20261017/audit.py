"""Recompute scores, weights, selection and serialized forecasts independently."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import torch
import experiment as e


def close(actual, expected, tol=1e-9):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tol)


def audit(smoke=False):
    e.setup()
    manifest = json.loads((e.ROOT/'manifest.json').read_text())
    source = e.PROJECT/'train_prices_15min.csv'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == e.SHA
    for path, sha in manifest['code_hashes'].items():
        assert hashlib.sha256((e.PROJECT/path).read_bytes()).hexdigest() == sha, path
    # Independently pivot the original quarters and reconstruct interpolation.
    official = pd.read_csv(source)
    keys = ['delivery_date', 'market_id', 'hour_index', 'quarter_index']
    assert not official.duplicated(keys).any() and len(official) == 363*384
    columns = pd.MultiIndex.from_product([[f'ZONE_{z}' for z in range(4)], range(1,25), range(1,5)])
    pivot = official.pivot(index='delivery_date', columns=keys[1:], values='price_eur_mwh').reindex(columns=columns).sort_index()
    y = pivot.to_numpy().reshape(363,4,24,4);h = y.mean(-1)
    uncorrected = np.asarray([[np.interp(np.arange(96)/4+.125, np.arange(24)+.5, zone).reshape(24,4) for zone in day] for day in h])
    baseline = uncorrected + h[...,None] - uncorrected.mean(-1,keepdims=True)
    raw = np.load(e.ROOT/'raw.npz')
    np.testing.assert_array_equal(raw['dates'], pivot.index.to_numpy())
    np.testing.assert_array_equal(raw['y'], y);close(raw['b'], baseline)
    months = np.asarray([str(date)[:7] for date in raw['dates']])
    daynums = np.asarray([int(str(date)[8:]) for date in raw['dates']])
    selected_paths = e.ROOT/'smoke' if smoke else e.ROOT/'folds'
    folders = sorted(path for path in selected_paths.iterdir() if path.is_dir())
    assert len(folders) == (2 if smoke else 8)
    checked, replayed, maximum_mean_error, max_replay = 0, 0, 0., 0.
    monthly = []
    for folder in folders:
        meta = json.loads((folder/'manifest.json').read_text());d = np.load(folder/'data.npz')
        nt, ns, nc, ne = (int(d[k]) for k in ['nt','ns','nc','ne'])
        assert ns == 15 and nt == meta['train_days'] and ne == meta['test_days']
        train = (months >= meta['train_start']) & (months <= meta['train_end'])
        select = (months == meta['validation']) & (daynums <= 15)
        cal = (months == meta['validation']) & (daynums > 15);test = months == meta['test']
        assert meta['train_end'] < meta['validation'] < meta['test']
        ids = np.flatnonzero(train | select | cal | test)
        np.testing.assert_array_equal(raw['dates'][ids], d['dates'])
        np.testing.assert_array_equal(y[ids], d['y']);close(baseline[ids],d['b'])
        assert train.sum() == nt and select.sum() == ns and cal.sum() == nc and test.sum() == ne
        weights = np.zeros(nt);train_months = months[train]
        for month in sorted(set(train_months)):
            pos = train_months == month;original = train & (months == month)
            bmae = float(np.abs(y[original]-baseline[original]).mean())
            close(bmae, meta['training_month_baselines'][month]['interpolation_mae'])
            weights[pos] = 1/(int(pos.sum())*bmae)
        weights /= weights.mean();close(weights,d['weights_ratio'],2e-7)
        close(np.ones(nt),d['weights_uniform'])
        # Equivalent objective on an asymmetric synthetic set and actual fitting months.
        errors = np.abs(y[train]-baseline[train]).mean((1,2,3))
        uniform_unscaled = np.mean([1/meta['training_month_baselines'][m]['interpolation_mae']/int((train_months==m).sum()) for m in train_months])
        close(float(np.mean(weights*errors))*uniform_unscaled*nt/len(set(train_months)),1.0)
        for key in ['rich','compact','cnn','curve']:
            original = raw[key]
            axes = tuple(range(original.ndim-1)) if key in ['rich','compact'] else (0,2) if key=='cnn' else (0,)
            mu = original[train].mean(axes,keepdims=True);sd = original[train].std(axes,keepdims=True).clip(.05)
            close(mu,d[key+'_mu']);close(sd,d[key+'_sd'])
            normalized = np.clip((original[ids]-mu)/sd,-15,15).astype('float32')
            if key=='compact':
                normalized = np.concatenate([normalized,np.broadcast_to(np.eye(4)[None,:,None,:],(len(ids),4,24,4))],-1).astype('float32')
            np.testing.assert_array_equal(normalized,d[key])
            if key in ['rich','compact']:
                expected = np.quantile(normalized[:nt].reshape(-1,normalized.shape[-1]),[.25,.5,.75],axis=0).astype('float32')
                np.testing.assert_array_equal(expected,d[key+'_knots'])
        selection = json.loads((folder/'selection.json').read_text());winners=selection['by_recipe'];rows=selection['candidates']
        assert len(rows) == (22 if smoke else 74)
        validation_baseline=float(np.abs(y[select]-baseline[select]).mean());close(validation_baseline,float(d['selection_interpolation_mae']))
        predictions={}
        for row in rows:
            cfg=row['config'];cid=str(cfg['id']);p=np.load(folder/'trials'/cid/'predictions.npy')
            assert p.shape==d['y'][nt:].shape and np.isfinite(p).all()
            close(float(np.abs(p[:ns]-y[select]).mean()),row['validation_mae'])
            close(row['validation_mae']/validation_baseline,row['selection_ratio'])
            violation=float(np.abs(p.mean(-1)-h[ids][nt:]).max())
            assert violation<1e-8;maximum_mean_error=max(maximum_mean_error,violation)
            predictions[cid]=p;checked+=1
        groups=sorted({(r['config']['family'],r['config']['objective']) for r in rows})
        for family,objective in groups:
            key=family+'_'+objective
            best=min([r for r in rows if (r['config']['family'],r['config']['objective'])==(family,objective)],key=lambda r:(r['selection_ratio'],str(r['config']['id'])))
            assert winners[key]['config']==best['config']
        for objective in ['uniform','ratio']:
            recipe='blend_'+objective;row=winners[recipe]
            for candidate in row['blend_candidates']:
                assert abs(sum(candidate['weights'].values())-1)<1e-12 and min(candidate['weights'].values())>=0
                blend=sum(predictions[str(winners[name]['config']['id'])][:ns]*w for name,w in candidate['weights'].items())
                close(float(np.abs(blend-y[select]).mean())/validation_baseline,candidate['selection_ratio'])
            expected=min(row['blend_candidates'],key=lambda v:v['selection_ratio'])
            assert row['weights']==expected['weights']
            blend=sum(predictions[str(winners[name]['config']['id'])]*w for name,w in row['weights'].items())
            saved=np.load(folder/'trials'/recipe/'predictions.npy');close(blend,saved)
            predictions[recipe]=saved
        eligible={name:row for name,row in winners.items() if name!='validation_selector'}
        selected=min(eligible,key=lambda name:(eligible[name]['selection_ratio'],name))
        assert winners['validation_selector']['chosen_recipe']==selected
        assert (folder/'selection.json').stat().st_mtime_ns <= (folder/'metrics.json').stat().st_mtime_ns
        report=json.loads((folder/'metrics.json').read_text())
        for name,row in winners.items():
            cid=str(row['config']['id']);p=predictions[cid][ns+nc:];err=np.abs(p-y[test])
            result=report['results'][name];bmae=float(np.abs(baseline[test]-y[test]).mean())
            close(err.mean(),result['mae']);close(bmae,result['interpolation_mae']);close(err.mean()/bmae,result['ratio'])
            close(err.mean((0,2,3)),result['by_zone']);close(err.mean((1,2,3)),result['daily_mae'])
            close(err.mean((0,2,3))/np.abs(baseline[test]-y[test]).mean((0,2,3)),result['ratio_by_zone'])
            monthly.append(dict(month=meta['test'],recipe=name,mae=float(err.mean()),ratio=float(err.mean())/bmae,days=ne))
        # Checkpoint replay for every distinct selected gradient model.
        for cid in sorted({str(row['config']['id']) for row in winners.values() if isinstance(row['config']['id'],int)}):
            ck=torch.load(folder/'trials'/cid/'model.pt',weights_only=False,map_location='cpu')
            cfg=ck['config'];net,feature=e.network(cfg,d);net.cuda().eval().load_state_dict(ck['state'])
            with torch.no_grad():
                rr=e.predict(net,cfg,torch.as_tensor(d[feature][nt:],device='cuda'),torch.as_tensor(d['curve'][nt:],device='cuda')).cpu().numpy().astype('float64')
            pp=d['b'][nt:]+20*(rr-rr.mean(-1,keepdims=True));delta=float(np.max(np.abs(pp-predictions[cid])))
            assert delta<1e-6;max_replay=max(max_replay,delta);replayed+=1
    out=e.ROOT/'smoke' if smoke else e.ROOT
    actual=pd.DataFrame(monthly);saved=pd.read_csv(out/'monthly_scores.csv')
    assert not saved.duplicated(['month','recipe']).any() and len(saved)==len(actual)
    merged=saved.merge(actual,on=['month','recipe'],validate='one_to_one',suffixes=('_saved','_audit'))
    close(merged.mae_saved,merged.mae_audit);close(merged.ratio_saved,merged.ratio_audit)
    summary=pd.read_csv(out/'summary.csv')
    for row in summary.to_dict('records'):
        group=actual[actual.recipe==row['recipe']]
        close(group.ratio.mean(),row['mean_monthly_ratio']);close(np.average(group.mae,weights=group.days),row['pooled_mae'])
        close(group.ratio.max(),row['worst_ratio'])
    result=dict(passed=True,smoke=smoke,folds=len(folders),candidate_predictions_recomputed=checked,
                selected_neural_models_replayed=replayed,max_replay_error=max_replay,max_hourly_mean_error=maximum_mean_error,
                checks='official data/source hashes, interpolation, chronology, train-only month weights/normalization/knots, candidate validation scores, family argmins, every blend candidate, recipe choice, test ratios, per-zone/daily/aggregate scores, selected serialization')
    e.write(out/'audit.json',result)
    if not smoke:
        lines=['# Average monthly ratio optimization','',
               'Completed developmental eight-fold experiment. Each candidate was stopped/selected using only the preceding month\'s first15days. Ratio means MAE / same-month mean-corrected-linear-interpolation MAE. Lower is better. All point predictions preserve supplied hourly means.','',
               '| Recipe | Average monthly ratio | Pooled MAE (EUR/MWh) | Worst monthly ratio |',
               '|---|---:|---:|---:|']
        for row in summary.sort_values('mean_monthly_ratio').to_dict('records'):
            lines.append(f"| {row['recipe']} | {row['mean_monthly_ratio']:.6f} | {row['pooled_mae']:.6f} | {row['worst_ratio']:.6f} |")
        lines += ['', '## Paired training-objective comparison','',
                  '| Family | Ordinary training | Monthly-ratio training | Ratio minus ordinary |',
                  '|---|---:|---:|---:|']
        indexed=summary.set_index('recipe')
        for family in e.FAMILIES+['ridge_rich','ridge_compact','blend']:
            ordinary=float(indexed.loc[family+'_uniform','mean_monthly_ratio']);relative=float(indexed.loc[family+'_ratio','mean_monthly_ratio'])
            lines.append(f'| {family} | {ordinary:.6f} | {relative:.6f} | {relative-ordinary:+.6f} |')
        lines += ['', '## Monthly results','','MONTHLY_TABLE','',
                  'The validation_selector row is a deployable rule chosen from prior validation results, not the best outer-test recipe. The lowest aggregate test result is exploratory evidence for future studies. These months have all been inspected during prior development, so results are not a blind benchmark. Search budgets/parameter counts vary across families. A lower aggregate alone does not establish statistical significance.','',
                  'Ratio training weights are proportional to1/(training-month days × training-month interpolation MAE), normalized to mean1. No outer-test denominator enters fitting or selection. Calibration dates are reserved and unused for point-model fitting.','',
                  'Historical same-fold linear reference: mean monthly ratio0.738303, pooledMAE4.273508 in results/normalized_validation_20261017/REPORT.md. It comes from the earlier broader search. Current paired uniform/ratio models share candidate spaces and are the appropriate control for the objective change.','',
                  'Reproduction: campaigns/monthly_ratio_20261017/run.py; fixed budget and constraints in campaigns/monthly_ratio_20261017/PROTOCOL.md. Per-fold models, predictions, selections, source hashes, monthly_scores.csv, summary.csv and audit.json are retained. No deployment was replaced.']
        # Avoid an optional report-format dependency in unattended finalization.
        table=actual.pivot(index='month',columns='recipe',values='ratio').round(6)
        markdown='| Month | '+' | '.join(table.columns)+' |\n|---|'+'---:|'*len(table.columns)+'\n'
        markdown+='\n'.join('| '+str(index)+' | '+' | '.join(f'{value:.6f}' for value in values)+' |' for index,values in zip(table.index,table.to_numpy()))
        # Replace the provisional pandas table with the dependency-free table.
        lines[lines.index('MONTHLY_TABLE')]=markdown
        (e.ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('AUDIT',json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args();audit(args.smoke)
