"""Reload the frozen winning artifact and evaluate every September delivery date."""
import csv,hashlib,json,pickle,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'catboost_vendor'))
import numpy as np
from reconstruction import read_targets,rich_features
from predict import bundle_prediction

root=Path('results');selection=json.loads((root/'monthly_best.json').read_text())
assert selection['selected_by']=='august_validation_mae'
assert selection['best_directory']==min(selection['validation_scores'],key=selection['validation_scores'].get)
directory=root/selection['best_directory'];report=json.loads((directory/'metrics.json').read_text())
assert report['split_counts']==dict(train=302,validation=31,test=30)
assert report['model_fit_days']==302 and report['final_refit_deferred']
assert report['split']['train']==['2025-10-01','2026-07-31']
assert report['split']['validation']==['2026-08-01','2026-08-31']
assert report['split']['test']==['2026-09-01','2026-09-30']
dates,y=read_targets('train_prices_15min.csv');h=y.mean(-1)
assert hashlib.sha256(Path('train_prices_15min.csv').read_bytes()).hexdigest()==report['source_sha256']
with (directory/'model.pkl').open('rb') as f:bundle=pickle.load(f)
# Confirm normalization is derived from the 302 training days at every nesting level.
train_x=rich_features(dates[:302],h[:302],dates[:302],y[:302]);mu=train_x.mean(0);sd=train_x.std(0).clip(.05)
levels=0;forest_roots=[]
def inspect(obj):
    global levels
    levels+=1
    assert np.allclose(obj['mu'],mu,atol=1e-6) and np.allclose(obj['sd'],sd,atol=1e-6)
    for item in obj['models']:
        if item['name']=='parent_ensemble':inspect(item['artifact'])
        elif item['name'].startswith('trees'):
            count=int(item['artifact'].estimators_[0].tree_.n_node_samples[0])
            assert count==302*96
            forest_roots.append(count)
inspect(bundle)
predictions=[]
for i in range(333,363):
    # The frozen model sees current hourly curves plus strictly earlier quarter history.
    pp=bundle_prediction(bundle,dates[i:i+1],h[i:i+1],dates[:i],y[:i]).reshape(4,24,4)
    predictions.append(pp)
pred=np.asarray(predictions)
cache=np.load(directory/'test_predictions.npz',allow_pickle=False)
assert np.array_equal(cache['dates'],dates[333:]) and np.array_equal(cache['observed'],y[333:])
max_difference=float(np.abs(pred-cache['predicted']).max())
numerical=dict(max_difference=max_difference,mean_absolute_difference=float(np.abs(pred-cache['predicted']).mean()),
    recomputed_mae=float(np.abs(pred-y[333:]).mean()),cached_mae=report['test']['mae'],
    percentile99_difference=float(np.quantile(np.abs(pred-cache['predicted']),.99)))
np.savez_compressed(directory/'audit_cpu_predictions.npz',predicted=pred,observed=y[333:],dates=dates[333:].astype('U10'))
(directory/'numerical_diagnostics.json').write_text(json.dumps(numerical,indent=2)+'\n')
print('NUMERICAL DIAGNOSTICS',json.dumps(numerical),flush=True)
# cuDNN TF32 in the original cached graph run causes sub-cent pointwise drift.
# Keep score agreement strict and verify the independent hourly constraint below.
assert max_difference<0.01,max_difference
mae=float(np.abs(pred-y[333:]).mean())
assert abs(mae-report['test']['mae'])<1e-5
csv_path=directory/'verified_september_predictions.csv'
with csv_path.open('w',newline='') as f:
    writer=csv.writer(f);writer.writerow(['delivery_date','market_id','hour_index','quarter_index','predicted_price_eur_mwh'])
    for day,date in enumerate(dates[333:]):
        for z in range(4):
            for hr in range(24):
                for q in range(4):writer.writerow([date,'ZONE_'+str(z),hr+1,q+1,f'{pred[day,z,hr,q]:.10f}'])
rows=list(csv.DictReader(csv_path.open()))
assert len(rows)==30*4*24*4
keys=[(r['delivery_date'],r['market_id'],int(r['hour_index']),int(r['quarter_index'])) for r in rows]
assert len(set(keys))==len(rows)
rounded=np.array([float(r['predicted_price_eur_mwh']) for r in rows]).reshape(30,4,24,4)
assert np.isfinite(rounded).all()
mean_error=float(np.abs(rounded.mean(-1)-h[333:]).max());assert mean_error<1e-4
sha=hashlib.sha256()
with (directory/'model.pkl').open('rb') as f:
    for chunk in iter(lambda:f.read(1024*1024),b''):sha.update(chunk)
result=dict(status='passed',selected_directory=directory.name,selection_basis='August MAE only',days=30,
    rows=len(rows),base_model_fit_days=302,normalization_days=302,inspected_bundle_levels=levels,
    forest_training_root_counts=forest_roots,artifact_sha256=sha.hexdigest(),
    september_mae=mae,reported_september_mae=report['test']['mae'],
    max_cached_vs_reloaded_prediction_difference=max_difference,max_exported_hourly_mean_error=mean_error,
    history='Current hourly curves and strictly earlier observed quarter history, sequential per delivery date',
    numerical_tolerance_eur_mwh=0.01,numerical_reason='Original GPU graph cache used cuDNN TF32; precision_diagnostics.json isolates the CPU/GPU drift',
    caveat='September was inspected during earlier development; this is not a fresh blind test.')
(directory/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
