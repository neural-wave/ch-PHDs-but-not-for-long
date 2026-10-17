"""Dependency-free deterministic reconstruction; run: python3 evaluate_baselines.py."""
import csv, math, json, hashlib
from collections import defaultdict
from pathlib import Path
root = Path(__file__).resolve().parent
source = root / 'train_prices_15min.csv'
groups = defaultdict(dict)
with source.open(newline='') as f:
    for row in csv.DictReader(f):
        key = (row['delivery_date'], row['market_id'])
        h, q = int(row['hour_index']), int(row['quarter_index'])
        assert 1 <= h <= 24 and 1 <= q <= 4
        assert (h,q) not in groups[key], 'Duplicate quarter'
        value = float(row['price_eur_mwh'])
        assert math.isfinite(value)
        groups[key][h,q] = value
assert len(groups) == 363 * 4
methods = ['hourly_mean', 'linear_interpolation', 'linear_interpolation_mean_corrected']
stats = defaultdict(list)
daily = []
max_violation = dict.fromkeys(methods, 0.0)
(root/'results').mkdir(exist_ok=True)
with (root/'results/baseline_predictions.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['delivery_date','market_id','hour_index','quarter_index','observed_price_eur_mwh']+methods)
    for (date,zone), values in sorted(groups.items()):
        assert len(values) == 96
        y = [values[h,q] for h in range(1,25) for q in range(1,5)]
        hourly = [sum(y[4*h:4*h+4])/4 for h in range(24)]
        # Hour centres are h+0.5, quarter centres are h+(q+0.5)/4.
        linear = []
        for h in range(24):
            for q in range(4):
                pos = h+(q+0.5)/4-0.5
                left = math.floor(pos)
                if pos <= 0: pred = hourly[0]
                elif pos >= 23: pred = hourly[-1]
                else: pred = hourly[left]*(1-(pos-left))+hourly[left+1]*(pos-left)
                linear.append(pred)
        flat = [hourly[h] for h in range(24) for _ in range(4)]
        corrected = [linear[4*h+q]+hourly[h]-sum(linear[4*h:4*h+4])/4 for h in range(24) for q in range(4)]
        predictions = dict(zip(methods,[flat,linear,corrected]))
        for method,pred in predictions.items():
            mae = sum(abs(a-b) for a,b in zip(y,pred))/96
            mse = sum((a-b)**2 for a,b in zip(y,pred))/96
            violation = max(abs(sum(pred[4*h:4*h+4])/4-hourly[h]) for h in range(24))
            max_violation[method] = max(max_violation[method],violation)
            stats[method,zone].append((mae,mse))
            daily.append([date,zone,method,mae,math.sqrt(mse),violation])
        for i,obs in enumerate(y):
            writer.writerow([date,zone,i//4+1,i%4+1,obs]+[predictions[m][i] for m in methods])
with (root/'results/baseline_daily_metrics.csv').open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['delivery_date','market_id','method','mae_eur_mwh','rmse_eur_mwh','max_hourly_mean_error']); w.writerows(daily)
summary = {'rows':len(groups)*96,'days':363,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'methods':{}}
for method in methods:
    by_zone = {z:sum(a for a,b in stats[method,z])/363 for z in sorted({z for _,z in groups})}
    summary['methods'][method] = {'mae_eur_mwh':sum(by_zone.values())/4,'mae_by_zone':by_zone,'rmse_eur_mwh':math.sqrt(sum(b for (m,z), pairs in stats.items() if m==method for a,b in pairs)/(363*4)),'max_hourly_mean_error':max_violation[method],'challenge_mean_constraint_pass':max_violation[method]<=0.0001}
(root/'results/baseline_metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
