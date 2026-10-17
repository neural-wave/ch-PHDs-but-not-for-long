"""Small metadata-only follow-up on the exported conditional tables."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/interpretability_deep_20261017'
r=list(csv.DictReader((OUT/'conditional_summary.csv').open()))
m=list(csv.DictReader((OUT/'conditional_monthly.csv').open()))
months=sorted(set(v['month'] for v in m));assert len(months)==8
zones=[f'ZONE_{z}' for z in range(4)];shapes=['Flat','Rising','Falling','Peak','Trough']
recipes=json.loads((ROOT/'results/model_ranking_20261017/champion_per_fold.json').read_text())['folds']
out=[]
for z in zones:
    a=[v for v in r if v['group']=='Zone' and v['value']==z]
    best=min([v for v in a if v['family']!='blend'],key=lambda v:float(v['equal_month_ratio']))
    blend=next(v for v in a if v['family']=='blend')
    wins=sum(min([v for v in m if v['group']=='Zone' and v['value']==z and v['month']==mo and v['family']!='blend'],key=lambda v:float(v['mae']))['family']==best['family'] for mo in months)
    standardized=[];cancellation=[]
    for mo in months:
        pooled=[v for v in m if v['month']==mo and v['group']=='Curve' and v['family']=='blend']
        weights={v['value']:int(v['hours'])/sum(int(t['hours']) for t in pooled) for v in pooled}
        subset={v['value'].split('/')[1]:v for v in m if v['month']==mo and v['group']=='ZoneCurve' and v['family']=='blend' and v['value'].startswith(z+'/')}
        assert set(subset)==set(shapes)
        den=sum(weights[s]*float(subset[s]['baseline_mae']) for s in shapes)
        standardized.append(sum(weights[s]*float(subset[s]['mae']) for s in shapes)/den)
        wr=next(v['family_weights'] for v in recipes if v['test_month']==mo)
        zone_rows={v['family']:v for v in m if v['month']==mo and v['group']=='Zone' and v['value']==z}
        weighted=sum(w*float(zone_rows[n.removeprefix('fixed_').removesuffix('_ratio')]['mae']) for n,w in wr.items())
        cancellation.append(weighted-float(zone_rows['blend']['mae']))
    out.append(dict(zone=z,best_single=best['family'],single_ratio=float(best['equal_month_ratio']),blend_ratio=float(blend['equal_month_ratio']),
        best_single_month_wins=wins,equal_month_shape_standardized_ratio=sum(standardized)/8,
        equal_month_cancellation_eur=sum(cancellation)/8,
        curve_mix_percent={s:100*int(next(v['hours'] for v in r if v['group']=='ZoneCurve' and v['value']==z+'/'+s and v['family']=='blend'))/int(blend['hours']) for s in shapes}))
(OUT/'zone_specialists.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
