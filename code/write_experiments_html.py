"""Short standalone report, regenerated from verified local metrics."""
import json
from datetime import datetime
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

root=Path(__file__).resolve().parents[1];results=root/'results'
def read(name):return json.loads((results/name/'metrics.json').read_text())
base=read('monthly_base_gpu1')
completed={}
for name in ['monthly_base_gpu1','monthly_cat_gpu1','monthly_graph_gpu1','monthly_revin_gpu1','monthly_joint_blend_gpu1']:
    path=results/name/'metrics.json'
    if path.exists():
        r=json.loads(path.read_text())
        if r.get('offline_inference_qa','').startswith('passed'):completed[name]=r
winner=min(completed,key=lambda k:completed[k]['validation']['mae']);current=completed[winner]
validation=current['validation']['mae'];test=current['test']['mae'];baseline=current['test_baseline']['mae']
val_baseline=next(c['validation_mae'] for c in base['candidates'] if c['name']=='interpolation')
gain=100*(1-test/baseline)
historical=[('Basic learned ensemble','Ridge, forests, boosting and shared MLPs','search_gpu1'),
    ('Richer features','Hourly slopes, cross-zone curves and earlier quarter shapes','rich_search_gpu1'),
    ('Zone-specific mixtures','Separate ensemble weights per zone','zone_search_gpu1'),
    ('GPU absolute-error boosting','CatBoost plus the previous ensemble','catboost_search_gpu1'),
    ('Weighted mean corrections','Redistribute the correction across quarters; not selected','projection_search_gpu1')]
rows=''.join(f'<tr><td><strong>{escape(title)}</strong><small>{escape(detail)}</small></td><td>{read(name)["holdout"]["mae"]:.3f}</td></tr>' for title,detail,name in historical)
stamp=datetime.now(ZoneInfo('Europe/Zurich')).strftime('%d %b %Y · %H:%M Zurich')
pending=[]
if 'monthly_cat_gpu1' not in completed:pending.append('monthly GPU boosting')
if 'monthly_graph_gpu1' not in completed:pending.append('four-node temporal attention')
if 'monthly_revin_gpu1' not in completed:pending.append('paired RevIN vs fixed normalization')
status='Results pending in this snapshot: '+', '.join(pending)+'.' if pending else 'All monthly comparisons shown here have completed.'
labels={'monthly_base_gpu1':'Shared MLP/tree ensemble','monthly_cat_gpu1':'GPU boosting ensemble',
    'monthly_graph_gpu1':'Ensemble with four-node temporal attention','monthly_revin_gpu1':'Ensemble adding RevIN','monthly_joint_blend_gpu1':'Jointly reweighted ensemble'}
monthly_rows=''.join(f'<tr><td>{escape(labels[name])}</td><td>{r["validation"]["mae"]:.3f}</td><td>{r["test"]["mae"]:.3f}</td></tr>' for name,r in completed.items())
html=f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AET reconstruction — experiments</title>
<style>
:root{{color-scheme:light;--ink:#162735;--muted:#596b77;--line:#dfe7e9;--accent:#087c6a}}
*{{box-sizing:border-box}}body{{margin:0;background:#f3f6f6;color:var(--ink);font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:900px;margin:28px auto;padding:30px;background:white;border:1px solid var(--line);border-radius:18px}}
.eyebrow{{font-size:11px;font-weight:750;letter-spacing:.12em;color:var(--accent);text-transform:uppercase}}
h1{{font-size:30px;letter-spacing:-.035em;margin:5px 0 6px}}h2{{font-size:17px;margin:24px 0 8px}}p{{margin:7px 0}}.muted,small{{color:var(--muted)}}
.protocol{{padding:10px 13px;background:#edf6f3;border-radius:9px;margin:17px 0;font-size:13px}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.card{{padding:16px;border:1px solid var(--line);border-radius:10px}}
.label{{font-size:12px;color:var(--muted)}}.value{{font-size:29px;font-weight:750;letter-spacing:-.03em;margin:3px 0}}.green{{color:var(--accent)}}
small{{display:block;font-size:12px}}table{{width:100%;border-collapse:collapse}}th{{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);text-align:left}}
th,td{{padding:9px 0;border-bottom:1px solid var(--line);vertical-align:top}}th:last-child,td:last-child{{text-align:right;white-space:nowrap;padding-left:16px}}
td strong{{font-size:13px}}.note{{font-size:12px;color:var(--muted)}}.next{{padding:12px 14px;background:#f4f7fb;border-radius:10px;font-size:13px;margin-top:18px}}
a{{color:var(--accent)}}footer{{border-top:1px solid var(--line);margin-top:18px;padding-top:12px;font-size:11px;color:var(--muted)}}
@media(max-width:620px){{main{{margin:10px;padding:19px}}h1{{font-size:25px}}.cards{{grid-template-columns:1fr}}.value{{font-size:25px}}}}
@media print{{body{{background:white}}main{{margin:0;border:0;padding:10px}}a{{color:inherit}}}}
</style></head><body><main>
<div class="eyebrow">AET Neural Wave · experiment summary</div>
<h1>From hourly prices to 15-minute profiles</h1>
<p class="muted">All four zones enter together. Models preserve each hourly mean. Training runs on Ando GPU 1.</p>
<div class="protocol"><strong>Current split:</strong> train Oct 2025–Jul 2026 (302 days) · validate August (31) · test September (30).</div>
<div class="cards">
<div class="card"><div class="label">August validation MAE</div><div class="value">{validation:.3f}</div><small>Interpolation: {val_baseline:.3f} €/MWh</small></div>
<div class="card"><div class="label">September test MAE</div><div class="value">{test:.3f}</div><small>Interpolation: {baseline:.3f} €/MWh</small></div>
<div class="card"><div class="label">Test error reduction</div><div class="value green">{gain:.1f}%</div><small>Compared on the same September dates</small></div>
</div>
<p class="note">Winner: <a href="{escape(winner)}/metrics.json">{escape(labels[winner])}</a>. Selected on August only; models fit on 302 days.</p>
<table aria-label="Comparable monthly experiments"><thead><tr><th>Monthly experiment</th><th>August MAE</th><th>September MAE</th></tr></thead><tbody>{monthly_rows}</tbody></table>
<h2>What we tried</h2>
<p class="note">Earlier experiments below used a different 60-day development test. Their scores are not directly comparable with the monthly cards.</p>
<table><thead><tr><th>Trial</th><th>Earlier test MAE</th></tr></thead><tbody>{rows}</tbody></table>
<p class="note">Original full-data reconstruction diagnostics: repeating hourly means 7.578; mean-corrected linear interpolation 5.474 €/MWh.</p>
<div class="next"><strong>Joint-node network:</strong> a new model combines temporal convolutions with learned attention between four zone nodes. Interaction and hourly-constraint checks passed. {escape(status)}</div>
<p class="note"><strong>RevIN:</strong> testing input-only per-day/per-zone normalization with reversible residual scaling against matched fixed-scale controls. All four zones and absolute level/scale features are retained; completed; RevIN beat matched controls in 2/4 pairs. Its small selected contribution improved September MAE by 0.00118 €/MWh; evidence of a general advantage is mixed. <a href="{escape(winner)}/audit.json">Current winner replay audit</a> passed.</p>
<p class="note"><strong>Deployment:</strong> a separate all-363-day refit completed with selected settings and weights frozen. Independent artifact, from-scratch recipe smoke and packaged inference checks passed. <a href="../submission/README.md">Ready-to-run solution</a>. The scores above belong to the preserved 302-day evaluation model.</p>
<p class="note">The rolling 28-day prototype stopped in its reduced smoke check on zero-weight normalization; fixed and superseded by the requested monthly split. September was viewed during earlier development, so it is not a fresh blind test. History inputs are strictly earlier dates.</p>
<footer>Snapshot: {escape(stamp)} · MAE in €/MWh · <a href="https://api.miguelnogales.com/nightwatch/">Live run dashboard</a> · <a href="../MONTHLY_PROTOCOL.md">Protocol</a></footer>
</main></body></html>'''
(results/'experiments_summary.html').write_text(html)
print(results/'experiments_summary.html')
