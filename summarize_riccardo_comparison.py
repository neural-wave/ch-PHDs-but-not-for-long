"""Add matched-month conclusions and separately scoped September comparisons."""
import csv
import json
from pathlib import Path
from statistics import fmean

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/riccardo_comparison_20261017'
OLD=ROOT/'results/normalized_validation_20261017'


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    assert json.loads((OUT/'independent_audit.json').read_text())['passed']
    compared=rows(OUT/'comparison.csv')
    allmonths=rows(OLD/'monthly_metrics.csv')+rows(OUT/'monthly_metrics.csv')
    lookup={(r['method'],r['test_month']):r for r in allmonths}
    ours=next(r for r in compared if r['method']=='linear')
    effects=[]
    for method in ['riccardo_reduced','riccardo_raw_hourly']:
        r=next(r for r in compared if r['method']==method)
        months=[f'2026-{m:02d}' for m in range(2,10)]
        effects.append(dict(method=method,
            rank_by_primary_score=next(i+1 for i,x in enumerate(compared) if x['method']==method),
            monthly_ratio_increase_vs_our_linear_percent=100*(float(r['mean_monthly_ratio'])/float(ours['mean_monthly_ratio'])-1),
            pooled_mae_increase_vs_our_linear_percent=100*(float(r['pooled_mae'])/float(ours['pooled_mae'])-1),
            months_beating_our_linear=sum(float(lookup[method,m]['mae'])<float(lookup['linear',m]['mae']) for m in months),
            months_beating_interpolation=sum(float(lookup[method,m]['mae_over_interpolation_mae'])<1 for m in months)))
    september=[]
    for r in allmonths:
        if r['test_month']=='2026-09':
            september.append(dict(method=r['method'],mae=float(r['mae']),
                ratio=float(r['mae_over_interpolation_mae']),selection='August 1–15; campaign budgets/configuration provenance differ',
                scope='September from eight-fold campaign'))
    for r in rows(OLD/'september_only.csv'):
        september.append(dict(method=r['method'],mae=float(r['mae']),ratio=float(r['mae_over_interpolation_mae']),
            selection=r['selection'],scope='September-only study; no eight-month estimate'))
    september.sort(key=lambda r:r['mae'])
    with (OUT/'september_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(september[0]));writer.writeheader();writer.writerows(september)
    (OUT/'conclusions.json').write_text(json.dumps(dict(effects_vs_our_linear=effects,september=september),indent=2)+'\n')
    appendix=['','## Direct comparison with our existing linear pipeline','',
        '| Imported variant | Rank among 11 pipelines | Change in primary ratio vs our linear | Change in pooled MAE | Months beating our linear |',
        '|---|---:|---:|---:|---:|']
    for r in effects:
        appendix.append(f"| {r['method']} | {r['rank_by_primary_score']} | {r['monthly_ratio_increase_vs_our_linear_percent']:+.2f}% | {r['pooled_mae_increase_vs_our_linear_percent']:+.2f}% | {r['months_beating_our_linear']}/8 |")
    appendix+=['','Positive changes mean higher error. Month wins are descriptive; dependent developmental folds do not support independent-fold inference.',
        '', 'Execution uses our existing CUDA/PyTorch 2.14.1+cu130 research environment. The upstream requirements pin PyTorch 2.8.0 and the committed run reports Apple MPS. Runtime/backend changes can affect optimization numerics; the smoke parity check compares original and adapted reduced-feature training in the same CUDA environment. We do not claim bitwise reproduction of the original MPS campaign.',
        '', '## September on identical dates, with selection differences retained','',
        'This table includes earlier September-only studies at the user’s request. All rows score the same September targets, but the joint blend uses full August selection and the campaigns differ in search scope; this is not a uniform selection-budget comparison. Single-month winners do not replace the primary eight-month ranking.',
        '', '| Method | September MAE EUR/MWh | September ratio | Evidence scope | Selection |',
        '|---|---:|---:|---|---|']
    for r in september:
        appendix.append(f"| {r['method']} | {r['mae']:.6f} | {r['ratio']:.6f} | {r['scope']} | {r['selection']} |")
    report=OUT/'REPORT.md'
    text=report.read_text().split('\n## Direct comparison with our existing linear pipeline')[0]
    report.write_text(text.rstrip()+'\n'+'\n'.join(appendix)+'\n')
    print(json.dumps(effects,indent=2))


if __name__=='__main__':
    main()
