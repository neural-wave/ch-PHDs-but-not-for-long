"""Recompute the completed selector comparison and plot expanding-fold errors."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / 'results/minimize_expanding_20261017'
OLD = PROJECT / 'results/monthly_quantiles_20261017'


def main():
    # This is reporting only: never use incomplete runs to choose new recipes.
    completion = json.loads((ROOT / 'completion.json').read_text())
    audit = json.loads((ROOT / 'audit.json').read_text())
    assert completion['complete'] and completion['audit_passed'] and audit['passed']
    assert audit['folds'] == 8 and audit['candidate_predictions_recomputed'] == 384
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for name, sha in manifest['code_hashes'].items():
        assert hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == sha, name
    assert hashlib.sha256((PROJECT / 'train_prices_15min.csv').read_bytes()).hexdigest() == manifest['source_sha256']
    monthly = pd.read_csv(ROOT / 'monthly_scores.csv')
    original = pd.read_csv(OLD / 'monthly_scores.csv')
    rows = []; interval_rows = []
    for fold in manifest['folds']:
        month = fold['test']; folder = ROOT / 'folds' / month
        data = np.load(folder / 'data.npz')
        meta = json.loads((folder / 'manifest.json').read_text())
        nt, ns, nc, ne = (int(data[k]) for k in ['nt', 'ns', 'nc', 'ne'])
        truth, hourly = data['y'][nt+ns+nc:], data['h'][nt+ns+nc:]
        unc = np.asarray([[np.interp(np.arange(96)/4+.125, np.arange(24)+.5, zone).reshape(24, 4)
                           for zone in day] for day in hourly])
        anchor = unc-unc.mean(-1, keepdims=True)+hourly[..., None]
        denominator = float(np.abs(anchor-truth).mean())
        assert denominator > 0
        before = np.load(OLD / 'folds' / month / 'selected/validation_selector/predictions.npz')
        after = np.load(folder / 'selected/validation_selector/predictions.npz')
        for item in [before, after]:
            np.testing.assert_array_equal(item['observed'], truth)
            np.testing.assert_array_equal(item['dates'], data['dates'][nt+ns+nc:])
            np.testing.assert_allclose(item['quantiles'][..., 5].mean(-1), hourly, rtol=0, atol=1e-8)
            assert item['quantiles'].shape == truth.shape+(11,)
            assert np.diff(item['quantiles'], axis=-1).min() >= -1e-9
        old_mae = float(np.abs(before['quantiles'][..., 5]-truth).mean())
        new_mae = float(np.abs(after['quantiles'][..., 5]-truth).mean())
        for frame, mae in [(original, old_mae), (monthly, new_mae)]:
            saved = frame[(frame.month == month) & (frame.recipe == 'validation_selector')].iloc[0]
            np.testing.assert_allclose([saved.mae, saved.ratio], [mae, mae/denominator], rtol=0, atol=1e-8)
        chosen = json.loads((folder / 'selection.json').read_text())['by_recipe']['validation_selector']
        row = dict(test_month=month, train_start=meta['train_start'], train_end=meta['train_end'],
                   training_months=len(meta['training_month_baselines']), training_days=nt,
                   selection_days=ns, calibration_days=nc, test_days=ne,
                   interpolation_mae=denominator, original_selector_mae=old_mae, new_selector_mae=new_mae,
                   original_selector_ratio=old_mae/denominator, new_selector_ratio=new_mae/denominator,
                   ratio_difference=(new_mae-old_mae)/denominator,
                   relative_mae_change=new_mae/old_mae-1,
                   chosen_recipe=chosen['chosen_recipe'])
        rows.append(row)
        prior_lower = prior_upper = None
        for coverage in [50, 80, 90, 95]:
            lower, upper = after[f'lower_{coverage}'], after[f'upper_{coverage}']
            assert np.isfinite(lower).all() and np.isfinite(upper).all() and (upper >= lower).all()
            if prior_lower is not None:
                assert (lower <= prior_lower+1e-9).all() and (upper >= prior_upper-1e-9).all()
            interval_rows.append(dict(test_month=month, test_days=ne, nominal=coverage,
                                      coverage=float(((truth >= lower) & (truth <= upper)).mean()),
                                      width=float((upper-lower).mean())))
            prior_lower, prior_upper = lower, upper
    table = pd.DataFrame(rows)
    assert table.training_months.tolist() == list(range(3, 11))
    assert int(table.test_days.sum()) == 241
    table.to_csv(ROOT / 'selector_comparison_by_training_size.csv', index=False)
    intervals = pd.DataFrame(interval_rows)
    intervals.to_csv(ROOT / 'selector_interval_comparison.csv', index=False)
    before = float(table.original_selector_ratio.mean())
    after = float(table.new_selector_ratio.mean())
    comparison = dict(original_selector_ratio=before, new_selector_ratio=after,
                      relative_ratio_change=after/before-1,
                      months_improved=int((table.ratio_difference < 0).sum()),
                      months_worsened=int((table.ratio_difference > 0).sum()),
                      months_tied=int((table.ratio_difference.abs() < 1e-12).sum()),
                      original_pooled_mae=float(np.average(table.original_selector_mae, weights=table.test_days)),
                      new_pooled_mae=float(np.average(table.new_selector_mae, weights=table.test_days)),
                      worst_month_ratio=float(table.new_selector_ratio.max()),
                      interpolation_error_reduction=1-after, folds=8, expanding_training_months=[3, 10],
                      target_quarters=92544, developmental=True,
                      interpretation='Training size and calendar month vary together; this is not an isolated training-size ablation.')
    for coverage in [50, 80, 90, 95]:
        group = intervals[intervals.nominal == coverage]
        comparison[f'coverage_{coverage}'] = float(np.average(group.coverage, weights=group.test_days))
        comparison[f'width_{coverage}'] = float(np.average(group.width, weights=group.test_days))
    (ROOT / 'verified_selector_comparison.json').write_text(json.dumps(comparison, indent=2)+'\n')
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [2.2, 1]}, sharex=True)
    xx = np.arange(8)
    axes[0].axhline(1., color='#888888', ls='--', lw=1, label='Corrected linear interpolation')
    axes[0].plot(xx, table.original_selector_ratio, 'o-', color='#6c7b95', label='Original pre-test selector')
    axes[0].plot(xx, table.new_selector_ratio, 'o-', color='#15736b', label='New pre-test selector')
    axes[0].set_ylabel('Monthly MAE / interpolation MAE')
    axes[0].set_title(f'Expanding training: mean ratio {before:.4f} → {after:.4f}')
    axes[0].legend(frameon=False, ncol=1, loc='upper right', bbox_to_anchor=(1, .94)); axes[0].grid(axis='y', alpha=.2)
    axes[1].bar(xx, table.ratio_difference, color=np.where(table.ratio_difference < 0, '#15736b', '#bf654e'))
    axes[1].axhline(0., color='#888888', lw=1); axes[1].set_ylabel('New − original ratio')
    axes[1].set_xticks(xx, [f"{r.test_month}\n{r.training_months} months\n{r.training_days} days"
                          for r in table.itertuples()])
    axes[1].set_xlabel('Test month and expanding training history')
    fig.text(.09, .015, 'Developmental walk-forward tests. Training size changes with calendar month; all choices use prior data.',
             fontsize=9, color='#666666')
    fig.tight_layout(rect=(0, .035, 1, 1))
    fig.savefig(ROOT / 'expanding_selector_comparison.png', dpi=180)
    fig.savefig(ROOT / 'expanding_selector_comparison.svg')
    plt.close(fig)
    print(json.dumps(comparison, indent=2))


if __name__ == '__main__':
    main()
