"""Grouped permutation reliance of selected temporal CNNs and their champion blend."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'campaigns/minimize_expanding_20261017'))
sys.path.insert(0, str(ROOT / 'code'))
import train as e
from analyze_feature_importance import GROUPS
import numpy as np
import torch

SOURCE = ROOT / 'results/minimize_expanding_20261017'
OUT = ROOT / 'results/champion_lag_importance_20261017'


def main():
    e.p.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    groups = {k: v for k, v in GROUPS.items() if k not in ['zone_identity', 'hour_identity']}
    groups['interpolation_lags1_2_7_14'] = list(range(345, 361))
    assert sorted(c for v in list(GROUPS.values())[:10] for c in v) == list(range(389))
    protocol = dict(groups=groups, repeats=20, seed=20261017, fitting=False,
                    method='Whole-day feature-group permutation within each full test month; all zones/hours move together. Only selected temporal CNN inputs change; all other experts and interpolation anchor remain fixed.',
                    scope='Eight-fold research champion; February has no temporal CNN and has zero ensemble effect. Standalone CNN aggregates use only participating folds.',
                    limitations='Developmental model reliance, not causal importance or retrained ablation. Correlated groups overlap and effects are not additive. Shuffled combinations can be out of distribution. No recalibration or submission changes.')
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    rows, audits = [], []
    for folder in sorted((SOURCE / 'folds').iterdir()):
        if not folder.is_dir():
            continue
        d = np.load(folder / 'data.npz')
        saved = np.load(folder / 'selected/fixed_blend_ratio/predictions.npz')
        sel = json.loads((folder / 'selection.json').read_text())['by_recipe']
        nt = int(d['nt'])
        dates = d['dates'][nt:]
        test = np.flatnonzero(np.isin(dates, saved['dates']))
        np.testing.assert_array_equal(dates[test], saved['dates'])
        anchor = d['b'][nt:]
        truth = d['y'][nt:][test]
        curve = torch.as_tensor(d['curve'][nt:], device='cuda')
        total = np.zeros(anchor.shape + (11,))
        cnn_total = np.zeros_like(total)
        cnns, cnn_weight = [], 0.
        with torch.no_grad():
            for family, weight in sel['fixed_blend_ratio']['weights'].items():
                for cid, seedweight in sel[family]['components'].items():
                    cp = torch.load(folder / 'trials' / cid / 'model.pt', weights_only=False, map_location='cpu')
                    net = e.Quantiles(cp['config'], d).cuda().eval()
                    net.load_state_dict(cp['state'])
                    w = weight * seedweight
                    x = torch.as_tensor(d[net.feature][nt:], device='cuda')
                    pred = e.physical(net(x, curve).cpu().numpy(), anchor)
                    total += w * pred
                    if cp['config']['family'] == 'cnn_rich':
                        cnns.append((net, w))
                        cnn_total += w * pred
                        cnn_weight += w
            replay = float(abs(total[test] - saved['quantiles']).max())
            assert replay < 1e-6, (folder.name, replay)
            baseline = abs(total[test, ..., 5] - truth).mean((1, 2, 3))
            denominator = float(abs(anchor[test] - truth).mean())
            audits.append(dict(month=folder.name, replay_error=replay, cnn_weight=cnn_weight))
            if not cnns:
                for group in groups:
                    rows.append(dict(month=folder.name, group=group, cnn_present=False,
                                     ensemble_delta_mae=0., ensemble_delta_ratio=0.))
                continue
            x0 = d['rich'][nt:]
            assert x0.shape[-1] == 389
            cnn_daily = abs(cnn_total[test, ..., 5] / cnn_weight - truth).mean((1, 2, 3))
            rng = np.random.default_rng(20261017)
            perms = [rng.permutation(len(test)) for _ in range(20)]
            for group, cols in groups.items():
                effects, cnn_effects = [], []
                for perm in perms:
                    altered = x0.copy()
                    donor = x0[test[perm]]
                    for c in cols:
                        altered[test, :, :, c] = donor[:, :, :, c]
                    x = torch.as_tensor(altered, device='cuda')
                    changed = np.zeros_like(total)
                    for net, w in cnns:
                        changed += w * e.physical(net(x, curve).cpu().numpy(), anchor)
                    new = (total - cnn_total + changed)[test]
                    assert np.isfinite(new).all() and np.diff(new, axis=-1).min() >= -1e-8
                    assert abs(new[..., 5].mean(-1) - anchor[test].mean(-1)).max() < 1e-4
                    effects.append(abs(new[..., 5] - truth).mean((1, 2, 3)) - baseline)
                    cnn_effects.append(abs(changed[test, ..., 5] / cnn_weight - truth).mean((1, 2, 3)) - cnn_daily)
                delta = np.asarray(effects)
                rows.append(dict(month=folder.name, group=group, cnn_present=True,
                                 ensemble_delta_mae=float(delta.mean()), ensemble_delta_ratio=float(delta.mean()/denominator),
                                 cnn_delta_mae=float(np.mean(cnn_effects)), permutation_sd=float(delta.mean(1).std()),
                                 daily_delta_mae=delta.mean(0).tolist()))
            (OUT / 'progress.json').write_text(json.dumps(dict(completed_month=folder.name)) + '\n')
            print('COMPLETED', folder.name, flush=True)
        del cnns
        torch.cuda.empty_cache()
    summary = []
    for group in groups:
        rr = [r for r in rows if r['group'] == group]
        present = [r for r in rr if r['cnn_present']]
        assert len(rr) == 8
        summary.append(dict(group=group, ensemble_delta_mae=float(np.mean([r['ensemble_delta_mae'] for r in rr])),
                            ensemble_delta_ratio=float(np.mean([r['ensemble_delta_ratio'] for r in rr])),
                            cnn_delta_mae=float(np.mean([r['cnn_delta_mae'] for r in present])), cnn_months=len(present)))
    (OUT / 'results.json').write_text(json.dumps(dict(summary=summary, monthly=rows, audits=audits, passed=True), indent=2) + '\n')
    lines = ['# Champion temporal-CNN feature reliance', '', protocol['method'], '', protocol['limitations'], '',
             '| Group | Ensemble MAE change | Equal-month ratio change | CNN MAE change (participating months) |', '|---|---:|---:|---:|']
    for r in sorted(summary, key=lambda v: -v['ensemble_delta_ratio']):
        lines.append(f"| {r['group']} | {r['ensemble_delta_mae']:+.6f} | {r['ensemble_delta_ratio']:+.6f} | {r['cnn_delta_mae']:+.6f} |")
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('AUDIT PASSED', flush=True)


if __name__ == '__main__':
    main()
