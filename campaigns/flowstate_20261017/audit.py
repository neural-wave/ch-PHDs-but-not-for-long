"""Independent scoring, causal-input checks, and serialized-checkpoint replay."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from experiment import (ROOT, PROJECT, START, TRAIN_END, VAL_END, SOURCE_SHA,
                        ConditionalCNN, configure, history_context, inputs, raw_features)


def main():
    configure()
    source = PROJECT / 'train_prices_15min.csv'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA
    groups = {}
    with source.open() as f:
        for r in csv.DictReader(f):
            date = r['delivery_date']; zone = int(r['market_id'][-1])
            hour, quarter = int(r['hour_index']) - 1, int(r['quarter_index']) - 1
            key = (zone, hour, quarter)
            assert key not in groups.setdefault(date, {})
            groups[date][key] = float(r['price_eur_mwh'])
    dates = np.array(sorted(groups))
    y = np.array([[[[groups[d][z, h, q] for q in range(4)] for h in range(24)]
                   for z in range(4)] for d in dates])
    h = y.mean(-1)
    p = np.load(ROOT / 'predictions.npz')
    np.testing.assert_array_equal(p['dates'], dates[TRAIN_END:])
    metrics = json.loads((ROOT / 'metrics.json').read_text())
    selection = json.loads((ROOT / 'selection.json').read_text())
    results = {}
    for name, declared in metrics['methods'].items():
        pred = p[name]
        assert pred.shape == (61, 4, 24, 4) and np.isfinite(pred).all()
        error = np.abs(pred[31:] - y[VAL_END:])
        by_zone = error.mean((0, 2, 3))
        # Official metric is equal weight across all four zones.
        mae = float(by_zone.mean())
        assert abs(mae - declared['september_mae']) < 1e-10
        np.testing.assert_allclose(by_zone, declared['by_zone'], atol=1e-10, rtol=0)
        assert np.abs(pred.mean(-1) - h[TRAIN_END:]).max() < 1e-8
        results[name] = dict(mae=mae, by_zone=by_zone.tolist())
    assert abs(results['interpolation']['mae'] - 7.090321234809023) < 1e-10
    assert abs(results['hourly_mean']['mae'] - 10.69536631944445) < 1e-10
    raw_train = raw_features(START, VAL_END)
    raw_eval = raw_features(TRAIN_END, len(dates))
    mu = raw_train[:TRAIN_END-START].mean((0, 2), keepdims=True)
    sd = raw_train[:TRAIN_END-START].std((0, 2), keepdims=True).clip(.05)
    replay, sensitivities = {}, {}
    for arm, chosen in selection['by_arm'].items():
        candidates = [r for r in selection['candidates'] if int(r['flowstate']) == int(arm)]
        assert chosen == min(candidates, key=lambda r: r['validation_mae'])
        ck = torch.load(ROOT / chosen['folder'] / 'model.pt', map_location='cpu', weights_only=False)
        np.testing.assert_array_equal(mu, ck['mu']); np.testing.assert_array_equal(sd, ck['sd'])
        x = np.clip((raw_eval - mu) / sd, -15, 15).astype('float32')
        for i in range(7):
            mask = raw_eval[:, 23 + 9*i + 8:23 + 9*i + 9]
            x[:, 23 + 9*i:23 + 9*i + 8] *= mask
            x[:, 23 + 9*i + 8:23 + 9*i + 9] = mask
        if arm == '0': x[:, -8:] = 0
        net = ConditionalCNN().cuda().eval(); net.load_state_dict(ck['state'])
        with torch.inference_mode():
            residual = net(torch.tensor(x, device='cuda')).cpu().numpy().astype('float64')
        residual -= residual.mean(-1, keepdims=True)
        name = 'conditional_flowstate' if arm == '1' else 'conditional_control'
        # Rebuild the deterministic interpolation anchor independently.
        anchor = np.array([[np.interp(np.arange(96)/4+.125, np.arange(24)+.5, zz)
                            for zz in hh] for hh in h[TRAIN_END:]]).reshape(-1, 4, 24, 4)
        anchor += h[TRAIN_END:, ..., None] - anchor.mean(-1, keepdims=True)
        replay[name] = float(np.abs(anchor + 20*residual - p[name]).max())
        assert replay[name] < 1e-7
        aug_mae = float(np.abs(p[name][:31] - y[TRAIN_END:VAL_END]).mean())
        assert abs(aug_mae - chosen['validation_mae']) < 2e-5
        perturbed = x.copy(); perturbed[:, 1, 92:] += .5
        with torch.inference_mode():
            change = net(torch.tensor(perturbed, device='cuda')).cpu().numpy().astype('float64') - residual
        # Hour24 of Zone1 affects the shape at Hour1 of Zone0 through full_curve.
        curve_effect = float(np.abs(change[:, 0, 0]).max())
        assert curve_effect > 1e-7
        sensitivities[name] = dict(distant_hourly_curve_effect=curve_effect)
        if arm == '1':
            perturbed = x.copy(); perturbed[:, -8:] = 0
            with torch.inference_mode():
                ablated = net(torch.tensor(perturbed, device='cuda')).cpu().numpy()
            sensitivities[name]['flowstate_feature_effect'] = float(np.abs(ablated - residual).max())
            assert sensitivities[name]['flowstate_feature_effect'] > 1e-7
    i = VAL_END
    f = np.load(ROOT / 'forecasts' / f'{dates[i]}.npz')['forecast']
    original = inputs(dates[i], h[i], dates, y, f)
    changed = y.copy(); changed[i:] += np.arange(4)*54321
    np.testing.assert_array_equal(original, inputs(dates[i], h[i], dates, changed, f))
    np.testing.assert_array_equal(history_context(dates[i], dates, y), history_context(dates[i], dates, changed))
    np.testing.assert_array_equal(original, inputs(dates[i], h[i], dates[:i], y[:i], f))
    # Inference for September was only invoked after selection was persisted.
    assert all((ROOT/'forecasts'/f'{d}.npz').stat().st_mtime >= selection['frozen_at_unix'] for d in dates[VAL_END:])
    audit = dict(passed=True, methods=results, replay_max_error=replay, sensitivities=sensitivities,
                 checks=['official CSV hash', 'all30 September days/4zones/96quarters',
                         'independent equal-zone MAE', 'hourly conservation', 'train-only statistics',
                         'August-only choice', 'choice persisted before September inference',
                         'query/future target perturbation', 'calendar-preserving context',
                         'checkpoint replay', 'distant current-hourly curve changes within-hour shape'])
    (ROOT / 'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    lines = ['# Exploratory FlowState reconstruction — 17 October 2026', '',
             'External pretrained weights; excluded from the challenge submission. September is developmental, not blind.', '',
             '| Method | September MAE (EUR/MWh) |', '|---|---:|']
    lines += [f'| {name} | {row["mae"]:.6f} |' for name, row in results.items()]
    lines += ['', 'Both conditional CNNs receive the entire current-day hourly curve, its slopes, quarter/weekday flags, '
              'and seven prior calendar days. The matched control has identical capacity with FlowState channels zeroed. '
              'FlowState r1.1 is frozen; 28 calendar days feed its native median 96-quarter forecast at scale0.25. '
              'The conditional CNN predicts mean-centered corrections around mean-corrected interpolation.', '',
              'Fit: October2025–July2026 after28day warmup. August selects stopping and one seed per arm. '
              'Three seeds17/43/71 per arm; no September choice. Audit passed.', '',
              'Evidence: metrics.json, audit.json, selection.json, model_receipt.json, predictions.npz and trials/*/model.pt.', '',
              'Pretraining-data overlap with official historical prices has not been audited. '
              'This comparison measures these particular adaptations; it does not establish FlowState’s best attainable error.']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__': main()
