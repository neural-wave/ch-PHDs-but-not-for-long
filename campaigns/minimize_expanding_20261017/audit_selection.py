"""Independent selection checks; this module never fits or changes predictions."""
import hashlib
import itertools
import json
import numpy as np
import torch


def audit_selection(e, folder, d, truth, validation_baseline, smoke, close):
    selection = json.loads((folder / 'selection.json').read_text())
    winners, rows = selection['by_recipe'], selection['candidates']
    assert len(rows) == (12 if smoke else 48)
    assert sorted(r['config']['id'] for r in rows) == sorted(c['id'] for c in e.configs()
            if not smoke or (c['lr'] == .001 and c['seed'] == 17))
    nt, ns = int(d['nt']), int(d['ns'])
    predictions = {}; replayed = 0; mean_error = replay_error = 0.
    for row in rows:
        cfg = row['config']; assert cfg['quantile_weight'] == 1 and cfg['preprocessing'] == 'fixed_physical'
        cid = str(cfg['id']); quantiles = np.load(folder / 'trials' / cid / 'quantiles.npy')
        assert quantiles.shape == d['y'][nt:].shape+(11,) and np.isfinite(quantiles).all()
        assert np.diff(quantiles, axis=-1).min() >= -1e-9
        violation = float(np.abs(quantiles[..., 5].mean(-1)-d['h'][nt:]).max())
        assert violation < 1e-8; mean_error = max(mean_error, violation)
        mae = float(np.abs(quantiles[:ns, ..., 5]-truth).mean())
        close(mae, row['validation_mae']); close(mae/validation_baseline, row['selection_ratio'])
        assert (quantiles[..., 9]-quantiles[..., 1]).mean() > 0
        curves = np.loadtxt(folder / 'trials' / cid / 'curves.csv', delimiter=',', skiprows=1)
        best = int(np.argmin(curves[:, 1]))
        assert row['best_epoch'] == int(curves[best, 0])
        close(curves[best, 1], row['selection_ratio'], 2e-6)
        predictions[cid] = quantiles
    selected_ids = set()
    for family, objective in itertools.product(e.p.FAMILIES, ['uniform', 'ratio']):
        name = f'fixed_{family}_{objective}'; saved = winners[name]
        singles = [r for r in rows if r['config']['family'] == family and r['config']['objective'] == objective]
        expected = {str(r['config']['id']): (r['selection_ratio'], {str(r['config']['id']): 1.}) for r in singles}
        for lr in sorted(set(r['config']['lr'] for r in singles)):
            pair = [r for r in singles if r['config']['lr'] == lr]
            if len(pair) != 2:
                continue
            cid = f'seedmean_{family}_{objective}_{lr:g}'
            parts = {str(r['config']['id']): .5 for r in pair}
            quantiles = sum(predictions[k]*w for k, w in parts.items())
            close(np.load(folder / 'trials' / cid / 'quantiles.npy'), quantiles)
            predictions[cid] = quantiles
            expected[cid] = (float(np.abs(quantiles[:ns, ..., 5]-truth).mean())/validation_baseline, parts)
        assert set(str(r['config']['id']) for r in saved['seed_candidates']) == set(expected)
        chosen = min(expected, key=lambda k: (expected[k][0], k))
        assert str(saved['config']['id']) == chosen and saved['components'] == expected[chosen][1]
        close(saved['selection_ratio'], expected[chosen][0])
        for r in saved['seed_candidates']:
            close(r['selection_ratio'], expected[str(r['config']['id'])][0])
        selected_ids.update(saved['components'])
    old_folder = e.OLD / 'folds' / folder.name
    old_d = np.load(old_folder / 'data.npz')
    old_selection = json.loads((old_folder / 'selection.json').read_text())['by_recipe']
    for name, row in winners.items():
        if not name.startswith('standard_'):
            continue
        src = e.PROJECT / row['source']
        assert hashlib.sha256(src.read_bytes()).hexdigest() == row['source_sha256']
        quantiles = np.load(src); cid = row['config']['id']
        close(quantiles, np.load(folder / 'trials' / cid / 'quantiles.npy'))
        close(float(np.abs(quantiles[:ns, ..., 5]-truth).mean())/validation_baseline, row['selection_ratio'])
        predictions[cid] = quantiles
        original = old_selection[row['original_recipe']]
        if original['config']['family'] != 'blend':
            checkpoint = torch.load(old_folder / 'trials' / str(original['config']['id']) / 'model.pt',
                                    weights_only=False, map_location='cpu')
            net = e.Quantiles(checkpoint['config'], old_d).cuda().eval(); net.load_state_dict(checkpoint['state'])
            with torch.no_grad():
                actual = e.physical(net(torch.as_tensor(old_d[net.feature][nt:], device='cuda'),
                                       torch.as_tensor(old_d['curve'][nt:], device='cuda')).cpu().numpy(), old_d['b'][nt:])
            delta = float(np.abs(actual-quantiles).max()); assert delta < 1e-6
            replay_error = max(replay_error, delta); replayed += 1
        else:
            expected = sum(np.load(old_folder / 'trials' / str(old_selection[k]['config']['id']) /
                                   'quantiles.npy')*w for k, w in original['weights'].items())
            close(expected, quantiles)
    # Reconstruct the frozen menu independently, including order for exact ties.
    for objective, kind in itertools.product(['uniform', 'ratio'], ['fixed', 'combined']):
        members = {n: r for n, r in winners.items() if r['config'].get('objective') == objective and
                   r['config']['family'] != 'blend' and 'blend_' not in n and n != 'validation_selector' and
                   (kind == 'combined' or n.startswith('fixed_'))}
        names = sorted(members); primary = 'fixed_linear_rich_' + objective
        menu = [{n: 1.} for n in names]
        menu += [{primary: w, other: 1-w} for other in names if other != primary for w in [.25, .5, .75]]
        top = sorted(names, key=lambda n: (members[n]['selection_ratio'], n))
        menu += [{n: 1/k for n in top[:k]} for k in [2, 3, 4, 6] if len(top) >= k]
        if kind == 'combined':
            menu += [{'standard_blend_'+objective: 1.}, {'fixed_blend_'+objective: 1.}]
        row = winners[kind+'_blend_'+objective]
        assert [v['weights'] for v in row['blend_candidates']] == menu
        ratios = []
        for weights, candidate in zip(menu, row['blend_candidates']):
            assert min(weights.values()) >= 0 and abs(sum(weights.values())-1) < 1e-12
            allq = sum(predictions[str(winners[n]['config']['id'])]*w for n, w in weights.items())
            ratio = float(np.abs(allq[:ns, ..., 5]-truth).mean())/validation_baseline
            close(ratio, candidate['selection_ratio']); ratios.append(ratio)
        assert row['weights'] == menu[int(np.argmin(ratios))]
        actual = sum(predictions[str(winners[n]['config']['id'])]*w for n, w in row['weights'].items())
        cid = row['config']['id']; close(actual, np.load(folder / 'trials' / cid / 'quantiles.npy'))
        predictions[cid] = actual
    empirical = np.quantile(d['y'][:nt]-d['b'][:nt], e.TAUS, axis=(0, 2)).transpose(1, 2, 0)
    empirical[..., :5] = np.minimum(empirical[..., :5], 0); empirical[..., 5] = 0
    empirical[..., 6:] = np.maximum(empirical[..., 6:], 0)
    predictions['interpolation'] = np.load(folder / 'trials/interpolation/quantiles.npy')
    close(predictions['interpolation'], d['b'][nt:, ..., None]+empirical[None, :, None, :, :])
    for cid in selected_ids:
        checkpoint = torch.load(folder / 'trials' / cid / 'model.pt', weights_only=False, map_location='cpu')
        net = e.Quantiles(checkpoint['config'], d).cuda().eval(); net.load_state_dict(checkpoint['state'])
        with torch.no_grad():
            quantiles = e.physical(net(torch.as_tensor(d[net.feature][nt:], device='cuda'),
                                      torch.as_tensor(d['curve'][nt:], device='cuda')).cpu().numpy(), d['b'][nt:])
        delta = float(np.abs(quantiles-predictions[cid]).max()); assert delta < 1e-6
        replay_error = max(replay_error, delta); replayed += 1
    return selection, predictions, len(rows), replayed, mean_error, replay_error
