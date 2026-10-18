"""Small checkpoint replay, causal-feature and import-isolation checks."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from pipeline_utils import (ROOT, HISTORY_COLUMNS, HOURLY_COLUMNS, check_causality,
                            feature_data, read_history, sha256, write_json)
from predict_prices import predict_arrays
from price_data import ZONES
import quantile_models as q


def verify(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    bundle = torch.load(checkpoint_dir / 'model.pt', weights_only=False, map_location='cpu')
    if bundle['run_metadata']['artifact_kind'] != 'isolated_trial':
        raise ValueError('These small checks expect a trial checkpoint')
    if bundle['training_days'] != 4 or bundle['calibration_days'] != 2:
        raise ValueError('Trial split changed')
    pairs = [(item['config']['family'], item['config']['seed']) for item in bundle['models']]
    expected = [(family, seed) for family in ['mlp_compact', 'cnn_rich', 'cnn2d_analog'] for seed in [17, 43]]
    if sorted(pairs) != sorted(expected) or any(item['epochs'] != 1 for item in bundle['models']):
        raise ValueError('The trial must fit all six selected components')
    if not all(torch.count_nonzero(item['state']['head.weight']) > 0 for item in bundle['models']):
        raise ValueError('A trial model has not updated its prediction head')
    dates, y = read_history(ROOT / 'data/train_prices_15min.csv')
    dates, y = dates[:8].astype(object), y[:8]
    hourly = y.mean(-1)
    causal_days = check_causality(dates, hourly, dates, y, [0, 3, 4, 5, 6])
    with np.load(checkpoint_dir / 'calibration_replay.npz') as saved:
        replay = predict_arrays(bundle, dates[4:6], hourly[4:6], dates[:6], y[:6])
        replay_error = float(np.abs(replay - saved['quantiles']).max())
        if replay_error > .0001:
            raise ValueError(f'Saved checkpoint replay failed: {replay_error}')
    source_map = json.loads((ROOT / 'provenance/source_map.json').read_text())
    imported_modules = {}
    for item in source_map['sources']:
        path = ROOT / item['copied_to']
        if sha256(path) != item['copied_sha256']:
            raise ValueError(f'Copied component changed: {path.name}')
        module = sys.modules[path.stem]
        if Path(module.__file__).resolve() != path.resolve():
            raise ValueError(f'A module was imported from outside the package: {path.stem}')
        imported_modules[path.stem] = item['copied_to']
    # Check that the 2D path can condition zone 0 on other zones and the full curve.
    data = feature_data(dates[6:7], hourly[6:7], dates[:6], y[:6])
    item = next(value for value in bundle['models'] if value['config']['family'] == 'cnn2d_analog')
    net = q.Quantiles(item['config'], data).eval()
    net.load_state_dict(item['state'])
    x = torch.tensor(data['cnn'], requires_grad=True)
    curve = torch.tensor(data['curve'], requires_grad=True)
    net(x, curve)[0, 0, :, 0, 5].square().sum().backward()
    grid_gradient = x.grad[:, :-net.aux].reshape(1, net.ch, 16, 96)
    other_zone_gradient = float(grid_gradient[:, 1:4].abs().sum())
    full_curve_gradient = float(curve.grad[:, 24:].abs().sum())
    if other_zone_gradient <= 0 or full_curve_gradient <= 0:
        raise ValueError('2D CNN lost cross-zone or full-curve conditioning')
    fixture = checkpoint_dir / 'fixture'
    fixture.mkdir()
    original = pd.read_csv(ROOT / 'data/train_prices_15min.csv')
    history = original[original.delivery_date < dates[6]]
    history.to_csv(fixture / 'history_15min.csv', index=False)
    pd.DataFrame(columns=HISTORY_COLUMNS).to_csv(fixture / 'empty_history.csv', index=False)
    rows = [(str(day), zone, hour + 1, hourly[index, z, hour])
            for index in [6, 7] for day in [dates[index]]
            for z, zone in enumerate(ZONES) for hour in range(24)]
    multi = pd.DataFrame(rows, columns=HOURLY_COLUMNS)
    single = multi[multi.delivery_date == dates[6]]
    single.to_csv(fixture / 'test_hourly.csv', index=False, float_format='%.15g')
    multi.to_csv(fixture / 'test_hourly_multiple.csv', index=False, float_format='%.15g')
    pd.concat([single, single.iloc[:1]]).to_csv(fixture / 'duplicate_hourly.csv', index=False)
    single.iloc[:-1].to_csv(fixture / 'incomplete_hourly.csv', index=False)
    nonfinite = single.copy()
    nonfinite.loc[nonfinite.index[0], 'hourly_price_eur_mwh'] = np.inf
    nonfinite.to_csv(fixture / 'nonfinite_hourly.csv', index=False)
    original[original.delivery_date <= dates[6]].to_csv(fixture / 'nonprior_history.csv', index=False)
    pd.concat([history, history.iloc[:1]]).to_csv(fixture / 'duplicate_history.csv', index=False)
    report = dict(passed=True, six_full_architectures_fitted=True, all_heads_updated=True,
                  training_dates=dates[:4].tolist(), calibration_dates=dates[4:6].tolist(),
                  fixture_dates=dates[6:8].tolist(), causal_poison_and_truncation_probes=causal_days,
                  checkpoint_replay_max_error=replay_error, copied_source_hashes_passed=True,
                  imports_resolve_inside_standalone_package=imported_modules,
                  cross_zone_grid_gradient=other_zone_gradient,
                  cross_zone_full_curve_gradient=full_curve_gradient)
    write_json(checkpoint_dir / 'invariants.json', report)
    print('INVARIANTS PASSED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    args = parser.parse_args()
    verify(args.checkpoint_dir)
