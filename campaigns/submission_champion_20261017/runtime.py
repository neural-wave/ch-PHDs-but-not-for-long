"""Offline champion inference; packaged beside sources/ and model.pt."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'sources'))
import numpy as np
import pandas as pd
import torch
import trainq as q
from reconstruction import ZONES, read_targets, baseline


def feature_data(dates, hourly, history_dates, history_y):
    raw = q.p.features(dates, hourly, history_dates, history_y)
    data = {key: np.clip(raw[key], -15, 15).astype('float32')
            for key in ['rich', 'compact', 'cnn', 'curve']}
    data['compact'] = np.concatenate([data['compact'], np.broadcast_to(
        np.eye(4)[None, :, None, :], (len(dates), 4, 24, 4))], -1).astype('float32')
    # These three selected architectures do not use additive knots.
    for key in ['rich', 'compact']:
        data[key + '_knots'] = np.zeros((3, data[key].shape[-1]), dtype='float32')
    return data


def predict_arrays(bundle, dates, hourly, history_dates, history_y, device='cpu'):
    torch.set_num_threads(1)
    data = feature_data(dates, hourly, history_dates, history_y)
    anchor = baseline(hourly)
    result = np.zeros((*anchor.shape, 11), dtype='float64')
    for item in bundle['models']:
        net = q.Quantiles(item['config'], data).to(device).eval()
        net.load_state_dict(item['state'])
        chunks = []
        with torch.no_grad():
            for start in range(0, len(dates), 16):
                ids = slice(start, start + 16)
                raw = net(torch.as_tensor(data[net.feature][ids], device=device),
                          torch.as_tensor(data['curve'][ids], device=device)).cpu().numpy()
                chunks.append(q.physical(raw, anchor[ids]))
        result += item['weight'] * np.concatenate(chunks)
    assert np.isfinite(result).all() and np.diff(result, axis=-1).min() >= -1e-8
    assert np.abs(result[..., 5].mean(-1) - hourly).max() <= .0001
    return result


def predict(model_path, hourly_path, history_path, output, quantile_output=None):
    bundle = torch.load(model_path, map_location='cpu', weights_only=False)
    df = pd.read_csv(hourly_path)
    keys = ['delivery_date', 'market_id', 'hour_index']
    assert not df.duplicated(keys).any(), 'Duplicate hourly keys'
    assert set(df.market_id) == set(ZONES) and set(df.hour_index) == set(range(1, 25))
    frame = df.pivot(index='delivery_date', columns=keys[1:], values='hourly_price_eur_mwh')
    frame = frame.reindex(columns=pd.MultiIndex.from_product([ZONES, range(1, 25)])).sort_index()
    dates = frame.index.to_numpy()
    assert len(dates) > 0
    hourly = frame.to_numpy().reshape(-1, 4, 24)
    assert np.isfinite(hourly).all()
    hd, hy = read_targets(history_path)
    assert all(str(day) < str(dates[0]) for day in hd), 'History must precede the first input date'
    quantiles = predict_arrays(bundle, dates, hourly, hd, hy)
    rows = []
    for i, day in enumerate(dates):
        for z, zone in enumerate(ZONES):
            for hour in range(24):
                for quarter in range(4):
                    values = quantiles[i, z, hour, quarter]
                    row = dict(delivery_date=day, market_id=zone, hour_index=hour+1,
                               quarter_index=quarter+1, predicted_price_eur_mwh=values[5])
                    if quantile_output:
                        row.update({f'q{tau:g}': value for tau, value in zip(q.TAUS, values)})
                        previous = None
                        for coverage, lo, hi in q.INTERVALS:
                            level = str(round(coverage * 100))
                            correction = np.asarray(bundle['calibration'][level]['correction'])[z, quarter]
                            lower, upper = values[lo] - correction, values[hi] + correction
                            if previous is not None:
                                lower, upper = min(lower, previous[0]), max(upper, previous[1])
                            row['lower_' + level], row['upper_' + level] = lower, upper
                            previous = lower, upper
                    rows.append(row)
    full = pd.DataFrame(rows)
    columns = keys + ['quarter_index', 'predicted_price_eur_mwh']
    full[columns].to_csv(output, index=False, float_format='%.10f')
    if quantile_output:
        full.to_csv(quantile_output, index=False, float_format='%.10f')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=str(HERE / 'model.pt'))
    parser.add_argument('--hourly', required=True)
    parser.add_argument('--history', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--quantiles-output')
    args = parser.parse_args()
    predict(args.model, args.hourly, args.history, args.output, args.quantiles_output)
