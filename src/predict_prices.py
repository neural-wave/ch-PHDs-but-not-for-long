"""Offline CPU inference for the six-component champion ensemble.

Adapted from campaigns/submission_champion_20261017/runtime.py; all prediction,
ensemble and calibration calculations retain the frozen implementation.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import quantile_models as q
from price_data import ZONES, baseline
from pipeline_utils import POINT_COLUMNS, feature_data, read_history, read_hourly


def predict_arrays(bundle, dates, hourly, history_dates, history_y, device='cpu'):
    torch.set_num_threads(1)
    if len(bundle['models']) != 6:
        raise ValueError('Expected all six ensemble components')
    if not np.isclose(sum(item['weight'] for item in bundle['models']), 1, atol=1e-12, rtol=0):
        raise ValueError('Ensemble weights must sum to one')
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
    if not np.isfinite(result).all() or np.diff(result, axis=-1).min() < -1e-8:
        raise ValueError('Invalid or unordered model quantiles')
    if np.abs(result[..., 5].mean(-1) - hourly).max() > .0001:
        raise ValueError('Hourly mean conservation failed')
    return result


def predict(model_path, hourly_path, history_path, output, quantile_output=None, allow_trial=False):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f'Missing checkpoint: {model_path}. See README.md for training commands.')
    bundle = torch.load(model_path, map_location='cpu', weights_only=False)
    if bundle.get('run_metadata', {}).get('artifact_kind') == 'isolated_trial' and not allow_trial:
        raise ValueError('This is a trial checkpoint. Pass --allow-trial for an execution check.')
    dates, hourly = read_hourly(hourly_path)
    hd, hy = read_history(history_path, before=dates[0])
    outputs = [Path(output)] + ([Path(quantile_output)] if quantile_output else [])
    inputs = {Path(path).resolve() for path in [model_path, hourly_path, history_path]}
    if any(path.resolve() in inputs for path in outputs) or len({p.resolve() for p in outputs}) != len(outputs):
        raise ValueError('Use distinct output paths, separate from all input files')
    quantiles = predict_arrays(bundle, dates, hourly, hd, hy)
    rows = []
    for i, day in enumerate(dates):
        for z, zone in enumerate(ZONES):
            for hour in range(24):
                for quarter in range(4):
                    values = quantiles[i, z, hour, quarter]
                    row = dict(delivery_date=day, market_id=zone, hour_index=hour + 1,
                               quarter_index=quarter + 1, predicted_price_eur_mwh=values[5])
                    if quantile_output:
                        row.update({f'q{tau:g}': value for tau, value in zip(q.TAUS, values)})
                        previous = None
                        for coverage, lo, hi in q.INTERVALS:
                            level = str(round(coverage * 100))
                            corrections = np.asarray(bundle['calibration'][level]['correction'])
                            if corrections.shape != (4, 4) or not np.isfinite(corrections).all() or (corrections < 0).any():
                                raise ValueError('Invalid calibration correction')
                            correction = corrections[z, quarter]
                            lower, upper = values[lo] - correction, values[hi] + correction
                            if previous is not None:
                                lower, upper = min(lower, previous[0]), max(upper, previous[1])
                            row['lower_' + level], row['upper_' + level] = lower, upper
                            previous = lower, upper
                    rows.append(row)
    full = pd.DataFrame(rows)
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    full[POINT_COLUMNS].to_csv(output, index=False, float_format='%.10f')
    if quantile_output:
        full.to_csv(quantile_output, index=False, float_format='%.10f')
    return full


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Explicit path to the fitted model.pt')
    parser.add_argument('--hourly', required=True)
    parser.add_argument('--history', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--quantiles-output')
    parser.add_argument('--allow-trial', action='store_true', help='Allow the explicitly labeled execution-test model')
    args = parser.parse_args()
    predict(args.model, args.hourly, args.history, args.output, args.quantiles_output, args.allow_trial)


if __name__ == '__main__':
    main()
