"""Independently check the jury point CSV and optional quantile/interval CSV."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_utils import POINT_COLUMNS, read_hourly, write_json
from price_data import ZONES
from quantile_models import TAUS


def check(hourly_path, prediction_path, quantile_path=None):
    dates, hourly = read_hourly(hourly_path)
    keys = POINT_COLUMNS[:-1]
    frame = pd.read_csv(prediction_path)
    if list(frame.columns) != POINT_COLUMNS:
        raise ValueError('Point CSV must have exactly the five required columns in order')
    expected = pd.MultiIndex.from_product([dates, ZONES, range(1, 25), range(1, 5)], names=keys)
    actual = pd.MultiIndex.from_frame(frame[keys])
    if len(frame) != len(expected) or actual.has_duplicates or not actual.equals(expected):
        raise ValueError('Missing, duplicate, unexpected or unordered point keys')
    values = frame.predicted_price_eur_mwh.to_numpy(dtype='float64').reshape(-1, 4, 24, 4)
    if not np.isfinite(values).all():
        raise ValueError('Predictions must be finite')
    mean_error = float(np.abs(values.mean(-1) - hourly).max())
    if mean_error > .0001:
        raise ValueError(f'Hourly mean error exceeds 0.0001: {mean_error}')
    result = dict(passed=True, dates=dates.tolist(), rows=len(frame), unique_keys=True,
                  finite=True, max_hourly_mean_error=mean_error)
    if quantile_path is not None:
        bands = pd.read_csv(quantile_path)
        qcolumns = [f'q{tau:g}' for tau in TAUS]
        interval_columns = [name + '_' + level for level in ['50', '80', '90', '95']
                            for name in ['lower', 'upper']]
        if list(bands.columns) != POINT_COLUMNS + qcolumns + interval_columns:
            raise ValueError('Unexpected quantile CSV schema')
        if not pd.MultiIndex.from_frame(bands[keys]).equals(expected):
            raise ValueError('Quantile and point keys differ')
        q = bands[qcolumns].to_numpy(dtype='float64')
        intervals = bands[interval_columns].to_numpy(dtype='float64')
        if not np.isfinite(q).all() or not np.isfinite(intervals).all():
            raise ValueError('Nonfinite quantiles or intervals')
        if (np.diff(q, axis=-1) < -1e-9).any():
            raise ValueError('Quantiles cross')
        if not np.allclose(q[:, 5], values.ravel(), atol=1e-9, rtol=0):
            raise ValueError('q50 and point predictions differ')
        previous = None
        for level in ['50', '80', '90', '95']:
            lower, upper = bands['lower_' + level].to_numpy(), bands['upper_' + level].to_numpy()
            if (lower > q[:, 5] + 1e-9).any() or (upper < q[:, 5] - 1e-9).any():
                raise ValueError('Interval excludes its median')
            if previous is not None and ((lower > previous[0] + 1e-9).any() or (upper < previous[1] - 1e-9).any()):
                raise ValueError('Calibrated intervals are not nested')
            previous = lower, upper
        result.update(ordered_quantiles=True, quantiles=11, nested_calibrated_intervals=[50, 80, 90, 95])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hourly', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--quantiles')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = check(args.hourly, args.predictions, args.quantiles)
    if args.report:
        write_json(args.report, result)
    print(result)


if __name__ == '__main__':
    main()
