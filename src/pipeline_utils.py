"""Shared standalone input checks and frozen feature preprocessing."""
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import torch

import feature_pipeline
from price_data import ZONES, read_targets

ROOT = Path(__file__).resolve().parent.parent
POINT_COLUMNS = ['delivery_date', 'market_id', 'hour_index', 'quarter_index',
                 'predicted_price_eur_mwh']
HOURLY_COLUMNS = ['delivery_date', 'market_id', 'hour_index', 'hourly_price_eur_mwh']
HISTORY_COLUMNS = ['delivery_date', 'market_id', 'hour_index', 'quarter_index', 'price_eur_mwh']


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def environment():
    return dict(python=sys.version, executable=sys.executable,
                system=platform.system(), machine=platform.machine(),
                torch=str(torch.__version__), numpy=np.__version__, pandas=pd.__version__,
                cuda_available=torch.cuda.is_available(), cuda_build=torch.version.cuda,
                threads=torch.get_num_threads())


def seed_everything(seed):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed)
    np.random.seed(seed)


def _read_frame(path, columns, allow_empty=False):
    frame = pd.read_csv(path)
    if set(frame.columns) != set(columns) or len(frame.columns) != len(columns):
        raise ValueError(f'{path}: expected columns {columns}')
    if frame.empty:
        if allow_empty:
            return frame
        raise ValueError(f'{path}: no input dates')
    dates = frame['delivery_date']
    parsed = pd.to_datetime(dates, format='%Y-%m-%d', errors='raise')
    if dates.isna().any() or not (parsed.dt.strftime('%Y-%m-%d') == dates).all():
        raise ValueError(f'{path}: delivery_date must be YYYY-MM-DD')
    if not frame.market_id.isin(ZONES).all():
        raise ValueError(f'{path}: market_id must be one of {ZONES}')
    for column, maximum in [('hour_index', 24), ('quarter_index', 4)]:
        if column in columns:
            values = pd.to_numeric(frame[column], errors='raise')
            if not np.isfinite(values).all() or not values.isin(range(1, maximum + 1)).all():
                raise ValueError(f'{path}: invalid {column}')
            frame[column] = values.astype('int64')
    price = columns[-1]
    frame[price] = pd.to_numeric(frame[price], errors='raise')
    if not np.isfinite(frame[price]).all():
        raise ValueError(f'{path}: prices must be finite')
    keys = columns[:-1]
    if frame.duplicated(keys).any():
        raise ValueError(f'{path}: duplicate input keys')
    rows_per_day = 384 if 'quarter_index' in columns else 96
    if not frame.groupby('delivery_date').size().eq(rows_per_day).all():
        raise ValueError(f'{path}: each date must contain exactly {rows_per_day} rows')
    return frame


def read_hourly(path):
    frame = _read_frame(path, HOURLY_COLUMNS)
    wide = frame.pivot(index='delivery_date', columns=['market_id', 'hour_index'],
                       values='hourly_price_eur_mwh')
    wide = wide.reindex(columns=pd.MultiIndex.from_product([ZONES, range(1, 25)])).sort_index()
    return wide.index.to_numpy(), wide.to_numpy(dtype='float64').reshape(-1, 4, 24)


def read_history(path, before=None):
    frame = _read_frame(path, HISTORY_COLUMNS, allow_empty=True)
    if frame.empty:
        return np.asarray([], dtype=object), np.empty((0, 4, 24, 4), dtype='float64')
    if before is not None and not (frame.delivery_date < str(before)).all():
        raise ValueError('History must strictly precede the first requested date')
    # Keep the frozen reader's ordering and conversion on valid inputs.
    return read_targets(path)


def feature_data(dates, hourly, history_dates, history_y, fit_mask=None):
    """Same fixed physical scaling, clipping and zone flags as the frozen fit."""
    raw = feature_pipeline.features(dates, hourly, history_dates, history_y)
    data = {key: np.clip(raw[key], -15, 15).astype('float32')
            for key in ['rich', 'compact', 'cnn', 'curve']}
    data['compact'] = np.concatenate([data['compact'], np.broadcast_to(
        np.eye(4)[None, :, None, :], (len(dates), 4, 24, 4))], -1).astype('float32')
    for key in ['rich', 'compact']:
        # Selected networks do not evaluate knots, but preserve state_dict buffers.
        data[key + '_knots'] = (np.zeros((3, data[key].shape[-1]), dtype='float32')
            if fit_mask is None else np.quantile(
                data[key][fit_mask].reshape(-1, data[key].shape[-1]), [.25, .5, .75],
                axis=0).astype('float32'))
    return data


def check_causality(dates, hourly, history_dates, history_y, indices):
    checked = []
    for index in sorted(set(indices)):
        day = dates[index:index + 1]
        h = hourly[index:index + 1]
        reference = feature_pipeline.features(day, h, history_dates, history_y)
        changed = history_y.copy()
        changed[history_dates >= dates[index]] += np.arange(4) * 123456
        poisoned = feature_pipeline.features(day, h, history_dates, changed)
        prior = history_dates < dates[index]
        truncated = feature_pipeline.features(day, h, history_dates[prior], history_y[prior])
        for key in reference:
            np.testing.assert_allclose(reference[key], poisoned[key], rtol=0, atol=1e-9)
            np.testing.assert_allclose(reference[key], truncated[key], rtol=0, atol=1e-9)
        checked.append(str(dates[index]))
    return checked
