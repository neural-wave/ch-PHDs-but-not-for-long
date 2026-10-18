"""Frozen champion components copied from interpolation_history.py.

Only required definitions are retained. See provenance/source_map.json.
Feature rules, layer definitions and tensor names are preserved.
"""
import numpy as np
import pandas as pd
from price_data import baseline


def interpolation_history(dates, history_dates, history_y, days=7):
    """Return (day, zone, hour, channel), ordered from lag 1 to lag N.

    Each lag contributes four signed errors, four absolute errors (EUR/MWh
    divided by 20), and an availability mask. Missing calendar dates stay missing;
    they are never replaced with an older available observation.
    """
    if days < 1:
        raise ValueError('days must be positive')
    history = {str(pd.Timestamp(d).date()): np.asarray(y) for d, y in zip(history_dates, history_y)}
    names = [f'interpolation_error_lag{lag}_{kind}{q}'
             for lag in range(1, days+1)
             for kind, quarters in [('signed_q', range(1,5)), ('absolute_q', range(1,5)), ('available', [''])]
             for q in quarters]
    out = np.zeros((len(dates),4,24,days*9), dtype=np.float64)
    for i, date in enumerate(dates):
        dt = pd.Timestamp(date)
        for lag in range(1, days+1):
            key = str((dt-pd.Timedelta(days=lag)).date())
            y = history.get(key)
            if y is None:
                continue
            error = (y-baseline(y.mean(-1)[None])[0])/20
            start = (lag-1)*9
            out[i,...,start:start+4] = error
            out[i,...,start+4:start+8] = np.abs(error)
            out[i,...,start+8] = 1
    return out, names
