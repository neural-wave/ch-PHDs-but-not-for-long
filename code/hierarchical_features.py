"""Causal contiguous daily lag counts for hierarchical HPO."""
import numpy as np
import pandas as pd
from reconstruction import rich_features


def lagged_features(dates, hourly, history_dates, history_y, lag_days=-1, rich=None):
    full = rich_features(dates,hourly,history_dates,history_y) if rich is None else rich
    assert full.shape == (len(dates)*96,389), 'Update the explicit feature schema before changing reconstruction.py'
    if lag_days == -1:
        return full
    assert lag_days >= 0
    # Existing schema: basic history287:307 and rich error/analog history345:381.
    # Keep every current-day and calendar feature; replace all history with N daily lags.
    current = full[:,np.r_[0:287,307:345,381:389]]
    if lag_days == 0:
        return current
    history = {str(d): yy-yy.mean(-1,keepdims=True) for d,yy in zip(history_dates,history_y)}
    extra = []
    for date in dates:
        dt = pd.Timestamp(str(date))
        lags = []
        for day in range(1,lag_days+1):
            key = str((dt-pd.Timedelta(days=day)).date())
            assert key < str(date)
            lags.append(history.get(key,np.zeros((4,24,4)))/20)
        extra.append(np.concatenate(lags,axis=-1).reshape(96,-1))
    return np.concatenate([current,np.concatenate(extra)],axis=1).astype(np.float32)
