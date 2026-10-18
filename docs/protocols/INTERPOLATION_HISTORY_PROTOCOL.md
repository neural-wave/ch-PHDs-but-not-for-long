# Seven-day interpolation-error input — 2026-10-17

User requested the previous seven days' interpolation errors as time-series input.
`code/interpolation_history.py` computes actual minus mean-corrected centred linear
interpolation, separately for each zone, hour and quarter. Ordered calendar lags
1 through 7 each contribute four signed residuals, four absolute residuals and
one availability flag. Errors are divided by 20 EUR/MWh consistently with the
existing features. Missing days use zero with availability zero; dates are not
compressed across DST gaps. Current/future targets are excluded by construction.

`compact_features(..., interpolation_days=7)` appends 63 channels to the existing
27 compact features. All 24 hours retain their own history. The CNN sees hours
as the temporal axis and the ordered day lags as channels; this is not a new
encoder over a flattened 672-quarter sequence. Absolute errors explicitly expose
movement unexplained by interpolation. Signed errors retain its direction/shape.
These are within-hour deviations; hourly prices already describe movement between
hourly means. Rich features previously included errors at lags 1/2/7/14 and
trailing medians, but not each of the preceding seven days separately.

Matched comparison is available through the existing simple-model runner:

```sh
export SIMPLE_SEARCH_ROOT=results/seven_day_interpolation_errors
export SIMPLE_SEARCH_FEATURES=compact,compact7
python code/simple_search.py --prepare
python code/simple_search.py --smoke
python code/simple_search.py --workers 8
```

Run training on an authorized research server and register long jobs with
Nightwatch. Defaults and existing artifacts are unchanged. The matched grid
contains 224 candidates: identical family, anchor, seed and optimizer options
for compact versus compact7. Select with August only; September is a previously
inspected development test. Only prior observations available on each query date
may be passed at inference. The offline submission has not been refit or changed.

`code/verify_interpolation_history.py` checks exact lag values, absolute magnitudes,
causal perturbations, past-only equivalence, missing-date masks, empty history,
unchanged compact prefix, four model families and output mean conservation.
No training comparison has been launched; no MAE improvement is established.
