# 1. Exact expert correction decomposition

Question: how do active experts combine to alter the interpolation shape?

For each fold/hour/quarter, retain baseline b, expert median p_i, frozen weight w_i and c_i=w_i(p_i-b). Verify p_blend=b+sum(c_i) and each expert's four corrections sum to zero. Include seed-level decomposition within selected averages. Compute physical contrasts a=(r1+r2)/2, u=(r1-r2)/2, v=(r3-r4)/2 for each correction; reconstruct (a+u,a-u,-a+v,-a-v).

Summarize signed and absolute contributions by month, zone, hour and calibration-defined ramp strata. Show baseline, expert corrections and truth in a waterfall/quarter plot. Magnitude is contribution, not marginal accuracy benefit. Extend existing cancellation and member-removal evidence rather than presenting it as new. Only claim exact quantile decomposition at the stage actually averaged; sorting and calibration can break a final-endpoint additive explanation.

Acceptance: replay tolerance documented, zero-sum checks pass, all eight folds covered, typical cases and labelled failures shown. Output contributions.csv, contrasts.csv, reconstruction_audit.json and figures. No fitting required.

See [shared protocol](PROTOCOL.md). Status: proposed, not executed.
