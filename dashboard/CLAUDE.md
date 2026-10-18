# Neural Wave 2026 dashboard

Offline, no-build, tabbed dashboard ("Neural Wave 2026 - PhDs but not for
long - Research Journal"). The **Input data visualization** tab (`#data`)
visualizes `../train_prices_15min.csv` (day-ahead
prices, EUR/MWh, 4 zones, 15-min resolution). The **Time series study** tab
presents the findings of `../TIMESERIES_INSIGHTS.md` as prose, tables and
charts. The **Historical progress** tab (`#history`) reports every trial in
`../ch-PHDs-but-not-for-long/` other than the delivered model. The **Final
ensemble** tab (`#final`) reports that repo's research champion on branch
`el_bicho/hackaton-research` and the frozen model refitted from it
(methodology, metrics and insights).

The page is a research report read by supervisors. Write its prose in
impersonal report voice: state findings, methods and caveats for a reader who
reviews the work, not one who carries it out.

Open `index.html` directly —
no server, no bundler, no npm, no CDN. That constraint drives every
architectural choice below; preserve it.

## Run / verify

- `open index.html` (or double-click) — the real acceptance bar, since the
  page must work over `file://`.
- `python3 -m http.server` from this directory — convenience only, for
  DevTools network inspection.
- No build step. Edit the `.js`/`.css`/`.html` files directly.

## Why classic `<script src>`, never `type="module"`

`<script src>` is a plain resource load and works over `file://`. ES module
`import` is fetched through the CORS-checked module loader and **fails**
over `file://` in Chrome ("Cross origin requests are only supported for
HTTP"). Load order in `index.html` is dependency order — `data.js` →
`chart.js` → `table.js` → `profile.js` → `deviation.js` → `compare.js` →
`study.js` → `experiment.js` → `ensemble.js` → `app.js` — since classic scripts share one global scope with no imports.

## Data

`data.js` is **generated** by `../scripts/prepare_data.py` from the sibling
CSV — never hand-edit it, rerun the script instead. It defines `const
DATASET = { zones, dates, prices: {ZONE_n: Float32Array}, stats }`.

- Flat per-zone index: `dayIdx*96 + (hour-1)*4 + (quarter-1)`, `hour` 1-24,
  `quarter` 1-4. `dayIdx` is the index into `DATASET.dates`.
- `DATASET.dates` has 363 entries and is **not contiguous** — two calendar
  days are missing (`2025-10-26`, `2026-03-29`, the DST changeovers). Always
  resolve a day via `DATASET.dates` / a date→index map, never by assuming
  `date + 1`.
- Hourly means are computed once and cached (`Chart.getHourlyPrices()`),
  not stored in `data.js` — other modules reuse that cache rather than
  recomputing.

## Page layout & tabs

- `<header class="app-header">` is a full-width band on `--header-plane`
  (a step off `--page-plane`). It holds `.app-title` (the round radar-scope
  `nerd-nerd-emoji.gif` left of the `<h1>` title) and the `#tab-bar` tablist.
- Each tab is a `<main class="tab-panel" id="tab-<id>" role="tabpanel">`.
  Inactive panels carry `hidden`, and `.tab-panel[hidden]` forces
  `display: none`.
- Tabs are underline-style in chrome green (`--frame`), the one hue UI
  chrome wears (see Theme below).
- `app.js` owns the active tab (`state.activeTab`) and mirrors it to
  `location.hash`, so a reload keeps the tab. The hash is either a tab id
  (`#data`, `#study`, `#history`, `#final`) or the id of an element inside a panel (for example
  the study TOC's `#study-zones`). `tabForHash()` resolves an anchor to the
  panel that contains it, so in-page links never bounce back to Data.
  `showTab()` is the only function that toggles panels.
- A bare `&sect;N` means a section of the current tab. A reference to another
  tab names it: "study &sect;N", "history &sect;N", "final &sect;N".

**Adding a tab:** add a `<button role="tab" data-tab="<id>">` to `#tab-bar`,
add a `<main class="tab-panel" id="tab-<id>" hidden>`, then add a `<id>:
() => {…}` entry to `TAB_ON_SHOW` at the top of `app.js`.

**Hidden-panel caveat:** canvas modules size themselves from
`getBoundingClientRect()`, which returns 0×0 inside a hidden panel. A render
while hidden is harmless, but every chart in a tab must re-render when its
tab is shown. That is what the `TAB_ON_SHOW` entry is for: the Data tab's
entry calls `rerenderChart()`. The ResizeObservers would eventually catch
up too, but only after a visible 150 ms blank.

## Time series study tab (`study.js`)

- The figures are **transcribed from `../TIMESERIES_INSIGHTS.md`**: they are
  whole-dataset statistics, not computed from `DATASET`. When that document
  changes, update both the prose and tables in `#tab-study` (`index.html`)
  and the data constants at the top of `study.js`.
- The tab opens with a Summary paragraph about its §1, then reorders the
  document: tab §1 = doc §7, tab §2 = doc §8, tab §3–§8 = doc §1–§6.
  Doc §9–§11 are left out. Every `&sect;N` reference to the study, on any
  tab, uses tab numbers.
- The tab ignores the dashboard filters. `Study` takes no `getState` and is
  rendered only from `TAB_ON_SHOW.study`, not from `rerenderChart()`.
- Two small chart factories live inside `study.js`: `columnChart()` (one
  pooled series, ≤24px columns with 4px rounded caps, selective direct
  labels) and `acfChart()` (one line per zone on a linear lag axis, with
  de-collided end labels). Reuse them for new study charts instead of
  adding modules.
- Figures pooled across zones use `--series-pooled` (neutral ink), never a
  zone hue, because they don't belong to any one zone. The dataviz validator
  flags it as "reads gray"; that is intended, since it isn't a categorical
  slot.
- The correlation heatmap and the inline data bars are plain HTML/CSS. A
  heatmap cell sets `--r` and mixes `--seq-high` (one hue, brighter = stronger)
  into the surface over r 0.4–1; the trivial diagonal is neutral. A bar cell
  sets `--v` (0–1) and `--c` (the zone colour).

## Historical progress and Final ensemble tabs (`experiment.js`, `ensemble.js`)

Both panels carry the `study` class to reuse the study tab's CSS, and both are
static like `Study`. The split is by role: Final holds the delivered model,
History every other trial. Content moves between them by section, so chart and
section IDs keep the prefix of the module that draws them (`cnn-*` →
`experiment.js`, `ens-*` → `ensemble.js`), not of the tab they sit in. Each tab
opens with a Summary (`#history-summary`, `#ens-summary`) that restates its
headline figures; update it whenever those figures change.

**Historical progress (`#tab-history`)** has three `.part-heading` parts under one
flat § numbering. `TAB_ON_SHOW.history` renders both `Experiment` and
`Ensemble`, because the RevIN charts live here.
- Part 1, §1–§7, the raw-hourly CNN (first iteration). Figures are **transcribed
  from `../ch-PHDs-but-not-for-long/experiments/*`** (`run_info.json`, metrics
  and month-difficulty CSVs), plus a flat-baseline MAE for Aug–Sep computed once
  from the training CSV. Rerunning it means updating the prose/tables and the
  constants in `experiment.js`.
- Part 2, §8, the model-family ranking from `MODEL_RANKING.md` plus the
  permutation importance of four earlier single-family models from
  `results/feature_importance_20261017/ANALYSIS.md`, and §9, the
  research ensemble's composition: the six experts and the linear experts' head
  (`docs/MODELS_AND_UNCERTAINTY.md`), selection and the per-fold blends
  (`results/model_ranking_20261017/champion_per_fold.json`), the
  member-removal diagnostics (`results/interpretability_deep_20261017/`), and
  "Feature evidence by frozen family". That subsection holds the closest
  per-family evidence from related models: the MLP and CNN columns of the §8
  permutation study (configs in `results/simple_5090/selection.json`) and the
  analog 2D CNN's cumulative input ablation (`results/parallel_2d_tuning_20261017/`,
  arms in `code/tune_parallel_cnn.py`). Every row states how its model differs from
  the frozen expert.
- Part 3, §10, the RevIN trial, from `results/champion_revin_20261017/`,
  `results/linear_compact_revin_20261017/`, `results/revin_quantiles_20261017/`
  and `results/revin_chainsaw_20261017/` (the chainsaw table). Its example-day
  band arrays are generated from `champion_revin_20261017/example_predictions.csv`.

**Final ensemble (`#tab-final`)** is kept short. It shows only the frozen
three-expert composition, so research-phase blends and experts the frozen model
does not use belong in history §9. §2 describes the delivered model
`champion_348fit_15cal_20261017` from `CHAMPION_FREEZE.json` and
`results/submission_champion_20261017/champion_submission/`
(`SPECIFICATION_CARD.md`, `audit.json`). Its scores are those of the research
champion `minimize_expanding_20261017/fixed_blend_ratio` from branch
`el_bicho/hackaton-research` of `../ch-PHDs-but-not-for-long/`, **transcribed**
from `results/minimize_expanding_20261017/` (`monthly_scores.csv`,
`monthly_zone_scores.csv`, `interval_metrics.csv`, `confidence_bins.csv`;
calibrated `fixed_blend_ratio` rows), `VALIDATION_PROTOCOL.md` (§3) and
`docs/MODELS_AND_UNCERTAINTY.md`. The curve-shape table and
best-single-family column (§5), the width-group ratios (§6) and the blend
diagnostics (§7) come from `results/interpretability_deep_20261017/`. The
input-sensitivity table (§7) is **derived**: the mean of the 2026-04…2026-09
rows of `results/champion_sensitivity_20261017/monthly.csv`, the folds that used
exactly the frozen recipe. Its MAE change is `original_target_mae −
original_mae`, and its band width is `median_width_change`. Over all eight months
the same means reproduce that folder's `summary.json`. Feature evidence on Final
must come from the frozen recipe; studies of other models go to History. A
per-expert study is prepared but not run
(`campaigns/frozen_recipe_importance_20261018/`). When its `REPORT.md` exists,
add its table to Final §7 and replace the "No feature-group study…" note. The
11 June band arrays (§6) are generated from
`illustrative_quantile_bands_20261017/example_predictions.csv`. Interval and
width-group MAEs are pooled over the 8 test months, while ratios are equal-month
means like the primary score. A new champion means updating this tab, history
§8–§9, and the constants in `ensemble.js`.

**Charts.** Regenerate the example-day band arrays rather than hand-edit them, and
read those days' observed prices from `DATASET`.
- `experiment.js` has its own `zoneLineChart()` (one line per zone on an
  ordered categorical axis; points from `contextFrom` on are drawn in the
  dotted secondary style). It copies `study.js`'s small canvas helpers rather
  than reaching into `Study`, so the modules stay independent.
- `ensemble.js` copies `experiment.js`'s
  canvas helpers; its `seriesLineChart()` takes an arbitrary series list
  (`{name, values, colorVar, dotted}`), used for ensemble vs interpolation
  (same `--series-pooled` hue, solid vs dotted), original vs RevIN, and per-zone
  ratios. `bandChart()` draws one day's quantile bands in the zone hue: the 90%
  and q25–q75 fills, q50 solid, observed dotted, and markers where observed falls
  outside the 90% interval. Paired panels share a `yDomain`; optional `windows`
  shade and label stretches in neutral ink.

## State ownership

`app.js` owns one `state` object (active tab, visible zones, date range,
deviation mode, compare windows, …) and is the only file that mutates it. Every
rendering module (`Chart`, `Profile`, `Deviation`, `Compare`, `TableView`)
is a classic-script IIFE taking a `getState()` callback at `init()` — they
read state, they never write it. `Study`, `Experiment` and `Ensemble` are the exceptions: they are
static and take no state.

`app.js`'s `rerenderChart()` is the single choke point called on any filter
change (zone toggle, date range, resolution). **A new chart module must be
added to it**, or it will silently go stale when filters change.

## Theme

- `style.css` defines one black instrument-panel theme in a single `:root`
  (`color-scheme: dark`); the page is black whatever the OS theme.
- Phosphor green (`--frame`, `--frame-dim`, `--glow`) is the chrome colour:
  card frames with HUD corner brackets, the active tab and switch, the title.
  Zone and series hues stay off green, so chrome and data never share a hue.
- The zone hues (cyan, orange, magenta, gold) pass the dataviz validator on
  adjacent pairs against `--surface-1`; rerun it before changing any of them.
- Type: `--font-mono` for chrome, labels, tables and figures, `--font-sans`
  for prose. Canvas text uses the same monospace stack (`FONT` constants and
  the `ctx.font` strings in each module).

## Chart conventions (keep new charts consistent with these)

- Zone → color is fixed (`--series-zone0..3` in `style.css`, mirrored as a
  small `ZONE_COLOR_VAR` map at the top of each rendering module — the
  duplication is intentional, there's no module system to share it through).
  Color never changes when a zone is toggled off; only visibility changes.
- Comparing two series of the *same* zone (hourly vs. 15-min, window A vs.
  B) uses one hue, two line styles: solid 2px = primary/bold series, dotted
  1.5px (`setLineDash([1,4])`, `globalAlpha 0.55`) = secondary/contextual
  series. Never a second hue for this.
- The zero-reference line is always solid, never dashed — a dashed line
  reads as a threshold/projection, which is misleading for "nothing."
- Read the `dataviz` skill before adding a new chart type, palette, or
  layout — the palette, mark specs, and the above conventions all come
  from it.

## Adding a new chart module

Follow the existing modules (`profile.js` is the simplest example,
`chart.js` the most complete): a classic-script IIFE exposing `{init,
render}`, canvas-based (no chart library), `init(opts)` wires DOM refs +
`getState`, `render()` reads `getState()` fresh each call. Wire it in
`app.js`: add DOM lookups in `init()`, call `.init()` once, add `.render()`
to `rerenderChart()` for a Data-tab chart, or to the tab's `TAB_ON_SHOW`
entry for a chart on another tab.
