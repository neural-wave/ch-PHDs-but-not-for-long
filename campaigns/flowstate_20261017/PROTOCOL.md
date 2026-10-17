# Exploratory FlowState with current-day hourly conditioning — 17 October 2026

User explicitly authorized trying pretrained FlowState despite challenge ineligibility,
then required current-day hourly information to influence quarter-hour shapes.
This campaign is separate from every submission/deployment artifact.

Use official price data only and no October2026 prices. FlowState is the sole
external pretrained component. Its pretraining overlap is not audited.
Pin IBM gift-flowstate source640f7c80a21a4e2253b6d6140d067452bb93e7d4;
resolve checkpoint ibm-research/flowstate r1.1 to a SHA in model_receipt.json.
Isolated venv inherits the existing GPU PyTorch and pins transformers4.50.3,
huggingface-hub0.30.2. Neither existing training env nor requirements are changed.

FlowState receives28 strictly prior calendar days at quarter-hour resolution,
preserving skipped DST dates as NaNs, and forecasts96quarters with scale0.25.
Its parameters remain frozen. Store native median and within-hour-centered median
as8 extra feature channels for all4zones. No future/query quarter targets enter inference.

Both matched conditional CNNs receive today’s4x24hourly prices, left/right slopes,
quarter/weekday indicators and seven calendar days of prior quarter prices,
hourly means and availability masks. A Conv1d94->16 kernel5 mixes local inputs;
Linear96->16 broadcasts the entire current-day hourly curve into every quarter’s
hidden activations; ReLU then Conv1d16->4 predicts a within-hour residual.
Center each group of4 residuals and add to mean-corrected interpolation.
The same-size control sets the8 normalized FlowState channels to0. Its unused
weights remain present, so both arms have the same total capacity.

Train indices28:302 (October2025–July2026 after warmup), August302:333 selects
stopping and seeds, September333:363 is developmental, never blind. Three paired
seeds17/43/71 per arm; width16, lr.001, AdamWweight_decay.01, batch32,
max300epochs/patience30. No seed ensemble and no September model selection.
Persist selections before September FlowState inference. Report projected-only
FlowState, matched conditional control/FlowState, interpolation and hourly-repeat.

Execution: existing RTX5090 root@93.91.156.85:55340, /workspace/HACKATON,
campaigns/flowstate_20261017/venv/bin/python, via current Nightwatch wrapper.
Model inference smoke precedes launch. Job gates full training on causal-input,
full-current-curve/FlowState gradient, conservation and2epoch fit smoke checks;
completion requires independent scoring, train-stat checks and checkpoint replay.
Outputs results/flowstate_20261017. This evaluates one bounded adaptation;
it does not measure an optimal FlowState architecture or fully fine-tuned model.
