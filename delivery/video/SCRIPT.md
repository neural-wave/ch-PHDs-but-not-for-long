# Two minute video script

Target recording length: **1 minute 50 seconds**, leaving ten seconds below the organizer's maximum. This narration describes the audited research result. Update the final delivery sentence after the final artifact is audited. The video has not been recorded or uploaded.

| Time | Visual | Narration |
| --- | --- | --- |
| 0:00-0:18 | Title and one hourly price expanded into four quarters | Our AET challenge is to turn an hourly electricity-price curve into a realistic quarter-hour curve. We know all 24 hourly prices for four zones. The difficult part is recovering the shape inside each hour while preserving its average. |
| 0:18-0:38 | Pipeline: hourly curves and prior history, interpolation, learned corrections, mean projection | We start from mean-corrected interpolation and learn the remaining corrections. The model uses the supplied hourly curves, calendar information, and quarter-hour history from earlier days. A final projection makes every reconstructed hour agree with its input. |
| 0:38-0:58 | Simple three-component diagram | Our strongest research ensemble combines a multilayer perceptron, a temporal convolutional network, and a convolutional model of historical analogs in the later folds. Earlier folds use their own prior-month selections. We also predict quantiles to describe uncertainty. |
| 0:58-1:20 | Monthly chart from the report, with baseline at 1 | Across eight expanding February-to-September developmental backtests, mean monthly error was 29.2 percent lower than mean-corrected interpolation. Pooled absolute error was 4.10 euros per megawatt-hour. These months were inspected during development, so this is not a blind-test claim. |
| 1:20-1:38 | Coverage number, then original/RevIN comparison | The calibrated 90 percent intervals covered 90.59 percent of the evaluated targets. RevIN widened intervals and increased coverage in our later tests, but worsened point accuracy, so we kept the original research champion. |
| 1:38-1:50 | Final command and output CSV, or clearly labelled interface screenshot | The jury interface takes hourly prices and strictly earlier history, then writes four finite predictions per hour. Final delivery requires an audited all-history fit and an offline execution check. |

If a real final-model demo is available, replace the final visual with an actual terminal recording and its output audit. Once completed, replace the last sentence with an accurate statement of the verified delivery. Do not present mock output as a model execution.

## Recording and upload

Use large labels and one visual at a time. Record the team's own narration, show units, and keep the developmental qualification beside the headline result. Use the report's monthly chart and the existing [original versus RevIN plot](../../results/champion_revin_20261017/full_day_normalized.png) with its illustrative-example qualification. Rehearse and measure the actual exported duration; the timings above are a plan.

Save the final file as `delivery/video/summary.mp4` or another clearly named video file, record its path and duration in [`submission_status.json`](../submission_status.json), and upload it to SwissTransfer. Add the working link to that status file and verify the download remains available through judging. Large video files stay outside ordinary Git; provide the link to the organizers with the final team handoff.
