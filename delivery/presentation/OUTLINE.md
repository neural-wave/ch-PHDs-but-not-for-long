# Five minute presentation outline

Prepared in advance for the presentation requested while judging takes place. This is an outline; final slides and rehearsal remain to be completed.

| Time | Visual | Main message |
| --- | --- | --- |
| 0:00-0:40 | One hourly average, four very different quarter-hour shapes | The task is shape reconstruction with an exact conservation constraint. All four hourly curves are known. |
| 0:40-1:40 | Baseline plus learned correction and projection | Use mean-corrected interpolation as a strong baseline. Learn residual shapes from official data, known hourly curves and strictly earlier history. Explain why the projection preserves the hourly mean. |
| 1:40-2:35 | Eight chronological folds and the monthly error chart | Fit from scratch; use the preceding month's first half for selection and second half for calibration. Report 0.707968 mean monthly error ratio and 4.103184 EUR/MWh pooled MAE. All months are developmental. |
| 2:35-3:25 | Ensemble components and uncertainty | Explain the later-fold MLP/temporal-CNN/analog-CNN blend. Show the calibrated 90% coverage of 90.59% and width of 18.8354 EUR/MWh, limited to the evaluated models. |
| 3:25-4:10 | Small ablation table | RevIN increased width/coverage but worsened point MAE in the matched later trials. Added complexity needs a measured benefit. The champion is unchanged. |
| 4:10-4:45 | Actual final inference command and output check | Show the real final bundle only after audit: offline execution, unique output keys, four quarters per hour, and mean tolerance. Distinguish final refit from backtest checkpoints. |
| 4:45-5:00 | Team and repository | Credit AET/official data, software and AI assistants. Finish with the result and how the jury can reproduce it. |

Use the report as the source of numbers. Add the final member list later. Rehearse within five minutes and update the demo section to the actual delivery status before presenting.
