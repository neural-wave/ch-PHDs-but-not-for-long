# HACKATON project memory

Objective: official-only four-zone quarter reconstruction; hourly means within 0.0001 EUR/MWh. No external/pretrained/October prices. October–July fitting, August selection, September development test; strictly prior observed history allowed.

Separate RTX5090 search completed 224 single models. Reported August-selected September MAE: linear 5.209794, additive 5.255107, MLP 5.343947, CNN 5.222494; independent audit passed. SIMPLE_SEARCH_PROTOCOL.md; results/simple_5090. Host root@93.91.156.85:55340, /workspace/HACKATON; Nightwatch2093a7ebed349785229d04bf completed. Prior ensemble 5.091209 remains stronger; no deployment replacement.

Verified 2026-10-17: simple linear11-quantile/hierarchical-median run completed 16 folds, exit0 in189.45s. Nightwatch0bf41a85e0867de319a9e14a; root@ssh1.vast.ai:24199, /workspace/HACKATON; log /root/.local/share/nightwatch-hackaton/0bf41a85e0867de319a9e14a.log. Offline joint medianMAE+pinball+L2,389 named features, explicit crossing correction. LINEAR_QUANTILES_PROTOCOL.md.

Final302-day September medianMAE5.226105; meanpinball1.505541. Learned50/90/95 interval coverage52.86/90.90/95.01%; widen-onlyCQR54.62/93.39/96.81%, with wider bands. Median-residualCP undercovers42.51/86.00/92.48%. Independent16-fold/192-interval audit passed; maxmeanerror1.14e-13. Local results/linear_quantiles_vast contains metrics,CSV,graphs,finalmodelJSON/coefficients/predictions. Learning overlays corrected without retraining; final graphs visually checked. No active quantile job.

MAE-only expanding16-fold results preserved at results/expanding_vast; starts with three training months. EXPANDING_PROTOCOL.md. Ando all363 refit not rechecked: Nightwatch816b1393f18c891ba8a8de11, /home/mnogales/Projects/HACKATON/results/deployment_all363_gpu1; DEPLOYMENT_REFIT.md. Baseline zone charts preserved; zones anonymized. Detailed previous startup: memory/records/20261017_before_quantile_completion_startup.md. Next optional interpretation/calibration review; preserve prior manifests and separate5090 context.
