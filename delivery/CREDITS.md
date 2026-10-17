# Credits and AI assistant usage

## Team

**PhDs but not for long**

| Member | Role |
| --- | --- |
| Daniel S. Vega | Paladin |
| Valentina Moretti | Qween |
| Riccardo Ghetti | Margrave |
| Miguel Nogales | Bard |

## Data and challenge

The official AET Neural Wave 2026 package supplies the only permitted price data. AET selected, processed and anonymized prices originating from the ENTSO-E Transparency Platform. Credit the official package; references to ENTSO-E identify the source and do not imply endorsement. See the [package README](../docs/reference/README.txt) and [challenge specification](../docs/reference/01_v02_AET_Neural_Wave_2026_challenge_EN.pdf).

No additional external prices, current-day quarter-hour targets or pretrained forecasting weights are used in the eligible research comparisons described here. The final submitted pipeline must preserve that restriction.

## Software

The repository uses Python with NumPy, pandas, SciPy, scikit-learn and PyTorch; the preserved deployment bundle also imports CatBoost. Exact model-runtime pins are in [`submission/requirements-catboost.txt`](../submission/requirements-catboost.txt) and its included [`requirements.txt`](../submission/requirements.txt). Graphviz is listed with the CatBoost environment. Report generation uses ReportLab and pypdf, listed separately in [`requirements-report.txt`](../requirements-report.txt); these are not inference dependencies. Preserve the notices of any third-party code actually distributed with the final package.

## AI assistant disclosure draft

OpenAI Codex assisted with repository organization, code and analysis work, documentation, and preparation of the report and video script. It operated on the supplied project files and saved experiment evidence. The numerical results quoted in these materials come from saved model evaluations and audits. Generating text does not supply additional price data or pretrained forecasting weights to the model.

Before publication, team members should review the report and verify that this disclosure accurately describes all assistants used, their roles, and the team's own contributions. Do not claim human review or independent validation until it has happened.

## License decision

The organizers state that the repository and final report will be open-sourced after the event. The team still needs to select a repository license and confirm how the official data and third-party notices should be handled. No license grant is invented by this preparation step.
