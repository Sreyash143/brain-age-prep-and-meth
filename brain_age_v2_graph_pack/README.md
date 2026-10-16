# Brain Age Estimation V2 — Graph Pack

This pack contains 40 PNG figures generated from the V2 ZIP's actual saved outputs:
- `results_v2/brainage_predictions.csv`
- `results_v2/test_predictions_raw_v2.csv`
- `results_v2/repeat_pairs.csv`
- `results_v2/final_statistics.json`

## Contents
- `01`–`08`: cohort and data composition
- `09`–`24`: prediction performance, calibration, errors, subgroup checks
- `25`–`28`: Brain Age Gap (BAG)
- `29`–`31`: model prediction uncertainty
- `32`–`35`: repeated-run BAG consistency
- `36`–`37`: saved headline metrics
- `38`–`40`: uncertainty and longitudinal exploratory plots
- `graph_index.csv`: figure title, filename, and notes
- `summary_metrics.json`: summary metrics and caveats
- `derived_predictions_with_errors.csv`: derived error fields for auditing
- `repeat_pairs_with_difference.csv`: repeat-pair differences

## Important interpretation cautions
1. The prediction table has 495 scan/run records from 165 unique subjects. Records from the same subject are not independent. Use subject-level splits and subject-level bootstrap confidence intervals for inferential claims.
2. The saved `final_statistics.json` reports MAE = 4.825 years, RMSE = 6.329 years, Pearson r = 0.789, R² = 0.593, raw MAE = 4.733 years, mean BAG = -0.117 years, and age-group accuracy = 0.709. However, directly recomputing MAE from the `predicted_age` and `age` columns gives about 4.733 years, whereas the saved `brain_age` column yields a different MAE. Confirm how the corrected MAE was calculated and its evaluation subset before quoting it.
3. The reported mean BAG and BAG SD in JSON appear to describe the BAG field. The BAG values are derived outputs, not clinical diagnoses.
4. The repeat-pair table contains 149 run pairs. Agreement analyses are descriptive and should not be interpreted as independent subjects without accounting for clustering.
5. CDR and MMSE subgroup plots are descriptive. Missing CDR values are excluded from group-specific boxplots; subgroup sample sizes should be reported.
6. Longitudinal trajectories are exploratory. These graphs show observed ages/predictions across scan days, not proof of biological aging rate or disease progression.
7. Graph 37 shows different metrics on a shared 0–1 display scale for convenience; do not interpret Pearson r, R², and age-group accuracy as interchangeable.
8. Graph 39 is an exploratory sample of subjects with at least 3 distinct scan days; the x-axis is scan day divided by 365.25 (approximate relative years from study day zero). Graph 40 summarizes intervals between distinct scan days.

## Suggested main-report figures
Start with 1, 7, 9, 11, 14, 19, 25, 26, 29, 32, 36, and 37. Add graphs 17, 23, 30, 34, 39, and 40 if relevant to your research questions. Do not include every graph in the main body; place secondary diagnostics in an appendix.
