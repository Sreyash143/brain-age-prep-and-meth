# V3 Brain-Age Project — Graph Pack

This pack contains 40 PNG figures generated from the supplied V3 project outputs and accompanying CSV/JSON result files. Figures are derived from the project's saved predictions and analyses; they are not simulated data.

## How to use
- Browse `graphs/` and select the figures suitable for the report.
- `graph_index.csv` lists each figure and its category.
- Figures are 180 DPI PNGs. They can be inserted into Word, Google Docs, or LaTeX.
- Relevant result tables are included as CSV/JSON alongside the figures.

## Important interpretation notes
1. **Raw versus corrected predictions:** raw predicted age (`predicted_age`) and bias-corrected brain age (`brain_age`) are different outputs. Do not label one as the other. The primary V3 `accuracy_m1.json` reports corrected MAE of about 3.76 years and raw MAE of about 3.27 years, but these are calculated on the pipeline's defined evaluation subsets; the raw model-comparison chart uses a common set of CN scans without the same correction/QC filtering and is not directly interchangeable with those headline metrics.
2. **Age-band MAE:** `accuracy_bands_m1.csv` is the saved result and is used directly for charts 13–14.
3. **Longitudinal slopes:** the slopes file has only a limited subset of subjects with qualifying longitudinal windows. Present these as exploratory/preliminary, and report sample sizes.
4. **Three-class age accuracy:** this is supplementary only; MAE/RMSE and calibration plots are more meaningful for a regression model.
5. **QC correlations:** association does not establish that a QC metric causes prediction error. Use these as diagnostic plots.
6. **Repeatability:** SEM/MDC95 are within-session repeatability quantities, not the same thing as age-prediction MAE.
7. The supplied `accuracy_m1.json` has a field called `MAE_alt_method_NOT_REPORTABLE`; do not use this as the model's headline accuracy. The methodology explicitly labels that alternative as not reportable.
8. Some chart names say "raw" and should remain so in captions. Do not silently relabel them as bias-corrected results.

## Recommended core report set
- 01 cohort counts
- 02 or 03 age distribution
- 04 subject-level folds
- 06 actual vs predicted raw age
- 07 corrected age vs chronological age
- 08 raw error distribution
- 10 BAG distribution
- 11 BAG vs age
- 13 MAE by age band
- 17 calibration slope by fold (optional)
- 21–23 repeatability and noise floor
- 24 uncertainty calibration (only if uncertainty is discussed)
- 25 QC flag counts
- 28–30 longitudinal slopes (only if the report discusses longitudinal analysis)
- 37 model comparison (state it is common-scan raw, uncorrected comparison)

All plots should be described as evidence from this V3 run, not proof of clinical validity. Brain Age Gap is a research biomarker and is not a clinical diagnosis.
