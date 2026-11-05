# Phase B outputs

Generated from the existing out-of-fold prediction CSVs and diagnostic bootstrap table.

- `tables/final_model_comparison.csv`: scan-weighted and subject-mean-prediction metrics.
- `tables/fold_wise_performance.csv`: fold-level MAE, RMSE and bias.
- `tables/age_stratified_subject_mean_mae.csv`: subject-mean-prediction MAE by age group.
- `tables/v51_raw_calibration_by_age.csv`: age-bin calibration and bias summary.
- `tables/uncertainty_analysis.csv`: association between predicted-age dispersion and absolute error.
- `figures/`: PNG (300 DPI) and PDF versions.
- `reports/uncertainty_findings.txt`: cautious interpretation of uncertainty analysis.

All plots are descriptive analyses of exported OOF predictions. The five outer folds are not
external validation. Bootstrap intervals are conditional on the exported predictions and do
not fully capture training-seed variability.
