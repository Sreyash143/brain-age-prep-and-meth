# V1 Brain Age Estimation — Graph Pack

Generated from the uploaded V1 output files:
- brainage_predictions.csv (495 rows)
- test_predictions_raw.csv (495 rows)
- repeat_pairs.csv (149 pairs)
- final_statistics.json

## Important interpretation notes
- Graphs use the saved prediction outputs and observed metadata; no synthetic observations were created.
- `predicted_age` is treated as the raw prediction, `brain_age` as the saved brain-age output, and `BAG` as the saved Brain Age Gap column.
- The summary metrics below are copied from final_statistics.json. Recompute metrics from the prediction CSV only after confirming the same correction/evaluation convention.
- Repeated scan/run observations from the same subject are not independent; treat subgroup plots as descriptive unless subject-level clustering is accounted for.
- Longitudinal trajectories are descriptive and should not be interpreted as clinical progression without a suitable longitudinal statistical design.

## Summary statistics from final_statistics.json
{
  "manifest_scans": 3336,
  "subjects": 1098,
  "train_scans": 2361,
  "val_scans": 480,
  "test_scans": 495,
  "test_cn_scans": 358,
  "models_trained": 3,
  "mae_years": 4.698356743708027,
  "rmse_years": 6.190173482393294,
  "pearson_r": 0.8026554575074245,
  "r2": 0.610897020650713,
  "raw_mae_years": 4.720532227076263,
  "mean_bag_years": 0.7355679726300728,
  "bag_sd_years": 6.246615001909709,
  "ensemble_prediction_sd_years": 0.9416872729297439,
  "age_group_accuracy": 0.7131313131313132,
  "age_group_macro_f1": 0.689335928505141,
  "repeat_pairs": 149,
  "sigma_sem_years": 0.8908756217849843,
  "mdc95_years": 2.469381237963142
}

## Contents
40 PNG graphs, graph_index.csv, derived_predictions_with_errors.csv, repeat_pairs_with_difference.csv, all_graphs.pdf, graph_contact_sheet.jpg.
