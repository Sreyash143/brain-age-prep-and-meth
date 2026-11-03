
### Population: qc_passed  (CN scans only; folds [0, 1, 2, 3, 4])

| model | n scans / subj | RAW MAE pooled [95% CI] | RAW MAE fold-mean | RAW RMSE | RAW R2 | CORR MAE pooled [95% CI] | CORR MAE fold-mean | CORR R2 | paired dMAE raw [CI] | paired dMAE corr [CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| v4_sfcn_zscore_ablation | 1896/767 | 3.110 [2.929,3.296] | 3.109 | 3.982 | 0.806 | 3.504 [3.299,3.719] | 3.491 | 0.752 | (baseline) | (baseline) |
| v4_sfcn_baseline_40ep | 1896/767 | 3.285 [3.092,3.483] | 3.281 | 4.208 | 0.783 | 3.790 [3.576,4.011] | 3.776 | 0.718 | +0.175 [+0.082,+0.269] | +0.287 [+0.168,+0.408] |
| v3_m1_recomputed | 1896/767 | 3.267 [3.083,3.456] | 3.259 | 4.173 | 0.787 | 3.765 [3.545,3.990] | 3.739 | 0.717 | +0.156 [+0.045,+0.268] | +0.261 [+0.109,+0.411] |

### Population: all_eligible  (CN scans only; folds [0, 1, 2, 3, 4])

| model | n scans / subj | RAW MAE pooled [95% CI] | RAW MAE fold-mean | RAW RMSE | RAW R2 | CORR MAE pooled [95% CI] | CORR MAE fold-mean | CORR R2 | paired dMAE raw [CI] | paired dMAE corr [CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| v4_sfcn_zscore_ablation | 2321/788 | 4.513 [4.155,4.889] | 4.518 | 6.650 | 0.551 | 5.664 [5.293,6.052] | 5.661 | 0.405 | (baseline) | (baseline) |
| v4_sfcn_baseline_40ep | 2321/788 | 4.771 [4.393,5.163] | 4.774 | 6.874 | 0.520 | 6.157 [5.773,6.563] | 6.154 | 0.319 | +0.259 [+0.092,+0.431] | +0.493 [+0.251,+0.748] |
| v3_m1_recomputed | 2321/788 | 4.753 [4.378,5.144] | 4.752 | 6.851 | 0.524 | 6.093 [5.686,6.523] | 6.095 | 0.312 | +0.241 [+0.032,+0.464] | +0.429 [+0.117,+0.756] |