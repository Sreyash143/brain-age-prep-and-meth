
### Population: qc_passed  (CN scans only; folds [0, 1, 2, 3, 4])

| model | n scans / subj | RAW MAE pooled [95% CI] | RAW MAE fold-mean | RAW RMSE | RAW R2 | CORR MAE pooled [95% CI] | CORR MAE fold-mean | CORR R2 | paired dMAE raw [CI] | paired dMAE corr [CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| v4_sfcn_zscore_ablation | 1896/767 | 3.110 [2.929,3.296] | 3.109 | 3.982 | 0.806 | 3.504 [3.299,3.719] | 3.491 | 0.752 | (baseline) | (baseline) |
| v5_z_khuber | 1896/767 | 3.054 [2.868,3.239] | 3.050 | 3.924 | 0.812 | 3.428 [3.230,3.637] | 3.416 | 0.766 | -0.056 [-0.142,+0.031] | -0.076 [-0.181,+0.029] |
| v5_z_khuber_ema | 1896/767 | 3.026 [2.842,3.211] | 3.021 | 3.913 | 0.813 | 3.431 [3.228,3.645] | 3.417 | 0.764 | -0.085 [-0.170,+0.001] | -0.073 [-0.181,+0.034] |

### Population: all_eligible  (CN scans only; folds [0, 1, 2, 3, 4])

| model | n scans / subj | RAW MAE pooled [95% CI] | RAW MAE fold-mean | RAW RMSE | RAW R2 | CORR MAE pooled [95% CI] | CORR MAE fold-mean | CORR R2 | paired dMAE raw [CI] | paired dMAE corr [CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| v4_sfcn_zscore_ablation | 2321/788 | 4.513 [4.155,4.889] | 4.518 | 6.650 | 0.551 | 5.664 [5.293,6.052] | 5.661 | 0.405 | (baseline) | (baseline) |
| v5_z_khuber | 2321/788 | 4.464 [4.108,4.839] | 4.466 | 6.588 | 0.559 | 5.635 [5.257,6.028] | 5.635 | 0.413 | -0.048 [-0.160,+0.065] | -0.029 [-0.177,+0.124] |
| v5_z_khuber_ema | 2321/788 | 4.492 [4.124,4.875] | 4.492 | 6.680 | 0.547 | 5.744 [5.348,6.145] | 5.743 | 0.389 | -0.020 [-0.150,+0.106] | +0.080 [-0.093,+0.263] |