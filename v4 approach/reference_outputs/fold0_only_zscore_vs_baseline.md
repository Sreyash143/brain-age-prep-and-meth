
### Population: qc_passed  (CN scans only; folds [0])

| model | n scans / subj | RAW MAE pooled [95% CI] | RAW MAE fold-mean | RAW RMSE | RAW R2 | CORR MAE pooled [95% CI] | CORR MAE fold-mean | CORR R2 | paired dMAE raw [CI] | paired dMAE corr [CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| v4_sfcn_baseline | 369/156 | 3.314 [2.899,3.762] | 3.314 | 4.263 | 0.821 | 3.358 [2.961,3.777] | 3.358 | 0.827 | (baseline) | (baseline) |
| v4_sfcn_zscore_ablation | 369/156 | 2.896 [2.529,3.296] | 2.896 | 3.734 | 0.863 | 3.185 [2.779,3.600] | 3.185 | 0.837 | -0.418 [-0.649,-0.178] | -0.173 [-0.499,+0.156] |

### Population: all_eligible  (CN scans only; folds [0])

| model | n scans / subj | RAW MAE pooled [95% CI] | RAW MAE fold-mean | RAW RMSE | RAW R2 | CORR MAE pooled [95% CI] | CORR MAE fold-mean | CORR R2 | paired dMAE raw [CI] | paired dMAE corr [CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| v4_sfcn_baseline | 479/158 | 5.028 [4.132,5.994] | 5.028 | 7.328 | 0.512 | 6.138 [5.178,7.236] | 6.138 | 0.318 | (baseline) | (baseline) |
| v4_sfcn_zscore_ablation | 479/158 | 4.347 [3.579,5.181] | 4.347 | 6.568 | 0.608 | 5.629 [4.900,6.425] | 5.629 | 0.475 | -0.682 [-1.353,-0.116] | -0.509 [-1.545,+0.359] |