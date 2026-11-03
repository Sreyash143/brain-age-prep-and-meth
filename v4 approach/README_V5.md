# V5 runbook (WSL, existing `wsl_env_312`)

Copy the contents of this folder into `/mnt/d/brain age prep and meth/v4 approach/`. It only adds new files (`v5_*.py`, `config_v5.py`, `tests/`, docs); no V4 file is overwritten. Then, from that directory:

```bash
cd "/mnt/d/brain age prep and meth/v4 approach"
source <path-to>/wsl_env_312/bin/activate
git status          # optional: commit or stash first; v5_train records the commit hash in the registry

# 0. tests (CPU, ~15 s) and a zero-cost protocol check
python -m pytest tests -q
python v5_train.py --experiment v5_z_khuber --folds 0,1,2,3,4 --dry_run   # asserts subject-disjoint splits, prints sizes

# 1. reproduce the baseline from existing predictions (no GPU)
python v5_report.py --tags v4_sfcn_zscore_ablation,v4_sfcn_baseline_40ep,v3_m1_recomputed \
   --baseline v4_sfcn_zscore_ablation --pred_dir v4/predictions \
   --qc_passed_from v4/predictions/bag_v4_sfcn_zscore_ablation_qc_passed.csv \
   --out_dir v5/results --name baseline_repro

# 2. ROUND 1 (needs your approval: ~6.5-7 GPU-hours; resumable, Ctrl+C safe)
python v5_train.py --experiment v5_z_khuber --folds 0,1,2,3,4 --seeds 2 --record_train_mae
python v5_predict.py --experiment v5_z_khuber --weights raw      # tag v5_z_khuber
python v5_predict.py --experiment v5_z_khuber --weights ema      # tag v5_z_khuber_ema
python v5_report.py --tags v4_sfcn_zscore_ablation,v5_z_khuber,v5_z_khuber_ema --baseline v4_sfcn_zscore_ablation \
   --pred_dir v4/predictions --pred_dir v5/predictions \
   --qc_passed_from v4/predictions/bag_v4_sfcn_zscore_ablation_qc_passed.csv --out_dir v5/results --name round1
# EMA vs raw weights is chosen from validation MAE in v5/experiment_registry.csv (best_val_mae_ema vs best_val_mae_raw), NOT from test.

# 3. ROUND 2 (only if round 1 passes the decision rules in AUDIT_AND_PLAN_V5.md; fold 0 only, ~6 GPU-h)
python v5_optuna.py --experiment v5_z_khuber --outer_fold 0 --n_trials 25 --trial_epochs 20
#   then the retrain command it prints; evaluate with  --folds 0  in v5_report.py
```

Reading the report: `paired dMAE` = (model - baseline); **negative = better than baseline**. RAW = model output, CORR = cross-fitted
bias-corrected. Never compare a RAW number to a CORR number.

Optional V4 fix (latent checkpoint-glob bug): `cd` to the project folder and `patch -p1 < v4_fix_checkpoint_glob.patch`
(back up `10_eval_canonical.py` first). Not needed for V5 scripts, which use their own strict matcher.

Files: `config_v5.py` experiments/search space | `v5_train.py` trainer (+EMA, registry) | `v5_predict.py` OOF predictions |
`v5_report.py` metrics, integrity checks, paired bootstrap | `v5_optuna.py` fixed-protocol search | `tests/` | `reference_outputs/` baseline reports.
