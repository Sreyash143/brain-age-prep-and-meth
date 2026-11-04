# V5.2 – subject-balanced sampling (one factor vs V5.1)

Copy the files in this folder over the same names in the project root (backup first; `v5_2.patch` is the exact diff, `patch -p1 < v5_2.patch`).
Nothing in V3/V4/V5/V5.1 outputs is touched; new checkpoints are `v5/checkpoints/v5_2_subject_balanced_f{fold}_s{seed}[.ema].pt`.

## What changed
* `v5_train.py`
  * `subject_balanced_weights()`: scan weight w_ij = 1/n_i (n_i = training scans of subject i), normalised to mean 1, computed from the training split only.
  * `make_loaders`: when `cfg["subject_balanced"]` a `WeightedRandomSampler` with these weights, `num_samples=len(tr)` (same epoch length as V5.1), `replacement=True`. Tail reweighting and subject balancing are mutually exclusive (ValueError).
  * `--dry_run` prints a per-fold weight report and asserts every subject has equal total sampling mass.
  * Hardening (defects found in audit): `SystemExit`/SIGTERM was logged as `completed` in the registry (now `failed`); a resume file written under a different config was silently resumed (now refused; old resume files without a hash still load).
* `config_v5.py`: `v5_2_subject_balanced` = V5 with `reweight_tail=False, subject_balanced=True` (identical to V5.1 otherwise; asserted by a test).
* `v5_optuna.py`: creates `v5/optuna/` at use time (test-order/cwd dependent `sqlite3 unable to open database file`).
* `tests/test_v5_2_sampling.py`: 10 tests. Full suite: 54 passed.

## Commands (from the project root, venv active)
```bash
python -m pytest tests -q                                   # ~2.5 min, CPU
python v5_train.py --experiment v5_2_subject_balanced --folds 0,1,2,3,4 --dry_run
# --- needs your approval: ~5.7 GPU-h like V5.1; resumable (Ctrl+C safe) ---
python v5_train.py --experiment v5_2_subject_balanced --folds 0,1,2,3,4 --seeds 2 --workers 2 --record_train_mae
python v5_predict.py --experiment v5_2_subject_balanced --weights raw      # tag v5_2_subject_balanced
python v5_predict.py --experiment v5_2_subject_balanced --weights ema      # tag v5_2_subject_balanced_ema
python v5_report.py --tags v4_sfcn_zscore_ablation,v5_z_khuber,v5_z_khuber_ema,v5_1_no_tail,v5_1_no_tail_ema,v5_2_subject_balanced,v5_2_subject_balanced_ema \
   --baseline v4_sfcn_zscore_ablation --pred_dir v4/predictions --pred_dir v5/predictions \
   --qc_passed_from v4/predictions/bag_v4_sfcn_zscore_ablation_qc_passed.csv --boot 10000 --out_dir v5/results --name v5_2_vs_all
# V5.2 vs V5.1 directly (paired, same subjects):
python v5_report.py --tags v5_1_no_tail,v5_2_subject_balanced --baseline v5_1_no_tail --pred_dir v5/predictions \
   --qc_passed_from v4/predictions/bag_v4_sfcn_zscore_ablation_qc_passed.csv --boot 10000 --out_dir v5/results --name v5_2_vs_v5_1
```
Choose raw vs EMA from validation MAE in `v5/experiment_registry.csv`, not from test.
