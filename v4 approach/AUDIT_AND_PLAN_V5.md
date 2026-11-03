# Brain-age V5: audit findings and improvement plan

Evidence base: your `results.zip` (predictions, canonical JSONs, tables, Optuna fold-0 JSON, QC review images) and
`v4_codes.zip` (all scripts). **Nothing was trained here** (no GPU, no MRI volumes in this session). Everything below
that is a number was computed from your prediction CSVs by `v5_report.py`, an independent re-implementation that
reproduces your published values exactly (see section 1).

## 0. Executive summary

* The V4 evaluation is sound on the things that matter most: subject-disjoint folds (no subject in two folds, in any file),
  cross-fitted bias correction that never sees the held-out fold's labels, QC flags that are image-based, a subject-cluster
  bootstrap. I found no bug that inflates your accuracy.
* The headline **3.491 is the fold-mean MAE of the bias-corrected brain age**. The raw model output has **MAE 3.110** on the
  same 1,896 scans (R² 0.806 vs 0.752). Correction raises MAE by ~0.39 y but removes age-dependent bias. Both are legitimate;
  they answer different questions and must be labelled separately. **You need to choose which is the primary endpoint before V5 results exist.**
* The Z-score gain is real and well supported (paired subject-level bootstrap), the other V4 differences are mostly not.
* One confirmed integrity problem: the `v4_sfcn_baseline` row is **old-config checkpoints in folds 1-4** (identical to the 40-epoch copy), so
  "Z-score vs baseline" is only a clean ablation on fold 0.
* Realistic headroom is small: **0.05-0.15 y of raw MAE**, from combining Z-score + Huber, EMA weights and (conditionally) a clean hyper-parameter search.
  Ensembling existing models and more seeds/TTA give essentially nothing (quantified below). Bigger ResNets lost badly.

## 1. Baseline reproduction (QC-passed, CN scans, 1,896 scans / 767 subjects / 1,351 sessions)

| quantity | published | recomputed here |
|---|---|---|
| corrected MAE, fold-mean | 3.491 | 3.491 |
| corrected MAE, pooled [subject-bootstrap 95% CI] | 3.504 [3.297, 3.713] | 3.504 [3.299, 3.719] |
| corrected RMSE / r / R² | 4.499 / 0.897 / 0.752 | 4.499 / 0.897 / 0.752 |
| **raw (uncorrected) MAE pooled / fold-mean** | 3.110 (Table B) | **3.110 / 3.109** |
| raw RMSE / R² | - | 3.982 / 0.806 |
| all-eligible corrected MAE | 5.664 | 5.664 (2,321 CN scans / 788 subjects) |
| V3 m1 / 40-epoch baseline / KL-Huber corrected MAE | 3.739 / 3.776 / 3.636 | 3.739 / 3.776 / 3.636 |

R² = 1 - SSE/SST about the mean chronological age (not squared correlation). Cohort reconciliation: the 2,705 QC-passed rows =
1,896 CN + 517 CDR>0 + 292 without CDR; evaluation uses CN only. 3,336 all-eligible rows = 2,705 + 631 QC-flagged (425 of them CN).
**The 5.664 vs 3.504 gap is the 425 flagged CN scans** (raw MAE 10.8 y vs 3.1 y). The review images show registration/skull-strip
failures flagged by template correlation, not by prediction error; QC code (`core_v4.qc_bad_table`) never reads predictions.

## 2. Findings, separated by type

### Confirmed software defects
| # | sev | where | what | impact |
|---|---|---|---|---|
| S1 | medium (latent) | `10_eval_canonical.py` `main()` ckpt glob `<exp>_f{k}_s*.pt` | also matches `<tag>.resume.pt` written by the patched `11_train_experiment.py`. Demonstrated in `tests/test_v5_train.py::test_strict_checkpoint_regex...` | re-running the evaluator after patched training crashes or double-counts models. Existing results are unaffected (n_models = 2 everywhere). Fix: `v4_fix_checkpoint_glob.patch` (not applied) |
| S2 | low | `12_optuna_search.py` `objective` | inner val split reseeded per trial (`SEED + fold + trial.number`) | see M3 |

### Data-integrity issues
| # | what | evidence | impact |
|---|---|---|---|
| D1 | `v4_sfcn_baseline` folds 1-4 predictions are bit-identical to `v4_sfcn_baseline_40ep` (max abs diff 0.0); only fold 0 differs (max 8.1 y) | `raw_*.csv` comparison | The "retrained 300-epoch baseline" with V3-m1 learning rates exists only for fold 0. Table A's baseline row mixes configs. `config_v4.py` says Z-score is "identical to baseline except input_mode", true only on fold 0. |
| D2 | `parse_name` in `00_build_manifest_and_folds.py` ignores `echo-N` and `run-NN` after the session tag | 2 colliding `(subject, day, run)` keys: OAS31065 day 44, OAS30339 day 66 (different files, same key) | negligible numerically (1 CN scan pair in the evaluation set). Folds are frozen so I did not touch it; `v5_report.py` keys on `file_path` and reports collisions. |

### Methodological concerns (not bugs)
* **M1 Raw vs corrected.** Inversion `brain_age = (pred - b)/a` with a≈0.78 stretches predictions by ~1.28x. By age band (raw / corrected MAE): ≤55: 3.75/3.40, 55-65: 3.05/3.78, 65-75: **2.74/3.51**, 75-85: 3.00/3.12, >85: **7.15/4.45**. Correction helps the sparse extremes and hurts the dense middle.
  Residual-vs-age slope: raw -0.211, corrected +0.009.
* **M2 Extremes.** Ages >85 (75 scans) are under-predicted by 7.1 y; ≤50 over-predicted by 6.3 y. Only 11 CN scans are >90. Extremes are 6% of scans but 13% of absolute error. Bins span 38-100 so there is no hard ceiling; the pretrained UKB head covers 42-82 and bins above 82 start as copies of the edge bin. Data scarcity, not a bug.
* **M3 Optuna protocol.** Each trial scored on a different validation subject set, so the "best" trial partly reflects the luckiest split (30 trials). MedianPruner(warmup=3) penalises slow-converging low-LR configs. Trials had no warm-up/scheduler unlike final training. Best `batch_size=2` sits on the lower edge of the grid. The study used `meannorm` input, so its result does not transfer to the Z-score model. Its best value (2.589) is an inner-validation number and is not comparable to held-out MAE.
* **M4 Nested evaluation.** A configuration tuned on folds 1-4 may only be judged on fold 0. Judging it on folds 1-4 reuses tuning subjects.
* **M5 Selection noise.** Early stopping/checkpoint choice uses ~12% of train subjects (~100 subjects) per fold; single-run curves are noisy (EMA addresses this).
* **M6 Bias-fit provenance (verified in code).** Correction is fitted on out-of-fold CN sessions of the other four folds. Those predictions come from models trained on the held-out fold's subjects, a second-order effect that cannot involve held-out labels. Acceptable; state it.
* **M7 Seeds share the inner split.** Both seeds use the same validation subjects (by design, fine for comparability), so seed averaging does not average out selection noise from the split.

### Unverified (need data I do not have)
* Training/validation curves (`v4/checkpoints/*.json` were not in the zip). Please send them: they tell us whether the ~50-epoch early stops are under- or over-fitting.
* Whether `qc_manual_review` edits changed the QC-passed set after the V4 predictions were made.

## 3. Statistics that decide what is worth training (paired, subject-cluster bootstrap, raw MAE, QC-passed CN)

| comparison | Δ MAE (A - B) | 95% CI | verdict |
|---|---|---|---|
| Z-score vs baseline, **fold 0 only** (same config) | -0.418 | [-0.655, -0.181] | real |
| Z-score vs 40-ep baseline, all folds | -0.175 | [-0.267, -0.081] | real; lower in all 5 folds |
| Z-score vs V3 m1, all folds | -0.156 | [-0.278, -0.039] | real |
| Z-score vs KL-Huber, all folds | -0.082 | [-0.178, +0.011] | not distinguishable |
| mean-ensemble Z-score + KL-Huber vs Z-score | -0.003 | - | no gain (residual corr 0.93) |

Seed/TTA averaging: the stored between-seed spread (mean `sd_seed` 0.50, i.e. mean |seed1-seed2| ≈ 1.0 y) implies a per-seed SD ≈ 0.9 y; averaging 5 instead of 2 seeds removes ≈0.24 y² of a 15.9 y² MSE, i.e. RMSE 3.98→~3.95 and MAE ≈ -0.02 y at best. TTA spread is smaller (0.40). Not worth the compute.

## 4. Ranked improvement plan

| rank | experiment | expected raw-MAE gain | cost | risk | why |
|---|---|---|---|---|---|
| 1 | **`v5_z_khuber`**: Z-score input + Huber term (never combined) with EMA(0.999) tracked; one run yields raw- and EMA-weight models | 0.05-0.15 (uncertain; overlap likely) | ≈ 6.5-7 GPU-h (V4 Z-score used 23,354 s for 10 checkpoints) | low | the two best single changes; EMA reduces selection/step noise at batch size 4 for free |
| 2 | `v5_report.py` paired comparison + decision rule (below) | evidence | minutes | none | the only way to tell 0.08 y from noise |
| 3 | `v5_optuna.py` fixed-split search on Z-score+Huber, outer fold 0 (25 trials x 20 ep) | 0.03-0.10, fold-0-only evidence | ≈ 6 GPU-h + retrain fold 0 | medium (noisy; boundary effects) | only if rank 1 gives a gain; reported on fold 0 only unless run per fold (5x) |
| 4 | control `v5_z_ctl_ema` (Z-score + EMA only) | attribution | ≈ 6.5 GPU-h | none | only needed to attribute the effect in the paper |
| 5 | more seeds / ensembling existing models | ≲0.02 | - | - | measured: no gain |
| 6 | ResNet-34/50 from scratch, DenseNet | negative | high | high | already 1.0-1.1 y worse |
| 7 | age-balanced tail loss / re-initialising the >82 bins | unknown, targets M2 | medium | medium | hypothesis; only after 1-3 |

## 5. Decision rules (fixed now, before any V5 number exists)

A V5 model replaces `v4_sfcn_zscore_ablation` only if **all** hold on QC-passed CN scans, folds evaluated per the nesting rule:
1. paired Δ pooled **raw** MAE ≤ -0.05 y with 95% subject-cluster CI entirely below 0;
2. lower raw MAE in ≥4 of 5 folds (or the evaluated fold if only one);
3. corrected MAE not worse (CI) and |residual-vs-age slope| not larger;
4. RMSE not worse; all-eligible reported separately, same direction.
EMA weights are used only if their mean best **validation** MAE (from checkpoint JSONs, never the held-out fold) is lower than raw weights'. Otherwise keep the existing model and report V5 as a negative result.

## 6. What I changed

New files only (isolated `v5/` namespace; V4 untouched): `config_v5.py`, `v5_train.py`, `v5_predict.py`, `v5_report.py`, `v5_optuna.py`, `tests/`,
`v4_fix_checkpoint_glob.patch` (optional), `reference_outputs/`. V5 reuses V4's `split_train_val`/`evaluate` so the inner split cannot drift. Training data protocol is unchanged.

## 7. Tests run here

`python -m pytest tests -q` -> 27 passed (16 evaluator tests on synthetic data, 10 trainer tests, 1 Optuna test, tiny stand-in network on CPU).
Mutation check: making the correction use the held-out fold makes `test_crossfit_never_uses_heldout_labels` fail. Not tested here: real SFCN training, GPU/bf16 path, the UKB checkpoint load, real npz loading (no data/GPU in this session). Run `--dry_run` first on your machine.

## 8. Remaining risks

Single cohort, ~1,100 subjects; CI half-width ≈ 0.2 y means sub-0.1 y differences are not detectable without pairing; tuned-config results valid on one fold only; clinical conclusions (2 conversion events) unchanged by anything here.
