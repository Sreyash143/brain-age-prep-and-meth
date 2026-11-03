# Brain-Age Reliability Pipeline — v3 Methodology

Contents: (1) what the data say is wrong in v2, (2) review of your v3 plan, (3) why the cited papers score better, (4) realistic targets, (5) execution order, (6) script inventory and what was tested.

---

## 1. Findings from your v2 outputs (computed from the CSVs you attached)

| Finding | Evidence | Consequence |
|---|---|---|
| **The bias "correction" made accuracy worse.** | CN MAE raw 4.52 → corrected 4.82. Predictions are compressed: pred ≈ 30 + 0.56·age on test CN. Dividing by the *training* slope (in-sample, ≈1) under-corrects; residual BAG-vs-age slope is −0.24 y/y (p≈6e-14). | Fit the correction on **out-of-fold** predictions (script 04b). The real fix is a less-compressed model. |
| **Errors are concentrated at the age extremes.** | MAE 6.1 for age <60 (mean BAG +3.8), 6.1 for age 80–90 (mean BAG −5.5). UKB pretraining covers 42–82 only; OASIS-3 spans 42–96. | Re-initialise the head over the OASIS range, tail-weighted sampling, unfreeze the backbone. |
| **A heavy tail of failures.** | 39/358 CN scans (11%) have \|BAG\|>10 y; MAE without them is 3.69. Repeat pairs such as OAS30094 differ by 11–13 y. | Almost certainly failed skull-strip/registration. Image-based QC (script 01). Never exclude by prediction error. |
| **Likely intensity-convention mismatch** (hypothesis, not proven). | The pretrained SFCN expects `x / mean(x)` (positive, background 0); v2 tensors are brain-only z-scores, and blocks 1–4 were frozen so the network could not adapt. | Zero-shot probe (script 02) settles it in ~20 min without training. |
| **Noise floor was under-estimated.** | v2 used only runs 1–2 of each session (38 sessions have 3+ runs). Pooled df-weighted σ_SEM = **1.81 y** (95% CI 1.22–2.39), MDC95 = 5.0 y; v2 reported 1.58 / 4.38. Robust (MAD) σ is only 0.56 → the SEM is driven by a few failed scans. ICC(A,1) = 0.94. | Report both plain and robust σ, and σ without QC-flagged sessions. |
| **Cell 18/19 of v2 leak.** | LME slopes were fitted on *all* visits (including post-baseline); BAG for training subjects was in-sample; `N=3` was a constant Cox covariate. The test split has only 34 subjects with ≥3 sessions. | v3 uses out-of-fold BAG for **every** subject (so survival can use the whole cohort) and baseline-window-only slopes. |
| **Your 3-class "accuracy" is not comparable to anything.** | Majority-class baseline 48.9%; 54% of scans lie within 5 y of a cut-off (60/75). | Report MAE; keep the 3-class number only as a footnote. |

Note on one tempting shortcut: regressing BAG on age and subtracting the trend (Beheshti-style) makes MAE look much better (3.56 on your own file) but it injects the true age into the prediction, so that MAE is **not an accuracy metric** (see your ref [12], de Lange & Cole). Script 04b computes it only to label it non-reportable.

## 2. Review of the v3 approach

**Keep as written:** Phases 0–2 (metric naming, 90% rejection, pre-registration, subject-level splitting), 3.2 repeat audit, Phase 5 principles, 6.1–6.8, 7, 8.1–8.3, 9.1–9.5, 10, 11, 12, 13.

**Changes / additions (all implemented):**

1. **Replace the single split with 5-fold subject-level cross-fitting.** Each scan is predicted only by models that never saw its subject. Gives (a) an honest accuracy estimate on ~all CN subjects instead of 122, (b) unbiased bias-correction parameters (fit on the other folds), (c) leakage-free BAG for the whole cohort, which is what makes the survival analysis possible at all. Caveat to state: the v2 test split had already been seen once; v3 decisions were made on folds/validation only.
2. **Fix the input convention before training** (script 02, zero-shot probe) — cheapest possible experiment with the largest suspected payoff.
3. **Head re-initialisation by bin copying** over 40–98 y instead of a fresh head; **full fine-tuning with layer-wise LR**, BatchNorm frozen in frozen blocks; **tail re-weighting**; KL + 0.1·L1 loss; GPU augmentation (the v2 torchio pipeline ran on CPU with `num_workers=0`).
4. **Pre-specified model set** (Step 4.8): m0 = v2-style baseline, m1 = v3 primary, m2 = m1 + repeat-consistency loss. A scratch 3D-ResNet is dropped from the default set (≈2k scans rarely suffice; keep your `resnet3d.py` if a reviewer wants a second family).
5. **Same-day σ_SEM is a lower bound for longitudinal noise** (no repositioning over months, scanner drift, non-linearity). Step 8.4 as written fixes the residual variance at σ_SEM². Script 06 does that (empirical-Bayes with known residual variance) **and** fits an unconstrained mixed model on the same windows, reporting the ratio of free residual SD to σ_SEM. If the ratio is well above 1, say so and use the σ sweep in script 08.
6. **Shrunk vs raw C-index will be ≈ equal by construction when SE is similar across subjects.** Shrunk = μ + rᵢ(raw − μ); if rᵢ is ~constant it is an affine copy of raw and a Cox model/C-index cannot tell them apart. The Step 9.6 primary test only has power if SE varies (different spans/visit counts). Script 07 prints CV(SE) and this warning. Consequently the defensible headline is **slope reliability and the noise floor** (Phase 6, 8.7), with the C-index comparison reported honestly as secondary. In a simulation I ran (300 subjects, 60 events, reliability 0.32), the paired difference was −0.007 (CI −0.026…+0.017).
7. **Paired bootstrap uses out-of-bag evaluation** (fit on the resample, score on subjects not drawn), not in-sample C-index.
8. **Time-dependent AUC** in script 07 is the naive (non-IPCW) version; install `scikit-survival` and swap in `cumulative_dynamic_auc` if censoring is heavy.
9. **Clinical block (9.7)** needs a CSV you supply (`subject_id`, APOE ε4, FreeSurfer hippocampus/ventricle columns); pass it with `--clinical`.
10. **Not automated** (needs metadata I don't have): 10.5 scanner/field-strength strata, 10.7 alternative CDR-matching rule, 3.3 within-session registration variant (needs your preprocessing code; the repeat-pair NCC in script 01 is the image-space proxy), 9.10 joint model (R `JM`/`joineRML` is the practical route).

## 3. Why the cited papers report better numbers

Figures marked ✔ were checked against the papers/abstracts; the others I did not verify — take them from each paper's results table.

| Ref | What they report | Why it is better than your setting |
|---|---|---|
| [4] Peng ✔ | MAE **2.14 y** on UK Biobank (N=14,503, 12,949 train); ensemble won PAC-2019 with 2.90 y (N=2,638) | ~13k training scans of a single, homogeneous protocol; narrow age window (bins 42–82); pretraining, augmentation, ensembling and a linear bias correction on held-out data. You fine-tune on ~2k scans with a head/range mismatch. |
| [5] Leonardsen ✔ | n=53,542 scans (ages 3–95); ~34k used for training; MAE ≈ **3.9 y** on unseen sites (n=2,553) | The honest comparison for *your* regime: even with 34k heterogeneous scans, external-site MAE is ~3.9. |
| [8] Lee ✔ | MRI model MAE **3.49 ± 0.16** in n=1,805 CN (ages 26–98) | A realistic target for a single-cohort CN set of this size. |
| [9] Gaser 2013 ✔ | "Accuracy rates of up to **81%**" predicting **MCI→AD conversion within 3 years** (ADNI, MCI patients) | **This is very likely the "80%" you remember.** It is a clinical classification accuracy in an MCI population, not an age-prediction accuracy. It is not comparable to your 70.9% age-group accuracy. |
| [1] Cole, [6] Bashyam, [7] Feng | MAE and r in healthy cohorts | Larger/wider training sets, healthy-only uniform samples; not verified here. |
| [13]–[15], [19] | Test–retest reliability comparisons of brain-age tools | Reliability is the gap your paper occupies; they rarely report MDC or slope reliability. |
| [16]–[18] | Uncertainty estimation | Relevant to Step 6.7 (calibration). |

So: **no cited paper reports a ≈80% age-group accuracy.** The age-prediction papers report MAE (2.1–3.9 y); the 81% figure is a conversion-classification accuracy. A reviewer will compare you on MAE and on conversion discrimination (C-index/AUC), not on the 3-class number. I would also not frame the paper as "we beat 80%".

## 4. Realistic targets

The following is my estimate, not a guarantee: fixing the input convention, the correction, the tail failures and the fine-tuning should plausibly bring CN MAE from 4.8 to roughly 3.3–3.9 y; 2.1 y is not attainable with ~2k scans. Relation between MAE and your 3-class metric (simulated with Gaussian errors on your age distribution):

| CN MAE | 4.8 | 4.0 | 3.5 | 3.0 | 2.5 |
|---|---|---|---|---|---|
| 3-class accuracy | ~74% | ~78% | ~80% | ~83% | ~85% |

(Your observed 70.9% at MAE 4.8 is below the Gaussian prediction because of the heavy tail and age-trend.) A 3-class accuracy near 80% therefore corresponds to MAE ≈ 3.5 — consistent with Lee et al. Clinical claims: BAG-slope C-index of 0.65–0.80 is plausible; ≥0.9 from brain age alone would indicate leakage.

## 5. Execution order

All commands run from the folder containing the scripts and `UKBiobank_deep_pretrain-master/`. Paths are set once in `config_v3.py`. `run_all.sh` lists the same sequence.

| # | Command | What it does | Checkpoint before continuing |
|---|---|---|---|
| 0 | `python 00_build_manifest_and_folds.py` | Manifest, cohort-flow counts, frozen 5-fold subject split, shared age bins, software versions, pre-registration template | Fill in `PREREGISTRATION_v3.md` now. Check `cohort_flow_v3.json`: events per fold, impaired subjects per fold. |
| 1 | `python 01_qc_scans.py` | Brain fraction, similarity to CN template, repeat-pair NCC; image-only flags | Open ~30 flagged and ~30 unflagged scans; freeze the QC rule. |
| 2 | `python 02_zero_shot_probe.py` | Pretrained SFCN, no training, 6 intensity conventions | Set `MEANNORM_CV` (or `input_mode='zscore'`) in `config_v3.py`. |
| 3 | `python 03_train_cv.py --variants m1 --folds 0 --seeds 1 --epochs 2` | Smoke test | Confirm no OOM (else `--bs 2 --bn_mode freeze`), time one epoch to budget the full run. |
| 4 | `python 03_train_cv.py --variants m0,m1,m2 --seeds 2` | 5 folds × (1 + 2 + 2) models; resumable | Compare per-fold best validation MAE of m0 vs m1. |
| 5 | `python 04a_predict_oof.py` | Out-of-fold predictions with 5-view TTA; ensemble and TTA spread | — |
| 6 | `python 04b_bias_correct_accuracy.py --variant m1` (also m0, m2) | Cross-fitted correction, MAE/RMSE/r/R² with subject-cluster CIs, age-band table, BAG by CDR | BAG-vs-age slope must be non-significant; BAG should rise with CDR. |
| 7 | `python 05_noise_floor.py --variant m1` (also m0, m2) | Pooled σ_SEM, MDC, cluster-bootstrap CIs, ICC, heteroscedasticity, CN vs impaired, calibration | Compare σ_SEM across m0/m1/m2 (the consistency-loss question). |
| 8 | `python 06_slopes.py --variant m1 --k 3 --min_span 2.0` | Window, raw/thresholded/shrunk/SNR slopes, SE, reliability, free mixed-model residual SD | Read `mean_reliability` and `residual_SD_free_over_sigma_SEM`. |
| 9 | `python 07_survival_validation.py --slopes v3/results/slopes_m1_k3_span2.csv --use_mmse` | Event-count rule, LRTs, repeated-CV C-index, OOB paired bootstrap, AUC(t), PH test, CDR-SB/MMSE endpoints | — |
| 10 | `python 08_robustness.py --variant m1` | k, σ bounds, span, σ outside the cohort, estimator comparison | — |
| 11 | `python 09_leakage_audit.py --variant m1 --min_span 2.0` | Automated Phase-11 checks (incl. scrambling post-window BAG to prove slopes don't change) | Must PASS before any number goes into the paper. |

Compute: I could not time training on your GPU. Full fine-tuning is heavier than v2's last-block training; with 25 models, plan in days, not hours, until you have timed one epoch. Reduce with `--seeds 1`.

## 6. Script inventory and test status

| File | Role | Status |
|---|---|---|
| `config_v3.py` | Paths, constants, variant definitions | Compiles |
| `core_v3.py` | SFCN, head expansion, soft labels, LLRD, GPU augmentation, data store | **Compiles only — not executed** (no PyTorch in my sandbox). |
| `00`, `01`, `02`, `03`, `04a` | Data, QC, probe, training, prediction | **Compile only — not executed** (need your data/GPU). |
| `04b`, `05`, `06`, `07`, `08`, `longitudinal_lib.py`, `09` | CPU analysis | Executed: 04b/05/06/08 on your v2 CSVs; 06→07 end-to-end on a simulated cohort (300 subjects, 60 events). 09 compiles only. |

Known risks to check on first run:
- `core_v3.SFCN` mirrors the official UKB layout from memory; `load_pretrained_sfcn` loads with `strict=True`, so a mismatch fails loudly with the key list — adapt layer names then.
- The z-score → mean-normalised conversion assumes background voxels are exactly 0 and uses a tunable CV; script 02 decides whether it helps.
- Your v2 `sfcn_model.py` / `brainage_utils.py` / `resnet3d.py` are **not** used; v3 is self-contained.
- Hyper-parameters (LRs, λ=0.1, consistency weight 0.5, bs 4) are reasonable defaults, not tuned.
- If OASIS-3 was among the pretraining cohorts of any public pretrained model you add (e.g. Leonardsen's), disclose it as potential leakage.

## 7. Manuscript fixes (Phase 13 additions)
Describe the BAG correction as cross-fitted; report MAE with CIs and the age-band table; replace the 3-class accuracy with MAE as headline; state that σ_SEM from same-day repeats is a lower bound; replace "subtract noise" language with shrinkage; state the shrunk-vs-raw invariance limitation; complete ref [19].
