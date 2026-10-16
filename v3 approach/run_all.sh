#!/usr/bin/env bash
# v3 execution order. Run from the folder containing these scripts (and ./UKBiobank_deep_pretrain-master/).
# Each GPU step is resumable. STOP at every "CHECKPOINT" and read the printed numbers before continuing.
set -e
python 00_build_manifest_and_folds.py                 # freeze folds, manifest, bins; then FILL IN v3/PREREGISTRATION_v3.md
python 01_qc_scans.py                                 # CHECKPOINT: eyeball flagged scans, freeze QC rule
python 02_zero_shot_probe.py                          # CHECKPOINT: set MEANNORM_CV / input_mode in config_v3.py
python 03_train_cv.py --variants m1 --folds 0 --seeds 1 --epochs 2    # smoke test (minutes). Time one epoch -> plan the full run
python 03_train_cv.py --variants m0,m1,m2 --seeds 2   # full run (hours-days, restart-safe)
python 04a_predict_oof.py --variants m0,m1,m2
for v in m0 m1 m2; do python 04b_bias_correct_accuracy.py --variant $v; done        # CHECKPOINT: accuracy + BAG-vs-age check
for v in m0 m1 m2; do python 05_noise_floor.py --variant $v; done                   # CHECKPOINT: sigma_SEM / MDC / ICC per model
python 06_slopes.py --variant m1 --k 3 --min_span 2.0
python 07_survival_validation.py --slopes v3/results/slopes_m1_k3_span2.csv --use_mmse
python 08_robustness.py --variant m1
python 09_leakage_audit.py --variant m1 --min_span 2.0                               # must PASS before writing numbers
