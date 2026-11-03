# -*- coding: utf-8 -*-
"""
config_v4.py -- V4 Phase 1 config: "freeze V3, build on it, never regenerate it."

VERIFIED against the actual methodology_v3.ipynb source (all 12 cells read directly,
not inferred from outputs) on this pass. Confirmed real V3 symbols this file now relies on:

  config_v3.py (imported as C in every V3 script):
    C.PREPROC_DIR, C.SCAN_META_CSV, C.CDR_CSV
    C.OUT, C.MANIFEST_CSV, C.FOLDS_CSV, C.BINS_JSON, C.QC_CSV
    C.CKPT_DIR, C.PRED_DIR, C.RES_DIR
    C.N_FOLDS, C.SEED, C.VAL_FRACTION, C.BIN_STEP, C.SIGMA, C.MEANNORM_CV
    C.AGE_BANDS, C.CLASS_CUTS, C.TTA_SHIFTS, C.SFCN_CKPT
    C.VARIANTS = {"m0": {...}, "m1": {...}, "m2": {...}}  -- each entry has (confirmed
        from 03_train_cv.py's cfg[...] accesses): unfreeze, lr_head, lr_backbone, llrd,
        l1, reweight, consistency, input_mode

  core_v3.py (imported as K):
    K.to_input(x, mode, cv), K.expected_age(lp, centers)
    K.load_pretrained_sfcn(ckpt_path, n_bins, bin_min, bin_step, device=dev)
        -- NOTE the real signature takes (n_bins, bin_min, bin_step), not just n_bins.
           Anything in V4 that reloads an SFCN must call it this way, not the 2-arg
           form in sfcn_model.py (that file was v2-era and is superseded by this).
    K.OLD_N_BINS, K.OLD_MIN, K.OLD_STEP   (UKB pretrained head: 40 bins, [42,82], step 1)
    K.ScanStore() with .add(paths, desc=...), .get(path), .arr (dict, .clear())
    K.ScanDS(store, df), K.PairDS(store, df)
    K.bin_centers(bins_dict, device), K.load_bins(path) -> dict(bin_min,bin_max,bin_step,sigma,n_bins)
    K.configure_trainable(model, unfreeze, lr_head, lr_backbone, llrd) -> (param_groups, trainable_names)
    K.tail_weights(ages), K.set_train_mode(model, trainable, bn_mode)
    K.gpu_augment(x), K.soft_labels(age, bins_dict, device), K.kl_loss(lp, y)
    K.shift_volume(x, shift)

  longitudinal_lib.py (imported as L):
    L.build_sessions(bag_df), L.build_windows(sessions, k, min_span_years, sigma)
    L.add_slope_variants(d, sigma), L.future_clinical_slopes(post, endpoint)

IMPORTANT: config_v3.py / core_v3.py / longitudinal_lib.py themselves are NOT present
in this sandbox -- only methodology_v3.ipynb's cell *usage* of them is. Every signature
above is transcribed verbatim from how the real V3 notebook calls them, across all 12
cells, so this is confirmed usage, not a guess. But V4 scripts below still need the
real .py files to exist on YOUR machine's PYTHONPATH (next to where you run V4) exactly
as they did for V3 -- nothing here recreates them. If any of the above turns out to be
stale (e.g. you've since refactored core_v3.py), the first V4 script you run will raise
an AttributeError/TypeError pointing at exactly which call needs to change.
"""

from pathlib import Path

# =========================================================
# Frozen V3 inputs -- read through config_v3 directly, not re-declared here,
# so V4 can never drift from whatever config_v3.py actually says (e.g. if you
# change C.SEED there, V4 picks it up automatically instead of silently using
# a stale copy).
# =========================================================
import config_v3 as C3   # noqa: E402  (V3's own config -- the single source of truth)

N_FOLDS = C3.N_FOLDS
SEED = C3.SEED
VAL_FRACTION = C3.VAL_FRACTION
AGE_BANDS = C3.AGE_BANDS
CLASS_CUTS = C3.CLASS_CUTS
SFCN_CKPT = C3.SFCN_CKPT
MANIFEST_CSV = C3.MANIFEST_CSV
FOLDS_CSV = C3.FOLDS_CSV
BINS_JSON = C3.BINS_JSON
QC_CSV = C3.QC_CSV

# =========================================================
# V4 outputs -- a sibling tree, never written into v3/
# =========================================================
OUT = Path("v4")
CKPT_DIR = OUT / "checkpoints"
PRED_DIR = OUT / "predictions"
RES_DIR = OUT / "results"
QC_REVIEW_DIR = OUT / "qc_review"
OPTUNA_DIR = OUT / "optuna"
for d in (OUT, CKPT_DIR, PRED_DIR, RES_DIR, QC_REVIEW_DIR, OPTUNA_DIR):
    d.mkdir(parents=True, exist_ok=True)

MAX_EPOCHS = 300
PATIENCE = 30
NUM_WORKERS = 4

# =========================================================
# EXPERIMENTS registry
# =========================================================
# Same dict shape as config_v3.VARIANTS (unfreeze/lr_head/lr_backbone/llrd/l1/
# reweight/consistency/input_mode) PLUS the new fields V4 needs: arch,
# activation, optimizer, loss, huber_delta, batch_size, description.
# core_v4.build_model() and 11_train_experiment.py read every field.
#
# arch        : "sfcn" (pretrained, transfer learning) | "resnet18" (no shipped
#               pretrained weights for this backbone -- trained from scratch;
#               see resnet3d.py's own docstring)
# activation  : "relu" | "gelu" -- swapped in AFTER pretrained weights load
#               (activations carry no parameters, so this never breaks loading)
# loss        : "kl" | "kl_l1" | "kl_huber"
# optimizer   : "adamw" | "adam"

EXPERIMENTS = {
    # ---- Phase 2A: architecture control (required baseline; v4_approach.txt
    # Phase 2.A says keep this fixed as "the reference against which V4 is measured") ----
    "v4_sfcn_baseline": dict(
        arch="sfcn", activation="relu", input_mode="meannorm",
        unfreeze="full", llrd=0.5, lr_head=3e-4, lr_backbone=3e-4,
        optimizer="adamw", weight_decay=1e-4, dropout=0.5,
        loss="kl_l1", l1_weight=0.1, huber_delta=1.0,
        reweight_tail=True, consistency=0.0, batch_size=4,
        description="Reproduces V3 m1's pretrained SFCN, full fine-tuning, and "
                     "layer-wise learning-rate decay, but uses V4's extended "
                     "training schedule and canonical evaluation pipeline.",
    ),

    # ---- Phase 2B: residual architecture experiment ----
    "v4_resnet18_residual": dict(
        arch="resnet18", activation="relu", input_mode="meannorm",
        unfreeze="full", llrd=1.0, lr_head=3e-4, lr_backbone=3e-4,
        optimizer="adamw", weight_decay=1e-4, dropout=0.5,
        loss="kl_l1", l1_weight=0.1, huber_delta=1.0,
        reweight_tail=True, consistency=0.0, batch_size=3,
        description="Residual-connection architecture (resnet3d.py 3D-ResNet18). "
                     "No pretrained checkpoint ships for this backbone in your "
                     "UKBiobank zip, so it trains from scratch -- per v4_approach.txt "
                     "Phase 2B, report its generalization gap (Table C: train-val MAE "
                     "gap, performance by outer fold) against v4_sfcn_baseline "
                     "explicitly, since 'lower training error but higher held-out MAE' "
                     "is not an improvement. '3D DenseNet' (Phase 2C) is NOT "
                     "implemented here -- flagged as future work, needs its own "
                     "architecture file and a GPU-memory budget check first.",
    ),

    # ---- Phase 3: activation ablation ----
    "v4_sfcn_gelu": dict(
        arch="sfcn", activation="gelu", input_mode="meannorm",
        unfreeze="full", llrd=0.5, lr_head=3e-4, lr_backbone=3e-4,
        optimizer="adamw", weight_decay=1e-4, dropout=0.5,
        loss="kl_l1", l1_weight=0.1, huber_delta=1.0,
        reweight_tail=True, consistency=0.0, batch_size=4,
        description="Identical to v4_sfcn_baseline except GELU replaces ReLU "
                     "after pretrained weights load. Everything else (data "
                     "pipeline, age-bin formulation) held fixed per Phase 3's "
                     "instruction to vary one factor at a time.",
    ),

    # ---- Phase 3: optimizer ablation ----
    "v4_sfcn_adam": dict(
        arch="sfcn", activation="relu", input_mode="meannorm",
        unfreeze="full", llrd=0.5, lr_head=3e-4, lr_backbone=3e-4,
        optimizer="adam", weight_decay=1e-4, dropout=0.5,
        loss="kl_l1", l1_weight=0.1, huber_delta=1.0,
        reweight_tail=True, consistency=0.0, batch_size=4,
        description="Identical to v4_sfcn_baseline except plain Adam (no "
                     "decoupled weight decay) replaces AdamW.",
    ),

    # ---- Phase 3: loss ablations ----
    "v4_sfcn_kl_only": dict(
        arch="sfcn", activation="relu", input_mode="meannorm",
        unfreeze="full", llrd=0.5, lr_head=3e-4, lr_backbone=3e-4,
        optimizer="adamw", weight_decay=1e-4, dropout=0.5,
        loss="kl", l1_weight=0.0, huber_delta=1.0,
        reweight_tail=True, consistency=0.0, batch_size=4,
        description="Identical to v4_sfcn_baseline except the 0.1*L1 term "
                     "(V3's own addition over plain UKB-style KL) is removed, "
                     "to test whether it is pulling its weight.",
    ),
    "v4_sfcn_kl_huber": dict(
        arch="sfcn", activation="relu", input_mode="meannorm",
        unfreeze="full", llrd=0.5, lr_head=3e-4, lr_backbone=3e-4,
        optimizer="adamw", weight_decay=1e-4, dropout=0.5,
        loss="kl_huber", l1_weight=0.0, huber_delta=1.0,
        reweight_tail=True, consistency=0.0, batch_size=4,
        description="Identical to v4_sfcn_baseline except the L1 term is "
                     "replaced by a Huber term (delta=1.0y). Per v4_approach.txt's "
                     "own caution, report it alongside raw MAE -- never let it "
                     "explain away a large |BAG| that Priority-1 QC should catch "
                     "instead; that is why 10_eval_canonical.py always reports "
                     "frac_abs_BAG_gt10 next to it.",
    ),

    # ---- Priority 2 (v3_problems.txt) / Phase 6 (v4_approach.txt): the
    # controlled input-convention ablation the zero-shot probe CANNOT settle on
    # its own (probe only tests the untouched pretrained backbone; after full
    # fine-tuning the network may adapt to either convention). ----
    "v4_sfcn_zscore_ablation": dict(
        arch="sfcn", activation="relu", input_mode="zscore",
        unfreeze="full", llrd=0.5, lr_head=3e-4, lr_backbone=3e-4,
        optimizer="adamw", weight_decay=1e-4, dropout=0.5,
        loss="kl_l1", l1_weight=0.1, huber_delta=1.0,
        reweight_tail=True, consistency=0.0, batch_size=4,
        description="Identical to v4_sfcn_baseline (same folds, seeds, "
                     "training settings) except input_mode='zscore'. This -- "
                     "not zero_shot_probe.csv's ~9.0y vs ~9.7-10.3y raw-MAE "
                     "numbers on the frozen backbone -- is the actual answer "
                     "to Priority 2. Compare directly against v4_sfcn_baseline "
                     "in Table A/B.",
    ),

    # ---- Frozen comparison copy of the pre-existing 40-epoch baseline.
    # The runner copies old baseline checkpoints to this tag before retraining
    # v4_sfcn_baseline with the extended schedule.
    "v4_sfcn_baseline_40ep": dict(
        arch="sfcn", activation="relu", input_mode="meannorm",
        unfreeze="full", llrd=1.0, lr_head=1.5e-4, lr_backbone=3e-4,
        optimizer="adamw", weight_decay=1e-4, dropout=0.5,
        loss="kl_l1", l1_weight=0.1, huber_delta=1.0,
        reweight_tail=True, consistency=0.0, batch_size=4,
        description="Checkpoint-preserved original V4 baseline before the "
                    "extended 300-epoch schedule. Use only if the old checkpoint "
                    "files were available and copied by run_v4_pipeline.sh.",
    ),

    # ---- Deeper transfer learning: pretrained UK Biobank SFCN plus 3
    # residual 3D convolution blocks at the final feature-map resolution.
    "v4_sfcn_deep_transfer": dict(
        arch="sfcn_deep", extra_blocks=3, activation="relu", input_mode="meannorm",
        unfreeze="full", llrd=0.5, lr_head=3e-4, lr_backbone=3e-4,
        optimizer="adamw", weight_decay=1e-4, dropout=0.5,
        loss="kl_l1", l1_weight=0.1, huber_delta=1.0,
        reweight_tail=True, consistency=0.0, batch_size=4,
        description="Transfer learning from the supplied UK Biobank-pretrained "
                    "SFCN, with three added residual 3D convolution blocks. "
                    "Only the added depth differs structurally from the baseline.",
    ),

    # ---- Higher-capacity ResNet comparisons. No matching pretrained ResNet
    # weights are supplied in this project archive; these train from scratch.
    "v4_resnet34_scratch": dict(
        arch="resnet34", activation="relu", input_mode="meannorm",
        unfreeze="full", llrd=1.0, lr_head=1e-4, lr_backbone=1e-4,
        optimizer="adamw", weight_decay=3e-4, dropout=0.5,
        loss="kl_l1", l1_weight=0.1, huber_delta=1.0,
        reweight_tail=True, consistency=0.0, batch_size=2,
        description="Deeper 3D ResNet-34 trained from scratch. This is a capacity "
                    "comparison, not a transfer-learning model.",
    ),
    "v4_resnet50_scratch": dict(
        arch="resnet50", activation="relu", input_mode="meannorm",
        unfreeze="full", llrd=1.0, lr_head=5e-5, lr_backbone=5e-5,
        optimizer="adamw", weight_decay=5e-4, dropout=0.5,
        loss="kl_l1", l1_weight=0.1, huber_delta=1.0,
        reweight_tail=True, consistency=0.0, batch_size=2,
        description="High-capacity 3D ResNet-50 trained from scratch. Compare "
                    "against ResNet-34 and the pretrained deeper SFCN; watch "
                    "the train-validation gap for overfitting.",
    ),
}

# Phase 4 (Optuna) search space, transcribed verbatim from v4_approach.txt's
# "Suggested initial ranges" table. Used by 12_optuna_search.py.
OPTUNA_SEARCH_SPACE = dict(
    lr_head=("loguniform", 1e-5, 1e-3),
    lr_backbone=("loguniform", 1e-6, 3e-4),
    weight_decay=("loguniform", 1e-6, 1e-3),
    dropout=("uniform", 0.2, 0.6),
    l1_weight=("uniform", 0.0, 0.3),
    llrd=("uniform", 0.5, 0.9),
    batch_size=("categorical", [2, 4, 8]),
    activation=("categorical", ["relu", "gelu"]),
)
OPTUNA_N_TRIALS_DEBUG = 30     # "about 30-50 trials on a single outer fold to debug"
OPTUNA_N_TRIALS_FULL = 50
OPTUNA_DEBUG_FOLD = 0
