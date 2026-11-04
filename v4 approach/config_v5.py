# -*- coding: utf-8 -*-
"""
config_v5.py -- V5 experiment namespace.  Reads V3/V4 config, writes ONLY under ./v5/.

Design rules (see README_V5.md):
  * V4 files, checkpoints and predictions are never written to.
  * Every V5 experiment is derived from the best V4 model (v4_sfcn_zscore_ablation)
    and changes ONE factor, so differences are attributable.
  * Same frozen folds, same per-fold inner validation split, same QC rule as V4.
"""
from pathlib import Path

import config_v3 as C3
import config_v4 as C4

OUT = Path("v5")
CKPT_DIR = OUT / "checkpoints"
PRED_DIR = OUT / "predictions"
RES_DIR = OUT / "results"
OPTUNA_DIR = OUT / "optuna"
REGISTRY_CSV = OUT / "experiment_registry.csv"

assert OUT.resolve() != C4.OUT.resolve() and OUT.resolve() != C3.OUT.resolve(), \
    "V5 output tree must differ from V3/V4 trees"
for _d in (OUT, CKPT_DIR, PRED_DIR, RES_DIR, OPTUNA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

BASE_EXPERIMENT = "v4_sfcn_zscore_ablation"     # current best (reference baseline)


def derive(**overrides):
    cfg = dict(C4.EXPERIMENTS[BASE_EXPERIMENT])
    cfg.update(overrides)
    return cfg


# EMA of weights: ema_decay=None disables.  When enabled the trainer saves TWO
# checkpoints from the SAME run (best raw weights, best EMA weights), so the EMA effect
# is a paired comparison and costs no extra training.
EXPERIMENTS = {
    # Round 1.  Z-score input (best single change) x Huber term (second-best single change).
    # They have never been combined.  Raw-weights checkpoint = tag "v5_z_khuber";
    # EMA checkpoint = tag "v5_z_khuber_ema" (see v5_predict.py).
    "v5_z_khuber": derive(
        loss="kl_huber", l1_weight=0.0, huber_delta=1.0, ema_decay=0.999,
        description="v4_sfcn_zscore_ablation with the L1 term replaced by Huber(delta=1) "
                    "(v4_sfcn_kl_huber's change). EMA(0.999) of weights tracked in the same run.",
    ),
        # V5.2.  Same as V5.1 (tail reweighting OFF, run as --name v5_1_no_tail --set reweight_tail=false) PLUS
    # subject-balanced sampling: scan weight 1/n_i from the TRAINING split only.  One factor vs V5.1.
    "v5_2_subject_balanced": derive(
        loss="kl_huber", l1_weight=0.0, huber_delta=1.0, ema_decay=0.999,
        reweight_tail=False, subject_balanced=True,
        description="V5.1 (no tail reweighting) + subject-balanced WeightedRandomSampler (w=1/n_scans_of_subject).",
    ),
    # Optional controls, only needed if the paper must attribute the effect to one factor.
    "v5_z_ctl_ema": derive(
        ema_decay=0.999,
        description="v4_sfcn_zscore_ablation re-run with EMA tracking only (control for EMA).",
    ),
    # V5.3: V5.1 baseline + mild age-aware sampling.
    # Tail reweighting and subject balancing remain disabled.
    "v5_3_mild_age_balanced": derive(
        loss="kl_huber",
        l1_weight=0.0,
        huber_delta=1.0,
        ema_decay=0.999,

        reweight_tail=False,
        subject_balanced=False,
        age_balanced=True,

        age_balance_bin_width=5,
        age_balance_exponent=0.25,
        age_balance_min_weight=0.5,
        age_balance_max_weight=2.0,

        description=(
            "V5.1 no-tail baseline + mild age-bin-balanced "
            "sampling, 5-year bins, exponent 0.25, "
            "weights capped at [0.5, 2.0]."
        ),
    ),
}

# Optuna search space for round 2 (only used by v5_optuna.py).  Centred on the V4 config,
# and the batch-size / dropout bounds are wider than V4's study, which hit the batch_size=2
# edge (see report).  Activation is fixed: GELU was clearly worse in V4.
OPTUNA_SPACE = dict(
    lr_head=("loguniform", 1e-4, 1e-3),
    lr_backbone=("loguniform", 3e-5, 6e-4),
    weight_decay=("loguniform", 1e-6, 1e-3),
    dropout=("uniform", 0.3, 0.6),
    llrd=("uniform", 0.3, 0.9),
    huber_delta=("categorical", [0.5, 1.0, 2.0]),
    batch_size=("categorical", [2, 4, 8]),
)
