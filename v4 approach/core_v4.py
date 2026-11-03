# -*- coding: utf-8 -*-
"""
core_v4.py -- new utilities V4 needs ON TOP OF core_v3. Everything that core_v3
(K) already does correctly for SFCN/meannorm-vs-zscore/soft-labels/KL is reused
verbatim through K, not reimplemented, per the standing instruction not to guess
at V3's private cv semantics. This file only adds what V3 never needed:
multi-architecture model construction (SFCN vs resnet18), an activation swap,
a Huber loss term, optimizer choice, and a trainable-param-group builder that
works for resnet18 too (K.configure_trainable's block-index parsing is SFCN-
specific -- see its real call site in 03_train_cv.py, which only ever builds
SFCN models).
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import core_v3 as K          # reused verbatim: to_input, soft_labels, kl_loss,
                              # ScanStore, ScanDS, PairDS, bin_centers, load_bins,
                              # expected_age, gpu_augment, shift_volume, tail_weights,
                              # set_train_mode, OLD_N_BINS/OLD_MIN/OLD_STEP
import config_v3 as C3
from sfcn_model import SFCN, load_pretrained_sfcn as _load_sfcn_2arg  # v2-era, SUPERSEDED for V4
from resnet3d import resnet18, resnet34, resnet50
from deep_sfcn import DeeperTransferSFCN


# =========================================================
# Model construction (arch + activation aware)
# =========================================================

def _swap_activation(module: nn.Module, kind: str):
    """In-place: replace every nn.ReLU in `module` with nn.GELU (or leave as
    ReLU). Activations carry no state, so this is always safe to do AFTER
    pretrained weights are loaded -- it cannot break load_state_dict."""
    if kind == "relu":
        return module
    if kind != "gelu":
        raise ValueError(f"unknown activation {kind!r}")
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.GELU())
        else:
            _swap_activation(child, kind)
    return module


def build_model(exp_cfg: dict, bins: dict, device):
    """
    exp_cfg : one row of config_v4.EXPERIMENTS
    bins    : dict from K.load_bins(config_v4.BINS_JSON) -- the FROZEN V3 bin
              scheme (bin_min, bin_max, bin_step, n_bins), shared by every V4
              model head so predictions are always on the same scale.

    Returns (model, trainable_param_names).
    """
    arch = exp_cfg["arch"]
    if arch in ("sfcn", "sfcn_deep"):
        # Transfer learning: load the actual UK Biobank-pretrained SFCN using
        # V3's strict checkpoint loader, including its copied age-bin head.
        base = K.load_pretrained_sfcn(
            C3.SFCN_CKPT, bins["n_bins"], bins["bin_min"], bins["bin_step"], device=device
        )
        if arch == "sfcn":
            model = base
            trainable = _configure_sfcn(model, exp_cfg)
        else:
            # Add trainable residual blocks after the pretrained feature
            # extractor while retaining the pretrained weights and head.
            model = DeeperTransferSFCN(
                base, extra_blocks=int(exp_cfg.get("extra_blocks", 3))
            )
            model.to(device)
            trainable = _configure_deep_sfcn(model, exp_cfg)
    elif arch in ("resnet18", "resnet34", "resnet50"):
        # The supplied archive contains pretrained SFCN weights, but no
        # matching ResNet checkpoint. These ResNet variants are trained from
        # scratch and must not be described as transfer-learning experiments.
        constructors = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50}
        model = constructors[arch](num_classes=bins["n_bins"], dropout=True)
        model.to(device)
        trainable = [n for n, _ in model.named_parameters()]
    else:
        raise ValueError(f"unknown arch {exp_cfg['arch']!r}")

    _swap_activation(model, exp_cfg["activation"])
    model.to(device)
    return model, trainable


def _configure_sfcn(model: SFCN, exp_cfg: dict):
    """
    SFCN-specific freeze/unfreeze, mirroring core_v3's configure_trainable
    semantics (confirmed call: K.configure_trainable(model, cfg["unfreeze"],
    cfg["lr_head"], cfg["lr_backbone"], cfg["llrd"])) but kept local here so
    core_v4 never has to assume core_v3's internal block-naming convention is
    unchanged -- this operates directly on the sfcn_model.SFCN module names
    (feature_extractor.conv_0..conv_5, classifier.*), which ARE fixed (they
    must match the pretrained checkpoint's state_dict).
    """
    # Accept the V3 config vocabulary as well as V4's names.
    # V3 used "all" for full fine-tuning and "last" for tail-only fine-tuning.
    unfreeze = {"all": "full", "last": "tail"}.get(exp_cfg["unfreeze"], exp_cfg["unfreeze"])
    if unfreeze not in ("tail", "full"):
        raise ValueError(f"unfreeze must be 'tail' or 'full' (V3 aliases: 'last'/'all'), got {unfreeze!r}")

    for name, param in model.named_parameters():
        if unfreeze == "full":
            param.requires_grad = True
        else:  # "tail" = V3's m0 style: final conv block (conv_5) + classifier only
            is_tail = name.startswith("classifier") or name.startswith("feature_extractor.conv_5")
            param.requires_grad = is_tail

    return [n for n, p in model.named_parameters() if p.requires_grad]


def _configure_deep_sfcn(model: DeeperTransferSFCN, exp_cfg: dict):
    """Freeze policy for the deeper SFCN: added residual blocks always train."""
    # Accept the V3 config vocabulary for compatibility with legacy configs.
    unfreeze = {"all": "full", "last": "tail"}.get(exp_cfg.get("unfreeze", "full"), exp_cfg.get("unfreeze", "full"))
    if unfreeze not in ("tail", "full"):
        raise ValueError(f"unfreeze must be 'tail' or 'full' (V3 aliases: 'last'/'all'), got {unfreeze!r}")
    for name, param in model.feature_extractor.named_parameters():
        if unfreeze == "full":
            param.requires_grad = True
        else:
            block_name = name.split(".")[0]
            block_idx = int(block_name.split("_")[1])
            param.requires_grad = block_idx >= 5
    for param in model.extra_blocks.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
    return [n for n, p in model.named_parameters() if p.requires_grad]


# =========================================================
# Optimizer construction (plain AdamW/Adam param groups, with optional LLRD)
# =========================================================

def build_optimizer(model, trainable_names, exp_cfg: dict):
    """
    Layer-wise LR decay (llrd < 1.0) scales the backbone LR down the deeper
    (earlier) a layer sits, going from classifier (full lr_head) backward
    through feature_extractor.conv_5 .. conv_0 (lr_backbone * llrd**depth).
    llrd == 1.0 reproduces plain two-group differential LR (V3 m1's scheme).
    """
    lr_head = exp_cfg["lr_head"]
    lr_backbone = exp_cfg["lr_backbone"]
    llrd = exp_cfg.get("llrd", 1.0)

    # order feature_extractor blocks shallow -> deep so decay compounds correctly
    block_order = ["conv_0", "conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

    groups = []
    for name, param in model.named_parameters():
        if name not in trainable_names:
            continue
        if name.startswith("classifier") or not name.startswith("feature_extractor"):
            groups.append({"params": [param], "lr": lr_head, "initial_lr": lr_head})
        else:
            block = name.split(".")[1]  # "feature_extractor.conv_3.0.weight" -> "conv_3"
            depth_from_output = len(block_order) - 1 - block_order.index(block) if block in block_order else 0
            lr = lr_backbone * (llrd ** depth_from_output)
            groups.append({"params": [param], "lr": lr, "initial_lr": lr})

    if exp_cfg["optimizer"] == "adamw":
        opt = torch.optim.AdamW(groups, weight_decay=exp_cfg["weight_decay"])
    elif exp_cfg["optimizer"] == "adam":
        opt = torch.optim.Adam(groups, weight_decay=exp_cfg["weight_decay"])
    else:
        raise ValueError(f"unknown optimizer {exp_cfg['optimizer']!r}")
    return opt


# =========================================================
# Loss (extends core_v3.kl_loss with an optional L1 or Huber term)
# =========================================================

def compute_loss(log_probs, soft_label, expected_age_pred, true_age, exp_cfg: dict):
    """
    loss == "kl"       : K.kl_loss(log_probs, soft_label) alone
    loss == "kl_l1"    : + l1_weight * |E[age] - age|                  (V3 m1's scheme)
    loss == "kl_huber" : + huber(E[age] - age, delta=huber_delta)      (new, Phase 3)
    """
    loss = K.kl_loss(log_probs, soft_label)
    kind = exp_cfg["loss"]
    if kind == "kl":
        return loss
    if kind == "kl_l1":
        return loss + exp_cfg["l1_weight"] * (expected_age_pred - true_age).abs().mean()
    if kind == "kl_huber":
        return loss + F.huber_loss(expected_age_pred, true_age, delta=exp_cfg["huber_delta"])
    raise ValueError(f"unknown loss {kind!r}")


def qc_bad_table(qc_df):
    """Return file_path + qc_bad using image-QC flags and optional manual review.

    CONFIRMED-FAILURE always excludes a scan. LOOKS-OK overrides an automated
    flag after visual review. Blank/unrecognized manual values preserve qc_flag.
    This function never uses prediction error to decide QC status.
    """
    if qc_df is None:
        return None
    if "file_path" not in qc_df.columns:
        raise ValueError("QC CSV must contain file_path")
    flags = pd.to_numeric(qc_df.get("qc_flag", 0), errors="coerce")
    if not hasattr(flags, "fillna"):
        flags = pd.Series(0, index=qc_df.index)
    bad = flags.fillna(0).eq(1)
    if "qc_manual_review" in qc_df.columns:
        manual = qc_df["qc_manual_review"].fillna("").astype(str).str.strip().str.upper()
        confirmed_fail = manual.isin(["CONFIRMED-FAILURE", "CONFIRMED_FAILURE", "FAIL"])
        looks_ok = manual.isin(["LOOKS-OK", "LOOKS_OK", "OK"])
        bad = (bad & ~looks_ok) | confirmed_fail
    return pd.DataFrame({"file_path": qc_df["file_path"], "qc_bad": bad.astype(bool)})


# =========================================================
# Parameter / memory accounting for Table C (Training efficiency)
# =========================================================

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(total_params=total, trainable_params=trainable)
