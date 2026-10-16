# =========================================================
# TRAIN ENSEMBLE v2 — Revision 2 (Section III-C)
# =========================================================
"""
Drop-in replacement for your original Cells 11-14 (train ensemble,
validation, test predictions, brain age + BAG). It expects everything
from your existing Cells 6-10 to already exist in the notebook namespace:

    manifest, train_df, val_df, test_df, train_cn, val_cn,
    BrainDataset, NUM_WORKERS, device, RESULTS_DIR, CHECKPOINT_DIR

Everything from Section III-B (systemic age-bias correction, linregress)
and III-A (subject-level split) is UNCHANGED and reused as-is; only the
model / loss / ensemble composition (III-C) changes.

WHAT CHANGED vs. your original Cell 11:
  1. 2 of the 3 ensemble members are pretrained SFCN (UKBiobank_deep_pretrain),
     fine-tuned with differential LR on the final conv block + head only.
  2. The 3rd member is a 3D-ResNet-18 (structurally different architecture,
     trained from scratch — no pretrained 3D-ResNet weights ship in the
     repo you provided; see resnet3d.py docstring).
  3. Loss is Gaussian soft-label + KL divergence over discretized age bins
     (nn.L1Loss() is gone).
  4. Batch sizes are raised to use the RTX 5080's 16GB VRAM headroom
     (SFCN: 8, ResNet3D: 6 — both still AMP fp16).
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import linregress
from sfcn_model import *
from resnet3d import *
from brainage_utils import *

from sfcn_model import (
    SFCN, load_pretrained_sfcn, freeze_sfcn_backbone,
    differential_lr_param_groups, expected_age_from_logprobs,
)
from resnet3d import resnet18
from brainage_utils import num2vect, my_KLDivLoss

# ---------------------------------------------------------
# Config — hardware-tailored for i9 / 128GB RAM / RTX 5080 16GB
# ---------------------------------------------------------
SFCN_CHECKPOINT = "./UKBiobank_deep_pretrain-master/brain_age/run_20190719_00_epoch_best_mae.p"
BATCH_SIZE_SFCN = 8
BATCH_SIZE_RESNET = 6
NUM_WORKERS = 0
MAX_EPOCHS = 60
PATIENCE = 8
BIN_STEP = 1
SIGMA = 1.0
USE_AMP = torch.cuda.is_available()
USE_COMPILE = torch.cuda.is_available()  # RTX 50-series (Blackwell) supports torch.compile

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Age-bin scheme, derived from YOUR data (not UKB's [42,82])
# ---------------------------------------------------------
_age_min = float(np.floor(train_cn["age"].min())) - 3
_age_max = float(np.ceil(train_cn["age"].max())) + 3
if (_age_max - _age_min) % BIN_STEP != 0:
    _age_max += BIN_STEP - ((_age_max - _age_min) % BIN_STEP)  # keep divisible by bin_step
BIN_RANGE = (_age_min, _age_max)
N_BINS = int((BIN_RANGE[1] - BIN_RANGE[0]) / BIN_STEP)
_, _bin_centers_np = num2vect(np.array([BIN_RANGE[0]]), BIN_RANGE, BIN_STEP, SIGMA)
BIN_CENTERS = torch.tensor(_bin_centers_np, dtype=torch.float32)

print("=" * 70)
print("AGE-BIN SCHEME (derived from train_cn, not hardcoded from UK Biobank)")
print("=" * 70)
print(f"bin_range={BIN_RANGE}, bin_step={BIN_STEP}, sigma={SIGMA}, n_bins={N_BINS}")

# ---------------------------------------------------------
# Datasets built ONCE, DataLoaders per-architecture on top of them
# (avoids re-caching the ~33GB volume set twice)
# ---------------------------------------------------------
train_ds = BrainDataset(train_cn, is_train=True, cache_in_ram=True)
val_ds = BrainDataset(val_cn, is_train=False, cache_in_ram=True)
test_ds = BrainDataset(test_df, is_train=False, cache_in_ram=True)


def make_loaders(batch_size):
    kwargs = dict(batch_size=batch_size, num_workers=NUM_WORKERS,
                  pin_memory=torch.cuda.is_available())
    return (
        DataLoader(train_ds, shuffle=True, **kwargs),
        DataLoader(val_ds, shuffle=False, **kwargs),
        DataLoader(test_ds, shuffle=False, **kwargs),
    )


# ---------------------------------------------------------
# Soft-label training / eval epoch (shared by both architectures)
# ---------------------------------------------------------
def run_epoch(model, loader, optimizer, train=True):
    model.train() if train else model.eval()
    total_loss, total_abs_error, n = 0.0, 0.0, 0
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP) if train else None
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            y_soft, _ = num2vect(targets.detach().cpu().numpy(), BIN_RANGE, BIN_STEP, SIGMA)
            y_soft = torch.tensor(y_soft, dtype=torch.float32, device=device)

            if train:
                optimizer.zero_grad(set_to_none=True)

            try:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
                    log_probs = model(inputs).reshape(inputs.shape[0], -1)
                    loss = my_KLDivLoss(log_probs, y_soft)

                if train:
                    if USE_AMP:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("CUDA OOM: skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                raise

            with torch.no_grad():
                pred_age = expected_age_from_logprobs(log_probs.detach().float(), BIN_CENTERS)
                total_abs_error += torch.abs(pred_age - targets).sum().item()

            bs = inputs.size(0)
            total_loss += float(loss.detach().item()) * bs
            n += bs

    if n == 0:
        return float("nan"), float("nan")
    return total_loss / n, total_abs_error / n


def train_one_model(model, optimizer, scheduler, train_loader, val_loader, tag, model_idx):
    best_path = os.path.join(CHECKPOINT_DIR, f"model_{model_idx}_{tag}_best.pt")
    best_val, patience_counter = float("inf"), 0
    history = {"train_mae": [], "val_mae": []}

    print("\n" + "=" * 70)
    print(f"MODEL {model_idx + 1} ({tag})")
    print("=" * 70)

    for epoch in range(MAX_EPOCHS):
        t0 = time.time()
        _, train_mae = run_epoch(model, train_loader, optimizer, train=True)
        _, val_mae = run_epoch(model, val_loader, optimizer, train=False)
        scheduler.step(val_mae)

        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)
        print(f"Epoch {epoch+1:02d}/{MAX_EPOCHS} | Train MAE={train_mae:.2f} | "
              f"Val MAE={val_mae:.2f} | LR={optimizer.param_groups[0]['lr']:.6f} | "
              f"{time.time()-t0:.1f}s")

        if val_mae < best_val:
            best_val, patience_counter = val_mae, 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1} (best Val MAE={best_val:.2f})")
                break
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    print(f"Model {model_idx+1} ({tag}) complete | Best Val MAE={best_val:.3f} years")
    return model, history


def maybe_compile(model):
    if USE_COMPILE:
        try:
            return torch.compile(model)
        except Exception as e:
            print(f"torch.compile unavailable ({e}); continuing uncompiled.")
    return model


# ---------------------------------------------------------
# Build + train the heterogeneous ensemble
# ---------------------------------------------------------
models, model_tags, histories = [], [], {}

sfcn_train_loader, sfcn_val_loader, sfcn_test_loader = make_loaders(BATCH_SIZE_SFCN)

for seed in (0, 1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    sfcn = load_pretrained_sfcn(SFCN_CHECKPOINT, n_bins=N_BINS, device=device)
    sfcn = freeze_sfcn_backbone(sfcn, unfreeze_from_block=5)
    sfcn = maybe_compile(sfcn)
    optimizer = torch.optim.AdamW(differential_lr_param_groups(sfcn, backbone_lr=1e-5, head_lr=1e-4),
                                   weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7)
    sfcn, hist = train_one_model(sfcn, optimizer, scheduler, sfcn_train_loader, sfcn_val_loader,
                                  tag="sfcn_pretrained", model_idx=len(models))
    models.append(sfcn); model_tags.append("sfcn_pretrained"); histories[len(models)-1] = hist

resnet_train_loader, resnet_val_loader, resnet_test_loader = make_loaders(BATCH_SIZE_RESNET)

torch.manual_seed(2)
np.random.seed(2)
resnet = resnet18(num_classes=N_BINS, dropout=True).to(device)
resnet = maybe_compile(resnet)
optimizer = torch.optim.AdamW(resnet.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7)
resnet, hist = train_one_model(resnet, optimizer, scheduler, resnet_train_loader, resnet_val_loader,
                                tag="resnet18_scratch", model_idx=len(models))
models.append(resnet); model_tags.append("resnet18_scratch"); histories[len(models)-1] = hist

print("\n" + "=" * 70)
print("ALL MODELS TRAINED")
for i, tag in enumerate(model_tags):
    print(f"Model {i+1} ({tag}) best Val MAE: {min(histories[i]['val_mae']):.3f} years")

# ---------------------------------------------------------
# Ensemble prediction helper (handles each model's own preferred batch size)
# ---------------------------------------------------------
_BATCH_BY_TAG = {"sfcn_pretrained": BATCH_SIZE_SFCN, "resnet18_scratch": BATCH_SIZE_RESNET}


@torch.no_grad()
def ensemble_predict(models, tags, df):
    all_model_preds = []
    for model, tag in zip(models, tags):
        model.eval()
        loader = DataLoader(BrainDataset(df, is_train=False), batch_size=_BATCH_BY_TAG[tag],
                             shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
        preds = []
        for inputs, _ in loader:
            inputs = inputs.to(device, non_blocking=True)
            log_probs = model(inputs)
            pred_age = expected_age_from_logprobs(log_probs, BIN_CENTERS)
            preds.extend(pred_age.detach().cpu().numpy().tolist())
        all_model_preds.append(np.asarray(preds, dtype=float))

    stacked = np.vstack(all_model_preds)
    out = df.copy()
    out["predicted_age"] = stacked.mean(axis=0)
    out["predicted_age_sd"] = stacked.std(axis=0)
    return out, stacked


# ---------------------------------------------------------
# Validation, test predictions (unchanged logic from your Cells 12-13)
# ---------------------------------------------------------
val_predictions, val_stacked = ensemble_predict(models, model_tags, val_cn)
val_mae = float(np.mean(np.abs(val_predictions["predicted_age"] - val_predictions["age"])))
val_rmse = float(np.sqrt(np.mean((val_predictions["predicted_age"] - val_predictions["age"]) ** 2)))
print(f"\nEnsemble validation MAE : {val_mae:.3f} years")
print(f"Ensemble validation RMSE: {val_rmse:.3f} years")

test_predictions, test_stacked = ensemble_predict(models, model_tags, test_df)
test_predictions.to_csv(os.path.join(RESULTS_DIR, "test_predictions_raw_v2.csv"), index=False)

# ---------------------------------------------------------
# Systemic age-bias correction — Section III-B, UNCHANGED
# ---------------------------------------------------------
train_predictions, _ = ensemble_predict(models, model_tags, train_cn)
x = train_predictions["age"].to_numpy(dtype=float)
y = train_predictions["predicted_age"].to_numpy(dtype=float)
slope, intercept, r_value, p_value, std_err = linregress(x, y)
if abs(slope) < 1e-12:
    raise RuntimeError("Bias-correction slope is too close to zero.")

test_predictions["brain_age"] = (test_predictions["predicted_age"] - intercept) / slope
test_predictions["BAG"] = test_predictions["brain_age"] - test_predictions["age"]
test_predictions.to_csv(os.path.join(RESULTS_DIR, "brainage_predictions_v2.csv"), index=False)

print("\n" + "=" * 70)
print("FINAL RESULTS — v2 HETEROGENEOUS PRETRAINED ENSEMBLE")
print("=" * 70)
print(f"Bias correction slope/intercept : {slope:.4f} / {intercept:.4f}")
test_mae = float(np.mean(np.abs(test_predictions["brain_age"] - test_predictions["age"])))
test_rmse = float(np.sqrt(np.mean((test_predictions["brain_age"] - test_predictions["age"]) ** 2)))
print(f"Test MAE  : {test_mae:.3f} years")
print(f"Test RMSE : {test_rmse:.3f} years")
print(f"Mean test BAG : {test_predictions['BAG'].mean():.3f} years")
