# =========================================================
# CV DIAGNOSTIC — Section III-A robustness addition
# =========================================================
"""
Standalone, SECONDARY diagnostic. It does NOT replace your production
train/val/test split (train_ensemble_v2.py) or touch test_df at all — it
only exists to put a confidence interval on the MAE you'd expect to see on
an unseen fold, since a single 70/15/15 split gives one point estimate that
is fragile with only a few hundred subjects.

Uses ONE pretrained-and-fine-tuned SFCN per fold (not the full 3-model
heterogeneous ensemble) to keep 5x the training cost reasonable — the
production ensemble's accuracy is dominated by the pretrained-SFCN
members anyway, so this is a fast, honest proxy for expected generalization.

Requires: manifest, train_df, val_df, test_df, BrainDataset, device,
NUM_WORKERS, CHECKPOINT_DIR already defined (same as train_ensemble_v2.py).
"""

import os
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from sfcn_model import load_pretrained_sfcn, freeze_sfcn_backbone, differential_lr_param_groups, expected_age_from_logprobs
from brainage_utils import num2vect, my_KLDivLoss

SFCN_CHECKPOINT = "./UKBiobank_deep_pretrain-master/brain_age/run_20190719_00_epoch_best_mae.p"
K_FOLDS = 5
BATCH_SIZE = 8
MAX_EPOCHS = 30       # fine-tuning a mostly-frozen backbone converges faster than from-scratch training
PATIENCE = 5
BIN_STEP, SIGMA = 1, 1.0
USE_AMP = torch.cuda.is_available()

# Pool everything EXCEPT the held-out test set's subjects.
cv_pool = pd_concat = __import__("pandas").concat([train_df, val_df], ignore_index=True)
cv_pool_cn = cv_pool[cv_pool["CDRTOT"] == 0.0].reset_index(drop=True)

gkf = GroupKFold(n_splits=K_FOLDS)
fold_maes = []

for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(cv_pool_cn, groups=cv_pool_cn["subject_id"])):
    fold_train = cv_pool_cn.iloc[tr_idx].reset_index(drop=True)
    fold_val = cv_pool_cn.iloc[va_idx].reset_index(drop=True)

    age_min = float(np.floor(fold_train["age"].min())) - 3
    age_max = float(np.ceil(fold_train["age"].max())) + 3
    bin_range = (age_min, age_max)
    n_bins = int((age_max - age_min) / BIN_STEP)
    _, bc_np = num2vect(np.array([age_min]), bin_range, BIN_STEP, SIGMA)
    bin_centers = torch.tensor(bc_np, dtype=torch.float32)

    train_loader = DataLoader(BrainDataset(fold_train, is_train=True, cache_in_ram=True),
                               batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                               pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(BrainDataset(fold_val, is_train=False, cache_in_ram=True),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=torch.cuda.is_available())

    torch.manual_seed(fold_idx)
    model = load_pretrained_sfcn(SFCN_CHECKPOINT, n_bins=n_bins, device=device)
    model = freeze_sfcn_backbone(model, unfreeze_from_block=5)
    optimizer = torch.optim.AdamW(differential_lr_param_groups(model, backbone_lr=1e-5, head_lr=1e-4), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_mae, patience_counter = float("inf"), 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            y_soft, _ = num2vect(targets.cpu().numpy(), bin_range, BIN_STEP, SIGMA)
            y_soft = torch.tensor(y_soft, dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
                log_probs = model(inputs).reshape(inputs.shape[0], -1)
                loss = my_KLDivLoss(log_probs, y_soft)
            loss.backward()
            optimizer.step()

        model.eval()
        abs_err, n = 0.0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                log_probs = model(inputs).reshape(inputs.shape[0], -1)
                pred_age = expected_age_from_logprobs(log_probs, bin_centers)
                abs_err += torch.abs(pred_age - targets).sum().item()
                n += targets.shape[0]
        val_mae = abs_err / n
        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae, patience_counter = val_mae, 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    print(f"Fold {fold_idx+1}/{K_FOLDS}: best Val MAE = {best_val_mae:.3f} years "
          f"(n_train={len(fold_train)}, n_val={len(fold_val)})")
    fold_maes.append(best_val_mae)

fold_maes = np.array(fold_maes)
mean_mae = fold_maes.mean()
sem_mae = fold_maes.std(ddof=1) / np.sqrt(K_FOLDS)
ci95 = 1.96 * sem_mae

print("\n" + "=" * 60)
print("5-FOLD SUBJECT-LEVEL CV — MAE ROBUSTNESS CHECK")
print("=" * 60)
print(f"Per-fold MAE: {np.round(fold_maes, 3).tolist()}")
print(f"Mean MAE: {mean_mae:.3f} years  (95% CI: {mean_mae-ci95:.3f} - {mean_mae+ci95:.3f})")
print("Compare this range against your single-split test MAE from train_ensemble_v2.py "
      "to see whether that point estimate is inside or outside the fold-to-fold spread.")
