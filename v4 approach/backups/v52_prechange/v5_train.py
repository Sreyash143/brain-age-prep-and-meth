# -*- coding: utf-8 -*-
"""
v5_train.py -- V5 trainer.  Same data protocol as V4's 11_train_experiment.py (it IMPORTS that
module's split/eval helpers so the subject-level split cannot drift), plus:

  * EMA of the weights (cfg["ema_decay"]); best-raw AND best-EMA checkpoints from one run
  * one row per run appended to v5/experiment_registry.csv (git commit, hyper-parameters,
    seed, best epoch, val MAE, runtime, peak memory, checkpoint path, status/failure reason)
  * writes only to v5/checkpoints; refuses names that collide with V4 experiments
  * --set key=value overrides (e.g. Optuna's best params) under a NEW --name

Protocol facts inherited unchanged from V4 (verified in code):
  - train pool = CN scans (CDRTOT==0) of the 4 other folds, QC-bad scans removed
  - inner validation = 12% of TRAIN SUBJECTS, split seeded by (C3.SEED + fold), identical across
    seeds/experiments; held-out fold is never used for early stopping or checkpoint choice
  - selection metric = raw (uncorrected) expected-age MAE on the inner validation subjects

Usage (from the project root, same venv):
    python v5_train.py --experiment v5_z_khuber --folds 0 --seeds 1 --dry_run
    python v5_train.py --experiment v5_z_khuber --folds 0,1,2,3,4 --seeds 2 --record_train_mae
"""
import argparse
import copy
import csv
import importlib
import json
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

import config_v3 as C3
import config_v4 as C4
import config_v5 as C5
import core_v3 as K
import core_v4 as K4

T = importlib.import_module("11_train_experiment")      # V4 helpers, reused verbatim


# ------------------------------------------------------------------ EMA
class EMA:
    """Exponential moving average of parameters; BatchNorm buffers are copied from the live model."""

    def __init__(self, model, decay):
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1.0 + step) / (10.0 + step))     # warm-up so early EMA is not dominated by init
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.mul_(d).add_(p.detach(), alpha=1.0 - d)
        for eb, b in zip(self.model.buffers(), model.buffers()):
            eb.copy_(b)


# ------------------------------------------------------------------ helpers
def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5).stdout.strip()
        return (out.stdout.strip() or "no-commit") + ("+dirty" if dirty else "")
    except Exception:
        return "not-a-git-repo"


REGISTRY_FIELDS = ["tag", "experiment", "status", "failure_reason", "git_commit", "arch", "init_checkpoint",
                   "input_mode", "loss", "hyperparameters_json", "fold", "seed", "best_val_epoch",
                   "best_val_mae_raw", "best_val_epoch_ema", "best_val_mae_ema", "epochs_run",
                   "runtime_seconds", "peak_gpu_memory_gb", "checkpoint_raw", "checkpoint_ema", "prediction_path"]


def registry_append(row):
    C5.REGISTRY_CSV.parent.mkdir(parents=True, exist_ok=True)
    new = not C5.REGISTRY_CSV.exists()
    with open(C5.REGISTRY_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def load_manifest_for_training():
    """Same manifest + QC rule as 11_train_experiment.main()."""
    man = pd.read_csv(C3.MANIFEST_CSV)
    qc = pd.read_csv(C3.QC_CSV) if C3.QC_CSV.exists() else None
    qc_bad = K4.qc_bad_table(qc)
    if qc_bad is None:
        return man
    man = man.merge(qc_bad, on="file_path", how="left")
    man["qc_bad"] = man["qc_bad"].fillna(False).astype(bool)
    print(f"QC filtering: excluded {int(man.qc_bad.sum())} of {len(man)} manifest scans from training/validation.")
    return man[~man.qc_bad].copy()


def make_loaders(cfg, tr, va, store, workers, dev):
    pin = dev.type == "cuda"
    train_ds, val_ds = K.ScanDS(store, tr), K.ScanDS(store, va)
    if cfg.get("reweight_tail"):
        sampler = WeightedRandomSampler(K.tail_weights(tr.age.values), num_samples=len(tr), replacement=True)
        tl = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler, num_workers=workers,
                        pin_memory=pin, drop_last=False)
    else:
        tl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=workers,
                        pin_memory=pin, drop_last=False)
    vl = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=workers, pin_memory=pin)
    return tl, vl


# ------------------------------------------------------------------ one training run
def train_one(name, cfg, fold, seed, man, store, bins, dev, *, epochs, patience, workers, cv,
              record_train_mae=False, force=False, save=True, ckpt_dir=None, build_model_fn=None):
    """Train one (fold, seed).  save=False => no files at all (used by Optuna trials).
    Returns a summary dict (always has 'status')."""
    ckpt_dir = Path(ckpt_dir or C5.CKPT_DIR)
    if save:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    build_model_fn = build_model_fn or K4.build_model
    tag = f"{name}_f{fold}_s{seed}"
    ckpt, ckpt_ema = ckpt_dir / f"{tag}.pt", ckpt_dir / f"{tag}.ema.pt"
    meta, resume_ckpt = ckpt_dir / f"{tag}.json", ckpt_dir / f"{tag}.resume.pt"

    if save and ckpt.exists() and meta.exists() and not force:
        print(f"[skip] {tag} already completed")
        return dict(status="skipped", tag=tag)
    if save and force:
        for p in (meta, resume_ckpt):
            if p.exists():
                p.unlink()

    torch.manual_seed(seed * 1000 + fold)
    np.random.seed(seed * 1000 + fold)

    pool = man[(man.fold != fold) & (man.CDRTOT == 0)]
    tr, va = T.split_train_val(pool, C3.SEED + fold, C3.VAL_FRACTION)     # identical to V4 (same function, same seed)
    assert not (set(tr.subject_id) & set(va.subject_id)), "train/val subject overlap"
    assert (tr.fold != fold).all() and (va.fold != fold).all(), "held-out fold leaked into training"
    store.add(pd.concat([tr, va]).file_path)

    centers = K.bin_centers(bins, dev)
    model, trainable = build_model_fn(cfg, bins, dev)
    opt = K4.build_optimizer(model, trainable, cfg)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, cooldown=1, min_lr=1e-7)
    ema = EMA(model, cfg["ema_decay"]) if cfg.get("ema_decay") else None
    tl, vl = make_loaders(cfg, tr, va, store, workers, dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    best, best_ep, bad, hist, step, start_epoch = 1e9, 0, 0, [], 0, 0
    best_e, best_e_ep = 1e9, 0
    t_start = time.time()

    if save and not force and resume_ckpt.exists():
        st = T._load_torch(resume_ckpt, dev)
        if isinstance(st, dict) and st.get("checkpoint_kind") == "v5_training_resume":
            model.load_state_dict(st["model_state_dict"])
            opt.load_state_dict(st["optimizer_state_dict"])
            sched.load_state_dict(st["scheduler_state_dict"])
            if ema is not None and st.get("ema_state_dict") is not None:
                ema.model.load_state_dict(st["ema_state_dict"])
            start_epoch, best, best_ep = int(st["next_epoch"]), float(st["best_val_mae"]), int(st["best_epoch"])
            best_e, best_e_ep = float(st["best_val_mae_ema"]), int(st["best_epoch_ema"])
            bad, hist, step = int(st["bad_epochs"]), list(st["history"]), int(st["step"])
            torch.set_rng_state(st["torch_rng_state"].cpu())
            np.random.set_state(st["numpy_rng_state"])
            print(f"[resume] {tag}: epoch {start_epoch + 1}/{epochs}, best raw val {best:.3f}")
        else:
            print(f"[WARN] {resume_ckpt} is not a v5 resume file; ignoring it and starting fresh")

    print(f"\n=== {tag} | train scans {len(tr)} ({tr.subject_id.nunique()} subj) | val scans {len(va)} "
          f"({va.subject_id.nunique()} subj) | EMA {cfg.get('ema_decay')}")
    status, reason = "completed", ""
    try:
        for ep in range(start_epoch, epochs):
            t0 = time.time()
            model.train()
            tot, n = 0.0, 0
            for x, age in tl:
                if ep == 0:                                   # one-epoch warm-up, as in V4
                    f = min(1.0, (step + 1) / max(1, len(tl)))
                    for g in opt.param_groups:
                        if "initial_lr" in g:
                            g["lr"] = g["initial_lr"] * f
                x, age = x.to(dev, non_blocking=True), age.to(dev)
                x = K.gpu_augment(K.to_input(x, cfg["input_mode"], cv))
                y = K.soft_labels(age, bins, dev)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    lp = model(x)
                ea = K.expected_age(lp, centers)
                loss = K4.compute_loss(lp, y, ea, age, cfg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                if ema is not None:
                    ema.update(model, step)
                tot += loss.item() * len(age)
                n += len(age)
                step += 1

            val_mae = T.evaluate(model, vl, cfg, centers, dev, cv)
            val_ema = T.evaluate(ema.model, vl, cfg, centers, dev, cv) if ema is not None else None
            train_mae = T.evaluate(model, tl, cfg, centers, dev, cv) if record_train_mae else None
            if ep >= 1:
                sched.step(val_mae)                           # scheduler follows RAW val MAE, as in V4
            hist.append(dict(epoch=ep + 1, train_loss=tot / max(n, 1), val_mae=val_mae, val_mae_ema=val_ema,
                             train_mae=train_mae, epoch_seconds=time.time() - t0))
            improved = False
            if val_mae < best:
                best, best_ep, improved = val_mae, ep + 1, True
                if save:
                    T._atomic_torch_save(model.state_dict(), ckpt)
            if val_ema is not None and val_ema < best_e:
                best_e, best_e_ep, improved = val_ema, ep + 1, True
                if save:
                    T._atomic_torch_save(ema.model.state_dict(), ckpt_ema)
            bad = 0 if improved else bad + 1
            print(f"ep {ep + 1:03d}/{epochs} | loss {tot / max(n, 1):.3f} | val {val_mae:.3f}"
                  + (f" | val_ema {val_ema:.3f}" if val_ema is not None else "")
                  + f" | lr {opt.param_groups[0]['lr']:.2e} | {time.time() - t0:.0f}s")
            if save:
                T._atomic_torch_save(dict(
                    checkpoint_kind="v5_training_resume", tag=tag, next_epoch=ep + 1, best_val_mae=best,
                    best_epoch=best_ep, best_val_mae_ema=best_e, best_epoch_ema=best_e_ep, bad_epochs=bad,
                    step=step, history=hist, model_state_dict=model.state_dict(),
                    optimizer_state_dict=opt.state_dict(), scheduler_state_dict=sched.state_dict(),
                    ema_state_dict=ema.model.state_dict() if ema is not None else None,
                    torch_rng_state=torch.get_rng_state(), numpy_rng_state=np.random.get_state()), resume_ckpt)
            if bad >= patience:
                print("early stop")
                break
    except KeyboardInterrupt:
        status, reason = "interrupted", "KeyboardInterrupt"
        print(f"\n[INTERRUPTED] {tag}: latest completed epoch checkpoint preserved.")
        raise

    except Exception as e:
        status, reason = "failed", f"{type(e).__name__}: {e}"
        traceback.print_exc()
        raise

    finally:
        runtime = time.time() - t_start
        peak = (torch.cuda.max_memory_allocated(dev) / 1e9) if dev.type == "cuda" else None
        summary = dict(tag=tag, experiment=name, status=status, failure_reason=reason, git_commit=git_commit(),
                       arch=cfg["arch"], init_checkpoint=str(C3.SFCN_CKPT) if cfg["arch"].startswith("sfcn") else "scratch",
                       input_mode=cfg["input_mode"], loss=cfg["loss"],
                       hyperparameters_json=json.dumps({k: v for k, v in cfg.items() if k != "description"}, default=str),
                       fold=fold, seed=seed, best_val_epoch=best_ep, best_val_mae_raw=best,
                       best_val_epoch_ema=best_e_ep if ema is not None else "",
                       best_val_mae_ema=best_e if ema is not None else "", epochs_run=len(hist),
                       runtime_seconds=round(runtime, 1), peak_gpu_memory_gb=peak,
                       checkpoint_raw=str(ckpt) if save else "", checkpoint_ema=str(ckpt_ema) if (save and ema is not None) else "",
                       prediction_path="")
        if save:
            registry_append(summary)
    if save:
        T._atomic_json_dump(dict(summary, cfg={k: v for k, v in cfg.items()}, history=hist, val_subjects=sorted(va.subject_id.unique()),
                                 n_train=len(tr), n_val=len(va), torch=torch.__version__, completed=True,
                                 **K4.count_params(model)), meta)
    print(f"{tag} done | best raw val {best:.3f} @ep{best_ep}"
          + (f" | best EMA val {best_e:.3f} @ep{best_e_ep}" if ema is not None else "") + f" | {runtime:.0f}s")
    del model, opt
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return summary


# ------------------------------------------------------------------ CLI
def parse_overrides(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=list(C5.EXPERIMENTS))
    ap.add_argument("--name", default=None, help="tag prefix; REQUIRED with --set so overridden runs never reuse a name")
    ap.add_argument("--set", action="append", help="override a config field, e.g. --set lr_head=4.6e-4")
    ap.add_argument("--folds", default=",".join(map(str, range(C3.N_FOLDS))))
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=C4.MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=C4.PATIENCE)
    ap.add_argument("--workers", type=int, default=C4.NUM_WORKERS)
    ap.add_argument("--cv", type=float, default=C3.MEANNORM_CV)
    ap.add_argument("--record_train_mae", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="build splits, assert subject-disjointness, print sizes, exit")
    args = ap.parse_args()

    cfg = dict(C5.EXPERIMENTS[args.experiment])
    over = parse_overrides(args.set)
    if over and not args.name:
        ap.error("--set requires --name")
    cfg.update(over)
    name = args.name or args.experiment
    if name in C4.EXPERIMENTS:
        ap.error(f"--name {name!r} collides with a V4 experiment")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev} torch={torch.__version__} git={git_commit()} cfg={ {k: v for k, v in cfg.items() if k != 'description'} }")
    man = load_manifest_for_training()
    bins = K.load_bins(C3.BINS_JSON)

    if args.dry_run:
        for f in map(int, args.folds.split(",")):
            pool = man[(man.fold != f) & (man.CDRTOT == 0)]
            tr, va = T.split_train_val(pool, C3.SEED + f, C3.VAL_FRACTION)
            assert not (set(tr.subject_id) & set(va.subject_id)) and (tr.fold != f).all() and (va.fold != f).all()
            print(f"fold {f}: train {len(tr)} scans/{tr.subject_id.nunique()} subj | val {len(va)}/{va.subject_id.nunique()} | "
                  f"held-out CN {int(((man.fold == f) & (man.CDRTOT == 0)).sum())} -- subject-disjoint OK")
        return

    store = K.ScanStore()
    for f in map(int, args.folds.split(",")):
        for s in range(args.seeds):
            train_one(name, cfg, f, s, man, store, bins, dev, epochs=args.epochs, patience=args.patience,
                      workers=args.workers, cv=args.cv, record_train_mae=args.record_train_mae, force=args.force)


if __name__ == "__main__":
    main()
