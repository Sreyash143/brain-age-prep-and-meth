# -*- coding: utf-8 -*-
"""
11_train_experiment.py -- V4 Phase 2/3 generalized trainer.

One script, driven entirely by one row of config_v4.EXPERIMENTS, replaces a
separate script per architecture/activation/optimizer/loss combination. This
is what makes Table A/B/C comparable across rows: every experiment goes
through the exact same training loop, the exact same frozen folds
(config_v3.FOLDS_CSV via the manifest), the exact same early-stopping split
fraction, and writes checkpoints/metadata in the exact same shape V3 used
(C.CKPT_DIR / f"{tag}.pt" + ".json"), so 10_eval_canonical.py and
13_build_results_tables.py don't need per-experiment special cases.

Usage:
    python 11_train_experiment.py --experiment v4_sfcn_baseline --folds 0,1,2,3,4 --seeds 2
    python 11_train_experiment.py --experiment v4_resnet18_residual --folds 0 --seeds 1 --epochs 2   # smoke test

Checkpoints: v4/checkpoints/<experiment>_f<fold>_s<seed>.pt (+ .json metadata,
including per-epoch wall-clock time and peak GPU memory for Table C).
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

import config_v3 as C3
import config_v4 as C4
import core_v3 as K
import core_v4 as K4


def split_train_val(df_train_cn, seed, frac):
    subj = np.array(sorted(df_train_cn.subject_id.unique()))
    rng = np.random.default_rng(seed)
    val_s = set(rng.choice(subj, size=max(5, int(len(subj) * frac)), replace=False))
    is_val = df_train_cn.subject_id.isin(val_s)
    return df_train_cn[~is_val].reset_index(drop=True), df_train_cn[is_val].reset_index(drop=True)


@torch.no_grad()
def evaluate(model, loader, cfg, centers, dev, cv):
    model.eval()
    err, n = 0.0, 0
    for x, age in loader:
        x, age = x.to(dev, non_blocking=True), age.to(dev)
        x = K.to_input(x, cfg["input_mode"], cv)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            lp = model(x)
        err += (K.expected_age(lp, centers) - age).abs().sum().item()
        n += len(age)
    return err / max(n, 1)


def _atomic_torch_save(obj, path):
    """Write checkpoints atomically so interruption cannot leave a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _atomic_json_dump(obj, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    tmp.replace(path)


def _load_torch(path, dev):
    # Explicit weights_only=False is needed for optimizer/RNG states in recent PyTorch.
    try:
        return torch.load(path, map_location=dev, weights_only=False)
    except TypeError:  # older PyTorch versions
        return torch.load(path, map_location=dev)


def train_one(experiment, fold, seed, man, store, bins, args, dev):
    cfg = C4.EXPERIMENTS[experiment]
    tag = f"{experiment}_f{fold}_s{seed}"
    ckpt = C4.CKPT_DIR / f"{tag}.pt"                 # best weights; canonical evaluator expects state_dict
    meta = C4.CKPT_DIR / f"{tag}.json"               # final metadata; its presence means completed
    resume_ckpt = C4.CKPT_DIR / f"{tag}.resume.pt"   # latest epoch + optimizer/scheduler state

    if ckpt.exists() and meta.exists() and not args.force:
        print(f"[skip] {tag} already completed (checkpoint + metadata found)")
        return

    # A force run deliberately starts fresh. Preserve old files before doing this in the pipeline.
    if args.force:
        start_epoch = 0
        # Prevent a stale completion marker from making an interrupted forced run
        # look completed on the next invocation.
        if meta.exists():
            meta.unlink()
        if resume_ckpt.exists():
            resume_ckpt.unlink()
    else:
        start_epoch = 0

    torch.manual_seed(seed * 1000 + fold)
    np.random.seed(seed * 1000 + fold)

    cn_train_pool = man[(man.fold != fold) & (man.CDRTOT == 0)]
    tr, va = split_train_val(cn_train_pool, C3.SEED + fold, C3.VAL_FRACTION)
    assert not (set(tr.subject_id) & set(va.subject_id))
    assert (tr.fold != fold).all() and (va.fold != fold).all(), "held-out fold leaked into training"
    store.add(pd.concat([tr, va]).file_path)

    centers = K.bin_centers(bins, dev)
    model, trainable = K4.build_model(cfg, bins, dev)
    opt = K4.build_optimizer(model, trainable, cfg)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5, cooldown=1, min_lr=1e-7
    )

    train_ds, val_ds = K.ScanDS(store, tr), K.ScanDS(store, va)
    if cfg.get("reweight_tail"):
        sampler = WeightedRandomSampler(K.tail_weights(tr.age.values), num_samples=len(tr), replacement=True)
        tl = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler, num_workers=args.workers,
                         pin_memory=True, drop_last=False)
    else:
        tl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=args.workers,
                         pin_memory=True, drop_last=False)
    vl = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=args.workers, pin_memory=True)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    best, bad, hist = 1e9, 0, []
    step, t_start = 0, time.time()

    # Resume the most recent fully completed epoch, including optimizer and scheduler.
    if not args.force and resume_ckpt.exists():
        state = _load_torch(resume_ckpt, dev)
        if isinstance(state, dict) and state.get("checkpoint_kind") == "v4_training_resume":
            model.load_state_dict(state["model_state_dict"])
            opt.load_state_dict(state["optimizer_state_dict"])
            sched.load_state_dict(state["scheduler_state_dict"])
            start_epoch = int(state["next_epoch"])
            best = float(state["best_val_mae"])
            bad = int(state["bad_epochs"])
            hist = list(state.get("history", []))
            step = int(state.get("step", 0))
            # Restore RNG only after constructing model, optimizer, datasets and loaders.
            if state.get("torch_rng_state") is not None:
                torch.set_rng_state(state["torch_rng_state"].cpu())
            if dev.type == "cuda" and state.get("cuda_rng_state_all") is not None:
                torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
            if state.get("numpy_rng_state") is not None:
                np.random.set_state(state["numpy_rng_state"])
            print(f"[resume] {tag}: continuing at epoch {start_epoch + 1}/{args.epochs}; "
                  f"best val MAE so far {best:.3f}; {len(hist)} completed epochs restored")
        else:
            print(f"[WARN] {resume_ckpt} is not a recognized full resume checkpoint; starting from best weights if available.")
            if ckpt.exists():
                model.load_state_dict(_load_torch(ckpt, dev))
                print(f"[warm-start] Loaded existing best model weights from {ckpt}; optimizer/epoch will restart.")
    elif not args.force and ckpt.exists() and not meta.exists():
        # Legacy interrupted runs saved only best weights. Preserve that learned model,
        # but be explicit that exact epoch/optimizer continuation is impossible from weights alone.
        model.load_state_dict(_load_torch(ckpt, dev))
        print(f"[warm-start] {tag}: loaded existing best weights. No full resume state exists, "
              "so optimizer/epoch restart; full resume checkpoints will now be saved each epoch.")

    print(f"\n=== {tag} | train scans {len(tr)} | val scans {len(va)} | "
          f"trainable params {sum(p.numel() for n, p in model.named_parameters() if n in trainable):,}")

    # If a run was stopped after its last saved epoch, this loop continues from the next epoch.
    for ep in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for x, age in tl:
            # Preserve the original one-epoch warmup, but do not repeat it after resuming.
            if ep == 0:
                f = min(1.0, (step + 1) / max(1, len(tl)))
                for g in opt.param_groups:
                    if "initial_lr" in g:
                        g["lr"] = g["initial_lr"] * f
            x, age = x.to(dev, non_blocking=True), age.to(dev)
            x = K.gpu_augment(K.to_input(x, cfg["input_mode"], args.cv))
            y = K.soft_labels(age, bins, dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                lp = model(x)
            ea = K.expected_age(lp, centers)
            loss = K4.compute_loss(lp, y, ea, age, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(age)
            n += len(age)
            step += 1

        val_mae = evaluate(model, vl, cfg, centers, dev, args.cv)
        train_mae = evaluate(model, tl, cfg, centers, dev, args.cv) if args.record_train_mae else None
        if ep >= 1:
            sched.step(val_mae)
        epoch_time = time.time() - t0
        hist.append(dict(epoch=ep + 1, train_loss=tot / max(n, 1), val_mae=val_mae,
                         train_mae=train_mae, epoch_seconds=epoch_time))
        improved = val_mae < best
        if improved:
            best, bad = val_mae, 0
            # Keep the best weights as a plain state_dict for the existing evaluation script.
            _atomic_torch_save(model.state_dict(), ckpt)
        else:
            bad += 1

        print(f"ep {ep + 1:02d}/{args.epochs} | loss {tot / max(n, 1):.3f} | val MAE {val_mae:.3f} | "
              f"lr {opt.param_groups[0]['lr']:.2e} | {epoch_time:.0f}s")

        # Atomic latest-state save after every finished epoch. Ctrl+C during an epoch
        # leaves the prior epoch's valid resume file intact.
        resume_state = dict(
            checkpoint_kind="v4_training_resume", tag=tag, experiment=experiment,
            fold=fold, seed=seed, next_epoch=ep + 1, best_val_mae=best,
            bad_epochs=bad, step=step, history=hist,
            model_state_dict=model.state_dict(), optimizer_state_dict=opt.state_dict(),
            scheduler_state_dict=sched.state_dict(), torch_rng_state=torch.get_rng_state(),
            cuda_rng_state_all=torch.cuda.get_rng_state_all() if dev.type == "cuda" else None,
            numpy_rng_state=np.random.get_state(),
            total_train_seconds_so_far=time.time() - t_start,
        )
        _atomic_torch_save(resume_state, resume_ckpt)

        if dev.type == "cuda":
            torch.cuda.empty_cache()
        if bad >= args.patience:
            print("early stop")
            break

    total_time = time.time() - t_start
    peak_mem_gb = (torch.cuda.max_memory_allocated(dev) / 1e9) if dev.type == "cuda" else None
    param_info = K4.count_params(model)
    _atomic_json_dump(dict(
        tag=tag, experiment=experiment, fold=fold, seed=seed,
        best_val_mae=best, input_mode=cfg["input_mode"], cv=args.cv, history=hist,
        n_train=len(tr), n_val=len(va), val_subjects=sorted(va.subject_id.unique()),
        total_train_seconds=total_time, mean_epoch_seconds=float(np.mean([h["epoch_seconds"] for h in hist])) if hist else None,
        peak_gpu_memory_gb=peak_mem_gb, completed=True, **param_info,
    ), meta)
    # Mark completed in the resume file too; leave it available for audit/recovery.
    if resume_ckpt.exists():
        final_state = _load_torch(resume_ckpt, dev)
        if isinstance(final_state, dict):
            final_state["completed"] = True
            _atomic_torch_save(final_state, resume_ckpt)
    print(f"{tag} done | best val MAE {best:.3f} | total {total_time:.0f}s | "
          f"peak mem {peak_mem_gb if peak_mem_gb else 'n/a (cpu)'} GB")
    del model, opt
    if dev.type == "cuda":
        torch.cuda.empty_cache()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=list(C4.EXPERIMENTS))
    ap.add_argument("--folds", default=",".join(map(str, range(C3.N_FOLDS))))
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=C4.MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=C4.PATIENCE)
    ap.add_argument("--workers", type=int, default=C4.NUM_WORKERS)
    ap.add_argument("--cv", type=float, default=C3.MEANNORM_CV)
    ap.add_argument("--record_train_mae", action="store_true",
                     help="also evaluate on the train split each epoch for Table C's train-val gap (slower)")
    ap.add_argument("--force", action="store_true", help="retrain even if checkpoint+metadata already exist")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    man = pd.read_csv(C3.MANIFEST_CSV)
    bins = K.load_bins(C3.BINS_JSON)
    qc = pd.read_csv(C3.QC_CSV) if C3.QC_CSV.exists() else None
    qc_bad = K4.qc_bad_table(qc)
    if qc_bad is not None:
        man = man.merge(qc_bad, on="file_path", how="left")
        man["qc_bad"] = man["qc_bad"].fillna(False).astype(bool)
        m = man[~man.qc_bad].copy()  # QC failures excluded from training/validation only
        print(f"QC filtering: excluded {int(man.qc_bad.sum())} of {len(man)} manifest scans "
              "from training/validation; all-eligible evaluation remains separate.")
    else:
        m = man

    store = K.ScanStore()
    for f in map(int, args.folds.split(",")):
        for s in range(args.seeds):
            train_one(args.experiment, f, s, m, store, bins, args, dev)


if __name__ == "__main__":
    main()
