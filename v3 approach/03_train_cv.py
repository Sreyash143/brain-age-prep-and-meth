"""
03_train_cv.py   (Phase 4 -- the GPU-heavy step; resumable)

For every (variant, fold, seed):
    train on CN scans of the other folds (minus a 12 % subject-level early-stopping split),
    early-stop on that held-out CN split, save the best checkpoint.
Fold k's models are later used ONLY to predict fold k's subjects (script 04) => every scan gets an out-of-fold prediction.

Variants (config_v3.VARIANTS):
    m0  v2-style baseline (last block + head, z-score input, KL loss)
    m1  v3 primary (all blocks, layer-wise LR, tail re-weighting, KL + 0.1*L1, GPU augmentation)
    m2  m1 + consistency loss on same-day repeat pairs from TRAINING subjects only

Examples
    python 03_train_cv.py --variants m1 --folds 0 --seeds 1 --epochs 2     # smoke test, ~minutes
    python 03_train_cv.py --variants m0,m1,m2 --seeds 2                    # full run (hours; restart-safe)

Memory: if CUDA OOM at --bs 4, use --bs 2 --bn_mode freeze.
"""
import argparse, json, os, time
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import config_v3 as C
import core_v3 as K


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


def train_one(variant, fold, seed, man, store, bins, args, dev):
    cfg = C.VARIANTS[variant]
    tag = f"{variant}_f{fold}_s{seed}"
    ckpt, meta = C.CKPT_DIR / f"{tag}.pt", C.CKPT_DIR / f"{tag}.json"
    if ckpt.exists() and meta.exists():
        print(f"[skip] {tag} already trained")
        return
    torch.manual_seed(seed * 1000 + fold)
    np.random.seed(seed * 1000 + fold)

    cn_train_pool = man[(man.fold != fold) & (man.CDRTOT == 0)]
    tr, va = split_train_val(cn_train_pool, C.SEED + fold, C.VAL_FRACTION)
    # Phase-11 integrity checks
    assert not (set(tr.subject_id) & set(va.subject_id))
    assert (tr.fold != fold).all() and (va.fold != fold).all(), "held-out fold leaked into training"
    store.add(pd.concat([tr, va]).file_path)

    centers = K.bin_centers(bins, dev)
    model = K.load_pretrained_sfcn(C.SFCN_CKPT, bins["n_bins"], bins["bin_min"], bins["bin_step"], device=dev)
    groups, trainable = K.configure_trainable(model, cfg["unfreeze"], cfg["lr_head"], cfg["lr_backbone"], cfg["llrd"])
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, min_lr=1e-7)

    train_ds, val_ds = K.ScanDS(store, tr), K.ScanDS(store, va)
    if cfg["reweight"]:
        sampler = WeightedRandomSampler(K.tail_weights(tr.age.values), num_samples=len(tr), replacement=True)
        tl = DataLoader(train_ds, batch_size=args.bs, sampler=sampler, num_workers=args.workers, pin_memory=True, drop_last=True)
    else:
        tl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    vl = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)

    pair_iter = None
    if cfg["consistency"] > 0:
        pds = K.PairDS(store, tr)                          # TRAINING subjects only
        print(f"  consistency pairs: {len(pds)}")
        pl = DataLoader(pds, batch_size=max(1, args.bs // 2), shuffle=True, num_workers=0, drop_last=True)
        pair_iter = iter(pl)

    best, bad, hist = 1e9, 0, []
    warm_steps = max(1, len(tl))                           # 1 epoch linear warm-up
    step = 0
    print(f"\n=== {tag} | train scans {len(tr)} | val scans {len(va)} | trainable: {trainable}")
    for ep in range(args.epochs):
        t0 = time.time()
        K.set_train_mode(model, trainable, args.bn_mode)
        tot, n = 0.0, 0
        for x, age in tl:
            if ep == 0:                                    # warm-up
                f = min(1.0, (step + 1) / warm_steps)
                for g in opt.param_groups:
                    g["lr"] = g["initial_lr"] * f
            x, age = x.to(dev, non_blocking=True), age.to(dev)
            x = K.gpu_augment(K.to_input(x, cfg["input_mode"], args.cv))
            y = K.soft_labels(age, bins, dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                lp = model(x)
            loss = K.kl_loss(lp, y)
            if cfg["l1"] > 0:
                loss = loss + cfg["l1"] * (K.expected_age(lp, centers) - age).abs().mean()
            loss.backward()

            if pair_iter is not None and step % 2 == 0:    # consistency step on repeat pairs
                try:
                    xp, agep = next(pair_iter)
                except StopIteration:
                    pair_iter = iter(pl)
                    xp, agep = next(pair_iter)
                B = xp.shape[0]
                xp = xp.to(dev).view(B * 2, 1, *xp.shape[-3:])
                agep = agep.to(dev).repeat_interleave(2)
                xp = K.gpu_augment(K.to_input(xp, cfg["input_mode"], args.cv))
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    lpp = model(xp)
                ea = K.expected_age(lpp, centers).view(B, 2)
                closs = K.kl_loss(lpp, K.soft_labels(agep, bins, dev)) + cfg["consistency"] * ((ea[:, 0] - ea[:, 1]) ** 2).mean()
                closs.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(age)
            n += len(age)
            step += 1

        val_mae = evaluate(model, vl, cfg, centers, dev, args.cv)
        if ep >= 1:
            sched.step(val_mae)
        hist.append(dict(epoch=ep + 1, train_loss=tot / max(n, 1), val_mae=val_mae))
        print(f"ep {ep + 1:02d}/{args.epochs} | loss {tot / max(n, 1):.3f} | val MAE {val_mae:.3f} | "
              f"lr {opt.param_groups[0]['lr']:.2e} | {time.time() - t0:.0f}s")
        if val_mae < best:
            best, bad = val_mae, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop")
                break
        torch.cuda.empty_cache()

    json.dump(dict(tag=tag, best_val_mae=best, input_mode=cfg["input_mode"], cv=args.cv, history=hist,
                   n_train=len(tr), n_val=len(va), val_subjects=sorted(va.subject_id.unique())), open(meta, "w"), indent=1)
    print(f"{tag} done | best val MAE {best:.3f}")
    del model, opt
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="m0,m1,m2")
    ap.add_argument("--folds", default=",".join(map(str, range(C.N_FOLDS))))
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--bn_mode", default="train", choices=["train", "freeze"])
    ap.add_argument("--cv", type=float, default=C.MEANNORM_CV)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(C.CKPT_DIR, exist_ok=True)
    man = pd.read_csv(C.MANIFEST_CSV)
    bins = K.load_bins(C.BINS_JSON)
    qc = pd.read_csv(C.QC_CSV) if C.QC_CSV.exists() else None
    if qc is not None:                                     # image-based exclusion applied to TRAINING scans only
        bad = set(qc[qc.qc_flag == 1].file_path)
        man["qc_flag"] = man.file_path.isin(bad).astype(int)
    store = K.ScanStore()
    for v in args.variants.split(","):
        for f in map(int, args.folds.split(",")):
            for s in range(args.seeds if v != "m0" else 1):
                m = man
                if qc is not None and "qc_flag" in man:
                    # exclude QC-failed scans from TRAINING/VALIDATION (they are still predicted & reported separately)
                    m = man[man.qc_flag == 0]
                train_one(v, f, s, m, store, bins, args, dev)


if __name__ == "__main__":
    main()
