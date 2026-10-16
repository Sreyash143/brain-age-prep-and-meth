"""
04a_predict_oof.py   (Phase 4.4-4.5 -- GPU)

Cross-fitted prediction: each scan is predicted ONLY by the models whose training set excluded that scan's subject
(fold k models -> fold k subjects). This applies equally to CN and impaired subjects, so noise comparisons between CDR
groups are not confounded by different ensemble sizes, and survival analyses can use the WHOLE cohort without leakage.

Per scan we store:
    predicted_age       mean over (seeds x TTA views)
    sd_seed             std across seed-means           (epistemic spread; NaN if only 1 seed)
    sd_tta              mean over seeds of std across views (augmentation spread)
    predicted_age_sd    std over all seed x view predictions (what 05_noise_floor uses for calibration)

Output: v3/predictions/raw_<variant>.csv
"""
import argparse, json, os
import numpy as np, pandas as pd, torch
import config_v3 as C
import core_v3 as K


@torch.no_grad()
def predict_models(models, df, store, cfg, cv, centers, dev, shifts, bs=4):
    """returns array (n_models, n_views, n_scans)"""
    out = np.zeros((len(models), len(shifts), len(df)), dtype=np.float32)
    for i in range(0, len(df), bs):
        x = torch.stack([store.get(p) for p in df.file_path.iloc[i:i + bs]]).squeeze(1).to(dev)
        x = K.to_input(x, cfg["input_mode"], cv)
        for si, sh in enumerate(shifts):
            xs = K.shift_volume(x, sh)
            for mi, m in enumerate(models):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    lp = m(xs)
                out[mi, si, i:i + len(xs)] = K.expected_age(lp, centers).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="m0,m1,m2")
    ap.add_argument("--n_tta", type=int, default=len(C.TTA_SHIFTS))
    ap.add_argument("--cv", type=float, default=C.MEANNORM_CV)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(C.PRED_DIR, exist_ok=True)
    man = pd.read_csv(C.MANIFEST_CSV)
    bins = K.load_bins(C.BINS_JSON)
    centers = K.bin_centers(bins, dev)
    store = K.ScanStore()
    shifts = C.TTA_SHIFTS[:args.n_tta]

    for v in args.variants.split(","):
        cfg = C.VARIANTS[v]
        parts = []
        for k in range(C.N_FOLDS):
            ck = sorted(C.CKPT_DIR.glob(f"{v}_f{k}_s*.pt"))
            if not ck:
                print(f"[warn] no checkpoints for {v} fold {k}; skipping")
                continue
            models = []
            for c in ck:
                m = K.load_pretrained_sfcn(C.SFCN_CKPT, bins["n_bins"], bins["bin_min"], bins["bin_step"], device=dev)
                sd = torch.load(c, map_location=dev)
                m.load_state_dict({kk.replace("_orig_mod.", ""): vv for kk, vv in sd.items()})
                models.append(m.eval())
            df = man[man.fold == k].reset_index(drop=True)
            store.arr.clear()
            store.add(df.file_path, desc=f"{v} fold {k}")
            P = predict_models(models, df, store, cfg, args.cv, centers, dev, shifts)      # (M,V,N)
            seed_means = P.mean(1)                                                         # (M,N)
            df["predicted_age"] = P.mean((0, 1))
            df["sd_seed"] = seed_means.std(0) if len(models) > 1 else np.nan
            df["sd_tta"] = P.std(1).mean(0)
            df["predicted_age_sd"] = P.reshape(-1, P.shape[-1]).std(0)
            df["n_models"] = len(models)
            parts.append(df)
            print(f"{v} fold {k}: {len(df)} scans, {len(models)} model(s) x {len(shifts)} views")
        pd.concat(parts).to_csv(C.PRED_DIR / f"raw_{v}.csv", index=False)
        print("saved", C.PRED_DIR / f"raw_{v}.csv")


if __name__ == "__main__":
    main()
