# -*- coding: utf-8 -*-
"""
10_eval_canonical.py -- V4 Phase 1, item 3-6 / v3_problems.txt Priority 3.

ONE documented evaluation pipeline for every V4 experiment (and, with
--include_v3, V3's own m0/m1/m2), producing BOTH populations explicitly and
by name so they can never again be silently conflated the way V3's raw
prediction CSV (2,321 CN scan rows) vs accuracy JSON (1,896) were:

    *_all_eligible   : every CN scan in the manifest with an out-of-fold
                        prediction, regardless of QC flag
    *_qc_passed      : the same, restricted to qc_flag == 0

Per population: cross-fitted bias correction (fit on out-of-fold CN sessions
of the OTHER folds only -- same method as V3's 04b, "invert"), fold-wise
MAE/RMSE/r/R2, pooled subject-cluster bootstrap 95% CI (not fold-SD -- per
v4_approach.txt Phase 1 item 5 and Table A's own instruction not to
substitute one for the other), raw vs bias-corrected MAE reported SEPARATELY
(a bias correction that increases MAE is not reported as an accuracy gain),
plus Table B's full metric set (mean BAG, BAG-vs-age slope, MAE by age band,
frac |BAG|>10y, repeat-scan SEM/MDC95/ICC, uncertainty calibration).

Usage:
    python 10_eval_canonical.py --experiment v4_sfcn_baseline
    python 10_eval_canonical.py --experiment v4_sfcn_baseline --include_v3    # also redo V3 m1 through this pipeline

Output: v4/predictions/{raw,bag}_<tag>_<population>.csv
        v4/results/canonical_<tag>_<population>.json
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import linregress, pearsonr, spearmanr

import config_v3 as C3
import config_v4 as C4
import core_v3 as K
import core_v4 as K4


# ---------------------------------------------------------------- predict

@torch.no_grad()
def predict_models(models, df, store, cfg, cv, centers, dev, shifts, bs=4):
    out = np.zeros((len(models), len(shifts), len(df)), dtype=np.float32)
    for i in range(0, len(df), bs):
        x = torch.stack([store.get(p) for p in df.file_path.iloc[i:i + bs]]).to(dev)
        x = K.to_input(x, cfg["input_mode"], cv)
        for si, sh in enumerate(shifts):
            xs = K.shift_volume(x, sh)
            for mi, m in enumerate(models):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    lp = m(xs)
                out[mi, si, i:i + len(xs)] = K.expected_age(lp, centers).cpu().numpy()
    return out


def run_oof_predictions(tag, ckpt_glob_fn, cfg, bins, man, dev, args):
    """ckpt_glob_fn(fold) -> list of checkpoint Paths for that fold. Returns
    one DataFrame with an out-of-fold predicted_age per manifest row."""
    centers = K.bin_centers(bins, dev)
    store = K.ScanStore()
    shifts = C3.TTA_SHIFTS[:args.n_tta]
    parts = []
    for k in range(C3.N_FOLDS):
        ck = ckpt_glob_fn(k)
        if not ck:
            print(f"[warn] no checkpoints for {tag} fold {k}; skipping")
            continue
        models = []
        for c in ck:
            m, _ = K4.build_model(cfg, bins, dev)
            sd = torch.load(c, map_location=dev)
            m.load_state_dict({kk.replace("_orig_mod.", ""): vv for kk, vv in sd.items()})
            models.append(m.eval())
        df = man[man.fold == k].reset_index(drop=True)
        store.arr.clear()
        store.add(df.file_path, desc=f"{tag} fold {k}")
        P = predict_models(models, df, store, cfg, args.cv, centers, dev, shifts)
        seed_means = P.mean(1)
        df["predicted_age"] = P.mean((0, 1))
        df["sd_seed"] = seed_means.std(0) if len(models) > 1 else np.nan
        df["sd_tta"] = P.std(1).mean(0)
        df["predicted_age_sd"] = P.reshape(-1, P.shape[-1]).std(0)
        df["n_models"] = len(models)
        parts.append(df)
        print(f"{tag} fold {k}: {len(df)} scans, {len(models)} model(s) x {len(shifts)} views")
    return pd.concat(parts) if parts else pd.DataFrame()


# ---------------------------------------------------------------- bias correction + metrics

def crossfit_correct(df):
    """Cross-fitted 'invert' correction (V3's non-circular method; the other
    'bag_regress' method is intentionally NOT exposed here -- its MAE is not
    a valid accuracy metric, see v3's own 04b docstring)."""
    df = df.copy()
    df["brain_age"] = np.nan
    params = {}
    sess = (df[df.CDRTOT == 0].groupby(["fold", "mri_label"])
            .agg(age=("age", "mean"), pred=("predicted_age", "mean")).reset_index())
    for k in sorted(df.fold.unique()):
        tr = sess[sess.fold != k]
        m = df.fold == k
        lr = linregress(tr.age, tr.pred)
        assert abs(lr.slope) > 1e-6
        df.loc[m, "brain_age"] = (df.loc[m, "predicted_age"] - lr.intercept) / lr.slope
        params[int(k)] = dict(slope=lr.slope, intercept=lr.intercept, n_sessions=int(len(tr)))
    df["BAG"] = df.brain_age - df.age
    return df, params


def band_table(cn):
    rows = []
    for lo, hi in C3.AGE_BANDS:
        g = cn[(cn.age >= lo) & (cn.age < hi)]
        if len(g):
            rows.append(dict(band=f"{lo}-{hi}", n=len(g), MAE=g.BAG.abs().mean(), mean_BAG=g.BAG.mean()))
    return pd.DataFrame(rows)


def three_class(df):
    t = np.digitize(df.age, C3.CLASS_CUTS)
    p = np.digitize(df.brain_age, C3.CLASS_CUTS)
    return float((t == p).mean())


def cluster_boot(df, fn, n=1000, seed=C3.SEED):
    rng = np.random.default_rng(seed)
    groups = {s: g for s, g in df.groupby("subject_id")}
    keys = np.array(list(groups))
    vals = [fn(pd.concat([groups[s] for s in rng.choice(keys, len(keys), replace=True)])) for _ in range(n)]
    return [float(x) for x in np.percentile(vals, [2.5, 97.5])]


def repeat_scan_metrics(df):
    """SEM / MDC95 / ICC(A,1) -- same formulas as V3's 05_noise_floor.py,
    reimplemented locally so this script is self-contained for Table B."""
    rows = []
    for lab, g in df.sort_values(["mri_label", "run"]).groupby("mri_label"):
        if len(g) < 2:
            continue
        b = g.BAG.to_numpy(float)
        rows.append(dict(ss=float(((b - b.mean()) ** 2).sum()), df=len(g) - 1, pair=b[:2]))
    if not rows:
        return dict(sigma_SEM=np.nan, MDC95=np.nan, ICC_A1=np.nan, n_repeat_sessions=0)
    S = pd.DataFrame(rows)
    sigma = float(np.sqrt(S.ss.sum() / S.df.sum())) if S.df.sum() > 0 else np.nan
    pairs = np.array([r["pair"] for r in rows if len(r["pair"]) == 2])
    icc = np.nan
    if len(pairs) >= 5:
        gm = pairs.mean()
        n, kk = pairs.shape
        ssr = kk * ((pairs.mean(1) - gm) ** 2).sum()
        ssc = n * ((pairs.mean(0) - gm) ** 2).sum()
        sst = ((pairs - gm) ** 2).sum()
        sse = sst - ssr - ssc
        msr, msc, mse = ssr / (n - 1), ssc / (kk - 1), sse / ((n - 1) * (kk - 1))
        icc = float((msr - mse) / (msr + (kk - 1) * mse + kk * (msc - mse) / n))
    return dict(sigma_SEM=sigma, MDC95=1.96 * np.sqrt(2) * sigma if np.isfinite(sigma) else np.nan,
                ICC_A1=icc, n_repeat_sessions=int(len(S)))


def uncertainty_calibration(df):
    g = df.dropna(subset=["predicted_age_sd"])
    if len(g) < 10:
        return dict(n=0, spearman_rho=np.nan, p=np.nan)
    sess = g.groupby("mri_label").agg(ens_sd=("predicted_age_sd", "mean"), BAG=("BAG", "mean"))
    sess_sd = g.sort_values(["mri_label", "run"]).groupby("mri_label").BAG.agg(lambda s: s.std(ddof=1) if len(s) > 1 else np.nan)
    j = sess.join(sess_sd.rename("within_sd")).dropna()
    if len(j) < 10:
        return dict(n=int(len(j)), spearman_rho=np.nan, p=np.nan)
    rho, p = spearmanr(j.ens_sd, j.within_sd)
    return dict(n=int(len(j)), spearman_rho=float(rho), p=float(p))


def compute_metrics(df_pop, label):
    """df_pop: already bias-corrected (has brain_age, BAG). CN-only metrics,
    fold-wise + pooled, for one population ('all_eligible' or 'qc_passed')."""
    cn = df_pop[df_pop.CDRTOT == 0]
    res = dict(population=label, n_CN_scans=len(cn), n_CN_subjects=int(cn.subject_id.nunique()))

    res["MAE"] = float(cn.BAG.abs().mean())
    res["MAE_CI95_bootstrap"] = cluster_boot(cn, lambda d: d.BAG.abs().mean())
    res["RMSE"] = float(np.sqrt((cn.BAG ** 2).mean()))
    res["pearson_r"] = float(pearsonr(cn.age, cn.brain_age)[0])
    res["R2"] = float(1 - ((cn.age - cn.brain_age) ** 2).sum() / ((cn.age - cn.age.mean()) ** 2).sum())
    res["raw_MAE_uncorrected"] = float((cn.predicted_age - cn.age).abs().mean())
    res["mean_BAG"] = float(cn.BAG.mean())
    s = linregress(cn.age, cn.BAG)
    res["BAG_vs_age_slope"], res["BAG_vs_age_p"] = float(s.slope), float(s.pvalue)
    res["frac_abs_BAG_gt10"] = float((cn.BAG.abs() > 10).mean())
    res["age_group_acc_3class"] = three_class(cn)
    res["age_group_acc_CI95"] = cluster_boot(cn, three_class)

    # fold-wise (Table C "performance by outer fold" / Table A "mean +/- SD across folds")
    fold_rows = []
    for k in sorted(cn.fold.unique()):
        g = cn[cn.fold == k]
        fold_rows.append(dict(fold=int(k), n=len(g), MAE=float(g.BAG.abs().mean()), RMSE=float(np.sqrt((g.BAG ** 2).mean()))))
    res["fold_wise"] = fold_rows
    res["MAE_mean_across_folds"] = float(np.mean([r["MAE"] for r in fold_rows]))
    res["MAE_sd_across_folds"] = float(np.std([r["MAE"] for r in fold_rows]))
    res["MAE_CI_NOTE"] = ("MAE_CI95_bootstrap above is the pooled subject-cluster bootstrap CI -- "
                           "use THIS for inference. MAE_sd_across_folds is descriptive spread only, "
                           "per v4_approach.txt/Table A: do not substitute fold-SD for a CI.")

    res["age_bands"] = band_table(cn).to_dict(orient="records")
    res.update(repeat_scan_metrics(cn))
    res["uncertainty_calibration"] = uncertainty_calibration(cn)

    imp = df_pop[df_pop.CDRTOT > 0].groupby("CDRTOT").BAG.agg(["count", "mean", "std"]).reset_index()
    res["BAG_by_CDR"] = imp.round(3).to_dict(orient="records")
    return res


# ---------------------------------------------------------------- main

def eval_one(tag, cfg, ckpt_glob_fn, bins, man, qc, dev, args):
    raw = run_oof_predictions(tag, ckpt_glob_fn, cfg, bins, man, dev, args)
    if raw.empty:
        print(f"[skip] {tag}: no predictions produced (no checkpoints found)")
        return
    raw.to_csv(C4.PRED_DIR / f"raw_{tag}.csv", index=False)

    qc_bad = K4.qc_bad_table(qc)
    if qc_bad is not None:
        qc_joined = raw.merge(qc_bad, on="file_path", how="left")
        qc_passed = qc_joined[~qc_joined.qc_bad.fillna(False)].drop(columns=["qc_bad"])
    else:
        qc_passed = None

    for label, population_df in [
        ("all_eligible", raw),
        ("qc_passed", qc_passed),
    ]:
        if population_df is None or population_df.empty:
            continue
        bag_df, bias_params = crossfit_correct(population_df)
        bag_df.to_csv(C4.PRED_DIR / f"bag_{tag}_{label}.csv", index=False)
        res = compute_metrics(bag_df, label)
        res["tag"], res["bias_params_per_fold"] = tag, bias_params
        print(f"\n--- {tag} / {label} ---")
        print(json.dumps({k: v for k, v in res.items() if k not in ("bias_params_per_fold", "fold_wise", "age_bands")}, indent=2))
        json.dump(res, open(C4.RES_DIR / f"canonical_{tag}_{label}.json", "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=list(C4.EXPERIMENTS))
    ap.add_argument("--n_tta", type=int, default=None)
    ap.add_argument("--cv", type=float, default=C3.MEANNORM_CV)
    ap.add_argument("--include_v3", action="store_true", help="also run V3's m1 through this same pipeline (for Table A's fair baseline row)")
    args = ap.parse_args()
    if args.n_tta is None:
        args.n_tta = len(C3.TTA_SHIFTS)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    man = pd.read_csv(C3.MANIFEST_CSV)
    bins = K.load_bins(C3.BINS_JSON)
    qc = pd.read_csv(C3.QC_CSV) if C3.QC_CSV.exists() else None

    cfg = C4.EXPERIMENTS[args.experiment]
    eval_one(args.experiment, cfg,
             lambda k: sorted(C4.CKPT_DIR.glob(f"{args.experiment}_f{k}_s*.pt")),
             bins, man, qc, dev, args)

    if args.include_v3:
        # C3.VARIANTS rows don't carry arch/activation (V3 only ever built
        # pretrained SFCN with ReLU) -- add them so core_v4.build_model works.
        v3_cfg = dict(C3.VARIANTS["m1"], arch="sfcn", activation="relu")
        eval_one("v3_m1_recomputed", v3_cfg,
                  lambda k: sorted(C3.CKPT_DIR.glob(f"m1_f{k}_s*.pt")),
                  bins, man, qc, dev, args)


if __name__ == "__main__":
    main()
