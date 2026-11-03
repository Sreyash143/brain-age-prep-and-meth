"""
05_noise_floor.py   (Phase 6 -- CPU)

Works on any predictions file that has: subject_id, mri_label, run, age, CDRTOT, BAG (+ optional predicted_age_sd, qc_flag,
scanner). Run it once per model variant (Step 6.8):
    python 05_noise_floor.py --variant m1
    python 05_noise_floor.py --bag path/to/brainage_predictions.csv --tag v2     # also works on your v2 file

Computes
  6.1 pooled within-session sigma_SEM (df-weighted; handles 3+ runs)         6.2 MDC95 = 1.96*sqrt(2)*sigma_SEM
  6.3 subject-cluster bootstrap CIs (1000)                                    6.4 ICC(A,1) from repeats + wide-cohort ICC
  6.5 heteroscedasticity (age, mean BAG, ensemble SD, QC)                     6.6 CN vs impaired: SEM ratio + CI, Levene
  6.7 uncertainty calibration (rank corr + quintile table)  and a robust sigma (MAD) + sigma without QC-flagged sessions,
      because heavy-tailed repeat differences (failed registrations) dominate a plain SD.
"""
import argparse, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr, levene, f as fdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config_v3 as C


def session_table(df):
    rows = []
    for lab, g in df.sort_values(["mri_label", "run"]).groupby("mri_label"):
        if len(g) < 2:
            continue
        b = g.BAG.to_numpy(float)
        rows.append(dict(mri_label=lab, subject_id=g.subject_id.iloc[0], age=g.age.mean(), CDRTOT=g.CDRTOT.iloc[0],
                         m=len(g), ss=float(((b - b.mean()) ** 2).sum()), df=len(g) - 1, mean_bag=b.mean(),
                         d12=b[0] - b[1], sd_within=float(b.std(ddof=1)),
                         ens_sd=float(g.predicted_age_sd.mean()) if "predicted_age_sd" in g else np.nan,
                         qc_flag=int(g.qc_flag.max()) if "qc_flag" in g else 0,
                         scanner=g.scanner.iloc[0] if "scanner" in g else None))
    return pd.DataFrame(rows)


def pooled_sem(s):
    return float(np.sqrt(s.ss.sum() / s.df.sum())) if len(s) and s.df.sum() > 0 else np.nan


def robust_sem(s):
    d = s.d12.to_numpy()
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)) if len(d) else np.nan


def boot(s, fn, n=1000, seed=C.SEED):
    rng = np.random.default_rng(seed)
    g = {k: v for k, v in s.groupby("subject_id")}
    keys = np.array(list(g))
    vals = [fn(pd.concat([g[k] for k in rng.choice(keys, len(keys), replace=True)])) for _ in range(n)]
    return [float(x) for x in np.nanpercentile(vals, [2.5, 97.5])]


def icc_from_pairs(df):
    rows = []
    for _, g in df.sort_values(["mri_label", "run"]).groupby("mri_label"):
        if len(g) >= 2:
            rows.append(g.BAG.to_numpy(float)[:2])
    X = np.array(rows)
    n, k = X.shape if len(rows) else (0, 2)
    if n < 5:
        return np.nan
    gm = X.mean()
    ssr = k * ((X.mean(1) - gm) ** 2).sum()
    ssc = n * ((X.mean(0) - gm) ** 2).sum()
    sst = ((X - gm) ** 2).sum()
    sse = sst - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def wide_icc(df, sigma):
    first = df.sort_values("day").groupby("subject_id").first() if "day" in df else df.groupby("subject_id").first()
    vb = float(first.BAG.var(ddof=1)) - sigma ** 2
    return float(max(vb, 0) / (max(vb, 0) + sigma ** 2)), float(first.BAG.var(ddof=1))


def main(bag_path, tag, exclude_qc):
    df = pd.read_csv(bag_path)
    if exclude_qc and C.QC_CSV.exists() and "qc_flag" not in df:
        df = df.merge(pd.read_csv(C.QC_CSV)[["file_path", "qc_flag", "ncc_repeat"]], on="file_path", how="left")
    S = session_table(df)
    R = {"tag": tag, "n_repeat_sessions": int(len(S)), "n_subjects": int(S.subject_id.nunique()),
         "sessions_with_3plus_runs": int((S.m >= 3).sum())}

    # 6.1-6.3
    sig = pooled_sem(S)
    R.update(sigma_SEM=sig, sigma_SEM_CI=boot(S, pooled_sem), MDC95=1.96 * np.sqrt(2) * sig,
             MDC95_CI=[1.96 * np.sqrt(2) * x for x in boot(S, pooled_sem)])
    R["sigma_SEM_robust_MAD"] = robust_sem(S)
    R["n_sessions_abs_diff_gt5y"] = int((S.d12.abs() > 5).sum())
    if S.qc_flag.sum() > 0:
        R["sigma_SEM_excluding_QC_flagged"] = pooled_sem(S[S.qc_flag == 0])
        R["n_QC_flagged_repeat_sessions"] = int(S.qc_flag.sum())
    # 6.4 ICC
    R["ICC_A1_repeats"] = icc_from_pairs(df)
    R["ICC_wide_cohort"], R["between_subject_var_first_session"] = wide_icc(df, sig)
    # 6.5 heteroscedasticity
    het = {}
    for col in ["age", "mean_bag", "ens_sd"]:
        if S[col].notna().sum() > 5:
            rho, p = spearmanr(S[col], S.sd_within)
            het[col] = dict(rho=float(rho), p=float(p))
    R["heteroscedasticity_spearman_vs_within_SD"] = het
    bands = []
    for lo, hi in C.AGE_BANDS:
        g = S[(S.age >= lo) & (S.age < hi)]
        if len(g) >= 5:
            bands.append(dict(band=f"{lo}-{hi}", n_sessions=len(g), sigma_SEM=pooled_sem(g)))
    R["sigma_SEM_by_age_band"] = bands
    # 6.6 CN vs impaired
    cn, im = S[S.CDRTOT == 0], S[S.CDRTOT > 0]
    if len(cn) > 5 and len(im) > 5:
        R["sigma_SEM_CN"], R["sigma_SEM_impaired"] = pooled_sem(cn), pooled_sem(im)
        R["n_CN_sessions"], R["n_impaired_sessions"] = int(len(cn)), int(len(im))
        stat, p = levene(cn.d12, im.d12)
        R["levene_p"] = float(p)
        ratio = lambda s: pooled_sem(s[s.CDRTOT > 0]) / pooled_sem(s[s.CDRTOT == 0])
        R["SEM_ratio_impaired_over_CN"] = float(ratio(S))
        R["SEM_ratio_CI"] = boot(S, ratio)
        if S.scanner.notna().any():
            R["scanner_by_group"] = pd.crosstab(S.scanner, S.CDRTOT > 0).to_dict()
    # 6.7 calibration
    cal = {}
    for name, g in [("all", S), ("CN", cn), ("impaired", im)]:
        g = g.dropna(subset=["ens_sd"])
        if len(g) >= 10:
            rho, p = spearmanr(g.ens_sd, g.sd_within)
            cal[name] = dict(n=int(len(g)), spearman_rho=float(rho), p=float(p))
            q = pd.qcut(g.ens_sd, 5, duplicates="drop")
            tab = g.groupby(q, observed=True).agg(pred_spread=("ens_sd", "mean"), observed_sd=("sd_within", "mean"), n=("ens_sd", "size"))
            tab.to_csv(C.RES_DIR / f"calibration_{tag}_{name}.csv")
            if name == "all":
                plt.figure(figsize=(4, 4)); plt.plot(tab.pred_spread, tab.observed_sd, "o-")
                plt.xlabel("mean ensemble spread (y)"); plt.ylabel("mean observed within-session SD (y)")
                plt.title(f"Uncertainty calibration ({tag})"); plt.tight_layout()
                plt.savefig(C.RES_DIR / f"calibration_{tag}.png", dpi=200); plt.close()
    R["uncertainty_calibration"] = cal

    print(json.dumps(R, indent=2, default=lambda o: o if not isinstance(o, np.generic) else o.item()))
    C.RES_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(C.RES_DIR / f"noise_floor_{tag}.json", "w"), indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    S.to_csv(C.RES_DIR / f"repeat_sessions_{tag}.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="m1")
    ap.add_argument("--bag", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--exclude_qc", action="store_true", help="merge QC flags to report sigma without flagged sessions")
    a = ap.parse_args()
    C.RES_DIR.mkdir(parents=True, exist_ok=True)
    main(a.bag or str(C.PRED_DIR / f"bag_{a.variant}.csv"), a.tag or a.variant, a.exclude_qc)
