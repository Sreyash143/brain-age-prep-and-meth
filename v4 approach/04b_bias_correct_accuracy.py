"""
04b_bias_correct_accuracy.py   (Phase 5 + Step 4.7 -- CPU)

Input : v3/predictions/raw_<variant>.csv   (needs: subject_id, mri_label, run, age, CDRTOT, fold, predicted_age, predicted_age_sd)
Output: v3/predictions/bag_<variant>.csv    (adds brain_age, BAG)   +   v3/results/accuracy_<variant>.json / .csv

Bias correction (the v2 bug):  v2 fitted  pred = a + b*age  on TRAINING scans, i.e. on data the network had already
fit. In-sample the slope is ~1, on unseen subjects it was 0.56 -> the correction under-corrected and left a
BAG-vs-age trend of -0.24 years/year. Here the line is fitted on OUT-OF-FOLD CN predictions of the OTHER folds
("cross-fitted"), after averaging repeat scans within each session, and then applied unchanged to the held-out fold.

Usage: python 04b_bias_correct_accuracy.py --variant m1 [--raw path.csv] [--exclude_qc]
"""
import argparse, json
import numpy as np, pandas as pd
from scipy.stats import linregress, pearsonr
import config_v3 as C


def crossfit_correct(df, method="invert"):
    """method='invert'      : brain_age = (pred - a) / b   (v2 / Peng-style). Uses NO chronological age at test time,
                                so MAE / r / R2 computed from it are legitimate accuracy metrics.
       method='bag_regress' : BAG_corr = (pred - age) - (a' + b'*age)   (Beheshti-style residualisation).
                                Uses the subject's TRUE age to build the prediction => brain_age = age + residual.
                                Good for an age-independent BAG in downstream models, but its MAE is NOT an accuracy
                                metric (it is inflated by construction; see de Lange & Cole 2020, ref [12]). Never report it."""
    df = df.copy()
    df["brain_age"] = np.nan
    params = {}
    sess = (df[df.CDRTOT == 0].groupby(["fold", "mri_label"])
            .agg(age=("age", "mean"), pred=("predicted_age", "mean")).reset_index())
    for k in sorted(df.fold.unique()):
        tr = sess[sess.fold != k]
        m = df.fold == k
        if method == "invert":
            lr = linregress(tr.age, tr.pred)
            assert abs(lr.slope) > 1e-6
            df.loc[m, "brain_age"] = (df.loc[m, "predicted_age"] - lr.intercept) / lr.slope
        else:
            lr = linregress(tr.age, tr.pred - tr.age)
            df.loc[m, "brain_age"] = df.loc[m, "age"] + (df.loc[m, "predicted_age"] - df.loc[m, "age"]) - (lr.intercept + lr.slope * df.loc[m, "age"])
        params[int(k)] = dict(slope=lr.slope, intercept=lr.intercept, n_sessions=int(len(tr)))
    df["BAG"] = df.brain_age - df.age
    return df, params


def band_table(cn):
    rows = []
    for lo, hi in C.AGE_BANDS:
        g = cn[(cn.age >= lo) & (cn.age < hi)]
        if len(g):
            rows.append(dict(band=f"{lo}-{hi}", n=len(g), MAE=g.BAG.abs().mean(), mean_BAG=g.BAG.mean()))
    return pd.DataFrame(rows)


def three_class(df):
    t = np.digitize(df.age, C.CLASS_CUTS)
    p = np.digitize(df.brain_age, C.CLASS_CUTS)
    return float((t == p).mean()), float(pd.Series(t).value_counts(normalize=True).max())


def cluster_boot(df, fn, n=1000, seed=C.SEED):
    rng = np.random.default_rng(seed)
    groups = {s: g for s, g in df.groupby("subject_id")}
    keys = np.array(list(groups))
    vals = []
    for _ in range(n):
        samp = rng.choice(keys, len(keys), replace=True)
        vals.append(fn(pd.concat([groups[s] for s in samp])))
    return np.percentile(vals, [2.5, 97.5])


def main(variant, raw, exclude_qc, method):
    raw = raw or (C.PRED_DIR / f"raw_{variant}.csv")
    df = pd.read_csv(raw)
    if exclude_qc and C.QC_CSV.exists():
        qc = pd.read_csv(C.QC_CSV)[["file_path", "qc_flag"]]
        df = df.merge(qc, on="file_path", how="left")
        print("excluding", int((df.qc_flag == 1).sum()), "image-QC-flagged scans (image-based rule only)")
        df = df[df.qc_flag != 1]
    df_alt, _ = crossfit_correct(df, "bag_regress" if method == "invert" else "invert")
    alt_mae = float(df_alt[df_alt.CDRTOT == 0].BAG.abs().mean())
    df, params = crossfit_correct(df, method)
    out = C.PRED_DIR / f"bag_{variant}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    cn = df[df.CDRTOT == 0]
    res = dict(variant=variant, n_CN_scans=len(cn), n_CN_subjects=int(cn.subject_id.nunique()))
    res["MAE"] = float(cn.BAG.abs().mean())
    res["MAE_CI95"] = cluster_boot(cn, lambda d: d.BAG.abs().mean()).tolist()
    res["correction_method"] = method
    res["MAE_alt_method_NOT_REPORTABLE"] = alt_mae
    res["RMSE"] = float(np.sqrt((cn.BAG ** 2).mean()))
    res["pearson_r"] = float(pearsonr(cn.age, cn.brain_age)[0])
    res["R2"] = float(1 - ((cn.age - cn.brain_age) ** 2).sum() / ((cn.age - cn.age.mean()) ** 2).sum())
    res["raw_MAE_uncorrected"] = float((cn.predicted_age - cn.age).abs().mean())
    res["median_abs_error"] = float(cn.BAG.abs().median())
    res["frac_abs_BAG_gt10"] = float((cn.BAG.abs() > 10).mean())
    s = linregress(cn.age, cn.BAG)
    res["BAG_vs_age_slope"], res["BAG_vs_age_p"] = float(s.slope), float(s.pvalue)
    res["mean_BAG_CN"] = float(cn.BAG.mean())
    acc, maj = three_class(cn)
    res["age_group_acc_3class"], res["age_group_majority_baseline"] = acc, maj
    res["age_group_acc_CI95"] = cluster_boot(cn, lambda d: three_class(d)[0]).tolist()
    sess = cn.groupby("mri_label").agg(age=("age", "first"), BAG=("BAG", "mean"), subject_id=("subject_id", "first"))
    res["session_averaged_MAE"] = float(sess.BAG.abs().mean())
    res["bias_params_per_fold"] = params
    bands = band_table(cn)
    imp = df[df.CDRTOT > 0].groupby("CDRTOT").BAG.agg(["count", "mean", "std"]).reset_index()
    cdr0 = df[df.CDRTOT == 0].BAG.agg(["count", "mean", "std"])

    print(json.dumps({k: v for k, v in res.items() if k != "bias_params_per_fold"}, indent=2))
    print("\nBy age band (CN):\n", bands.round(2).to_string(index=False))
    print("\nBAG by CDR (should increase with CDR):\n", imp.round(2).to_string(index=False), "\nCDR 0 mean BAG:", round(cdr0["mean"], 2))
    if method == "bag_regress":
        print("\n[NOTE] --method bag_regress: MAE/R2 above are NOT valid accuracy metrics (true age enters the prediction). "
              "Report accuracy from --method invert; use bag_regress BAG only as an age-independent covariate.")
    if res["BAG_vs_age_p"] < 0.05 and method == "invert":
        print("\n[CHECK FAILED] BAG still trends with age in CN (Step 5.3). Fix before survival analyses.")
    if exclude_qc is False and res["frac_abs_BAG_gt10"] > 0.03:
        print(f"[WARN] {100 * res['frac_abs_BAG_gt10']:.1f}% of CN scans have |BAG|>10 y -> inspect preprocessing/QC.")

    C.RES_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(C.RES_DIR / f"accuracy_{variant}.json", "w"), indent=2)
    bands.to_csv(C.RES_DIR / f"accuracy_bands_{variant}.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="m1")
    ap.add_argument("--raw", default=None)
    ap.add_argument("--exclude_qc", action="store_true")
    ap.add_argument("--method", default="invert", choices=["invert", "bag_regress"])
    a = ap.parse_args()
    main(a.variant, a.raw, a.exclude_qc, a.method)
