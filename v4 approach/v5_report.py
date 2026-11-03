# -*- coding: utf-8 -*-
"""
v5_report.py -- automated, leakage-aware evaluation + paired comparison from prediction CSVs.

Needs only pandas / numpy / scipy (no torch, no GPU, no project imports), so it can be run anywhere
and re-run on existing V3/V4 predictions to reproduce their reported numbers.

INPUT  raw_<tag>.csv  (out-of-fold, one row per scan) with columns:
       subject_id, day, run, mri_label, file_path, age, CDRTOT, fold, predicted_age [, sex]

WHAT IT DOES (per tag, per population = qc_passed and all_eligible, kept SEPARATE)
  1. validates the files (subject in exactly one fold, unique file_path, no NaN, identical cohort
     across tags, key collisions reported)
  2. cross-fitted 'invert' bias correction, identical to V4's 10_eval_canonical.crossfit_correct:
        for held-out fold k: fit  pred = a*age + b  on CN sessions of the OTHER folds,
        brain_age = (pred - b)/a        (held-out labels never enter the fit)
  3. reports RAW and CORRECTED metrics separately; never mixes them:
        MAE pooled (all CN scans)  |  MAE fold-mean (mean of 5 per-fold MAEs)  |  RMSE  |  Pearson r
        R2 = 1 - SSE/SST, SST around the mean chronological age  (NOT squared correlation)
        95% CI = subject-cluster bootstrap of the POOLED MAE (resample subjects, keep all their scans)
  4. residual bias: slope of (pred-age) on age, MAE and signed error per age band, by sex
  5. paired subject-cluster bootstrap of delta-MAE (tag - baseline) on identical scans

Usage:
  python v5_report.py --tags v4_sfcn_zscore_ablation,v4_sfcn_kl_huber --baseline v4_sfcn_zscore_ablation \
      --pred_dir v4/predictions --pred_dir v5/predictions --qc_csv v3/qc_scans_v3.csv --out_dir v5/results
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr

SEED = 42
REQ = ["subject_id", "day", "run", "mri_label", "file_path", "age", "CDRTOT", "fold", "predicted_age"]
BANDS = [(0, 55), (55, 65), (65, 75), (75, 85), (85, 120)]


# ----------------------------------------------------------------------------- io / QC
def load_raw(tag, pred_dirs):
    for d in pred_dirs:
        p = Path(d) / f"raw_{tag}.csv"
        if p.exists():
            return pd.read_csv(p), p
    raise FileNotFoundError(f"raw_{tag}.csv not found in {list(map(str, pred_dirs))}")


def qc_bad_paths(qc_df):
    """Same rule as core_v4.qc_bad_table: qc_flag==1 is bad; manual review can override
    (CONFIRMED-FAILURE always bad, LOOKS-OK clears an automatic flag). Image-based only."""
    if "file_path" not in qc_df.columns:
        raise ValueError("QC CSV needs a file_path column")
    flags = pd.to_numeric(qc_df["qc_flag"], errors="coerce") if "qc_flag" in qc_df.columns else pd.Series(0, index=qc_df.index)
    bad = flags.fillna(0).eq(1)
    if "qc_manual_review" in qc_df.columns:
        m = qc_df["qc_manual_review"].fillna("").astype(str).str.strip().str.upper()
        bad = (bad & ~m.isin(["LOOKS-OK", "LOOKS_OK", "OK"])) | m.isin(["CONFIRMED-FAILURE", "CONFIRMED_FAILURE", "FAIL"])
    return set(qc_df.loc[bad, "file_path"])


# ----------------------------------------------------------------------------- validation
def validate(df, tag):
    res = []

    def chk(name, ok, detail=""):
        res.append(dict(tag=tag, check=name, status="PASS" if ok else "FAIL", detail=detail))

    missing = [c for c in REQ if c not in df.columns]
    chk("required columns present", not missing, f"missing={missing}")
    if missing:
        return res
    chk("file_path is unique (one row per scan)", not df.file_path.duplicated().any(),
        f"duplicates={int(df.file_path.duplicated().sum())}")
    chk("every subject belongs to exactly one fold (no repeat-scan leakage)", df.groupby("subject_id").fold.nunique().max() == 1)
    chk("folds 0..4 all present", set(df.fold.unique()) == set(range(5)), f"folds={sorted(df.fold.unique())}")
    cn = df[df.CDRTOT == 0]
    chk("no NaN age / prediction among CN scans", not cn[["age", "predicted_age"]].isna().any().any())
    k = int(df.duplicated(["subject_id", "day", "run"]).sum())
    res.append(dict(tag=tag, check="(subject,day,run) key collisions [informational]", status="WARN" if k else "PASS",
                    detail=f"{k} colliding rows; file_path is used as the scan key here"))
    return res


# ----------------------------------------------------------------------------- bias correction
def crossfit_correct(df):
    """Identical to V4 10_eval_canonical.crossfit_correct (V3's non-circular 'invert' method)."""
    df = df.copy()
    df["brain_age"] = np.nan
    params = {}
    sess = (df[df.CDRTOT == 0].groupby(["fold", "mri_label"])
            .agg(age=("age", "mean"), pred=("predicted_age", "mean")).reset_index())
    for k in sorted(df.fold.unique()):
        tr = sess[sess.fold != k]
        lr = linregress(tr.age, tr.pred)
        if abs(lr.slope) < 1e-6:
            raise ValueError("degenerate bias-correction slope")
        m = df.fold == k
        df.loc[m, "brain_age"] = (df.loc[m, "predicted_age"] - lr.intercept) / lr.slope
        params[int(k)] = dict(slope=float(lr.slope), intercept=float(lr.intercept), n_sessions=int(len(tr)))
    return df, params


# ----------------------------------------------------------------------------- statistics
def subject_sums(subj, values):
    g = pd.DataFrame({"s": subj.values, "v": np.asarray(values, float)}).groupby("s").v.agg(["sum", "count"])
    return g["sum"].to_numpy(), g["count"].to_numpy().astype(float), g.index.to_numpy()


def boot_ratio_ci(sums, counts, B=10000, seed=SEED):
    """95% CI of sum(sums)/sum(counts) when whole SUBJECTS are resampled with replacement."""
    rng = np.random.default_rng(seed)
    n, out = len(sums), []
    for _ in range(0, B, 2000):
        idx = rng.integers(0, n, size=(min(2000, B - len(out)), n))
        out.extend(sums[idx].sum(1) / counts[idx].sum(1))
    return [float(x) for x in np.percentile(out, [2.5, 97.5])], np.asarray(out)


def err_metrics(cn, col, B):
    e = (cn[col] - cn.age).to_numpy(float)
    s_abs, cnt, _ = subject_sums(cn.subject_id, np.abs(e))
    ci, _ = boot_ratio_ci(s_abs, cnt, B)
    s_sq, cnt2, _ = subject_sums(cn.subject_id, e ** 2)
    rmse_ci = [float(np.sqrt(x)) for x in boot_ratio_ci(s_sq, cnt2, B)[0]]
    fold_mae = {int(k): float(np.abs(g[col] - g.age).mean()) for k, g in cn.groupby("fold")}
    sst = float(((cn.age - cn.age.mean()) ** 2).sum())
    return dict(MAE_pooled=float(np.abs(e).mean()), MAE_pooled_CI95=ci,
                MAE_fold_mean=float(np.mean(list(fold_mae.values()))), MAE_fold_sd=float(np.std(list(fold_mae.values()))),
                MAE_by_fold=fold_mae, RMSE=float(np.sqrt((e ** 2).mean())), RMSE_CI95=rmse_ci,
                pearson_r=float(pearsonr(cn.age, cn[col])[0]), R2_SSE_over_SST=float(1 - (e ** 2).sum() / sst))


def band_table(cn):
    rows = []
    for lo, hi in BANDS:
        g = cn[(cn.age > lo) & (cn.age <= hi)] if lo else cn[cn.age <= hi]
        if len(g):
            rows.append(dict(band=f"({lo},{hi}]", n=int(len(g)),
                             raw_MAE=float((g.predicted_age - g.age).abs().mean()), raw_bias=float((g.predicted_age - g.age).mean()),
                             corr_MAE=float((g.brain_age - g.age).abs().mean()), corr_bias=float((g.brain_age - g.age).mean())))
    return rows


def paired_delta(a, b, B):
    """a, b: DataFrames indexed by file_path with columns subject_id, age, <col>. Returns delta=MAE_a-MAE_b."""
    d = (a["err"] - b["err"])
    s, c, _ = subject_sums(a.subject_id, d)
    ci, draws = boot_ratio_ci(s, c, B, seed=SEED + 1)
    return dict(delta_MAE=float(d.mean()), CI95=ci, p_delta_lt_0=float((draws < 0).mean()),
                n_scans=int(len(d)), n_subjects=int(len(s)))


# ----------------------------------------------------------------------------- main analysis
def analyse(tags, baseline, pred_dirs, qc_bad, qc_passed_from, folds, B, intersect):
    checks, raws = [], {}
    for t in tags:
        df, p = load_raw(t, pred_dirs)
        raws[t] = df
        checks += validate(df, t)
        print(f"loaded {t}: {p} {df.shape}")

    keysets = {t: set(df.file_path) for t, df in raws.items()}
    same = len({frozenset(v) for v in keysets.values()}) == 1
    checks.append(dict(tag="ALL", check="identical scan set across all tags (comparable cohorts)", status="PASS" if same else ("WARN" if intersect else "FAIL"),
                       detail="" if same else f"sizes={ {t: len(v) for t, v in keysets.items()} }"))
    if not same:
        if not intersect:
            return checks, None
        common = set.intersection(*keysets.values())
        raws = {t: df[df.file_path.isin(common)].copy() for t, df in raws.items()}

    out = dict(definitions=dict(
        cohort="cognitively normal scans (CDRTOT==0); scans with missing CDR or CDRTOT>0 are not evaluated",
        MAE_pooled="mean |pred-age| over all CN scans of the population (one scan = one unit)",
        MAE_fold_mean="unweighted mean of the 5 per-fold MAEs (the 'headline' value in V4 Table A is of this kind)",
        CI="95% percentile bootstrap of the pooled value, resampling SUBJECTS (all scans of a subject together), 10^4 draws",
        R2="1 - SSE/SST (predictive R2 about the mean chronological age), not squared Pearson r",
        raw="predicted_age, model output with no correction",
        corrected="brain_age = (pred - b)/a with a,b fitted on other-fold CN sessions only (cross-fitted)"),
        populations={}, folds_evaluated=folds)

    for pop in ("qc_passed", "all_eligible"):
        if pop == "qc_passed" and qc_bad is None and qc_passed_from is None:
            continue
        res_pop, scans = {}, {}
        for t in tags:
            df = raws[t]
            if pop == "qc_passed":
                df = df[df.file_path.isin(qc_passed_from)] if qc_passed_from is not None else df[~df.file_path.isin(qc_bad)]
            df, params = crossfit_correct(df)
            cn = df[(df.CDRTOT == 0) & df.fold.isin(folds)].copy()
            r = dict(n_CN_scans=int(len(cn)), n_CN_subjects=int(cn.subject_id.nunique()),
                     n_CN_sessions=int(cn.mri_label.nunique()), n_scans_all_CDR=int(len(df)),
                     raw=err_metrics(cn, "predicted_age", B), corrected=err_metrics(cn, "brain_age", B),
                     residual_vs_age_slope_raw=float(linregress(cn.age, cn.predicted_age - cn.age).slope),
                     residual_vs_age_slope_corrected=float(linregress(cn.age, cn.brain_age - cn.age).slope),
                     mean_error_raw=float((cn.predicted_age - cn.age).mean()),
                     by_age_band=band_table(cn),
                     by_sex={str(k): dict(n=int(len(g)), raw_MAE=float((g.predicted_age - g.age).abs().mean()),
                                          corr_MAE=float((g.brain_age - g.age).abs().mean()))
                             for k, g in cn.groupby(cn.sex.fillna("NA"))} if "sex" in cn.columns else {},
                     bias_params_per_fold=params)
            res_pop[t] = r
            scans[t] = cn.set_index("file_path")
        if baseline in scans:
            for t in tags:
                if t == baseline:
                    continue
                for kind, col in (("raw", "predicted_age"), ("corrected", "brain_age")):
                    a, b = scans[t].copy(), scans[baseline].copy()
                    a["err"], b["err"] = (a[col] - a.age).abs(), (b[col] - b.age).abs()
                    b = b.loc[a.index]
                    res_pop[t].setdefault("paired_vs_baseline", {})[kind] = dict(baseline=baseline, **paired_delta(a, b, B))
        out["populations"][pop] = res_pop
    return checks, out


def to_markdown(out, tags):
    lines = []
    for pop, res in out["populations"].items():
        lines += [f"\n### Population: {pop}  (CN scans only; folds {out['folds_evaluated']})\n",
                  "| model | n scans / subj | RAW MAE pooled [95% CI] | RAW MAE fold-mean | RAW RMSE | RAW R2 | CORR MAE pooled [95% CI] | CORR MAE fold-mean | CORR R2 | paired dMAE raw [CI] | paired dMAE corr [CI] |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for t in tags:
            r = res[t]
            pr = r.get("paired_vs_baseline", {})
            f = lambda d: f"{d['delta_MAE']:+.3f} [{d['CI95'][0]:+.3f},{d['CI95'][1]:+.3f}]" if d else "(baseline)"
            lines.append(f"| {t} | {r['n_CN_scans']}/{r['n_CN_subjects']} | {r['raw']['MAE_pooled']:.3f} [{r['raw']['MAE_pooled_CI95'][0]:.3f},{r['raw']['MAE_pooled_CI95'][1]:.3f}] "
                         f"| {r['raw']['MAE_fold_mean']:.3f} | {r['raw']['RMSE']:.3f} | {r['raw']['R2_SSE_over_SST']:.3f} "
                         f"| {r['corrected']['MAE_pooled']:.3f} [{r['corrected']['MAE_pooled_CI95'][0]:.3f},{r['corrected']['MAE_pooled_CI95'][1]:.3f}] "
                         f"| {r['corrected']['MAE_fold_mean']:.3f} | {r['corrected']['R2_SSE_over_SST']:.3f} | {f(pr.get('raw'))} | {f(pr.get('corrected'))} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True, help="comma-separated tags (raw_<tag>.csv)")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--pred_dir", action="append", required=True)
    ap.add_argument("--qc_csv", default=None, help="QC csv with file_path, qc_flag [, qc_manual_review]")
    ap.add_argument("--qc_passed_from", default=None, help="alternative: a CSV whose file_path column lists the QC-passed scans")
    ap.add_argument("--folds", default="0,1,2,3,4", help="evaluate only these outer folds (needed when a config was tuned on other folds)")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--intersect", action="store_true", help="if cohorts differ, evaluate the common scans (flagged WARN)")
    ap.add_argument("--out_dir", default="v5/results")
    ap.add_argument("--name", default="comparison")
    a = ap.parse_args(argv)

    tags = a.tags.split(",")
    if a.baseline not in tags:
        tags = [a.baseline] + tags
    qc_bad = qc_passed = None
    if a.qc_csv:
        qc_bad = qc_bad_paths(pd.read_csv(a.qc_csv))
    elif a.qc_passed_from:
        qc_passed = set(pd.read_csv(a.qc_passed_from).file_path)
    checks, out = analyse(tags, a.baseline, a.pred_dir, qc_bad, qc_passed, [int(x) for x in a.folds.split(",")], a.boot, a.intersect)

    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks).to_csv(od / f"{a.name}_integrity_checks.csv", index=False)
    for c in checks:
        print(f"[{c['status']}] {c['tag']}: {c['check']} {c['detail']}")
    if out is None or any(c["status"] == "FAIL" for c in checks):
        print("Integrity FAIL -- metrics not written.")
        return 1
    (od / f"{a.name}.json").write_text(json.dumps(out, indent=2))
    md = to_markdown(out, tags)
    (od / f"{a.name}.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
