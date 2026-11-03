"""
07_survival_validation.py   (Phase 9 -- CPU)

    python 07_survival_validation.py --slopes v3/results/slopes_m1_k3_span2.csv [--clinical extra.csv] [--n_repeats 20] [--n_boot 1000]

Primary-endpoint rule (pre-registered): >= 30 events -> survival is primary, continuous endpoint parallel;
fewer events -> continuous endpoint primary, survival exploratory. Printed at the top.

Models (all share age_base, sex, bag_base = "floor"):
    floor | floor+raw | floor+shrunk | floor+raw+shrunk   (+ optional clinical block: baseline MMSE, APOE4, hippocampus, ventricles ...)
Reported:  likelihood-ratio tests (nested), repeated stratified subject-level CV C-index (ridge-Cox, penalty by inner CV),
           paired OUT-OF-BAG bootstrap of  C(shrunk) - C(raw),  time-dependent AUC at 2 and 4 y, Schoenfeld PH test,
           continuous endpoint (future CDR-SB / MMSE slope) with HC3 standard errors.
"""
import argparse, json, warnings
import numpy as np, pandas as pd
from scipy.stats import chi2
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.statistics import proportional_hazard_test
import statsmodels.api as sm
import config_v3 as C
import longitudinal_lib as L

warnings.filterwarnings("ignore")
T, E = "time_to_event", "event"
FLOOR = ["age_base", "sex", "bag_base"]
GRID = [0.01, 0.1, 0.5, 1.0]


def fit_cox(train, cols, pen):
    mu, sd = train[cols].mean(), train[cols].std(ddof=0).replace(0, 1)
    z = ((train[cols] - mu) / sd).assign(**{T: train[T].values, E: train[E].values})
    m = CoxPHFitter(penalizer=pen, l1_ratio=0.0).fit(z, T, E)
    return m, mu, sd


def risk(model, mu, sd, df, cols):
    return model.predict_partial_hazard(((df[cols] - mu) / sd)).to_numpy()


def cidx(model, mu, sd, df, cols):
    return concordance_index(df[T], -risk(model, mu, sd, df, cols), df[E])


def tune_pen(train, cols, seed=0):
    skf = StratifiedKFold(3, shuffle=True, random_state=seed)
    best, bp = -1, GRID[-1]
    for pen in GRID:
        cs = []
        for tr, te in skf.split(train, train[E]):
            try:
                m, mu, sd = fit_cox(train.iloc[tr], cols, pen)
                if train.iloc[te][E].sum() >= 2:
                    cs.append(cidx(m, mu, sd, train.iloc[te], cols))
            except Exception:
                pass
        if cs and np.mean(cs) > best:
            best, bp = np.mean(cs), pen
    return bp


def td_auc(t_arr, e_arr, rk, horizon):
    case = (t_arr <= horizon) & (e_arr == 1)
    ctrl = t_arr > horizon
    if case.sum() < 3 or ctrl.sum() < 3:
        return np.nan
    a, b = rk[case][:, None], rk[ctrl][None, :]
    return float(((a > b).sum() + 0.5 * (a == b).sum()) / (a.size * b.shape[1]))


def build_models(df, clinical_cols):
    M = {"floor": FLOOR, "floor+raw": FLOOR + ["S_raw"], "floor+shrunk": FLOOR + ["S_shrunk"],
         "floor+raw+shrunk": FLOOR + ["S_raw", "S_shrunk"]}
    if clinical_cols:
        M["clinical"] = FLOOR + clinical_cols
        M["clinical+shrunk"] = FLOOR + clinical_cols + ["S_shrunk"]
    return M


def survival_block(df, models, n_repeats, n_boot, out):
    ev = int(df[E].sum())
    out["n_subjects"], out["n_events"] = len(df), ev
    # --- LRT (unpenalized, full data, nested)
    ll = {}
    for name, cols in models.items():
        try:
            ll[name] = (CoxPHFitter().fit(df[cols + [T, E]], T, E).log_likelihood_, len(cols))
        except Exception as e:
            ll[name] = (np.nan, len(cols))
    lrt = {}
    for small, big in [("floor", "floor+raw"), ("floor", "floor+shrunk"), ("floor+raw", "floor+raw+shrunk"),
                       ("floor+shrunk", "floor+raw+shrunk"), ("clinical", "clinical+shrunk")]:
        if small in ll and big in ll and np.isfinite(ll[small][0]) and np.isfinite(ll[big][0]):
            stat = 2 * (ll[big][0] - ll[small][0]); k = ll[big][1] - ll[small][1]
            lrt[f"{small} -> {big}"] = dict(chi2=float(stat), df=int(k), p=float(chi2.sf(max(stat, 0), k)))
    out["LRT_unpenalized"] = lrt
    out["unpenalized_covariates_per_10_events_ok"] = {n: bool(len(c) <= ev / 10) for n, c in models.items()}

    # --- repeated stratified subject-level CV
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=C.SEED)
    rows, oof_risk = [], {n: np.full(len(df), np.nan) for n in models}
    for fi, (tr, te) in enumerate(rskf.split(df, df[E])):
        dtr, dte = df.iloc[tr], df.iloc[te]
        if dte[E].sum() < 2:
            continue
        row = {"fold": fi}
        for name, cols in models.items():
            try:
                pen = tune_pen(dtr, cols, seed=fi)
                m, mu, sd = fit_cox(dtr, cols, pen)
                row[name] = cidx(m, mu, sd, dte, cols)
                if fi < 5:                                   # first repeat -> pooled OOF risk for AUC(t)
                    oof_risk[name][te] = risk(m, mu, sd, dte, cols)
            except Exception:
                row[name] = np.nan
        rows.append(row)
    cv = pd.DataFrame(rows)
    cv.to_csv(C.RES_DIR / "cv_cindex.csv", index=False)
    out["CV_cindex_mean"] = {n: float(cv[n].mean()) for n in models}
    out["CV_cindex_sd_across_folds"] = {n: float(cv[n].std()) for n in models}
    if {"floor+raw", "floor+shrunk"} <= set(cv.columns):
        dlt = (cv["floor+shrunk"] - cv["floor+raw"]).dropna()
        out["CV_paired_diff_shrunk_minus_raw"] = dict(mean=float(dlt.mean()), pct2_5=float(np.percentile(dlt, 2.5)),
                                                      pct97_5=float(np.percentile(dlt, 97.5)),
                                                      note="spread over CV folds; folds overlap, so use the bootstrap below for inference")
    # --- time-dependent AUC on pooled first-repeat OOF risks
    auc = {}
    for n in models:
        r = oof_risk[n]; ok = ~np.isnan(r)
        auc[n] = {f"AUC_{h}y": td_auc(df[T].values[ok], df[E].values[ok], r[ok], h) for h in (2, 4)}
    out["time_dependent_AUC_naive_no_IPCW"] = auc

    # --- paired OUT-OF-BAG bootstrap: shrunk vs raw (identical subjects in each resample)
    rng = np.random.default_rng(C.SEED)
    pens = {n: tune_pen(df, models[n]) for n in ("floor+raw", "floor+shrunk")}
    diffs = []
    for b in range(n_boot):
        idx = rng.integers(0, len(df), len(df))
        oob = np.setdiff1d(np.arange(len(df)), idx)
        dtr, dte = df.iloc[idx], df.iloc[oob]
        if dtr[E].sum() < 5 or dte[E].sum() < 3:
            continue
        try:
            cr = cidx(*fit_cox(dtr, models["floor+raw"], pens["floor+raw"]), dte, models["floor+raw"])
            cs = cidx(*fit_cox(dtr, models["floor+shrunk"], pens["floor+shrunk"]), dte, models["floor+shrunk"])
            diffs.append(cs - cr)
        except Exception:
            pass
    diffs = np.array(diffs)
    if len(diffs) >= 100:
        out["PRIMARY_paired_OOB_bootstrap_C_shrunk_minus_raw"] = dict(
            n_valid=int(len(diffs)), mean=float(diffs.mean()), CI95=[float(x) for x in np.percentile(diffs, [2.5, 97.5])],
            excludes_zero=bool(np.percentile(diffs, 2.5) > 0 or np.percentile(diffs, 97.5) < 0))
    else:
        out["PRIMARY_paired_OOB_bootstrap_C_shrunk_minus_raw"] = f"only {len(diffs)} valid resamples -- too few events"
    # --- PH assumption
    try:
        m = CoxPHFitter().fit(df[models["floor+shrunk"] + [T, E]], T, E)
        ph = proportional_hazard_test(m, df[models["floor+shrunk"] + [T, E]], time_transform="rank").summary
        out["schoenfeld_p"] = {k: float(v) for k, v in ph["p"].items()}
        out["hazard_ratio_shrunk_per_unit_slope"] = float(np.exp(m.params_["S_shrunk"]))
    except Exception as e:
        out["schoenfeld_p"] = f"failed: {str(e)[:80]}"


def continuous_block(df, post, out):
    res = {}
    for endpoint, base_col in [("CDRSUM", "base_CDRSUM"), ("MMSE", "base_MMSE")]:
        fut = L.future_clinical_slopes(post, endpoint)
        d = df.set_index("subject_id").join(fut, how="inner").dropna(subset=[fut.name])
        d["baseline_score"] = [post[s][base_col] for s in d.index]
        covs = FLOOR + ([ "baseline_score"] if d.baseline_score.std() > 0 else [])
        d = d.dropna(subset=covs + ["S_raw", "S_shrunk"])
        if len(d) < 15:
            res[endpoint] = f"only {len(d)} subjects with >=2 post-window sessions"
            continue
        y = d[fut.name]
        base = sm.OLS(y, sm.add_constant(d[covs])).fit()
        r = dict(n=int(len(d)), R2_covariates_only=float(base.rsquared))
        for name in ("S_raw", "S_shrunk"):
            X = sm.add_constant(d[covs + [name]])
            m = sm.OLS(y, X).fit(cov_type="HC3")
            ry = sm.OLS(y, sm.add_constant(d[covs])).fit().resid
            rx = sm.OLS(d[name], sm.add_constant(d[covs])).fit().resid
            r[name] = dict(coef=float(m.params[name]), se_HC3=float(m.bse[name]), p=float(m.pvalues[name]),
                           R2=float(m.rsquared), delta_R2=float(m.rsquared - base.rsquared),
                           partial_r=float(np.corrcoef(rx, ry)[0, 1]))
        res[endpoint] = r
    out["continuous_endpoint"] = res


def main(a):
    d = pd.read_csv(a.slopes)
    tag = a.slopes.split("slopes_")[-1].replace(".csv", "")
    post = pd.read_pickle(a.slopes.replace("slopes_", "post_").replace(".csv", ".pkl"))
    clin = []
    if a.clinical:
        extra = pd.read_csv(a.clinical)                        # subject_id + numeric baseline columns (apoe4, hippo, vent, ...)
        d = d.merge(extra, on="subject_id", how="left")
        clin = [c for c in extra.columns if c != "subject_id"]
        d = d.dropna(subset=clin)
    d["baseline_MMSE_win"] = [post[s]["base_MMSE"] for s in d.subject_id]
    if d.baseline_MMSE_win.notna().all() and "baseline_MMSE" not in clin and a.use_mmse:
        clin = clin + ["baseline_MMSE_win"]
    ev = int(d[E].sum())
    rule = "SURVIVAL primary (>=30 events), continuous parallel" if ev >= 30 else "CONTINUOUS primary (<30 events), survival EXPLORATORY"
    print(f"events={ev} of {len(d)} subjects -> {rule}")
    out = dict(tag=tag, primary_endpoint_rule=rule)
    cv_se = float(d.SE_S.std() / d.SE_S.mean())
    out["SE_coefficient_of_variation"] = cv_se
    out["shrinkage_invariance_note"] = (
        "Shrunk = mu + r_i*(raw - mu). If SE_S is similar across subjects (CV of SE small), r_i is ~constant, shrunk is an affine "
        "copy of raw, and a Cox model/C-index cannot distinguish them (expected paired difference ~0). "
        "A shrunk-vs-raw C-index gain is only possible through heterogeneity in SE (different spans / visit counts). "
        f"Observed CV(SE) = {cv_se:.2f}.")
    print(out["shrinkage_invariance_note"])
    if ev >= 8:
        survival_block(d, build_models(d, clin), a.n_repeats, a.n_boot, out)
    else:
        out["survival"] = "fewer than 8 events: Cox models not fitted"
    continuous_block(d, post, out)
    C.RES_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(C.RES_DIR / f"validation_{tag}.json", "w"), indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    print(json.dumps(out, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slopes", required=True)
    ap.add_argument("--clinical", default=None)
    ap.add_argument("--use_mmse", action="store_true", help="add baseline MMSE to the clinical block")
    ap.add_argument("--n_repeats", type=int, default=20)
    ap.add_argument("--n_boot", type=int, default=1000)
    main(ap.parse_args())
