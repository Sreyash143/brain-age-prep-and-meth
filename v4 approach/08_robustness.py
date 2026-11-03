"""
08_robustness.py   (Phase 10 -- CPU)

Re-runs the slope + survival pipeline over a grid and writes one robustness table (v3/results/robustness_<variant>.csv):
    10.1 window size k in {3,4,5}          10.2 sigma at its bootstrap CI bounds     10.3 minimum window span {0,1,2} y
    10.4 sigma estimated on subjects OUTSIDE the survival cohort (and random split-halves) to show the noise floor is not
         driven by the subjects being analysed
    10.6 estimators: raw, thresholded, shrunk, SNR (C-index by repeated CV, same folds)
Not automated here (needs metadata you may not have): 10.5 scanner/field-strength strata, 10.7 alternative CDR-matching rule.

    python 08_robustness.py --variant m1
"""
import argparse, importlib, json, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
warnings.filterwarnings("ignore")
import config_v3 as C
import longitudinal_lib as L
V = importlib.import_module("07_survival_validation")
N = importlib.import_module("05_noise_floor")


def quick(d, n_repeats):
    ev = int(d.event.sum())
    r = dict(n=len(d), events=ev)
    if ev < 8:
        return r
    models = {"floor": V.FLOOR, "raw": V.FLOOR + ["S_raw"], "thr": V.FLOOR + ["S_thr"], "shrunk": V.FLOOR + ["S_shrunk"], "snr": V.FLOOR + ["S_snr"]}
    rs = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=C.SEED)
    acc = {m: [] for m in models}
    for fi, (tr, te) in enumerate(rs.split(d, d.event)):
        if d.iloc[te].event.sum() < 2:
            continue
        for m, cols in models.items():
            try:
                pen = V.tune_pen(d.iloc[tr], cols, seed=fi)
                acc[m].append(V.cidx(*V.fit_cox(d.iloc[tr], cols, pen), d.iloc[te], cols))
            except Exception:
                pass
    for m in models:
        r[f"C_{m}"] = float(np.mean(acc[m])) if acc[m] else np.nan
    return r


def main(a):
    bag = pd.read_csv(a.bag or C.PRED_DIR / f"bag_{a.variant}.csv")
    nj = json.load(open(a.noise_json or C.RES_DIR / f"noise_floor_{a.variant}.json"))
    sess = L.build_sessions(bag)
    sigmas = {"point": nj["sigma_SEM"], "ci_low": nj["sigma_SEM_CI"][0], "ci_high": nj["sigma_SEM_CI"][1]}
    rows = []
    for k in (3, 4, 5):
        for span in (0.0, 1.0, 2.0):
            for sname, sg in sigmas.items():
                if sname != "point" and (k != 3 or span != a.primary_span):
                    continue                                    # CI-bound sweep only around the primary configuration
                d, _ = L.build_windows(sess, k=k, min_span_years=span, sigma=sg)
                if len(d) < 15:
                    rows.append(dict(k=k, min_span=span, sigma=sname, n=len(d), note="too few subjects"))
                    continue
                d, info = L.add_slope_variants(d, sg)
                r = dict(k=k, min_span=span, sigma=sname, sigma_value=sg, mean_reliability=info["mean_reliability"],
                         frac_exceeding_threshold=info["frac_exceeding_threshold"]); r.update(quick(d, a.n_repeats))
                rows.append(r); print(r)
    # 10.4 sigma from subjects outside the survival cohort + split-half stability
    d3, _ = L.build_windows(sess, k=3, min_span_years=a.primary_span, sigma=nj["sigma_SEM"])
    S = N.session_table(bag)
    out_cohort = S[~S.subject_id.isin(d3.subject_id)]
    extra = dict(sigma_outside_survival_cohort=N.pooled_sem(out_cohort) if len(out_cohort) > 20 else np.nan,
                 n_sessions_outside=int(len(out_cohort)))
    rng = np.random.default_rng(C.SEED); subs = S.subject_id.unique()
    halves = []
    for _ in range(200):
        h = set(rng.choice(subs, len(subs) // 2, replace=False))
        halves.append(N.pooled_sem(S[S.subject_id.isin(h)]))
    extra.update(sigma_split_half_median=float(np.median(halves)), sigma_split_half_IQR=[float(x) for x in np.percentile(halves, [25, 75])])
    print(extra)
    pd.DataFrame(rows).to_csv(C.RES_DIR / f"robustness_{a.variant}.csv", index=False)
    json.dump(extra, open(C.RES_DIR / f"robustness_sigma_{a.variant}.json", "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="m1")
    ap.add_argument("--bag", default=None)
    ap.add_argument("--noise_json", default=None)
    ap.add_argument("--primary_span", type=float, default=2.0)
    ap.add_argument("--n_repeats", type=int, default=5)
    main(ap.parse_args())
