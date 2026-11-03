"""
06_slopes.py   (Phases 7-8 -- CPU)

Builds baseline windows, raw / thresholded / shrunk / SNR slopes, slope SE, reliability, and writes the survival cohort.

    python 06_slopes.py --variant m1 --k 3 --min_span 2.0
    python 06_slopes.py --bag v3/predictions/bag_m1.csv --noise_json v3/results/noise_floor_m1.json --k 3

sigma (residual SD of a single scan's BAG) comes from 05_noise_floor.py (pooled sigma_SEM) unless --sigma is given.
IMPORTANT caveat built into the output: same-day sigma_SEM is a LOWER BOUND on between-visit noise (no repositioning
across months, no scanner drift, no true non-linearity). The unconstrained mixed model below estimates the residual SD
actually present in the windows; if it is much larger than sigma_SEM, say so and run the sensitivity analysis in 08.
"""
import argparse, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")
import config_v3 as C
import longitudinal_lib as L


def main(a):
    bag = pd.read_csv(a.bag or C.PRED_DIR / f"bag_{a.variant}.csv")
    if a.sigma is None:
        nj = json.load(open(a.noise_json or C.RES_DIR / f"noise_floor_{a.variant}.json"))
        sigma = nj["sigma_SEM"]
    else:
        sigma = a.sigma
    print(f"sigma_SEM used = {sigma:.3f} y")
    sessions = L.build_sessions(bag)
    d, post = L.build_windows(sessions, k=a.k, min_span_years=a.min_span, sigma=sigma)
    if len(d) < 10:
        raise SystemExit(f"Only {len(d)} eligible subjects for k={a.k}, min_span={a.min_span}")
    d, info = L.add_slope_variants(d, sigma)
    info.update(k=a.k, min_span=a.min_span, sigma=sigma, n_events=int(d.event.sum()),
                median_window_span=float(d.window_span_years.median()),
                n_subjects_before_span_rule=int(L.build_windows(sessions, k=a.k, min_span_years=0, sigma=sigma)[0].shape[0]))

    # slope reliability headline
    info["reliability_flag"] = "LOW (<0.5): slope is close to unusable as an individual marker" if info["mean_reliability"] < 0.5 else "adequate (>=0.5)"
    info["SE_larger_than_abs_slope"] = bool(info["median_SE"] > info["median_abs_slope"])

    # unconstrained mixed model on the SAME baseline-window sessions (sensitivity; Step 8.4 caveat)
    try:
        ws = []
        for sid in d.subject_id:
            g = sessions[sessions.subject_id == sid].sort_values("day").iloc[:a.k].copy()
            g["t"] = (g.day - g.day.iloc[0]) / 365.25
            ws.append(g)
        ws = pd.concat(ws)
        mm = smf.mixedlm("BAG ~ t", ws, groups=ws["subject_id"], re_formula="~t").fit(reml=True)
        info["mixedlm_residual_SD_free"] = float(np.sqrt(mm.scale))
        info["mixedlm_slope_SD_free"] = float(np.sqrt(mm.cov_re.iloc[1, 1]))
        info["residual_SD_free_over_sigma_SEM"] = float(np.sqrt(mm.scale) / sigma)
    except Exception as e:
        info["mixedlm_error"] = str(e)[:200]

    tag = f"{a.variant}_k{a.k}" + (f"_span{a.min_span:g}" if a.min_span else "")
    C.RES_DIR.mkdir(parents=True, exist_ok=True)
    d.to_csv(C.RES_DIR / f"slopes_{tag}.csv", index=False)
    pd.to_pickle(post, C.RES_DIR / f"post_{tag}.pkl")
    json.dump(info, open(C.RES_DIR / f"slope_info_{tag}.json", "w"), indent=2)
    print(json.dumps(info, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
    ax[0].scatter(d.S_raw, d.S_shrunk, s=12); lim = [d.S_raw.min(), d.S_raw.max()]
    ax[0].plot(lim, lim, "k--", lw=.8); ax[0].set_xlabel("raw slope (y/y)"); ax[0].set_ylabel("shrunk slope (y/y)")
    ax[1].scatter(d.S_raw.abs(), d.SE_S, s=12); ax[1].plot([0, d.S_raw.abs().max()], [0, d.S_raw.abs().max()], "k--", lw=.8)
    ax[1].set_xlabel("|raw slope|"); ax[1].set_ylabel("slope SE")
    plt.tight_layout(); plt.savefig(C.RES_DIR / f"slopes_{tag}.png", dpi=200)
    print("saved", C.RES_DIR / f"slopes_{tag}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="m1")
    ap.add_argument("--bag", default=None)
    ap.add_argument("--noise_json", default=None)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--min_span", type=float, default=0.0)
    main(ap.parse_args())
