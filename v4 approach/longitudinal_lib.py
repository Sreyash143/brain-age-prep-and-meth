"""
longitudinal_lib.py -- Phases 7-8 as reusable functions (used by 06, 07, 08).

Key design points
  * Session-level BAG = mean of repeat scans; the session's noise variance is sigma^2 / n_runs.
  * Baseline window = first k sessions, ALL with CDR == 0, plus >= 1 later session with a CDR.
    Everything that feeds survival (slope, intercept, SE, shrinkage) uses window sessions ONLY.
  * Time zero for survival = LAST window session. Event = first later session with CDR > 0; otherwise censored at last visit.
  * Shrunk slope = empirical-Bayes estimate with the residual variance FIXED at sigma_SEM^2 (method-of-moments tau^2).
    With times centred within subject, intercept and slope are uncorrelated, so this equals the random-slope BLUP of a
    mixed model whose residual variance is constrained to sigma_SEM^2 -- without needing a solver that can fix it.
"""
import numpy as np, pandas as pd


def build_sessions(bag):
    """bag: scan-level frame with BAG. Returns one row per session."""
    g = bag.groupby("mri_label")
    s = g.agg(subject_id=("subject_id", "first"), day=("day", "first"), age=("age", "first"), sex=("sex", "first"),
              CDRTOT=("CDRTOT", "first"), CDRSUM=("CDRSUM", "first"), MMSE=("MMSE", "first"),
              BAG=("BAG", "mean"), n_runs=("BAG", "size")).reset_index()
    return s.sort_values(["subject_id", "day"]).reset_index(drop=True)


def wls_slope(t, y, w, sigma):
    """weighted straight-line fit; w = number of repeat scans per session (precision weights)."""
    w = np.asarray(w, float)
    tb = np.average(t, weights=w)
    yb = np.average(y, weights=w)
    sxx = float((w * (t - tb) ** 2).sum())
    if sxx <= 0:
        return np.nan, np.nan, tb, yb, 0.0
    b = float((w * (t - tb) * (y - yb)).sum() / sxx)
    return b, float(sigma / np.sqrt(sxx)), tb, yb, sxx


def build_windows(sessions, k=3, min_span_years=0.0, sigma=1.0, sigma_cn_only=None):
    """Returns (subject_table, post_sessions_dict). One row per eligible subject."""
    rows, post = [], {}
    for sid, g in sessions.groupby("subject_id"):
        g = g.sort_values("day")
        if len(g) < k + 1:
            continue
        win, later = g.iloc[:k], g.iloc[k:]
        if win.CDRTOT.isna().any() or not (win.CDRTOT == 0).all():
            continue
        later = later[later.CDRTOT.notna()]
        if len(later) == 0:
            continue
        t = (win.day.to_numpy(float) - win.day.iloc[0]) / 365.25
        span = float(t[-1] - t[0])
        if span < min_span_years or span <= 0:
            continue
        b, se, tb, yb, sxx = wls_slope(t, win.BAG.to_numpy(float), win.n_runs.to_numpy(float), sigma)
        if not np.isfinite(b):
            continue
        t_last_day = win.day.iloc[-1]
        conv = later[later.CDRTOT > 0]
        if len(conv):
            ev, tt = 1, (conv.day.iloc[0] - t_last_day) / 365.25
        else:
            ev, tt = 0, (later.day.iloc[-1] - t_last_day) / 365.25
        if tt <= 0:
            continue
        rows.append(dict(subject_id=sid, k=k, window_span_years=span, S_raw=b, SE_S=se,
                         bag_base=yb + b * (t[-1] - tb),            # fitted BAG at the last window session
                         bag_mean=yb, age_base=float(win.age.iloc[-1]), sex=win.sex.iloc[0],
                         n_window_scans=int(win.n_runs.sum()), time_to_event=float(tt), event=int(ev)))
        p = later.copy()
        p["t_since_window"] = (p.day - t_last_day) / 365.25
        post[sid] = dict(post=p, base_CDRSUM=float(win.CDRSUM.iloc[-1]) if win.CDRSUM.notna().iloc[-1] else np.nan,
                         base_MMSE=float(win.MMSE.iloc[-1]) if win.MMSE.notna().iloc[-1] else np.nan)
    d = pd.DataFrame(rows)
    if len(d):
        d["sex"] = d.sex.map({"M": 0, "F": 1}).fillna(0)
    return d, post


def eb_shrink(S, se):
    """DerSimonian-Laird tau^2 with KNOWN per-subject sampling variance se^2; returns shrunk slopes, tau2, mu, reliability."""
    S, v = np.asarray(S, float), np.asarray(se, float) ** 2
    w0 = 1 / v
    mu0 = (w0 * S).sum() / w0.sum()
    Q = (w0 * (S - mu0) ** 2).sum()
    denom = w0.sum() - (w0 ** 2).sum() / w0.sum()
    tau2 = max(0.0, (Q - (len(S) - 1)) / denom)
    w = 1 / (v + tau2)
    mu = (w * S).sum() / w.sum()
    rel_i = tau2 / (tau2 + v) if tau2 > 0 else np.zeros_like(v)
    shrunk = mu + rel_i * (S - mu)
    return shrunk, float(tau2), float(mu), rel_i, float(np.mean(rel_i))


def add_slope_variants(d, sigma):
    d = d.copy()
    d["S_thr"] = np.where(d.S_raw.abs() > 1.96 * d.SE_S, d.S_raw, 0.0)
    d["S_snr"] = d.S_raw / d.SE_S
    sh, tau2, mu, rel_i, rel = eb_shrink(d.S_raw, d.SE_S)
    d["S_shrunk"], d["reliability_i"] = sh, rel_i
    info = dict(tau2=tau2, tau=float(np.sqrt(tau2)), mu=mu, mean_reliability=rel,
                median_SE=float(d.SE_S.median()), median_abs_slope=float(d.S_raw.abs().median()),
                frac_exceeding_threshold=float((d.S_thr != 0).mean()), n=int(len(d)),
                corr_raw_shrunk=float(np.corrcoef(d.S_raw, d.S_shrunk)[0, 1]))
    return d, info


def future_clinical_slopes(post, endpoint):
    """OLS slope of CDRSUM or MMSE on time since window end, using POST-window sessions only (>= 2 needed)."""
    out = {}
    for sid, o in post.items():
        p = o["post"].dropna(subset=[endpoint])
        if len(p) >= 2 and p.t_since_window.nunique() >= 2:
            out[sid] = float(np.polyfit(p.t_since_window, p[endpoint], 1)[0])
    return pd.Series(out, name=f"future_{endpoint}_slope")
