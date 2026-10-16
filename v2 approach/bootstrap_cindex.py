# =========================================================
# BOOTSTRAP C-INDEX SIGNIFICANCE TEST — Section III-F.5 / III-G.5
# =========================================================
"""
Drop-in addition for your Cell 18/19 equivalent (longitudinal + survival
analysis), to be run AFTER you already have `surv_df`, `cph_raw`, `cph_clean`,
`c_raw`, `c_clean` from that cell.

WHY THIS REPLACES "DeLong's test": DeLong's test is designed to compare two
correlated AUROC curves from a binary classifier — it does not directly
apply to Harrell's C-index from two Cox models. Your own paper's methodology
(III-F.5/III-G.5) already specifies the correct approach: resample subjects
with replacement, refit both Cox models, compute the C-index difference,
repeat 1000x, and take the 2.5th/97.5th percentiles as the 95% CI. That is
exactly what `bootstrap_cindex_difference` below does.
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.exceptions import ConvergenceError
import warnings


def bootstrap_cindex_difference(surv_df: pd.DataFrame, cols_raw, cols_clean,
                                 duration_col="time_to_event", event_col="event_observed",
                                 n_boot=1000, seed=42):
    """
    Returns a dict with: c_raw, c_clean, observed_delta, boot_deltas (array),
    ci_lower, ci_upper, n_successful, p_value_approx (two-sided, from the
    fraction of bootstrap deltas that cross 0).
    """
    rng = np.random.default_rng(seed)
    n = len(surv_df)

    # Observed (point) estimate on the full data — same as your Cell 18/19.
    cph_raw = CoxPHFitter().fit(surv_df[cols_raw], duration_col=duration_col, event_col=event_col)
    cph_clean = CoxPHFitter().fit(surv_df[cols_clean], duration_col=duration_col, event_col=event_col)
    c_raw = concordance_index(surv_df[duration_col], -cph_raw.predict_partial_hazard(surv_df), surv_df[event_col])
    c_clean = concordance_index(surv_df[duration_col], -cph_clean.predict_partial_hazard(surv_df), surv_df[event_col])
    observed_delta = c_clean - c_raw

    boot_deltas = []
    n_failed = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)  # resample subjects with replacement
            resample = surv_df.iloc[idx].reset_index(drop=True)

            # Need at least a handful of events in the resample for Cox to be meaningful.
            if resample[event_col].sum() < 5:
                n_failed += 1
                continue

            try:
                m_raw = CoxPHFitter().fit(resample[cols_raw], duration_col=duration_col, event_col=event_col)
                m_clean = CoxPHFitter().fit(resample[cols_clean], duration_col=duration_col, event_col=event_col)
                c_r = concordance_index(resample[duration_col], -m_raw.predict_partial_hazard(resample), resample[event_col])
                c_c = concordance_index(resample[duration_col], -m_clean.predict_partial_hazard(resample), resample[event_col])
                boot_deltas.append(c_c - c_r)
            except (ConvergenceError, np.linalg.LinAlgError, ZeroDivisionError, Exception):
                n_failed += 1
                continue

    boot_deltas = np.array(boot_deltas)
    n_successful = len(boot_deltas)

    if n_successful < 100:
        print(f"WARNING: only {n_successful}/{n_boot} bootstrap resamples converged. "
              f"CI below is unreliable with this few successful resamples — your survival "
              f"cohort may be too small (few conversion events) for this test to be meaningful.")

    ci_lower, ci_upper = np.percentile(boot_deltas, [2.5, 97.5]) if n_successful > 0 else (np.nan, np.nan)

    # Approximate two-sided p-value: 2 * min(P(delta<=0), P(delta>=0)) from the bootstrap distribution.
    if n_successful > 0:
        p_le0 = float(np.mean(boot_deltas <= 0))
        p_ge0 = float(np.mean(boot_deltas >= 0))
        p_value_approx = min(1.0, 2 * min(p_le0, p_ge0))
    else:
        p_value_approx = np.nan

    result = dict(
        c_raw=c_raw, c_clean=c_clean, observed_delta=observed_delta,
        boot_deltas=boot_deltas, ci_lower=ci_lower, ci_upper=ci_upper,
        n_successful=n_successful, n_failed=n_failed, p_value_approx=p_value_approx,
    )

    print("=" * 70)
    print("BOOTSTRAP C-INDEX DIFFERENCE (clean - raw)")
    print("=" * 70)
    print(f"C-index raw   : {c_raw:.4f}")
    print(f"C-index clean : {c_clean:.4f}")
    print(f"Observed delta: {observed_delta:+.4f}")
    print(f"Bootstrap 95% CI on delta: [{ci_lower:+.4f}, {ci_upper:+.4f}]  "
          f"(from {n_successful}/{n_boot} successful resamples)")
    print(f"Approx. two-sided p-value: {p_value_approx:.4f}")
    if n_successful > 0:
        if ci_lower > 0:
            print("-> CI excludes 0: clean speed significantly OUTPERFORMS raw speed.")
        elif ci_upper < 0:
            print("-> CI excludes 0: clean speed significantly UNDERPERFORMS raw speed.")
        else:
            print("-> CI includes 0: no significant difference detected at alpha=0.05.")
    print("=" * 70)

    return result


# Example usage (place after your existing Cell 18/19 Cox-model code):
#
#   cols_raw = ["time_to_event", "event_observed", "S_raw", "C", "age", "sex", "N"]
#   cols_clean = ["time_to_event", "event_observed", "S_clean", "C", "age", "sex", "N"]
#   result = bootstrap_cindex_difference(surv_df, cols_raw, cols_clean, n_boot=1000)
