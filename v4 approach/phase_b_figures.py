#!/usr/bin/env python3
"""
Phase B: publication-oriented analysis and figures for the OASIS brain-age project.

Run from the project root:
    python phase_b/phase_b_figures.py

Prerequisite:
    Run the existing v5_diagnostics.py first so that the CSV analysis tables
    exist in v5/analysis/results/.

This script does not retrain models or modify prediction files.
It writes figures/tables/reports under phase_b/.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(".").resolve()
PRED = ROOT / "v5" / "predictions"
RESULTS = ROOT / "v5" / "analysis" / "results"
QC_REFERENCE = ROOT / "v4" / "predictions" / "bag_v4_sfcn_zscore_ablation_qc_passed.csv"
OUT = ROOT / "phase_b"
FIG = OUT / "figures"
TABLE = OUT / "tables"
REPORT = OUT / "reports"

for p in (FIG, TABLE, REPORT):
    p.mkdir(parents=True, exist_ok=True)

# Prediction files and labels follow v5_diagnostics.py.
FILES = {
    "V4": "raw_v4_sfcn_zscore_ablation.csv",
    "V5 RAW": "raw_v5_z_khuber.csv",
    "V5 EMA": "raw_v5_z_khuber_ema.csv",
    "V5.1 RAW": "raw_v5_1_no_tail.csv",
    "V5.1 EMA": "raw_v5_1_no_tail_ema.csv",
    "V5.2 RAW": "raw_v5_2_subject_balanced.csv",
    "V5.2 EMA": "raw_v5_2_subject_balanced_ema.csv",
    "V5.3 RAW": "raw_v5_3_mild_age_balanced.csv",
    "V5.3 EMA": "raw_v5_3_mild_age_balanced_ema.csv",
}
FINAL = "V5.1 RAW"
AGE_ORDER = ["<50", "50-59", "60-69", "70-79", "80-89", "90+"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.titlesize": 14,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Run the existing v5_diagnostics.py first and check the prediction paths."
        )
    return path


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=300)
    plt.savefig(FIG / name.replace(".png", ".pdf"))
    plt.close()


def load_predictions():
    # Use the same explicit QC-passed cognitively-normal reference cohort
    # as the existing v5_diagnostics.py. Do not assume prediction CSVs are
    # already restricted to QC-passed scans.
    qc = pd.read_csv(require_file(QC_REFERENCE))
    needed_qc = {"file_path", "subject_id", "CDRTOT", "age"}
    missing_qc = needed_qc - set(qc.columns)
    if missing_qc:
        raise ValueError(f"QC reference missing columns: {sorted(missing_qc)}")
    qc_cn = qc[(qc["CDRTOT"] == 0) & qc["file_path"].notna()].copy()
    qc_paths = set(qc_cn["file_path"])
    if len(qc_paths) != len(qc_cn):
        raise ValueError("Duplicate file_path entries in QC reference cohort")

    data = {}

    for label, filename in FILES.items():

    # V4 predictions are stored separately from the V5 predictions.
        if label == "V4":
            path = require_file(QC_REFERENCE)
        else:
            path = require_file(PRED / filename)

        df = pd.read_csv(path)
        needed = {"subject_id", "file_path", "age", "CDRTOT",
                  "predicted_age", "fold"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"{label}: missing columns {sorted(missing)} in {path}")
        df = df[df["file_path"].isin(qc_paths)].copy()
        df = df[df["CDRTOT"] == 0].copy()
        df = df.dropna(subset=["subject_id", "file_path", "age",
                               "predicted_age", "fold"])
        if set(df["file_path"]) != qc_paths:
            raise ValueError(
                f"{label}: QC cohort mismatch; missing "
                f"{len(qc_paths - set(df['file_path']))} scans and extra "
                f"{len(set(df['file_path']) - qc_paths)} scans."
            )
        df["error"] = df["predicted_age"] - df["age"]
        df["abs_error"] = df["error"].abs()
        df["sq_error"] = df["error"] ** 2
        df["age_group"] = pd.cut(
            df["age"], [-np.inf, 50, 60, 70, 80, 90, np.inf],
            labels=AGE_ORDER, right=False
        )
        data[label] = df

    # Ensure a common cohort and fold mapping before plotting comparisons.
    if len(qc_paths) != 1896 or qc_cn["subject_id"].nunique() != 767:
        warnings.warn(
            f"QC reference cohort is {len(qc_paths)} scans / "
            f"{qc_cn['subject_id'].nunique()} subjects, not the expected "
            "1,896 / 767. Verify the cohort definition."
        )
    ref = data["V4"].set_index("file_path").sort_index()
    for label, df in data.items():
        cur = df.set_index("file_path").sort_index()
        if not cur.index.equals(ref.index):
            raise ValueError(f"{label}: scan IDs differ from V4.")
        if not cur["fold"].equals(ref["fold"]):
            raise ValueError(f"{label}: fold assignments differ from V4.")
        if not cur["subject_id"].equals(ref["subject_id"]):
            raise ValueError(f"{label}: subject assignments differ from V4.")
    return data


def model_comparison(data):
    rows = []
    for label, df in data.items():
        rows.append({
            "model": label,
            "n_scans": len(df),
            "n_subjects": df.subject_id.nunique(),
            "MAE": df.abs_error.mean(),
            "RMSE": np.sqrt(df.sq_error.mean()),
            "R2": 1 - df.sq_error.sum() /
                 ((df.age - df.age.mean()) ** 2).sum(),
            "bias": df.error.mean(),
            "weighting": "scan_weighted",
        })
        subj = (df.groupby("subject_id")
                  .agg(age=("age", "mean"),
                       predicted_age=("predicted_age", "mean"))
                  .reset_index())
        err = subj.predicted_age - subj.age
        rows.append({
            "model": label,
            "n_scans": len(df),
            "n_subjects": len(subj),
            "MAE": err.abs().mean(),
            "RMSE": np.sqrt((err ** 2).mean()),
            "R2": 1 - (err ** 2).sum() /
                 ((subj.age - subj.age.mean()) ** 2).sum(),
            "bias": err.mean(),
            "weighting": "subject_mean_prediction",
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABLE / "final_model_comparison.csv", index=False)
    return out


def foldwise_figure(data):
    rows = []
    for label, df in data.items():
        for fold, g in df.groupby("fold"):
            rows.append({
                "model": label, "fold": int(fold),
                "MAE": g.abs_error.mean(),
                "RMSE": np.sqrt(g.sq_error.mean()),
                "bias": g.error.mean(),
            })
    fold = pd.DataFrame(rows)
    fold.to_csv(TABLE / "fold_wise_performance.csv", index=False)

    models = list(FILES.keys())
    x = np.arange(len(models))
    width = 0.14
    fig, ax = plt.subplots(figsize=(12, 5.5))
    folds = sorted(fold.fold.unique())
    for i, f in enumerate(folds):
        vals = [fold[(fold.model == m) & (fold.fold == f)].MAE.iloc[0]
                for m in models]
        ax.bar(x + (i - (len(folds)-1)/2) * width, vals,
               width, label=f"Fold {f}")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_ylabel("MAE (years)")
    ax.set_title("Fold-wise MAE across model variants")
    ax.legend(ncol=5, frameon=False)
    savefig("fig01_foldwise_mae.png")


def prediction_plot(data):
    df = data[FINAL]
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    hb = ax.hexbin(df.age, df.predicted_age, gridsize=35,
                   mincnt=1, bins="log")
    lo = min(df.age.min(), df.predicted_age.min())
    hi = max(df.age.max(), df.predicted_age.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5,
            label="Identity line")
    ax.set(xlabel="Chronological age (years)",
           ylabel="Predicted age (years)",
           title="V5.1 RAW: predicted versus chronological age")
    fig.colorbar(hb, ax=ax, label="Log scan count")
    ax.legend(frameon=False)
    savefig("fig02_predicted_vs_actual.png")


def residual_plots(data):
    df = data[FINAL]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(df.error, bins=35, edgecolor="black", linewidth=.35)
    axes[0].axvline(0, linestyle="--", linewidth=1.2)
    axes[0].set(xlabel="Prediction error (predicted − actual, years)",
                 ylabel="Number of scans",
                 title="Residual distribution")
    axes[1].scatter(df.age, df.error, s=13, alpha=.35)
    axes[1].axhline(0, linestyle="--", linewidth=1.2)
    axes[1].set(xlabel="Chronological age (years)",
                ylabel="Prediction error (years)",
                title="Residuals versus chronological age")
    savefig("fig03_residual_distribution_and_age.png")

    # Residual distribution by age band.
    fig, ax = plt.subplots(figsize=(8.5, 5))
    vals = [df.loc[df.age_group == band, "error"].dropna().values
            for band in AGE_ORDER]
    ax.boxplot(vals, tick_labels=AGE_ORDER, showfliers=False)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set(xlabel="Chronological age group", ylabel="Prediction error (years)",
           title="V5.1 RAW residuals by age group")
    savefig("fig04_residuals_by_age_group.png")


def age_stratified(data):
    rows = []
    for label, df in data.items():
        # Subject-mean prediction metric within each age group.
        subj = (df.groupby(["age_group", "subject_id"], observed=True)
                  .agg(age=("age", "mean"),
                       predicted_age=("predicted_age", "mean"))
                  .reset_index())
        subj["abs_error"] = (subj.predicted_age - subj.age).abs()
        for band, g in subj.groupby("age_group", observed=True):
            rows.append({
                "model": label, "age_group": str(band),
                "n_subjects": len(g), "MAE": g.abs_error.mean()
            })
    out = pd.DataFrame(rows)
    out.to_csv(TABLE / "age_stratified_subject_mean_mae.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label in FILES:
        g = out[out.model == label].set_index("age_group").reindex(AGE_ORDER)
        ax.plot(AGE_ORDER, g.MAE, marker="o", linewidth=1.5, label=label)
    ax.set(xlabel="Chronological age group",
           ylabel="Subject-mean-prediction MAE (years)",
           title="Age-stratified MAE by model")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    savefig("fig05_age_stratified_mae.png")


def calibration_bias(data):
    df = data[FINAL].copy()
    # Ten-year age bins provide an interpretable descriptive calibration view.
    edges = np.arange(20, 101, 10)
    df["calibration_bin"] = pd.cut(df.age, edges, right=False)
    cal = (df.groupby("calibration_bin", observed=True)
             .agg(n_scans=("error", "size"),
                  mean_actual_age=("age", "mean"),
                  mean_predicted_age=("predicted_age", "mean"),
                  mean_bias=("error", "mean"),
                  sd_error=("error", "std"))
             .reset_index())
    cal["calibration_bin"] = cal.calibration_bin.astype(str)
    cal.to_csv(TABLE / "v51_raw_calibration_by_age.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(cal.mean_actual_age, cal.mean_predicted_age,
                 marker="o", label="Observed mean prediction")
    lo = min(cal.mean_actual_age.min(), cal.mean_predicted_age.min())
    hi = max(cal.mean_actual_age.max(), cal.mean_predicted_age.max())
    axes[0].plot([lo, hi], [lo, hi], linestyle="--", label="Identity")
    axes[0].set(xlabel="Mean chronological age (years)",
                ylabel="Mean predicted age (years)",
                title="Calibration by 10-year age bins")
    axes[0].legend(frameon=False)

    axes[1].errorbar(cal.mean_actual_age, cal.mean_bias,
                     yerr=cal.sd_error, marker="o", capsize=3)
    axes[1].axhline(0, linestyle="--")
    axes[1].set(xlabel="Mean chronological age (years)",
                ylabel="Mean prediction bias (years)",
                title="Age-dependent bias (error SD bars)")
    savefig("fig06_calibration_and_bias.png")


def bootstrap_plot():
    path = RESULTS / "paired_subject_bootstrap.csv"
    require_file(path)
    b = pd.read_csv(path)
    needed = {"comparison", "metric", "age_group",
              "mean_MAE_difference", "CI95_lower", "CI95_upper"}
    if needed - set(b.columns):
        raise ValueError(f"Bootstrap table missing: {needed - set(b.columns)}")

    # Subject-mean-prediction comparisons only; negative favours V5.1 RAW.
    sub = b[(b.metric == "subject_mean_prediction") &
            (b.age_group.astype(str) == "ALL")].copy()
    # Keep comparisons with V5.1 RAW as model_a.
    sub = sub[sub.comparison.str.startswith("V5_1_RAW -") |
              sub.comparison.str.startswith("V5.1 RAW -")]
    if sub.empty:
        # The diagnostics script may label it with underscore names.
        sub = b[(b.metric == "subject_mean_prediction") &
                (b.age_group.astype(str) == "ALL") &
                (b.model_a.astype(str).str.contains("V5_1_RAW"))].copy()
    if sub.empty:
        warnings.warn("No V5.1 RAW-first overall bootstrap comparisons found; skipping figure.")
        return

    sub = sub.sort_values("mean_MAE_difference")
    y = np.arange(len(sub))
    est = sub.mean_MAE_difference.to_numpy()
    lo = sub.CI95_lower.to_numpy()
    hi = sub.CI95_upper.to_numpy()
    fig, ax = plt.subplots(figsize=(8.5, max(4, .55 * len(sub))))
    ax.errorbar(est, y, xerr=[est-lo, hi-est], fmt="o", capsize=4)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(sub.comparison)
    ax.set_xlabel("Paired subject-level MAE difference (years)")
    ax.set_title("V5.1 RAW paired-bootstrap comparisons\nNegative values favour V5.1 RAW")
    savefig("fig07_paired_bootstrap_ci.png")


def uncertainty_analysis(data):
    rows = []
    for label, df in data.items():
        if "predicted_age_sd" not in df.columns:
            continue
        v = df[["predicted_age_sd", "abs_error"]].dropna()
        rho = v.corr(method="spearman").iloc[0, 1] if len(v) > 1 else np.nan
        rows.append({
            "model": label, "n_scans": len(v),
            "mean_predicted_age_sd": v.predicted_age_sd.mean(),
            "mean_absolute_error": v.abs_error.mean(),
            "spearman_rho_uncertainty_vs_absolute_error": rho,
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABLE / "uncertainty_analysis.csv", index=False)

    if not out.empty:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.bar(out.model, out.spearman_rho_uncertainty_vs_absolute_error)
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_ylabel("Spearman correlation")
        ax.set_title("Prediction dispersion versus absolute error")
        ax.tick_params(axis="x", rotation=35)
        savefig("fig08_uncertainty_error_association.png")

    # Write a cautious interpretation for the report.
    final_row = out[out.model == FINAL]
    if not final_row.empty:
        rho = final_row.iloc[0].spearman_rho_uncertainty_vs_absolute_error
        text = (
            f"V5.1 RAW uncertainty/error Spearman correlation: {rho:.4f}. "
            "This is an association between the exported prediction-dispersion "
            "measure and absolute prediction error. A weak correlation does not "
            "support a claim of reliable uncertainty estimation or calibrated "
            "prediction intervals. The dispersion measure should not be described "
            "as validated epistemic or aleatoric uncertainty without additional "
            "validation."
        )
    else:
        text = (
            "The prediction files do not contain predicted_age_sd, so the "
            "uncertainty/error association could not be calculated."
        )
    (REPORT / "uncertainty_findings.txt").write_text(text + "\n", encoding="utf-8")


def model_metric_dashboard(data):
    """Compare all models using subject-mean-prediction metrics."""
    rows = []
    for label, df in data.items():
        subj = (
            df.groupby("subject_id")
              .agg(
                  age=("age", "mean"),
                  predicted_age=("predicted_age", "mean")
              )
              .reset_index()
        )
        err = subj.predicted_age - subj.age

        rows.append({
            "model": label,
            "MAE": err.abs().mean(),
            "RMSE": np.sqrt((err ** 2).mean()),
            "R2": 1 - (err ** 2).sum() /
                 ((subj.age - subj.age.mean()) ** 2).sum(),
            "bias": err.mean()
        })

    metrics = pd.DataFrame(rows)
    metrics.to_csv(TABLE / "model_metric_dashboard.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    x = np.arange(len(metrics))
    labels = metrics.model.tolist()

    plots = [
        ("MAE", "MAE (years)", "Subject-level MAE"),
        ("RMSE", "RMSE (years)", "Subject-level RMSE"),
        ("R2", "R²", "Subject-level R²"),
        ("bias", "Bias (years)", "Subject-level prediction bias")
    ]

    for ax, (column, ylabel, title) in zip(axes.flat, plots):
        ax.bar(x, metrics[column])
        if column == "bias":
            ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)

    savefig("fig09_model_metric_dashboard.png")


def fold_heatmap(data):
    """Display MAE consistency across the five outer folds."""
    rows = []

    for label, df in data.items():
        for fold, group in df.groupby("fold"):
            rows.append({
                "model": label,
                "fold": int(fold),
                "MAE": group.abs_error.mean()
            })

    fold_df = pd.DataFrame(rows)
    pivot = fold_df.pivot(
        index="model", columns="fold", values="MAE"
    )
    pivot.to_csv(TABLE / "fold_mae_heatmap.csv")

    fig, ax = plt.subplots(figsize=(8.5, 6))
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"Fold {f}" for f in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(
                j, i, f"{pivot.iloc[i, j]:.2f}",
                ha="center", va="center", fontsize=8
            )

    ax.set_title("Fold-wise MAE heatmap across model variants")
    ax.set_xlabel("Outer cross-validation fold")
    ax.set_ylabel("Model")
    fig.colorbar(im, ax=ax, label="MAE (years)")

    savefig("fig10_fold_mae_heatmap.png")


def subject_bland_altman(data):
    """Bland–Altman agreement plot using one averaged value per subject."""
    df = data[FINAL]

    subj = (
        df.groupby("subject_id")
          .agg(
              actual_age=("age", "mean"),
              predicted_age=("predicted_age", "mean")
          )
          .reset_index()
    )

    subj["mean_age"] = (
        subj.actual_age + subj.predicted_age
    ) / 2

    subj["difference"] = (
        subj.predicted_age - subj.actual_age
    )

    mean_diff = subj.difference.mean()
    sd_diff = subj.difference.std(ddof=1)

    lower = mean_diff - 1.96 * sd_diff
    upper = mean_diff + 1.96 * sd_diff

    subj.to_csv(
        TABLE / "subject_level_bland_altman.csv",
        index=False
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        subj.mean_age,
        subj.difference,
        alpha=0.45,
        s=18
    )

    ax.axhline(mean_diff, linestyle="-",
               label=f"Mean difference = {mean_diff:.2f}")
    ax.axhline(lower, linestyle="--",
               label=f"Lower 95% limits = {lower:.2f}")
    ax.axhline(upper, linestyle="--",
               label=f"Upper 95% limits = {upper:.2f}")

    ax.set(
        xlabel="Mean of chronological and predicted age (years)",
        ylabel="Predicted − chronological age (years)",
        title="Subject-level Bland–Altman agreement: V5.1 RAW"
    )
    ax.legend(frameon=False, fontsize=8)

    savefig("fig11_subject_bland_altman.png")


def absolute_error_ecdf(data):
    """Cumulative distribution of absolute prediction errors."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for label, df in data.items():
        errors = np.sort(df.abs_error.dropna().values)
        if len(errors) == 0:
            continue

        cumulative = np.arange(1, len(errors) + 1) / len(errors)
        ax.plot(errors, cumulative, linewidth=1.5, label=label)

    ax.set(
        xlabel="Absolute prediction error (years)",
        ylabel="Cumulative proportion of scans",
        title="Cumulative distribution of absolute prediction error"
    )
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.2)
    ax.legend(ncol=3, fontsize=8, frameon=False)

    savefig("fig12_absolute_error_ecdf.png")


def age_group_cohort_distribution(data):
    """Show scan and unique-subject counts in each age group."""
    df = data[FINAL]

    scan_counts = (
        df.groupby("age_group", observed=True)
          .size()
          .reindex(AGE_ORDER, fill_value=0)
    )

    subject_counts = (
        df.groupby("age_group", observed=True)
          .subject_id.nunique()
          .reindex(AGE_ORDER, fill_value=0)
    )

    out = pd.DataFrame({
        "age_group": AGE_ORDER,
        "n_scans": scan_counts.values,
        "n_subjects": subject_counts.values
    })

    out.to_csv(TABLE / "age_group_cohort_distribution.csv", index=False)

    x = np.arange(len(AGE_ORDER))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width / 2, out.n_scans, width, label="Scans")
    ax.bar(x + width / 2, out.n_subjects, width, label="Unique subjects")

    ax.set_xticks(x)
    ax.set_xticklabels(AGE_ORDER)
    ax.set_xlabel("Chronological age group")
    ax.set_ylabel("Count")
    ax.set_title("Evaluation cohort distribution by age group")
    ax.legend(frameon=False)

    savefig("fig13_age_group_cohort_distribution.png")


def age_stratified_bias(data):
    """
    Subject-level bias by age group.
    Each subject contributes one averaged prediction per age group.
    Error bars are approximate 95% CIs using the standard error.
    """
    df = data[FINAL]

    subj = (
        df.groupby(["age_group", "subject_id"], observed=True)
          .agg(
              age=("age", "mean"),
              predicted_age=("predicted_age", "mean")
          )
          .reset_index()
    )

    subj["bias"] = subj.predicted_age - subj.age

    rows = []
    for band in AGE_ORDER:
        group = subj[subj.age_group.astype(str) == band]
        n = len(group)

        if n == 0:
            continue

        mean_bias = group.bias.mean()
        sd = group.bias.std(ddof=1) if n > 1 else np.nan
        se = sd / np.sqrt(n) if n > 1 else np.nan

        rows.append({
            "age_group": band,
            "n_subjects": n,
            "mean_bias": mean_bias,
            "sd_bias": sd,
            "ci95_lower": mean_bias - 1.96 * se
                if np.isfinite(se) else np.nan,
            "ci95_upper": mean_bias + 1.96 * se
                if np.isfinite(se) else np.nan
        })

    out = pd.DataFrame(rows)
    out.to_csv(TABLE / "age_stratified_bias.csv", index=False)

    x = np.arange(len(out))
    y = out.mean_bias.to_numpy()
    lower = y - out.ci95_lower.to_numpy()
    upper = out.ci95_upper.to_numpy() - y

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(
        x, y, yerr=[lower, upper],
        fmt="o", capsize=4, linewidth=1.5
    )
    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(out.age_group)
    ax.set(
        xlabel="Chronological age group",
        ylabel="Mean prediction bias (years)",
        title="V5.1 RAW age-stratified prediction bias"
    )
    ax.grid(axis="y", alpha=0.2)

    savefig("fig14_age_stratified_bias.png")


def subject_prediction_plot(data):
    """Predicted versus actual age after averaging repeated scans per subject."""
    df = data[FINAL]

    subj = (
        df.groupby("subject_id")
          .agg(
              actual_age=("age", "mean"),
              predicted_age=("predicted_age", "mean")
          )
          .reset_index()
    )

    subj.to_csv(TABLE / "subject_level_predictions.csv", index=False)

    lo = min(subj.actual_age.min(), subj.predicted_age.min())
    hi = max(subj.actual_age.max(), subj.predicted_age.max())

    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(
        subj.actual_age,
        subj.predicted_age,
        gridsize=30,
        mincnt=1,
        bins="log"
    )

    ax.plot(
        [lo, hi], [lo, hi],
        linestyle="--", linewidth=1.5,
        label="Identity line"
    )

    ax.set(
        xlabel="Mean chronological age (years)",
        ylabel="Mean predicted age (years)",
        title="Subject-level predicted versus actual age: V5.1 RAW"
    )

    fig.colorbar(hb, ax=ax, label="Log subject count")
    ax.legend(frameon=False)

    savefig("fig15_subject_predicted_vs_actual.png")

def main():
    print("Loading prediction files...")
    data = load_predictions()
    ref = data[FINAL]
    print(f"Common cohort: {len(ref)} scans; {ref.subject_id.nunique()} subjects")
    if len(ref) != 1896 or ref.subject_id.nunique() != 767:
        warnings.warn(
            "The selected V5.1 RAW cohort is not 1,896 scans / 767 subjects. "
            "Check filtering and source prediction files before interpreting figures."
        )

    print("Creating comparison table...")
    comparison = model_comparison(data)
    print(comparison[comparison.weighting == "scan_weighted"].to_string(index=False))

    print("Creating figures...")
    foldwise_figure(data)
    prediction_plot(data)
    residual_plots(data)
    age_stratified(data)
    calibration_bias(data)
    bootstrap_plot()
    uncertainty_analysis(data)
    model_metric_dashboard(data)
    fold_heatmap(data)
    subject_bland_altman(data)
    absolute_error_ecdf(data)
    age_group_cohort_distribution(data)
    age_stratified_bias(data)
    subject_prediction_plot(data)
    readme = """# Phase B outputs

Generated from the existing out-of-fold prediction CSVs and diagnostic bootstrap table.

- `tables/final_model_comparison.csv`: scan-weighted and subject-mean-prediction metrics.
- `tables/fold_wise_performance.csv`: fold-level MAE, RMSE and bias.
- `tables/age_stratified_subject_mean_mae.csv`: subject-mean-prediction MAE by age group.
- `tables/v51_raw_calibration_by_age.csv`: age-bin calibration and bias summary.
- `tables/uncertainty_analysis.csv`: association between predicted-age dispersion and absolute error.
- `figures/`: PNG (300 DPI) and PDF versions.
- `reports/uncertainty_findings.txt`: cautious interpretation of uncertainty analysis.

All plots are descriptive analyses of exported OOF predictions. The five outer folds are not
external validation. Bootstrap intervals are conditional on the exported predictions and do
not fully capture training-seed variability.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(f"\nPhase B figures and tables written to: {OUT}")


if __name__ == "__main__":
    main()
