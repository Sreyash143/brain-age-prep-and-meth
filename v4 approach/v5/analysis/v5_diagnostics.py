#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ============================================================
# CONFIGURATION
# ============================================================

BASE = "v4/predictions/bag_v4_sfcn_zscore_ablation_qc_passed.csv"

FILES = {
    "V4": "v4/predictions/raw_v4_sfcn_zscore_ablation.csv",

    "V5_RAW": "v5/predictions/raw_v5_z_khuber.csv",
    "V5_EMA": "v5/predictions/raw_v5_z_khuber_ema.csv",

    "V5_1_RAW": "v5/predictions/raw_v5_1_no_tail.csv",
    "V5_1_EMA": "v5/predictions/raw_v5_1_no_tail_ema.csv",

    "V5_2_RAW": "v5/predictions/raw_v5_2_subject_balanced.csv",
    "V5_2_EMA": "v5/predictions/raw_v5_2_subject_balanced_ema.csv",

    "V5_3_RAW": "v5/predictions/raw_v5_3_mild_age_balanced.csv",
    "V5_3_EMA": "v5/predictions/raw_v5_3_mild_age_balanced_ema.csv",
}

OUT = "v5/analysis/results"

os.makedirs(OUT, exist_ok=True)

SEED = 42
N_BOOT = 10000

AGE_BINS = [-np.inf, 50, 60, 70, 80, 90, np.inf]

AGE_LABELS = [
    "<50",
    "50-59",
    "60-69",
    "70-79",
    "80-89",
    "90+",
]


# ============================================================
# 1. LOAD AND VALIDATE COMMON QC-PASSED CN COHORT
# ============================================================

print("\n========== LOADING REFERENCE COHORT ==========")

qc = pd.read_csv(BASE)

required_qc_columns = {
    "file_path",
    "subject_id",
    "CDRTOT",
    "age",
}

missing = required_qc_columns - set(qc.columns)

if missing:
    raise ValueError(
        f"Reference cohort missing columns: {missing}"
    )

qc_cn = qc[
    (qc["CDRTOT"] == 0)
    & qc["file_path"].notna()
].copy()

qc_cn_paths = set(qc_cn["file_path"])

if len(qc_cn_paths) != len(qc_cn):
    raise ValueError(
        "Duplicate file_path entries in reference CN cohort"
    )

print("QC-passed scans:", len(qc))
print("QC-passed CN scans:", len(qc_cn))
print("QC-passed CN subjects:", qc_cn["subject_id"].nunique())

models = {}

for name, path in FILES.items():

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name}: prediction file not found: {path}"
        )

    df = pd.read_csv(path)

    required_columns = {
        "subject_id",
        "file_path",
        "age",
        "CDRTOT",
        "predicted_age",
        "fold",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{name}: missing columns: {missing}"
        )

    # Restrict to the exact QC-passed CN reference cohort.
    df = df[
        df["file_path"].isin(qc_cn_paths)
    ].copy()

    # Remove incomplete predictions.
    df = df.dropna(
        subset=[
            "subject_id",
            "file_path",
            "age",
            "predicted_age",
            "fold",
        ]
    )

    # Validate cognitive status.
    df = df[df["CDRTOT"] == 0].copy()

    if df["file_path"].duplicated().any():
        raise ValueError(
            f"{name}: duplicate file_path predictions"
        )

    if set(df["file_path"]) != qc_cn_paths:
        missing_paths = qc_cn_paths - set(df["file_path"])
        extra_paths = set(df["file_path"]) - qc_cn_paths

        raise ValueError(
            f"{name}: cohort mismatch. "
            f"Missing={len(missing_paths)}, "
            f"Extra={len(extra_paths)}"
        )

    # Calculate prediction errors.
    df["error"] = (
        df["predicted_age"] - df["age"]
    )

    df["abs_error"] = df["error"].abs()

    df["sq_error"] = df["error"] ** 2

    # Assign age group using actual scan age.
    df["age_group"] = pd.cut(
        df["age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=False,
    )

    models[name] = df

    print(
        f"{name:12s}: "
        f"{len(df):4d} scans | "
        f"{df['subject_id'].nunique():4d} subjects"
    )


# ============================================================
# 2. VERIFY IDENTICAL SCAN SETS AND FOLD ASSIGNMENTS
# ============================================================

print("\n========== VALIDATING COMMON SCAN SET ==========")

reference_name = "V4"

reference = (
    models[reference_name]
    .set_index("file_path")
    .sort_index()
)

for name, df in models.items():

    current = (
        df.set_index("file_path")
        .sort_index()
    )

    if not current.index.equals(reference.index):
        raise ValueError(
            f"{name}: scan set differs from {reference_name}"
        )

    if not current["fold"].equals(reference["fold"]):
        raise ValueError(
            f"{name}: fold assignments differ from {reference_name}"
        )

    if not current["subject_id"].equals(reference["subject_id"]):
        raise ValueError(
            f"{name}: subject assignments differ from {reference_name}"
        )

print("All models use identical scans.")
print("All models use identical subject assignments.")
print("All models use identical fold assignments.")


# ============================================================
# 3. METRIC HELPER
# ============================================================

def metrics(df):

    return {
        "n_scans": len(df),

        "n_subjects": df["subject_id"].nunique(),

        "MAE": mean_absolute_error(
            df["age"],
            df["predicted_age"],
        ),

        "RMSE": np.sqrt(
            mean_squared_error(
                df["age"],
                df["predicted_age"],
            )
        ),

        "R2": r2_score(
            df["age"],
            df["predicted_age"],
        ),

        "bias": df["error"].mean(),
    }


# ============================================================
# 4. OVERALL SCAN-LEVEL AND SUBJECT-BALANCED METRICS
# ============================================================

print("\n========== CALCULATING OVERALL METRICS ==========")

overall_results = []

subject_overall = {}

for name, df in models.items():

    # --------------------------------------------------------
    # Scan-weighted metrics
    # --------------------------------------------------------

    scan = metrics(df)

    scan["model"] = name
    scan["weighting"] = "scan_weighted"

    overall_results.append(scan)

    # --------------------------------------------------------
    # Subject-level metrics
    #
    # Average predictions and actual ages across each
    # subject's scans. Each subject then contributes equally.
    # --------------------------------------------------------

    subj = (
        df.groupby("subject_id")
        .agg(
            age=("age", "mean"),
            predicted_age=("predicted_age", "mean"),
        )
        .reset_index()
    )

    subj["error"] = (
        subj["predicted_age"] - subj["age"]
    )

    subj["abs_error"] = subj["error"].abs()

    subj["sq_error"] = subj["error"] ** 2

    subject_overall[name] = (
        subj[["subject_id", "abs_error", "error"]]
        .rename(
            columns={
                "abs_error": "MAE",
                "error": "bias",
            }
        )
    )

    subject_metrics = {
        "n_scans": len(df),

        "n_subjects": len(subj),

        "MAE": subj["abs_error"].mean(),

        "RMSE": np.sqrt(
            subj["sq_error"].mean()
        ),

        "R2": r2_score(
            subj["age"],
            subj["predicted_age"],
        ),

        "bias": subj["error"].mean(),

        "model": name,

        "weighting": "subject_mean_prediction",
    }

    overall_results.append(subject_metrics)


overall_df = pd.DataFrame(overall_results)

overall_df.to_csv(
    f"{OUT}/overall_comparison.csv",
    index=False,
)


# ============================================================
# 5. FOLD-WISE PERFORMANCE
# ============================================================

print("\n========== CALCULATING FOLD METRICS ==========")

fold_results = []

for name, df in models.items():

    for fold, g in df.groupby("fold"):

        row = metrics(g)

        row.update({
            "model": name,
            "fold": int(fold),
        })

        fold_results.append(row)


fold_df = pd.DataFrame(fold_results)

fold_df.to_csv(
    f"{OUT}/fold_comparison.csv",
    index=False,
)


# ============================================================
# 6. AGE-STRATIFIED SCAN-WEIGHTED AND SUBJECT-BALANCED
# ============================================================

print("\n========== CALCULATING AGE GROUP METRICS ==========")

age_results = []

subject_age = {}

for name, df in models.items():

    # --------------------------------------------------------
    # Scan-level age metrics
    # --------------------------------------------------------

    for group, g in df.groupby(
        "age_group",
        observed=True,
    ):

        if len(g) == 0:
            continue

        row = metrics(g)

        row.update({
            "model": name,
            "age_group": str(group),
            "weighting": "scan_weighted",
        })

        age_results.append(row)

    # --------------------------------------------------------
    # Subject-balanced age metrics
    #
    # First average scans within each subject and age group.
    # Then calculate metrics across subjects.
    # --------------------------------------------------------

    for group, g in df.groupby(
        "age_group",
        observed=True,
    ):

        if len(g) == 0:
            continue

        sg = (
            g.groupby("subject_id")
            .agg(
                age=("age", "mean"),
                predicted_age=("predicted_age", "mean"),
            )
            .reset_index()
        )

        sg["error"] = (
            sg["predicted_age"] - sg["age"]
        )

        sg["abs_error"] = sg["error"].abs()

        sg["sq_error"] = sg["error"] ** 2

        row = {
            "n_scans": len(g),

            "n_subjects": len(sg),

            "MAE": sg["abs_error"].mean(),

            "RMSE": np.sqrt(
                sg["sq_error"].mean()
            ),

            "R2": r2_score(
                sg["age"],
                sg["predicted_age"],
            ) if len(sg) >= 2 else np.nan,

            "bias": sg["error"].mean(),

            "model": name,

            "age_group": str(group),

            "weighting": "subject_balanced",
        }

        age_results.append(row)

    # Store one mean absolute error per subject per age group.
    subject_age[name] = (
        df.groupby(
            ["age_group", "subject_id"],
            observed=True,
        )
        .agg(
            MAE=("abs_error", "mean"),
            bias=("error", "mean"),
        )
        .reset_index()
    )


age_df = pd.DataFrame(age_results)

age_df.to_csv(
    f"{OUT}/age_stratified_errors.csv",
    index=False,
)


# ============================================================
# 7. SUBJECT-LEVEL ERROR AND UNCERTAINTY
# ============================================================

print("\n========== CALCULATING SUBJECT-LEVEL ERRORS ==========")

subject_results = []

for name, df in models.items():

    aggregation = {
        "n_scans": ("abs_error", "size"),
        "mean_abs_error": ("abs_error", "mean"),
        "mean_error": ("error", "mean"),
        "mean_age": ("age", "mean"),
    }

    if "predicted_age_sd" in df.columns:
        aggregation["mean_uncertainty"] = (
            "predicted_age_sd",
            "mean",
        )

    subject_df = (
        df.groupby("subject_id")
        .agg(**aggregation)
        .reset_index()
    )

    subject_df["model"] = name

    subject_results.append(subject_df)


subject_df = pd.concat(
    subject_results,
    ignore_index=True,
)

subject_df.to_csv(
    f"{OUT}/subject_level_errors.csv",
    index=False,
)


# ============================================================
# 8. UNCERTAINTY VERSUS ACTUAL ERROR
# ============================================================

print("\n========== UNCERTAINTY ANALYSIS ==========")

uncertainty_results = []

for name, df in models.items():

    if "predicted_age_sd" not in df.columns:
        continue

    valid = df.dropna(
        subset=[
            "predicted_age_sd",
            "abs_error",
        ]
    )

    correlation = valid[
        ["predicted_age_sd", "abs_error"]
    ].corr(
        method="spearman"
    ).iloc[0, 1]

    uncertainty_results.append({
        "model": name,

        "n_scans": len(valid),

        "mean_uncertainty": (
            valid["predicted_age_sd"].mean()
        ),

        "mean_absolute_error": (
            valid["abs_error"].mean()
        ),

        "spearman_uncertainty_error": correlation,
    })


uncertainty_df = pd.DataFrame(
    uncertainty_results
)

uncertainty_df.to_csv(
    f"{OUT}/uncertainty_analysis.csv",
    index=False,
)



# ============================================================
# 9. PAIRED CLUSTER BOOTSTRAP
# ============================================================

print("\n========== PAIRED CLUSTER BOOTSTRAP ==========")

# Positive difference: model A has higher error.
# Negative difference: model A has lower error.
#
# Two distinct overall estimands:
#
# 1. scan_weighted:
#    Pooled absolute error over all scans.
#    Subjects are resampled as clusters.
#
# 2. subject_mean_prediction:
#    Average predictions and actual ages within each subject,
#    then calculate MAE across subjects.
#
# Age-specific analysis:
#    Average predictions and actual ages within each
#    subject-age-group, then calculate MAE across subjects.


comparisons = [
    ("V5_1_RAW", "V4"),
    ("V5_1_RAW", "V5_RAW"),
    ("V5_1_RAW", "V5_2_RAW"),
    ("V5_1_RAW", "V5_3_RAW"),

    ("V5_1_RAW", "V5_1_EMA"),
    ("V5_2_RAW", "V5_2_EMA"),
    ("V5_3_RAW", "V5_3_EMA"),
    ("V5_RAW", "V5_EMA"),
]

bootstrap_results = []

rng = np.random.default_rng(SEED)


def percentile_ci(values):
    return np.percentile(values, [2.5, 97.5])


def paired_cluster_bootstrap(
    a,
    b,
    model_a,
    model_b,
    metric,
    group_label="ALL",
):
    """
    Paired bootstrap over subjects.

    Both models use identical subject samples in each
    bootstrap replicate.

    For scan-weighted MAE, each sampled subject contributes
    all of their scans.

    For subject-mean-prediction MAE, each subject contributes
    one absolute error.
    """

    paired = a.merge(
        b,
        on="subject_id",
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )

    if len(paired) < 2:
        return

    n = len(paired)

    if metric == "subject_mean_prediction":

        differences = (
            paired["MAE_a"].to_numpy()
            - paired["MAE_b"].to_numpy()
        )

        observed = differences.mean()

        boot = np.empty(N_BOOT)

        for i in range(N_BOOT):
            sample = rng.integers(0, n, size=n)
            boot[i] = differences[sample].mean()

    elif metric == "scan_weighted":

        # Each row represents one subject.
        # Error sums and scan counts are retained so that
        # subjects with more scans contribute proportionally
        # to the pooled scan MAE.

        sum_a = paired["abs_sum_a"].to_numpy()
        sum_b = paired["abs_sum_b"].to_numpy()
        count_a = paired["n_scans_a"].to_numpy()
        count_b = paired["n_scans_b"].to_numpy()

        observed = (
            sum_a.sum() / count_a.sum()
            - sum_b.sum() / count_b.sum()
        )

        boot = np.empty(N_BOOT)

        for i in range(N_BOOT):

            sample = rng.integers(0, n, size=n)

            mae_a = (
                sum_a[sample].sum()
                / count_a[sample].sum()
            )

            mae_b = (
                sum_b[sample].sum()
                / count_b[sample].sum()
            )

            boot[i] = mae_a - mae_b

    else:
        raise ValueError(f"Unknown bootstrap metric: {metric}")

    lo, hi = percentile_ci(boot)

    bootstrap_results.append({
        "model_a": model_a,
        "model_b": model_b,
        "comparison": f"{model_a} - {model_b}",
        "metric": metric,
        "age_group": group_label,
        "n_subjects": n,
        "mean_MAE_difference": observed,
        "CI95_lower": lo,
        "CI95_upper": hi,
    })


# ------------------------------------------------------------
# Prepare subject-level scan error sums for scan bootstrap.
# ------------------------------------------------------------

scan_subject = {}

for name, df in models.items():

    scan_subject[name] = (
        df.groupby("subject_id")
        .agg(
            abs_sum=("abs_error", "sum"),
            n_scans=("abs_error", "size"),
        )
        .reset_index()
        .rename(
            columns={
                "abs_sum": "abs_sum",
                "n_scans": "n_scans",
            }
        )
    )


# ------------------------------------------------------------
# Prepare subject-mean-prediction data by age group.
# This matches the subject_balanced age metric.
# ------------------------------------------------------------

subject_age_prediction = {}

for name, df in models.items():

    result = (
        df.groupby(
            ["age_group", "subject_id"],
            observed=True,
        )
        .agg(
            age=("age", "mean"),
            predicted_age=("predicted_age", "mean"),
        )
        .reset_index()
    )

    result["MAE"] = (
        result["predicted_age"] - result["age"]
    ).abs()

    subject_age_prediction[name] = result[
        ["age_group", "subject_id", "MAE"]
    ]


# ------------------------------------------------------------
# Run all overall and age-specific comparisons.
# ------------------------------------------------------------

for model_a, model_b in comparisons:

    # Overall scan-weighted MAE.
    a = scan_subject[model_a].rename(
        columns={
            "abs_sum": "abs_sum_a",
            "n_scans": "n_scans_a",
        }
    )

    b = scan_subject[model_b].rename(
        columns={
            "abs_sum": "abs_sum_b",
            "n_scans": "n_scans_b",
        }
    )

    paired_cluster_bootstrap(
        a,
        b,
        model_a,
        model_b,
        metric="scan_weighted",
    )

    # Overall subject-mean-prediction MAE.
    paired_cluster_bootstrap(
        subject_overall[model_a][
            ["subject_id", "MAE"]
        ],
        subject_overall[model_b][
            ["subject_id", "MAE"]
        ],
        model_a,
        model_b,
        metric="subject_mean_prediction",
    )

    # Age-specific subject-mean-prediction MAE.
    for group in AGE_LABELS:

        a = subject_age_prediction[model_a].copy()
        b = subject_age_prediction[model_b].copy()

        a = a[
            a["age_group"].astype(str) == group
        ][["subject_id", "MAE"]]

        b = b[
            b["age_group"].astype(str) == group
        ][["subject_id", "MAE"]]

        paired_cluster_bootstrap(
            a,
            b,
            model_a,
            model_b,
            metric="subject_mean_prediction",
            group_label=group,
        )


bootstrap_df = pd.DataFrame(bootstrap_results)

bootstrap_df.to_csv(
    f"{OUT}/paired_subject_bootstrap.csv",
    index=False,
)

print(
    "Bootstrap comparisons generated:",
    len(bootstrap_df),
)

print(
    "Bootstrap replicates per comparison:",
    N_BOOT,
)


# ============================================================
# 10. PRINT RESULTS
# ============================================================

print("\n\n")
print("=" * 90)
print("OVERALL COMPARISON - SCAN WEIGHTED")
print("=" * 90)

print(
    overall_df[
        overall_df["weighting"] == "scan_weighted"
    ][
        [
            "model",
            "n_scans",
            "n_subjects",
            "MAE",
            "RMSE",
            "R2",
            "bias",
        ]
    ].sort_values("MAE").to_string(index=False)
)


print("\n")
print("=" * 90)
print("OVERALL COMPARISON - SUBJECT MEAN PREDICTION")
print("=" * 90)

print(
    overall_df[
        overall_df["weighting"] == "subject_mean_prediction"
    ][
        [
            "model",
            "n_subjects",
            "MAE",
            "RMSE",
            "R2",
            "bias",
        ]
    ].sort_values("MAE").to_string(index=False)
)


print("\n")
print("=" * 90)
print("FOLD-WISE MAE")
print("=" * 90)

print(
    fold_df.pivot(
        index="fold",
        columns="model",
        values="MAE",
    ).to_string()
)


print("\n")
print("=" * 90)
print("AGE MAE - SCAN WEIGHTED")
print("=" * 90)

print(
    age_df[
        age_df["weighting"] == "scan_weighted"
    ].pivot(
        index="age_group",
        columns="model",
        values="MAE",
    ).to_string()
)


print("\n")
print("=" * 90)
print("AGE MAE - SUBJECT BALANCED")
print("=" * 90)

print(
    age_df[
        age_df["weighting"] == "subject_balanced"
    ].pivot(
        index="age_group",
        columns="model",
        values="MAE",
    ).to_string()
)


print("\n")
print("=" * 90)
print("OVERALL PAIRED SUBJECT-LEVEL BOOTSTRAP")
print("=" * 90)

print(
    bootstrap_df[
        bootstrap_df["age_group"] == "ALL"
    ][
        [
            "comparison",
            "n_subjects",
            "mean_MAE_difference",
            "CI95_lower",
            "CI95_upper",
        ]
    ].to_string(index=False)
)


print("\n")
print("=" * 90)
print("AGE-SPECIFIC PAIRED SUBJECT-LEVEL BOOTSTRAP")
print("=" * 90)

print(
    bootstrap_df[
        bootstrap_df["age_group"] != "ALL"
    ].to_string(index=False)
)


print("\n")
print("=" * 90)
print("UNCERTAINTY ANALYSIS")
print("=" * 90)

print(
    uncertainty_df.to_string(index=False)
)

print("\n")
print("=" * 90)
print("OVERALL PAIRED CLUSTER BOOTSTRAP")
print("=" * 90)

print(
    bootstrap_df[
        bootstrap_df["age_group"] == "ALL"
    ][
        [
            "comparison",
            "metric",
            "n_subjects",
            "mean_MAE_difference",
            "CI95_lower",
            "CI95_upper",
        ]
    ].to_string(index=False)
)

print("\n")
print("=" * 90)
print("AGE-SPECIFIC PAIRED SUBJECT BOOTSTRAP")
print("=" * 90)

print(
    bootstrap_df[
        bootstrap_df["age_group"] != "ALL"
    ][
        [
            "comparison",
            "age_group",
            "n_subjects",
            "mean_MAE_difference",
            "CI95_lower",
            "CI95_upper",
        ]
    ].to_string(index=False)
)

# ============================================================
# 11. FINAL OUTPUT SUMMARY
# ============================================================

print("\n")
print("=" * 90)
print("DIAGNOSTICS COMPLETED")
print("=" * 90)

print("Evaluation cohort:")
print("Scans:", len(qc_cn_paths))
print("Subjects:", qc_cn["subject_id"].nunique())

print("\nModels evaluated:")
for name in models:
    print("-", name)

print("\nOutput files:")

for filename in [
    "overall_comparison.csv",
    "fold_comparison.csv",
    "age_stratified_errors.csv",
    "subject_level_errors.csv",
    "uncertainty_analysis.csv",
    "paired_subject_bootstrap.csv",
]:

    print(f"{OUT}/{filename}")

print("\nDiagnostics completed successfully.")