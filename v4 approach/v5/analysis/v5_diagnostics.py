import os
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = "v4/predictions/bag_v4_sfcn_zscore_ablation_qc_passed.csv"

FILES = {
    "V4": "v4/predictions/raw_v4_sfcn_zscore_ablation.csv",
    "V5_RAW": "v5/predictions/raw_v5_z_khuber.csv",
    "V5_EMA": "v5/predictions/raw_v5_z_khuber_ema.csv",
    "V5_1_RAW": "v5/predictions/raw_v5_1_no_tail.csv",
    "V5_1_EMA": "v5/predictions/raw_v5_1_no_tail_ema.csv",
}

OUT = "v5/analysis/results"
os.makedirs(OUT, exist_ok=True)

SEED = 42
N_BOOT = 10000

# --------------------------------------------------
# 1. Load and validate common QC-passed CN cohort
# --------------------------------------------------

qc = pd.read_csv(BASE)
qc_paths = set(qc["file_path"].dropna())

print("QC-passed scans in reference:", len(qc_paths))

models = {}

for name, path in FILES.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing prediction file: {path}")

    df = pd.read_csv(path)

    df = df[df["file_path"].isin(qc_paths)].copy()
    df = df[df["CDRTOT"] == 0].copy()

    df = df.dropna(
        subset=[
            "subject_id", "file_path", "age",
            "predicted_age", "fold"
        ]
    )

    if df["file_path"].duplicated().any():
        raise ValueError(f"{name}: duplicate file_path predictions")

    df["error"] = df["predicted_age"] - df["age"]
    df["abs_error"] = df["error"].abs()
    df["sq_error"] = df["error"] ** 2

    df["age_group"] = pd.cut(
        df["age"],
        bins=[-np.inf, 50, 60, 70, 80, 90, np.inf],
        labels=["<50", "50-59", "60-69",
                "70-79", "80-89", "90+"],
        right=False
    )

    models[name] = df

    print(
        f"{name}: {len(df)} scans, "
        f"{df.subject_id.nunique()} subjects"
    )

# Verify identical scan sets and fold assignments
reference_name = list(models)[0]
reference = models[reference_name].set_index("file_path")

for name, df in models.items():
    current = df.set_index("file_path")

    if set(current.index) != set(reference.index):
        raise ValueError(
            f"{name}: scan set differs from {reference_name}"
        )

    if not current["fold"].equals(reference["fold"]):
        raise ValueError(f"{name}: fold assignments differ")

print("\nAll models use the same scan set and fold assignments.")

# --------------------------------------------------
# 2. Metric helper
# --------------------------------------------------

def metrics(df):
    return {
        "n_scans": len(df),
        "n_subjects": df.subject_id.nunique(),
        "MAE": mean_absolute_error(df.age, df.predicted_age),
        "RMSE": np.sqrt(mean_squared_error(df.age, df.predicted_age)),
        "R2": r2_score(df.age, df.predicted_age),
        "bias": df.error.mean(),
    }


# --------------------------------------------------
# 3. Overall scan-level and subject-balanced metrics
# --------------------------------------------------

overall_results = []

for name, df in models.items():
    scan = metrics(df)
    scan["model"] = name
    scan["weighting"] = "scan_weighted"
    overall_results.append(scan)

    # Each subject contributes equally, regardless of scan count.
    subj = (
        df.groupby("subject_id")
        .agg(
            age=("age", "mean"),
            predicted_age=("predicted_age", "mean"),
        )
        .reset_index()
    )

    subj["error"] = subj.predicted_age - subj.age
    subj["abs_error"] = subj.error.abs()
    subj["sq_error"] = subj.error ** 2

    sm = {
        "n_scans": len(df),
        "n_subjects": len(subj),
        "MAE": subj.abs_error.mean(),
        "RMSE": np.sqrt(subj.sq_error.mean()),
        "R2": r2_score(subj.age, subj.predicted_age),
        "bias": subj.error.mean(),
        "model": name,
        "weighting": "subject_mean_prediction",
    }
    overall_results.append(sm)

overall_df = pd.DataFrame(overall_results)
overall_df.to_csv(f"{OUT}/overall_comparison.csv", index=False)

# --------------------------------------------------
# 4. Fold-wise performance
# --------------------------------------------------

fold_results = []

for name, df in models.items():
    for fold, g in df.groupby("fold"):
        row = metrics(g)
        row.update({"model": name, "fold": int(fold)})
        fold_results.append(row)

fold_df = pd.DataFrame(fold_results)
fold_df.to_csv(f"{OUT}/fold_comparison.csv", index=False)

# --------------------------------------------------
# 5. Age-stratified scan-weighted and subject-balanced
# --------------------------------------------------

age_results = []

for name, df in models.items():
    for group, g in df.groupby("age_group", observed=True):
        if len(g) == 0:
            continue

        row = metrics(g)
        row.update({
            "model": name,
            "age_group": str(group),
            "weighting": "scan_weighted",
        })
        age_results.append(row)

        # Within each age group, first average scans per subject.
        # Subjects are then weighted equally.
        sg = (
            g.groupby("subject_id")
            .agg(
                age=("age", "mean"),
                predicted_age=("predicted_age", "mean"),
            )
            .reset_index()
        )

        sg["error"] = sg.predicted_age - sg.age
        sg["abs_error"] = sg.error.abs()
        sg["sq_error"] = sg.error ** 2

        row = {
            "n_scans": len(g),
            "n_subjects": len(sg),
            "MAE": sg.abs_error.mean(),
            "RMSE": np.sqrt(sg.sq_error.mean()),
            "bias": sg.error.mean(),
            "model": name,
            "age_group": str(group),
            "weighting": "subject_balanced",
        }
        age_results.append(row)

age_df = pd.DataFrame(age_results)
age_df.to_csv(f"{OUT}/age_stratified_errors.csv", index=False)

# --------------------------------------------------
# 6. Subject-level error and uncertainty
# --------------------------------------------------

subject_results = []

for name, df in models.items():
    subject_df = (
        df.groupby("subject_id")
        .agg(
            n_scans=("abs_error", "size"),
            mean_abs_error=("abs_error", "mean"),
            mean_error=("error", "mean"),
            mean_age=("age", "mean"),
            mean_uncertainty=("predicted_age_sd", "mean"),
        )
        .reset_index()
    )

    subject_df["model"] = name
    subject_results.append(subject_df)

subject_df = pd.concat(subject_results, ignore_index=True)
subject_df.to_csv(f"{OUT}/subject_level_errors.csv", index=False)

# --------------------------------------------------
# 7. Uncertainty versus actual error
# --------------------------------------------------

uncertainty_results = []

for name, df in models.items():
    valid = df.dropna(
        subset=["predicted_age_sd", "abs_error"]
    )

    correlation = valid[
        ["predicted_age_sd", "abs_error"]
    ].corr(method="spearman").iloc[0, 1]

    uncertainty_results.append({
        "model": name,
        "n_scans": len(valid),
        "mean_uncertainty": valid.predicted_age_sd.mean(),
        "mean_absolute_error": valid.abs_error.mean(),
        "spearman_uncertainty_error": correlation,
    })

uncertainty_df = pd.DataFrame(uncertainty_results)
uncertainty_df.to_csv(
    f"{OUT}/uncertainty_analysis.csv", index=False
)

# --------------------------------------------------
# 8. Paired subject-level bootstrap
# --------------------------------------------------

# Subject-level MAE within each actual scan-age group.
# Each subject contributes one mean absolute error per group.
subject_age = {}

for name, df in models.items():
    subject_age[name] = (
        df.groupby(["age_group", "subject_id"], observed=True)
        .agg(
            MAE=("abs_error", "mean"),
            bias=("error", "mean"),
        )
        .reset_index()
    )

bootstrap_results = []
rng = np.random.default_rng(SEED)

comparisons = [
    ("V5_RAW", "V4"),
    ("V5_EMA", "V4"),
    ("V5_1_RAW", "V4"),
    ("V5_1_EMA", "V4"),
    ("V5_1_RAW", "V5_RAW"),
    ("V5_1_EMA", "V5_EMA"),
]

for model_a, model_b in comparisons:
    for group in models[model_a]["age_group"].cat.categories:
        a = subject_age[model_a]
        b = subject_age[model_b]

        a = a[a.age_group == group][["subject_id", "MAE"]]
        b = b[b.age_group == group][["subject_id", "MAE"]]

        paired = a.merge(
            b, on="subject_id", suffixes=("_a", "_b")
        )

        if len(paired) < 2:
            continue

        differences = (
            paired["MAE_a"] - paired["MAE_b"]
        ).to_numpy()

        observed = differences.mean()

        boot = np.empty(N_BOOT)
        n = len(differences)

        for i in range(N_BOOT):
            sample = rng.integers(0, n, size=n)
            boot[i] = differences[sample].mean()

        lo, hi = np.percentile(boot, [2.5, 97.5])

        bootstrap_results.append({
            "model_a": model_a,
            "model_b": model_b,
            "comparison": f"{model_a} - {model_b}",
            "age_group": str(group),
            "n_subjects": n,
            "mean_MAE_difference": observed,
            "CI95_lower": lo,
            "CI95_upper": hi,
        })

bootstrap_df = pd.DataFrame(bootstrap_results)
bootstrap_df.to_csv(
    f"{OUT}/paired_subject_bootstrap.csv", index=False
)

# --------------------------------------------------
# 9. Print results
# --------------------------------------------------

print("\n========== OVERALL COMPARISON ==========")
print(overall_df.to_string(index=False))

print("\n========== FOLD MAE ==========")
print(
    fold_df.pivot(
        index="fold", columns="model", values="MAE"
    ).to_string()
)

print("\n========== AGE MAE: SCAN WEIGHTED ==========")
print(
    age_df[age_df.weighting == "scan_weighted"]
    .pivot(index="age_group", columns="model", values="MAE")
    .to_string()
)

print("\n========== AGE MAE: SUBJECT BALANCED ==========")
print(
    age_df[age_df.weighting == "subject_balanced"]
    .pivot(index="age_group", columns="model", values="MAE")
    .to_string()
)

print("\n========== PAIRED SUBJECT BOOTSTRAP ==========")
print(bootstrap_df.to_string(index=False))

print("\n========== UNCERTAINTY ==========")
print(uncertainty_df.to_string(index=False))

print("\nResults saved to:", OUT)
