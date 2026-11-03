
import os
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = "v4/predictions/bag_v4_sfcn_zscore_ablation_qc_passed.csv"

FILES = {
    "V4": "v4/predictions/raw_v4_sfcn_zscore_ablation.csv",
    "V5_RAW": "v5/predictions/raw_v5_z_khuber.csv",
    "V5_EMA": "v5/predictions/raw_v5_z_khuber_ema.csv",
}

OUT = "v5/analysis/results"
os.makedirs(OUT, exist_ok=True)

# --------------------------------------------------
# 1. Load exact QC-passed scan set
# --------------------------------------------------

qc = pd.read_csv(BASE)

qc_paths = set(qc["file_path"].dropna())

print("QC-passed scans:", len(qc_paths))

models = {}

for name, path in FILES.items():
    df = pd.read_csv(path)

    df = df[df["file_path"].isin(qc_paths)].copy()
    df = df[df["CDRTOT"] == 0].copy()

    df = df.dropna(
        subset=["subject_id", "file_path", "age",
                "predicted_age", "fold"]
    )

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

# --------------------------------------------------
# 2. Fold-wise performance
# --------------------------------------------------

fold_results = []

for name, df in models.items():
    for fold, g in df.groupby("fold"):

        fold_results.append({
            "model": name,
            "fold": int(fold),
            "n_scans": len(g),
            "n_subjects": g.subject_id.nunique(),
            "MAE": mean_absolute_error(
                g.age, g.predicted_age
            ),
            "RMSE": np.sqrt(mean_squared_error(
                g.age, g.predicted_age
            )),
            "R2": r2_score(g.age, g.predicted_age),
            "bias": g.error.mean(),
        })

fold_df = pd.DataFrame(fold_results)
fold_df.to_csv(
    f"{OUT}/fold_comparison.csv", index=False
)

# --------------------------------------------------
# 3. Age-stratified performance
# --------------------------------------------------

age_results = []

for name, df in models.items():
    for group, g in df.groupby("age_group", observed=True):

        if len(g) == 0:
            continue

        age_results.append({
            "model": name,
            "age_group": str(group),
            "n_scans": len(g),
            "n_subjects": g.subject_id.nunique(),
            "MAE": g.abs_error.mean(),
            "RMSE": np.sqrt(g.sq_error.mean()),
            "bias": g.error.mean(),
        })

age_df = pd.DataFrame(age_results)
age_df.to_csv(
    f"{OUT}/age_stratified_errors.csv", index=False
)

# --------------------------------------------------
# 4. Subject-level error analysis
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

subject_df = pd.concat(
    subject_results, ignore_index=True
)

subject_df.to_csv(
    f"{OUT}/subject_level_errors.csv", index=False
)

# --------------------------------------------------
# 5. Uncertainty versus actual error
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
# 6. Overall summary
# --------------------------------------------------

summary = []

for name, df in models.items():

    summary.append({
        "model": name,
        "scans": len(df),
        "subjects": df.subject_id.nunique(),
        "MAE": df.abs_error.mean(),
        "RMSE": np.sqrt(df.sq_error.mean()),
        "R2": r2_score(df.age, df.predicted_age),
        "bias": df.error.mean(),
    })

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    f"{OUT}/overall_comparison.csv", index=False
)

# --------------------------------------------------
# 7. Print results
# --------------------------------------------------

print("\n========== OVERALL COMPARISON ==========")
print(summary_df.to_string(index=False))

print("\n========== FOLD COMPARISON ==========")
print(
    fold_df.pivot(
        index="fold",
        columns="model",
        values="MAE"
    ).to_string()
)

print("\n========== AGE STRATIFIED ERRORS ==========")
print(
    age_df.pivot(
        index="age_group",
        columns="model",
        values="MAE"
    ).to_string()
)

print("\n========== UNCERTAINTY ANALYSIS ==========")
print(uncertainty_df.to_string(index=False))

print("\nResults saved to:", OUT)

