import pandas as pd
import numpy as np

qc = pd.read_csv(
    "v4/predictions/bag_v4_sfcn_zscore_ablation_qc_passed.csv"
)

files = {
    "V4": "v4/predictions/bag_v4_sfcn_zscore_ablation_qc_passed.csv",
    "V5_RAW": "v5/predictions/raw_v5_z_khuber.csv",
    "V5_EMA": "v5/predictions/raw_v5_z_khuber_ema.csv",
}

valid_paths = set(qc["file_path"])
frames = []

for model, path in files.items():
    df = pd.read_csv(path)
    df = df[df["file_path"].isin(valid_paths)].copy()
    df = df[df["CDRTOT"] == 0].copy()

    if model == "V4":
        df["prediction"] = df["brain_age"]
    else:
        df["prediction"] = df["predicted_age"]

    df["model"] = model
    frames.append(df[[
        "subject_id", "file_path", "age", "prediction", "model"
    ]])

data = pd.concat(frames, ignore_index=True)

data["error"] = data["prediction"] - data["age"]
data["abs_error"] = data["error"].abs()

data["age_group"] = pd.cut(
    data["age"],
    bins=[-np.inf, 50, 60, 70, 80, 90, np.inf],
    labels=["<50", "50-59", "60-69", "70-79", "80-89", "90+"],
    right=False
)

# Each subject receives equal total weight within each age group.
data["subject_scans_in_group"] = (
    data.groupby(["model", "age_group", "subject_id"])["file_path"]
        .transform("count")
)

data["weight"] = 1 / data["subject_scans_in_group"]

results = []

for (model, group), d in data.groupby(
    ["model", "age_group"], observed=True
):
    results.append({
        "model": model,
        "age_group": group,
        "scans": len(d),
        "subjects": d["subject_id"].nunique(),
        "weighted_MAE": np.average(d["abs_error"], weights=d["weight"]),
        "weighted_bias": np.average(d["error"], weights=d["weight"]),
        "median_absolute_error": d["abs_error"].median()
    })

result = pd.DataFrame(results)

print(result.to_string(index=False))

result.to_csv(
    "v5/analysis/results/age_stratified_subject_weighted.csv",
    index=False
)

scan_data = data[data["model"] == "V5_RAW"].copy()

subject_spans = (
    scan_data.groupby("subject_id")
    .agg(
        scans=("file_path", "count"),
        min_age=("age", "min"),
        max_age=("age", "max"),
        mean_age=("age", "mean"),
        age_groups=("age_group", "nunique")
    )
)

subject_spans["age_span"] = (
    subject_spans["max_age"] - subject_spans["min_age"]
)

print("\nSubjects spanning multiple age groups:")
print(
    subject_spans[subject_spans["age_groups"] > 1]
    .sort_values("age_span", ascending=False)
    .head(30).to_string()
)

print("\nAge-span distribution:")
print(subject_spans["age_span"].describe())

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N_BOOTSTRAPS = 10000

# First calculate one MAE per subject, per model, per age group.
subject_errors = (
    data.groupby(
        ["age_group", "model", "subject_id"],
        observed=True
    )
    .agg(
        subject_MAE=("abs_error", "mean"),
        scans=("file_path", "count")
    )
    .reset_index()
)

results = []

for age_group in data["age_group"].dropna().unique():

    subset = subject_errors[
        subject_errors["age_group"] == age_group
    ]

    pivot = subset.pivot(
        index="subject_id",
        columns="model",
        values="subject_MAE"
    ).dropna(subset=["V4", "V5_RAW", "V5_EMA"])

    if len(pivot) == 0:
        continue

    v4 = pivot["V4"].to_numpy()
    n = len(v4)

    for model in ["V5_RAW", "V5_EMA"]:

        difference = pivot[model].to_numpy() - v4

        bootstrap_differences = np.empty(N_BOOTSTRAPS)

        for i in range(N_BOOTSTRAPS):
            indices = rng.integers(0, n, size=n)
            bootstrap_differences[i] = difference[indices].mean()

        results.append({
            "age_group": str(age_group),
            "comparison": f"{model} vs V4",
            "subjects": n,
            "V4_MAE": v4.mean(),
            "V5_MAE": pivot[model].mean(),
            "MAE_difference": difference.mean(),
            "CI_lower": np.percentile(bootstrap_differences, 2.5),
            "CI_upper": np.percentile(bootstrap_differences, 97.5)
        })

bootstrap_results = pd.DataFrame(results)

print("\nPAIRED SUBJECT-LEVEL BOOTSTRAP")
print(bootstrap_results.round(4).to_string(index=False))

bootstrap_results.to_csv(
    "v5/analysis/results/age_group_paired_bootstrap.csv",
    index=False
)