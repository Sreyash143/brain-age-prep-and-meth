# -*- coding: utf-8 -*-
"""
13_build_results_tables.py -- builds Table A/B/C/D exactly as specified in
v4_results_listing_order.txt, from 10_eval_canonical.py's and
11_train_experiment.py's saved outputs. Run this LAST, after training +
canonical eval for every experiment you want in the comparison.

Usage:
    python 13_build_results_tables.py --experiments v4_sfcn_baseline,v4_resnet18_residual,v4_sfcn_optuna_best \
        --include_v3

Output (one CSV per table, qc_passed population unless noted):
    v4/results/table_A_primary_comparison.csv
    v4/results/table_B_calibration_reliability.csv
    v4/results/table_C_efficiency_generalization.csv
    v4/results/table_D_longitudinal.csv       (only if longitudinal outputs are found)
"""
import argparse
import glob
import json

import numpy as np
import pandas as pd

import config_v4 as C4


def load_canonical(tag, population):
    p = C4.RES_DIR / f"canonical_{tag}_{population}.json"
    if not p.exists():
        return None
    return json.load(open(p))


def load_training_meta(tag):
    """Aggregate per-fold/seed training metadata jsons for one experiment."""
    files = sorted(glob.glob(str(C4.CKPT_DIR / f"{tag}_f*_s*.json")))
    if not files:
        return None
    metas = [json.load(open(f)) for f in files]
    return metas


# ---------------------------------------------------------------- Table A

def build_table_a(tags, population="qc_passed"):
    rows = []
    for tag in tags:
        res = load_canonical(tag, population)
        if res is None:
            rows.append(dict(Model=tag, Input="—", Parameters="—", MAE="To calculate",
                              RMSE="—", Pearson_r="—", R2="—",
                              Note=f"no canonical_{tag}_{population}.json found -- run 10_eval_canonical.py first"))
            continue
        metas = load_training_meta(tag)
        params = metas[0]["total_params"] if metas else None
        mean_ci = res["MAE_CI95_bootstrap"]
        rows.append(dict(
            Model=tag,
            Input=res.get("bias_params_per_fold", {}),  # population carries input_mode implicitly via tag's cfg
            Parameters=f"{params:,}" if params else "To calculate",
            MAE_mean=round(res["MAE_mean_across_folds"], 3),
            MAE_sd_across_folds=round(res["MAE_sd_across_folds"], 3),
            MAE_bootstrap_CI95=f"[{mean_ci[0]:.3f}, {mean_ci[1]:.3f}]",
            RMSE=round(res["RMSE"], 3),
            Pearson_r=round(res["pearson_r"], 3),
            R2=round(res["R2"], 3),
            n_CN_scans=res["n_CN_scans"],
            n_CN_subjects=res["n_CN_subjects"],
            population=population,
        ))
    df = pd.DataFrame(rows)
    df.to_csv(C4.RES_DIR / "table_A_primary_comparison.csv", index=False)
    print("\n=== Table A -- Primary model comparison ===")
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------- Table B

def build_table_b(tags, population="qc_passed"):
    rows = []
    for tag in tags:
        res = load_canonical(tag, population)
        if res is None:
            continue
        rows.append(dict(
            Model=tag,
            MAE=round(res["MAE"], 3),
            RMSE=round(res["RMSE"], 3),
            raw_MAE_uncorrected=round(res["raw_MAE_uncorrected"], 3),
            Mean_BAG=round(res["mean_BAG"], 3),
            BAG_vs_age_slope=round(res["BAG_vs_age_slope"], 4),
            BAG_vs_age_p=round(res["BAG_vs_age_p"], 4),
            pct_abs_BAG_gt10y=round(100 * res["frac_abs_BAG_gt10"], 2),
            sigma_SEM=round(res["sigma_SEM"], 3) if np.isfinite(res.get("sigma_SEM", np.nan)) else None,
            MDC95=round(res["MDC95"], 3) if np.isfinite(res.get("MDC95", np.nan)) else None,
            ICC_A1=round(res["ICC_A1"], 3) if np.isfinite(res.get("ICC_A1", np.nan)) else None,
            uncertainty_calibration_rho=res["uncertainty_calibration"]["spearman_rho"],
            MAE_by_age_band=res["age_bands"],
            population=population,
        ))
    df = pd.DataFrame(rows)
    df.to_csv(C4.RES_DIR / "table_B_calibration_reliability.csv", index=False)
    print("\n=== Table B -- Calibration and reliability ===")
    print(df.drop(columns=["MAE_by_age_band"]).to_string(index=False))
    return df


# ---------------------------------------------------------------- Table C

def build_table_c(tags):
    rows = []
    for tag in tags:
        metas = load_training_meta(tag)
        if not metas:
            rows.append(dict(Model=tag, Note="no training metadata found -- run 11_train_experiment.py first"))
            continue
        best = min(metas, key=lambda m: m["best_val_mae"])
        all_val_gaps = [
            (m["history"][-1]["train_mae"] - m["history"][-1]["val_mae"])
            for m in metas if m["history"] and m["history"][-1].get("train_mae") is not None
        ]
        qc_res = load_canonical(tag, "qc_passed")
        all_res = load_canonical(tag, "all_eligible")
        rows.append(dict(
            Model=tag,
            n_parameters=f"{best['total_params']:,}",
            mean_epoch_seconds=round(float(np.mean([m["mean_epoch_seconds"] for m in metas])), 1),
            total_training_seconds_all_fold_seed=round(float(np.sum([m["total_train_seconds"] for m in metas])), 1),
            peak_gpu_memory_gb=round(max((m["peak_gpu_memory_gb"] or 0) for m in metas), 2),
            best_val_MAE=round(best["best_val_mae"], 3),
            train_val_MAE_gap=round(float(np.mean(all_val_gaps)), 3) if all_val_gaps else "run with --record_train_mae",
            MAE_qc_passed=round(qc_res["MAE"], 3) if qc_res else None,
            MAE_all_eligible=round(all_res["MAE"], 3) if all_res else None,
            qc_sensitivity_delta=(round(all_res["MAE"] - qc_res["MAE"], 3) if (qc_res and all_res) else None),
            n_checkpoints_fold_seed=len(metas),
        ))
    df = pd.DataFrame(rows)
    df.to_csv(C4.RES_DIR / "table_C_efficiency_generalization.csv", index=False)
    print("\n=== Table C -- Training efficiency and generalization ===")
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------- Table D

def build_table_d(tags):
    rows = []
    for tag in tags:
        for f in sorted(glob.glob(str(C4.RES_DIR.parent.parent / "v3" / "results" / f"slope_info_{tag}_*.json"))) + \
                  sorted(glob.glob(str(C4.RES_DIR / f"slope_info_{tag}_*.json"))):
            info = json.load(open(f))
            val = {}
            vpath = f.replace("slope_info_", "validation_")
            try:
                val = json.load(open(vpath))
            except FileNotFoundError:
                pass
            rows.append(dict(
                Model=tag,
                config=f.split("slope_info_")[-1].replace(".json", ""),
                n_eligible_subjects=info.get("n_subjects_before_span_rule"),
                n_events=info.get("n_events"),
                min_observation_span_years=info.get("min_span"),
                median_window_span_years=info.get("median_window_span"),
                mean_slope_reliability=info.get("mean_reliability"),
                reliability_flag=info.get("reliability_flag"),
                primary_endpoint_rule=val.get("primary_endpoint_rule", "not run"),
                clinical_endpoint="survival (C-index)" if val.get("primary_endpoint_rule", "").startswith("SURVIVAL")
                                   else "continuous (CDR-SB/MMSE slope)",
            ))
    if not rows:
        print("\n=== Table D -- Longitudinal analysis ===\nNo longitudinal outputs found for these tags "
              "(run 06_slopes.py / 07_survival_validation.py on a V4 model's bag_<tag>_qc_passed.csv first). Skipped.")
        return None
    df = pd.DataFrame(rows)
    df.to_csv(C4.RES_DIR / "table_D_longitudinal.csv", index=False)
    print("\n=== Table D -- Longitudinal analysis ===")
    print(df.to_string(index=False))
    print("\nNOTE per v4_results_listing_order.txt: a brain-age accuracy improvement must NOT be presented "
          "as a clinical-prediction improvement unless the clinical endpoint above is independently adequately "
          "powered (>=30 events for survival) and evaluated on its own terms.")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", required=True, help="comma-separated experiment tags, e.g. v4_sfcn_baseline,v4_resnet18_residual")
    ap.add_argument("--include_v3", action="store_true", help="prepend v3_m1_recomputed (requires --include_v3 was used in 10_eval_canonical.py)")
    ap.add_argument("--population", default="qc_passed", choices=["qc_passed", "all_eligible"])
    args = ap.parse_args()

    tags = args.experiments.split(",")
    if args.include_v3:
        tags = ["v3_m1_recomputed"] + tags

    build_table_a(tags, args.population)
    build_table_b(tags, args.population)
    build_table_c(tags)
    build_table_d(tags)


if __name__ == "__main__":
    main()
