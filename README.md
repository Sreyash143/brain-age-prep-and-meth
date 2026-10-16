# Brain Age Estimation and Longitudinal MRI Analysis

## Project Directory Structure

The project contains the preprocessing pipeline, clinical datasets, pretrained models, multiple experimental approaches (V1–V4), evaluation scripts, model checkpoints, and visualization packs for brain age estimation using the OASIS-3 MRI dataset.

```text
brain-age-prep-and-methodology/
│
├── 📓 preprocessing.ipynb
├── 📓 preprocessing(old).ipynb
├── 📓 code updated.ipynb
│
├── 📄 brainage.pdf
├── 📄 references.txt
├── 📄 deleted meth.txt
├── 📄 v3 approach.txt
├── 📄 v3 problems.txt
├── 📄 v4 approach.txt
├── 📄 v4 results listing order.txt
│
├── 🧠 mni152_brain_1mm.nii.gz
├── 📊 OASIS3_UDSb4_cdr_final.csv
├── 📊 OASIS3_UDSd1_diagnoses_filtered.csv
├── 📊 okarun143_8_17_2026_12_36_37.csv
│
├── 📦 UKBiobank_deep_pretrain-master.zip
├── 📦 brain_age_v1_graph_pack.zip
├── 📦 brain_age_v2_graph_pack.zip
├── 📦 brain_age_v3_graph_pack.zip
├── 📦 brain_age_v1_v2_v3_comparison_pack.zip
├── 📦 v2 approach.zip
├── 📦 v3 approach.zip
├── 📦 v4 approach.zip
├── 📦 files.zip
├── 📦 results_notebook.zip
│
├── 📁 T1/                         # Raw T1-weighted MRI scans
├── 📁 T1 preprocessed/            # V1 preprocessed scans
├── 📁 T1 preprocessed v2/         # V2 preprocessed scans
│
├── 📁 v1 preprocessing/
│   └── 🧠 mni152_brain_1mm.nii.gz
│
├── 📁 v1 approach/
│   ├── 📓 brain_age_pipeline_fixed prep.ipynb
│   ├── 📓 brain_age_pipeline_fixed except prep.ipynb
│   ├── 📓 methodology.ipynb
│   ├── 📓 section_111_a.ipynb
│   ├── 🐍 brain_age_pipeline_fixed (1).py
│   ├── 📁 checkpoints_notebook/
│   └── 📁 results_notebook/
│
├── 📁 v2 approach/
│   ├── 📓 preprocessing.ipynb
│   ├── 📓 preprocessing_v2.ipynb
│   ├── 📓 methodology.ipynb
│   ├── 📓 methodology_fixed.ipynb
│   ├── 📓 methodology_recovery.ipynb
│   ├── 📓 train_ensemble_v2.ipynb
│   │
│   ├── 🐍 preprocessing_v2.py
│   ├── 🐍 train_ensemble_v2.py
│   ├── 🐍 brainage_utils.py
│   ├── 🐍 sfcn_model.py
│   ├── 🐍 resnet3d.py
│   ├── 🐍 bootstrap_cindex.py
│   ├── 🐍 cv_diagnostic.py
│   │
│   ├── 📊 brainage_predictions_v2.csv
│   ├── 📊 OASIS3_UDSb4_cdr_final.csv
│   ├── 📊 OASIS3_UDSd1_diagnoses_filtered.csv
│   ├── 📊 okarun143_8_17_2026_12_36_37.csv
│   │
│   ├── 🧠 mni152_brain_1mm.nii.gz
│   ├── 🧠 run_20190719_00_epoch_best_mae.p
│   │
│   ├── 📦 results_v2.zip
│   ├── 📁 UKBiobank_deep_pretrain-master/
│   ├── 📁 checkpoints_v2/
│   ├── 📁 results_v2/
│   └── 📁 __pycache__/
│
├── 📁 v3 approach/
│   ├── 📓 methodology_v3.ipynb
│   ├── 📝 METHODOLOGY_v3.md
│   ├── 🐍 config_v3.py
│   ├── 🐍 core_v3.py
│   ├── 🐍 longitudinal_lib.py
│   │
│   ├── 🐍 00_build_manifest_and_folds.py
│   ├── 🐍 01_qc_scans.py
│   ├── 🐍 02_zero_shot_probe.py
│   ├── 🐍 03_train_cv.py
│   ├── 🐍 04a_predict_oof.py
│   ├── 🐍 04b_bias_correct_accuracy.py
│   ├── 🐍 05_noise_floor.py
│   ├── 🐍 06_slopes.py
│   ├── 🐍 07_survival_validation.py
│   ├── 🐍 08_robustness.py
│   ├── 🐍 09_leakage_audit.py
│   ├── 🐚 run_all.sh
│   │
│   ├── 📊 OASIS3_*.csv
│   ├── 📊 okarun143_*.csv
│   ├── 🧠 mni152_brain_1mm.nii.gz
│   ├── 📁 UKBiobank_deep_pretrain-master/
│   ├── 📁 v3/                     # V3 outputs and results
│   └── 📁 __pycache__/
│
├── 📁 v4 approach/
│   ├── 📓 methodology_v3.ipynb
│   ├── 📝 METHODOLOGY_v3.md
│   ├── 📝 README_V4_IMPROVEMENTS.md
│   │
│   ├── 🐍 config_v3.py
│   ├── 🐍 config_v4.py
│   ├── 🐍 core_v3.py
│   ├── 🐍 core_v4.py
│   ├── 🐍 deep_sfcn.py
│   ├── 🐍 sfcn_model.py
│   ├── 🐍 resnet3d.py
│   ├── 🐍 longitudinal_lib.py
│   ├── 🐍 qc_review_thumbnails.py
│   │
│   ├── 🐍 00_*.py – 13_*.py     # V4 pipeline scripts
│   ├── 🐚 run_all.sh
│   ├── 🐚 run_v4_pipeline.sh
│   │
│   ├── 📊 OASIS3_*.csv
│   ├── 📊 okarun143_*.csv
│   ├── 🧠 mni152_brain_1mm.nii.gz
│   ├── 📁 UKBiobank_deep_pretrain-master/
│   ├── 📁 v3/                     # Carried-forward V3 outputs
│   ├── 📁 v4/                     # V4 outputs and results
│   ├── 📁 logs/
│   └── 📁 __pycache__/
│
├── 📁 v4 checkpoints/
│   ├── 🧠 v4_sfcn_baseline_f*_s*.pt
│   │                              # 5 folds × 2 seeds = 10 checkpoints
│   ├── 🧠 v4_resnet18_residual_f0_s*.pt
│   └── 📋 *.json                  # Per-checkpoint training metadata
│
├── 📁 brain_age_v1_graph_pack/
│   └── 📊 V1 result graphs
│
├── 📁 brain_age_v2_graph_pack/
│   └── 📊 V2 result graphs
│
├── 📁 brain_age_v3_graph_pack/
│   └── 📊 V3 result graphs
│
├── 📁 brain_age_v1_v2_v3_comparison/
│   ├── 🖼️ 01_* – 16_*             # Comparison plots (.png)
│   ├── 📄 README.md
│   ├── 📊 common_test_set_metrics.csv
│   ├── 📊 age_band_metrics.csv
│   ├── 📊 cdr_group_metrics.csv
│   ├── 📊 sex_group_metrics.csv
│   ├── 📊 mae_subject_cluster_bootstrap_ci.csv
│   ├── 📊 matched_predictions_all_versions.csv
│   ├── 🖼️ graph_contact_sheet.jpg
│   └── 📄 all_graphs.pdf
│
├── 📁 drive-download-20260916T124138Z-1-001/
│   ├── 📓 code updated.ipynb
│   ├── 📓 preprocessing.ipynb
│   ├── 📓 section_111_a.ipynb
│   ├── 📓 train_overnight.ipynb
│   ├── 🐍 code updated.py
│   ├── 🐍 train_overnight.py
│   ├── 🐍 update_ipynb.py
│   └── 🐍 scratch_check.py
│
├── 📁 myenv/                       # Python virtual environment
├── 📁 wsl_env/                     # WSL Python environment
└── 📁 wsl_env_312/                 # WSL Python 3.12 environment
```

## Directory Overview

| Directory / File | Description |
|---|---|
| `T1/` | Raw T1-weighted MRI scans used as input data. |
| `T1 preprocessed/` | MRI scans processed using the V1 preprocessing pipeline. |
| `T1 preprocessed v2/` | MRI scans processed using the V2 preprocessing pipeline. |
| `v1 approach/` | Initial brain age estimation implementation and notebook-based experiments. |
| `v2 approach/` | Updated preprocessing, SFCN/ResNet models, ensemble training, and prediction generation. |
| `v3 approach/` | Cross-validation, bias correction, longitudinal analysis, robustness evaluation, and leakage auditing. |
| `v4 approach/` | Extended experimental pipeline with updated architectures, training configurations, and evaluation scripts. |
| `v4 checkpoints/` | Saved model weights and associated training metadata. |
| `brain_age_v*_graph_pack/` | Version-specific evaluation graphs and visualizations. |
| `brain_age_v1_v2_v3_comparison/` | Comparative metrics, subgroup evaluations, confidence intervals, and plots across V1–V3. |
| `OASIS3_*.csv` | Clinical and diagnostic information from OASIS-3. |
| `okarun143_*.csv` | Subject and MRI scan manifest information. |
| `mni152_brain_1mm.nii.gz` | MNI152 reference brain template used in spatial preprocessing. |
| `UKBiobank_deep_pretrain-master/` | Pretrained model resources used by the pipeline. |
| `myenv/`, `wsl_env/`, `wsl_env_312/` | Local Python virtual environments for different development configurations. |

## Experimental Versions

- **V1 — Initial pipeline:** Baseline preprocessing and brain age estimation.
- **V2 — Model and preprocessing improvements:** Updated preprocessing, SFCN/ResNet implementations, and ensemble experiments.
- **V3 — Longitudinal evaluation pipeline:** Cross-validation, out-of-fold predictions, bias correction, noise-floor estimation, longitudinal slope analysis, survival validation, robustness checks, and leakage auditing.
- **V4 — Extended model experiments:** Revised architectures and training configurations, including SFCN and ResNet-based experiments, with saved checkpoints and evaluation outputs.

## Key Results and Comparison Resources

The `brain_age_v1_v2_v3_comparison/` directory contains resources for comparing the three earlier approaches:

- `common_test_set_metrics.csv` — Performance metrics on a common test set.
- `age_band_metrics.csv` — Performance across chronological age groups.
- `cdr_group_metrics.csv` — Performance across Clinical Dementia Rating groups.
- `sex_group_metrics.csv` — Performance by sex.
- `mae_subject_cluster_bootstrap_ci.csv` — Subject-clustered bootstrap confidence intervals for MAE.
- `matched_predictions_all_versions.csv` — Matched predictions across model versions.
- `all_graphs.pdf` — Consolidated visualization report.
- `graph_contact_sheet.jpg` — Overview of the generated comparison plots.

## Notes

- File and directory names are preserved as provided in the project inventory.
- Wildcard entries such as `OASIS3_*.csv`, `okarun143_*.csv`, and `00_*.py – 13_*.py` represent groups of files rather than literal filenames.
- The directory tree is a structural overview; verify actual file names and locations against the current repository.
- For a public GitHub repository, consider excluding raw MRI data, large model checkpoints, virtual environments, temporary files, and sensitive or restricted clinical data. Follow the OASIS-3 data-use requirements before publishing any dataset or derived data.
