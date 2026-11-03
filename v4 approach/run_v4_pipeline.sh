
#!/usr/bin/env bash

set -Eeuo pipefail

# ============================================================
# V4 BRAIN AGE ESTIMATION PIPELINE
# Run from the project directory containing the Python scripts.
# ============================================================

PYTHON="${PYTHON:-python}"

# Set to 1 to run Optuna; set to 0 to skip it.
# Optuna was already run in the supplied log. Keep it off by default to avoid repeating
# the expensive search; set RUN_OPTUNA=1 explicitly when you want another search.
RUN_OPTUNA="${RUN_OPTUNA:-0}"

# IMPORTANT:
# This pipeline intentionally does NOT pass --force.
# The trainer must support loading a checkpoint to resume an interrupted fold.
# A weights-only checkpoint can restore model weights, but cannot restore the
# exact epoch/optimizer/scheduler state. See the accompanying explanation.

# All predefined V4 experiments in config_v4.py
BASE_EXPERIMENTS=(
    "v4_sfcn_baseline"
    "v4_resnet18_residual"
    "v4_sfcn_gelu"
    "v4_sfcn_adam"
    "v4_sfcn_kl_only"
    "v4_sfcn_kl_huber"
    "v4_sfcn_zscore_ablation"
    "v4_sfcn_deep_transfer"
    "v4_resnet34_scratch"
    "v4_resnet50_scratch"
)

# The Optuna-selected configuration must be added manually to config_v4.py.
OPTUNA_EXPERIMENT="v4_sfcn_optuna_best"

# Output log
mkdir -p logs
LOG_FILE="logs/v4_pipeline_$(date +%Y%m%d_%H%M%S).log"

# Log both terminal output and a persistent log file.
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "V4 PIPELINE STARTED: $(date)"
echo "Log file: $LOG_FILE"
echo "============================================================"

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "[ERROR] Required file not found: $1"
        exit 1
    fi
}

pause_for_review() {
    echo
    echo "$1"
    read -r -p "Type YES to continue: " answer
    if [[ "$answer" != "YES" ]]; then
        echo "[STOPPED] Pipeline paused by user."
        exit 0
    fi
}

# Check core project files.
for file in \
    config_v3.py \
    config_v4.py \
    core_v3.py \
    core_v4.py \
    11_train_experiment.py \
    12_optuna_search.py \
    10_eval_canonical.py \
    13_build_results_tables.py \
    qc_review_thumbnails.py
do
    require_file "$file"
done

# ============================================================
# PHASE 1: IMAGE QUALITY CONTROL
# ============================================================

echo
echo "========== PHASE 1: IMAGE QUALITY CONTROL =========="

"$PYTHON" qc_review_thumbnails.py

echo
echo "QC thumbnail generation completed."
echo "Review the generated PNG grids:"
echo "  1. 30 lowest-NCC flagged scans"
echo "  2. 30 random unflagged scans"
echo "  3. 15 worst same-day repeat pairs"
echo
echo "Update qc_scans_v3.csv with your manual confirmations."
echo "Check that the QC CSV is at the location configured in config_v3.py."

pause_for_review \
"Have you manually reviewed the images and saved the QC decisions?"

# Ensure the configured QC CSV exists.
QC_CSV=$("$PYTHON" -c \
    'import config_v3; print(config_v3.QC_CSV)')

require_file "$QC_CSV"

echo "[OK] QC file found: $QC_CSV"

# Preserve the existing 40-epoch baseline under a separate experiment tag
# before the original baseline is retrained with the longer schedule.
OLD_BASELINE_AVAILABLE=0
if compgen -G "v4/checkpoints/v4_sfcn_baseline_f*_s*.pt" > /dev/null; then
    if ! compgen -G "v4/checkpoints/v4_sfcn_baseline_40ep_f*_s*.pt" > /dev/null; then
        for src in v4/checkpoints/v4_sfcn_baseline_f*_s*.pt; do
            dst="${src/v4_sfcn_baseline_/v4_sfcn_baseline_40ep_}"
            cp -f "$src" "$dst"
        done
        for src in v4/checkpoints/v4_sfcn_baseline_f*_s*.json; do
            [[ -f "$src" ]] || continue
            dst="${src/v4_sfcn_baseline_/v4_sfcn_baseline_40ep_}"
            cp -f "$src" "$dst"
            # Keep metadata's tag consistent with its copied filename.
            sed -i 's/v4_sfcn_baseline/v4_sfcn_baseline_40ep/g' "$dst"
        done
        echo "[OK] Preserved existing baseline checkpoints as v4_sfcn_baseline_40ep."
    else
        echo "[INFO] v4_sfcn_baseline_40ep checkpoint copies already exist; leaving them unchanged."
    fi
    OLD_BASELINE_AVAILABLE=1
fi

# Back up all existing checkpoints before the longer training overwrites them.
if compgen -G "v4/checkpoints/*.pt" > /dev/null; then
    BACKUP_DIR="v4/checkpoints_backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    cp -a v4/checkpoints/. "$BACKUP_DIR"/
    echo "[OK] Existing checkpoints backed up to $BACKUP_DIR"
fi

# ============================================================
# PHASE 2: BASELINE AND ABLATION TRAINING
# ============================================================

echo
echo "========== PHASE 2: BASELINE AND ABLATIONS =========="
echo "Training budget: up to 300 epochs per model; early-stopping patience 30."
echo "Resume-safe mode: completed folds are skipped; interrupted folds resume from .resume.pt."
echo "The best model remains in .pt for evaluation; full training state is saved in .resume.pt."

for EXP in "${BASE_EXPERIMENTS[@]}"; do
    echo
    echo "------------------------------------------------------------"
    echo "Training experiment: $EXP"
    echo "------------------------------------------------------------"

    "$PYTHON" 11_train_experiment.py \
        --experiment "$EXP" \
        --folds 0,1,2,3,4 \
        --seeds 2
done

echo
echo "[OK] All predefined V4 experiments have completed training."

# ============================================================
# PHASE 3: OPTUNA HYPERPARAMETER SEARCH
# ============================================================

if [[ "$RUN_OPTUNA" == "1" ]]; then
    echo
    echo "========== PHASE 3: OPTUNA DEBUG SEARCH =========="

    "$PYTHON" 12_optuna_search.py \
        --base_experiment v4_sfcn_baseline \
        --mode debug

    echo
    echo "Optuna debug search completed."
    echo "Inspect the JSON file(s) in v4/optuna/."
    echo "Select the best parameters and add a new row named:"
    echo "  $OPTUNA_EXPERIMENT"
    echo "to config_v4.py, using the existing experiment format."

    pause_for_review \
    "Have you added the Optuna-selected configuration to config_v4.py?"

    # Verify the new experiment exists in the registry.
    "$PYTHON" -c \
        "import config_v4; assert '$OPTUNA_EXPERIMENT' in config_v4.EXPERIMENTS, 'Experiment missing from config_v4.py'; print('Optuna configuration verified.')"

    echo
    echo "Training the registered Optuna configuration."

    "$PYTHON" 11_train_experiment.py \
        --experiment "$OPTUNA_EXPERIMENT" \
        --folds 0,1,2,3,4 \
        --seeds 2
fi

# ============================================================
# PHASE 4: CANONICAL EVALUATION
# ============================================================

echo
echo "========== PHASE 4: CANONICAL EVALUATION =========="

ALL_EXPERIMENTS=("${BASE_EXPERIMENTS[@]}")
if [[ "$OLD_BASELINE_AVAILABLE" == "1" ]]; then
    ALL_EXPERIMENTS+=("v4_sfcn_baseline_40ep")
fi

if [[ "$RUN_OPTUNA" == "1" ]]; then
    ALL_EXPERIMENTS+=("$OPTUNA_EXPERIMENT")
fi

# Evaluate the first experiment and include the original V3 m1 baseline.
FIRST=1

for EXP in "${ALL_EXPERIMENTS[@]}"; do
    echo
    echo "------------------------------------------------------------"
    echo "Canonical evaluation: $EXP"
    echo "------------------------------------------------------------"

    if [[ "$FIRST" == "1" ]]; then
        "$PYTHON" 10_eval_canonical.py \
            --experiment "$EXP" \
            --include_v3
        FIRST=0
    else
        "$PYTHON" 10_eval_canonical.py \
            --experiment "$EXP"
    fi
done

echo
echo "[OK] Canonical evaluation finished."
echo "Check v4/predictions/ and v4/results/."

# ============================================================
# PHASE 5: LONGITUDINAL ANALYSIS
# ============================================================

echo
echo "========== PHASE 5: LONGITUDINAL ANALYSIS =========="

# These scripts must be available in the project directory.
# Run them only after confirming their expected input/output paths.
for file in \
    05_noise_floor.py \
    06_slopes.py \
    07_survival_validation.py
do
    require_file "$file"
done

echo
echo "IMPORTANT:"
echo "The longitudinal scripts must read the V4 prediction files:"
echo "  v4/predictions/bag_<experiment>_qc_passed.csv"
echo
echo "Check each script's input path before running it."
echo "If it defaults to V3 predictions, change its input configuration"
echo "or add the appropriate input argument before proceeding."

pause_for_review \
"Have you verified the longitudinal scripts will process V4 QC-passed predictions?"

# The commands below assume each script reads its input/output paths
# from its own configuration and requires no command-line arguments.
"$PYTHON" 05_noise_floor.py
"$PYTHON" 06_slopes.py
"$PYTHON" 07_survival_validation.py

echo
echo "[OK] Longitudinal scripts completed."

# ============================================================
# PHASE 6: BUILD FINAL RESULTS TABLES
# ============================================================

echo
echo "========== PHASE 6: FINAL RESULTS TABLES =========="

EXPERIMENT_LIST=$(IFS=,; echo "${ALL_EXPERIMENTS[*]}")

"$PYTHON" 13_build_results_tables.py \
    --experiments "$EXPERIMENT_LIST" \
    --include_v3

echo
echo "============================================================"
echo "V4 PIPELINE COMPLETED: $(date)"
echo "============================================================"
echo
echo "Review the following outputs:"
echo "  v4/checkpoints/"
echo "  v4/predictions/"
echo "  v4/results/table_A_primary_comparison.csv"
echo "  v4/results/table_B_calibration_reliability.csv"
echo "  v4/results/table_C_efficiency_generalization.csv"
echo "  v4/results/table_D_longitudinal.csv"
echo
echo "Full execution log: $LOG_FILE"
