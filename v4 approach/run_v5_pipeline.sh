#!/usr/bin/env bash
# V5 brain-age pipeline runner for WSL.
# Safe by default: tests, split validation and baseline reproduction only.
# Training is started only when RUN_TRAIN=1 is set.
# Optional Optuna stage is started only when RUN_OPTUNA=1 is set.

set -Eeuo pipefail

# ---- Configuration (override with environment variables if needed) ----
PROJECT_DIR="${PROJECT_DIR:-/mnt/d/brain age prep and meth/v4 approach}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$PROJECT_DIR/wsl_env_312/bin/activate}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/v5/logs}"
RUN_TRAIN="${RUN_TRAIN:-0}"
RUN_OPTUNA="${RUN_OPTUNA:-0}"
TRAIN_FOLDS="${TRAIN_FOLDS:-0,1,2,3,4}"
TRAIN_SEEDS="${TRAIN_SEEDS:-2}"
BOOTSTRAP_REPS="${BOOTSTRAP_REPS:-10000}"

# Optional second-stage tuning settings. This is exploratory for outer fold 0 only.
OPTUNA_FOLD="${OPTUNA_FOLD:-0}"
OPTUNA_TRIALS="${OPTUNA_TRIALS:-25}"
OPTUNA_EPOCHS="${OPTUNA_EPOCHS:-20}"

mkdir -p "$LOG_DIR"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/v5_pipeline_${RUN_ID}.log"

# Log all output to terminal and a timestamped file.
exec > >(tee -a "$LOG_FILE") 2>&1

on_error() {
  local exit_code=$?
  echo
  echo "[ERROR] Pipeline failed with exit code $exit_code at line ${BASH_LINENO[0]:-unknown}."
  echo "[INFO] Log: $LOG_FILE"
  echo "[INFO] Fix the reported issue, then rerun the script. Completed stages are not automatically repeated by this script."
  exit "$exit_code"
}
trap on_error ERR

stage() {
  echo
  echo "================================================================"
  echo "[$(date '+%F %T')] $*"
  echo "================================================================"
}

need_file() {
  if [[ ! -f "$1" ]]; then
    echo "[ERROR] Required file not found: $1"
    echo "Copy the contents of v5_pkg/ into PROJECT_DIR first."
    exit 2
  fi
}

stage "V5 pipeline starting"
echo "Project: $PROJECT_DIR"
echo "Log:     $LOG_FILE"
echo "RUN_TRAIN=$RUN_TRAIN | RUN_OPTUNA=$RUN_OPTUNA"

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "[ERROR] Project directory does not exist: $PROJECT_DIR"
  echo "Set PROJECT_DIR to the directory containing config_v3.py, config_v4.py and the V4 project files."
  exit 2
fi
cd "$PROJECT_DIR"

# Activate the existing environment if it is not already active.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -f "$VENV_ACTIVATE" ]]; then
    # shellcheck disable=SC1090
    source "$VENV_ACTIVATE"
  else
    echo "[ERROR] Virtual environment activation script not found: $VENV_ACTIVATE"
    echo "Set VENV_ACTIVATE to your actual wsl_env_312/bin/activate path."
    exit 2
  fi
fi

echo "Python: $(command -v python)"
python --version

# Check required V5 and V4 files before executing anything.
for f in config_v3.py config_v4.py core_v3.py core_v4.py v5_train.py config_v5.py v5_predict.py v5_report.py v5_optuna.py; do
  need_file "$PROJECT_DIR/$f"
done
need_file "$PROJECT_DIR/v4/predictions/bag_v4_sfcn_zscore_ablation_qc_passed.csv"

# Capture repository state when available; this is informational only.
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Git commit: $(git rev-parse --short HEAD)"
  echo "Git status (short):"
  git status --short || true
else
  echo "[INFO] No Git repository detected; continuing."
fi

stage "1/4: Unit tests"
python -m pytest tests -q

stage "2/4: Split and subject-leakage dry run"
python v5_train.py \
  --experiment v5_z_khuber \
  --folds "$TRAIN_FOLDS" \
  --dry_run

stage "3/4: Reproduce V4 baseline report"
python v5_report.py \
  --tags v4_sfcn_zscore_ablation,v4_sfcn_baseline_40ep,v3_m1_recomputed \
  --baseline v4_sfcn_zscore_ablation \
  --pred_dir v4/predictions \
  --qc_passed_from v4/predictions/bag_v4_sfcn_zscore_ablation_qc_passed.csv \
  --out_dir v5/results \
  --name baseline_repro

echo "[INFO] Baseline report should be checked before training. See v5/results/baseline_repro.md and integrity checks."

if [[ "$RUN_TRAIN" != "1" ]]; then
  stage "Preflight complete; training not started"
  echo "Tests, split validation and baseline reproduction have run."
  echo "Review v5/results/baseline_repro.md and baseline_repro_integrity_checks.csv."
  echo "When ready, rerun with: RUN_TRAIN=1 bash run_v5_pipeline.sh"
  echo "Log: $LOG_FILE"
  exit 0
fi

stage "4/4: Train V5 (this can take several GPU-hours)"
python v5_train.py \
  --experiment v5_z_khuber \
  --folds "$TRAIN_FOLDS" \
  --seeds "$TRAIN_SEEDS" \
  --record_train_mae

stage "Generate out-of-fold predictions using raw weights"
python v5_predict.py --experiment v5_z_khuber --weights raw

stage "Generate out-of-fold predictions using EMA weights"
python v5_predict.py --experiment v5_z_khuber --weights ema

stage "Compare V5 raw and EMA predictions against V4"
python v5_report.py \
  --tags v4_sfcn_zscore_ablation,v5_z_khuber,v5_z_khuber_ema \
  --baseline v4_sfcn_zscore_ablation \
  --pred_dir v4/predictions \
  --pred_dir v5/predictions \
  --qc_passed_from v4/predictions/bag_v4_sfcn_zscore_ablation_qc_passed.csv \
  --boot "$BOOTSTRAP_REPS" \
  --out_dir v5/results \
  --name round1

stage "Round 1 complete"
echo "Review v5/results/round1.md and v5/results/round1_integrity_checks.csv."
echo "Choose raw vs EMA using validation MAE in v5/experiment_registry.csv, not held-out test performance."
echo "Do not run Optuna automatically; first apply the decision rules in AUDIT_AND_PLAN_V5.md."

if [[ "$RUN_OPTUNA" == "1" ]]; then
  stage "Optional Optuna search (outer fold $OPTUNA_FOLD only)"
  python v5_optuna.py \
    --experiment v5_z_khuber \
    --outer_fold "$OPTUNA_FOLD" \
    --n_trials "$OPTUNA_TRIALS" \
    --trial_epochs "$OPTUNA_EPOCHS"
  echo "[IMPORTANT] Optuna output is exploratory for the selected outer fold. It does not establish a five-fold tuned-model result."
  echo "Use the retraining command printed by v5_optuna.py, then evaluate only the appropriate held-out fold(s)."
fi

echo
echo "[DONE] V5 pipeline stages requested have completed."
echo "Log: $LOG_FILE"
