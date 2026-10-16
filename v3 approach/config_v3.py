"""
config_v3.py -- single source of truth for paths and constants.
Edit the PATHS block once; every other v3 script imports from here.
"""
from pathlib import Path

# ------------------------------------------------------------------ PATHS
ROOT = Path("/mnt/d/brain age prep and meth")
PREPROC_DIR = ROOT / "T1 preprocessed v2"                      # *_preproc.npz tensors (160x192x160)
SCAN_META_CSV = ROOT / "okarun143_8_17_2026_12_36_37.csv"      # Label, Age, M/F
CDR_CSV = ROOT / "OASIS3_UDSb4_cdr_final.csv"                  # MRI Label, age at visit, CDRTOT, CDRSUM, MMSE
SFCN_CKPT = Path("./UKBiobank_deep_pretrain-master/brain_age/run_20190719_00_epoch_best_mae.p")

OUT = Path("./v3")                                             # everything v3 writes lives here
MANIFEST_CSV = OUT / "manifest_v3.csv"
FOLDS_CSV = OUT / "folds_v3.csv"                               # FROZEN subject->fold assignment
BINS_JSON = OUT / "bins_v3.json"
QC_CSV = OUT / "qc_scans_v3.csv"
CKPT_DIR = OUT / "checkpoints"
PRED_DIR = OUT / "predictions"
RES_DIR = OUT / "results"

# ------------------------------------------------------------------ CONSTANTS
SEED = 42
N_FOLDS = 5
VOL_SHAPE = (160, 192, 160)
BIN_STEP = 1.0
SIGMA = 1.0                      # Gaussian soft-label width (years)
VAL_FRACTION = 0.12              # share of TRAIN subjects (per fold) held out for early stopping
AGE_BANDS = [(0, 60), (60, 70), (70, 80), (80, 120)]
CLASS_CUTS = (60, 75)            # v2's derived 3-class age groups (kept only for comparability)

# Training variants (pre-specify BEFORE looking at survival results -- Step 4.8)
#   m0 = v2-style baseline: last block + head only, z-score input, KL only
#   m1 = v3 primary: full fine-tune w/ layer-wise LR, tail re-weighting, KL + L1
#   m2 = m1 + same-day-repeat consistency loss (labelled variant)
VARIANTS = {
    "m0": dict(unfreeze="last", input_mode="zscore", reweight=False, l1=0.0, consistency=0.0,
               lr_head=1e-4, lr_backbone=1e-5, llrd=1.0),
    "m1": dict(unfreeze="all", input_mode="meannorm", reweight=True, l1=0.1, consistency=0.0,
               lr_head=3e-4, lr_backbone=3e-4, llrd=0.5),
    "m2": dict(unfreeze="all", input_mode="meannorm", reweight=True, l1=0.1, consistency=0.5,
               lr_head=3e-4, lr_backbone=3e-4, llrd=0.5),
}
# CV of the brightness convention found by 02_zero_shot_probe.py (edit after running it)
MEANNORM_CV = 0.35

TTA_SHIFTS = [(0, 0, 0), (2, 0, 0), (-2, 0, 0), (0, 2, 0), (0, -2, 0)]   # voxel shifts, no flips
