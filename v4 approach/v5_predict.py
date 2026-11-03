# -*- coding: utf-8 -*-
"""
v5_predict.py -- out-of-fold predictions for V5 checkpoints (writes only v5/predictions/).

Re-uses 10_eval_canonical.run_oof_predictions (same TTA shifts, same seed/TTA aggregation, same
columns) but selects checkpoints with a STRICT regex, so '<tag>.resume.pt' training-state files
(which 10_eval_canonical's glob '<exp>_f{k}_s*.pt' would also match) can never be loaded as weights.

    python v5_predict.py --experiment v5_z_khuber --weights raw     # tag v5_z_khuber
    python v5_predict.py --experiment v5_z_khuber --weights ema     # tag v5_z_khuber_ema
Then:  python v5_report.py ...
"""
import argparse
import importlib
import re
from argparse import Namespace

import pandas as pd
import torch

import config_v3 as C3
import config_v5 as C5
import core_v3 as K

E = importlib.import_module("10_eval_canonical")


def ckpt_paths(experiment, fold, weights):
    suffix = r"\.ema\.pt" if weights == "ema" else r"\.pt"
    pat = re.compile(rf"^{re.escape(experiment)}_f{fold}_s(\d+){suffix}$")
    return sorted(p for p in C5.CKPT_DIR.iterdir() if pat.match(p.name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--base", default=None, help="config key to read the architecture from (default: --experiment)")
    ap.add_argument("--weights", choices=["raw", "ema"], default="raw")
    ap.add_argument("--n_tta", type=int, default=len(C3.TTA_SHIFTS))
    ap.add_argument("--cv", type=float, default=C3.MEANNORM_CV)
    args = ap.parse_args()

    cfg = dict(C5.EXPERIMENTS[args.base or args.experiment])
    tag = args.experiment + ("_ema" if args.weights == "ema" else "")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    man = pd.read_csv(C3.MANIFEST_CSV)
    bins = K.load_bins(C3.BINS_JSON)

    # every fold must have the same number of checkpoints, otherwise folds are not comparable
    counts = {k: len(ckpt_paths(args.experiment, k, args.weights)) for k in range(C3.N_FOLDS)}
    print("checkpoints per fold:", counts)
    if len(set(counts.values())) > 1:
        print("[WARN] folds have different numbers of checkpoints; predictions are NOT comparable across folds")

    raw = E.run_oof_predictions(tag, lambda k: ckpt_paths(args.experiment, k, args.weights), cfg, bins, man, dev,
                                Namespace(n_tta=args.n_tta, cv=args.cv))
    if raw.empty:
        raise SystemExit("no predictions produced")
    out = C5.PRED_DIR / f"raw_{tag}.csv"
    raw.to_csv(out, index=False)
    print("wrote", out, raw.shape)


if __name__ == "__main__":
    main()
