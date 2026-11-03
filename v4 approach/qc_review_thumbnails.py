# -*- coding: utf-8 -*-
"""
qc_review_thumbnails.py -- v3_problems.txt Priority 1 / v4_approach.txt Phase 6.

"Before making the network larger, inspect the flagged and unflagged MRI
volumes for registration errors, skull-stripping failures, inconsistent
brain masks, and intensity problems." V3's own 01_qc_scans.py prints this
instruction but nothing in the archive actually produced images to look at
-- 631/3336 scans were flagged by ncc_template/brain_frac/ncc_repeat alone
and apparently never eyeballed. This script closes that gap: it saves PNG
thumbnail grids (3 orthogonal mid-slices per scan) for

    - the 30 LOWEST-ncc_template flagged scans   (most likely true failures)
    - 30 RANDOM unflagged scans                  (baseline "what good looks like")
    - the 15 worst same-day repeat pairs by ncc_repeat (if any exist)

so a human can actually confirm the QC rule is catching registration/
skull-strip failures and not, say, unusually atrophied-but-valid brains.

This produces NO exclusion decision by itself -- per the standing rule, QC
exclusions must come from what a human sees in these images, never from
a model's prediction error.

Usage:
    python qc_review_thumbnails.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config_v3 as C3
import config_v4 as C4


def load(p):
    with np.load(p) as z:
        return z["data"].astype(np.float32)


def save_grid(paths, labels, out_path, title, ncols=5):
    nrows = int(np.ceil(len(paths) / ncols))
    fig, axes = plt.subplots(nrows, ncols * 3, figsize=(ncols * 4.5, nrows * 1.8))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for i, (p, lab) in enumerate(zip(paths, labels)):
        try:
            v = load(p)
        except Exception as e:
            row, col = divmod(i, ncols)
            axes[row, col * 3].text(0.5, 0.5, f"load failed:\n{e}", ha="center", va="center", fontsize=6)
            continue
        cx, cy, cz = (s // 2 for s in v.shape)
        row, col = divmod(i, ncols)
        for j, sl in enumerate([v[cx, :, :], v[:, cy, :], v[:, :, cz]]):
            ax = axes[row, col * 3 + j]
            ax.imshow(np.rot90(sl), cmap="gray")
            ax.axis("off")
            if j == 0:
                ax.set_title(lab, fontsize=6, loc="left")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("saved", out_path)


def main():
    qc = pd.read_csv(C3.QC_CSV)
    man = pd.read_csv(C3.MANIFEST_CSV)
    rng = np.random.default_rng(C3.SEED)

    # 30 lowest-ncc_template flagged scans
    flagged = qc[qc.qc_flag == 1].sort_values("ncc_template").head(30)
    if len(flagged):
        labs = [f"{r.mri_label}\nncc={r.ncc_template:.2f}" for r in flagged.itertuples()]
        save_grid(flagged.file_path.tolist(), labs, C4.QC_REVIEW_DIR / "flagged_worst30.png",
                  "30 lowest-ncc_template FLAGGED scans (expect: registration/skull-strip failures)")
    else:
        print("[warn] no flagged scans found in", C3.QC_CSV)

    # 30 random unflagged scans
    unflagged = qc[qc.qc_flag == 0]
    if len(unflagged):
        sample = unflagged.sample(n=min(30, len(unflagged)), random_state=C3.SEED)
        labs = [f"{r.mri_label}\nncc={r.ncc_template:.2f}" for r in sample.itertuples()]
        save_grid(sample.file_path.tolist(), labs, C4.QC_REVIEW_DIR / "unflagged_sample30.png",
                  "30 random UNFLAGGED scans (baseline: what a clean scan looks like)")

    # 15 worst same-day repeat pairs by ncc_repeat
    if "ncc_repeat" in qc.columns:
        worst_rep = (qc.dropna(subset=["ncc_repeat"])
                     .drop_duplicates("mri_label")
                     .sort_values("ncc_repeat").head(15))
        if len(worst_rep):
            pair_paths, pair_labs = [], []
            for lab in worst_rep.mri_label:
                g = man[man.mri_label == lab].sort_values("run")
                if len(g) >= 2:
                    pair_paths += [g.file_path.iloc[0], g.file_path.iloc[1]]
                    pair_labs += [f"{lab} run1", f"{lab} run2"]
            if pair_paths:
                save_grid(pair_paths, pair_labs, C4.QC_REVIEW_DIR / "worst_repeat_pairs15.png",
                          "15 worst same-day repeat pairs by ncc_repeat (large BAG differences in v2 came from here)",
                          ncols=4)

    print(f"\nAll grids written to {C4.QC_REVIEW_DIR}/. Open them and, for each flagged scan, mark:\n"
          "  CONFIRMED-FAILURE (bad registration/skull-strip/mask) vs LOOKS-OK (keep qc_flag but reconsider threshold).\n"
          "Record the result as a column in qc_scans_v3.csv (e.g. qc_manual_review) BEFORE freezing which exclusion "
          "rule 03_train_cv.py / 10_eval_canonical.py use for TRAINING. Never adjust qc_flag based on which scans "
          "happen to have large prediction error -- that is the circularity v3_problems.txt Priority 1 warns against.")


if __name__ == "__main__":
    main()
