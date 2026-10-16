"""
01_qc_scans.py   (Steps 1.4, 3.2 -- image-only QC; CPU, no GPU needed)

Why: in v2, 39 of 358 test CN scans (11%) had |BAG| > 10 y and a handful of same-day repeats differed by 7-13 y.
Errors that large are almost never biology; they are failed skull-strips / registrations. This script flags them
using IMAGE properties only (never the model's error -- error-based exclusion would be circular).

Per scan:   brain_frac (non-zero voxel fraction), z_mean, z_std, nan_frac, ncc_template (similarity to a CN mean template)
Per session with >=2 runs:   ncc_repeat (similarity between run 1 and run 2 after registration)
Flags:      qc_flag = 1 when a robust z-score (median/MAD) of ncc_template or brain_frac is beyond +-QC_Z, or ncc_repeat is low.

Output: v3/qc_scans_v3.csv  (merge on file_path).  ALWAYS eyeball ~30 flagged and ~30 unflagged scans before fixing exclusion rules.
"""
import argparse
import numpy as np, pandas as pd
from tqdm import tqdm
import config_v3 as C


def load(p):
    with np.load(p) as z:
        return z["data"].astype(np.float32)


def ncc(a, b):
    m = (a != 0) | (b != 0)
    x, y = a[m], b[m]
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-8))


def robust_z(s):
    med = s.median()
    mad = (s - med).abs().median() * 1.4826 + 1e-9
    return (s - med) / mad


def main(qc_z, n_template):
    man = pd.read_csv(C.MANIFEST_CSV)
    rng = np.random.default_rng(C.SEED)
    cn = man[man.CDRTOT == 0]
    tmpl_paths = rng.choice(cn.file_path.values, size=min(n_template, len(cn)), replace=False)
    template = np.mean([load(p) for p in tqdm(tmpl_paths, desc="template")], axis=0)

    rows = []
    for p in tqdm(man.file_path, desc="per-scan QC"):
        v = load(p)
        rows.append(dict(file_path=p, brain_frac=float((v != 0).mean()), z_mean=float(v[v != 0].mean()) if (v != 0).any() else np.nan,
                         z_std=float(v[v != 0].std()) if (v != 0).any() else np.nan,
                         nan_frac=float(np.isnan(v).mean()), ncc_template=ncc(v, template)))
    qc = pd.DataFrame(rows)

    # same-session repeat NCC (run1 vs run2)
    rep = {}
    for lab, g in tqdm(man.sort_values("run").groupby("mri_label"), desc="repeat NCC"):
        if len(g) >= 2:
            rep[lab] = ncc(load(g.file_path.iloc[0]), load(g.file_path.iloc[1]))
    qc = qc.merge(man[["file_path", "mri_label"]], on="file_path")
    qc["ncc_repeat"] = qc.mri_label.map(rep)

    qc["z_ncc_template"] = robust_z(qc.ncc_template)
    qc["z_brain_frac"] = robust_z(qc.brain_frac)
    low_rep = qc.ncc_repeat < (qc.ncc_repeat.median() - qc_z * (qc.ncc_repeat - qc.ncc_repeat.median()).abs().median() * 1.4826)
    qc["qc_flag"] = ((qc.z_ncc_template < -qc_z) | (qc.z_brain_frac.abs() > qc_z) | (qc.nan_frac > 0) | low_rep.fillna(False)).astype(int)
    qc.to_csv(C.QC_CSV, index=False)

    print(f"scans {len(qc)} | flagged {qc.qc_flag.sum()} ({100 * qc.qc_flag.mean():.1f}%)")
    print(qc[["brain_frac", "ncc_template", "ncc_repeat"]].describe().round(3))
    print("Next: open the 30 lowest-ncc_template scans in a viewer; confirm they are failed registrations/skull-strips.")
    print("Then freeze the rule (QC_Z) in the pre-registration and apply it once.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc_z", type=float, default=4.0)
    ap.add_argument("--n_template", type=int, default=100)
    a = ap.parse_args()
    main(a.qc_z, a.n_template)
