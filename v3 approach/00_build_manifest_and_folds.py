"""
00_build_manifest_and_folds.py   (Phases 0-2 of the v3 plan)

Creates, ONCE:
  * manifest_v3.csv   one row per scan (subject, session, run, age, sex, CDR, CDRSUM, MMSE, path)
  * folds_v3.csv      frozen subject -> fold map (stratified by age band, sex, ever-impaired)
  * bins_v3.json      age-bin scheme shared by every model (so heads have identical shape)
  * cohort_flow_v3.json  counts for Figure 1
  * PREREGISTRATION_v3.md  dated template (fill in BEFORE running any training)
  * environment_v3.json    software versions

Usage:  python 00_build_manifest_and_folds.py            (refuses to overwrite a frozen split)
        python 00_build_manifest_and_folds.py --force    (only if you really mean it)
"""
import argparse, glob, json, os, re, sys, zipfile, datetime, platform
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
import config_v3 as C


def parse_name(path):
    m = re.search(r"sub-(OAS\d+)_sess?-d(\d+)(?:_run-(\d+))?", os.path.basename(path))
    if not m:
        raise ValueError(path)
    sid, day, run = m.group(1), m.group(2), m.group(3) or "01"
    return dict(subject_id=sid, day=int(day), run=run, mri_label=f"{sid}_MR_d{day}", file_path=path)


def build_manifest():
    files = sorted(glob.glob(os.path.join(str(C.PREPROC_DIR), "**", "*_preproc.npz"), recursive=True))
    if not files:
        sys.exit(f"No *_preproc.npz under {C.PREPROC_DIR}")
    ok = [f for f in files if zipfile.is_zipfile(f)]
    print(f"npz found {len(files)} | valid {len(ok)} | corrupt {len(files) - len(ok)}")
    rows, bad = [], 0
    for f in ok:
        try:
            rows.append(parse_name(f))
        except ValueError:
            bad += 1
    print("unparseable filenames:", bad)
    man = pd.DataFrame(rows)

    meta = (pd.read_csv(C.SCAN_META_CSV)[["Label", "Age", "M/F"]]
            .rename(columns={"Label": "mri_label", "Age": "age_meta", "M/F": "sex"})
            .drop_duplicates("mri_label"))
    cdr = (pd.read_csv(C.CDR_CSV)[["MRI Label", "age at visit", "CDRTOT", "CDRSUM", "MMSE"]]
           .rename(columns={"MRI Label": "mri_label", "age at visit": "age_cdr"})
           .drop_duplicates("mri_label"))
    man = man.merge(meta, on="mri_label", how="left").merge(cdr, on="mri_label", how="left")
    man["age"] = pd.to_numeric(man["age_cdr"], errors="coerce").fillna(pd.to_numeric(man["age_meta"], errors="coerce"))
    man = man.drop(columns=["age_meta", "age_cdr"])
    return man


def main(force):
    os.makedirs(C.OUT, exist_ok=True)
    if C.FOLDS_CSV.exists() and not force:
        sys.exit(f"{C.FOLDS_CSV} already exists and is FROZEN. Use --force only if you accept invalidating all results.")

    man = build_manifest()
    flow = {"scans_with_npz": int(len(man)), "subjects_with_npz": int(man.subject_id.nunique())}

    # --- Step 1.3: CDR alignment audit (rule used = exact MRI-label match in your precomputed CDR file)
    flow["scans_missing_age"] = int(man.age.isna().sum())
    flow["scans_missing_CDR"] = int(man.CDRTOT.isna().sum())
    flow["sessions_missing_CDR"] = int(man[man.CDRTOT.isna()].mri_label.nunique())
    man = man.dropna(subset=["age"]).reset_index(drop=True)
    # scans with no CDR cannot be labelled CN/impaired -> excluded from training AND longitudinal windows
    man["has_cdr"] = man.CDRTOT.notna()

    # --- Step 1.2: repeat audit (same session, >=2 runs). Runs are distinguished only by run number here.
    runs = man.groupby("mri_label").run.nunique()
    flow["sessions"] = int(len(runs))
    flow["sessions_with_repeat"] = int((runs >= 2).sum())
    flow["sessions_with_3plus_runs"] = int((runs >= 3).sum())
    man["n_runs"] = man.mri_label.map(man.groupby("mri_label").size())

    sess_per_subj = man.groupby("subject_id").mri_label.nunique()
    flow["subjects_3plus_sessions"] = int((sess_per_subj >= 3).sum())
    cn_scans = man[(man.CDRTOT == 0)]
    flow["CN_scans"] = int(len(cn_scans))
    flow["CN_subjects"] = int(cn_scans.subject_id.nunique())
    flow["impaired_scans"] = int((man.CDRTOT > 0).sum())

    # --- Step 2: subject-level stratified folds
    subj = man.sort_values("day").groupby("subject_id").agg(
        age0=("age", "first"), sex=("sex", "first"),
        ever_impaired=("CDRTOT", lambda s: int((s > 0).any())), n_scans=("age", "size")).reset_index()
    subj["band"] = pd.cut(subj.age0, [0, 65, 75, 200], labels=["<65", "65-75", ">75"]).astype(str)
    subj["strat"] = subj.band + "|" + subj.sex.fillna("NA").astype(str) + "|" + subj.ever_impaired.astype(str)
    vc = subj.strat.value_counts()
    small = vc[vc < C.N_FOLDS].index
    subj.loc[subj.strat.isin(small), "strat"] = "other"
    if (subj.strat == "other").sum() and (subj.strat == "other").sum() < C.N_FOLDS:
        subj.loc[subj.strat == "other", "strat"] = vc.index[0]
    skf = StratifiedKFold(C.N_FOLDS, shuffle=True, random_state=C.SEED)
    subj["fold"] = -1
    for k, (_, te) in enumerate(skf.split(subj, subj.strat)):
        subj.loc[subj.index[te], "fold"] = k
    assert (subj.fold >= 0).all()
    subj[["subject_id", "fold", "strat"]].to_csv(C.FOLDS_CSV, index=False)
    man = man.merge(subj[["subject_id", "fold"]], on="subject_id")
    # integrity: no subject in two folds
    assert man.groupby("subject_id").fold.nunique().max() == 1
    man.sort_values(["subject_id", "day", "run"]).to_csv(C.MANIFEST_CSV, index=False)

    # --- shared age-bin scheme from CN ages only
    lo = float(np.floor(cn_scans.age.min())) - 4
    hi = float(np.ceil(cn_scans.age.max())) + 4
    json.dump(dict(bin_min=lo, bin_max=hi, bin_step=C.BIN_STEP, sigma=C.SIGMA, n_bins=int((hi - lo) / C.BIN_STEP)),
              open(C.BINS_JSON, "w"), indent=2)

    flow["fold_subject_counts"] = subj.fold.value_counts().sort_index().to_dict()
    flow["fold_CN_scan_counts"] = man[man.CDRTOT == 0].fold.value_counts().sort_index().to_dict()
    flow["fold_impaired_subject_counts"] = subj[subj.ever_impaired == 1].fold.value_counts().sort_index().to_dict()
    json.dump(flow, open(C.OUT / "cohort_flow_v3.json", "w"), indent=2, default=int)
    print(json.dumps(flow, indent=2, default=int))

    # --- software versions (Step 3.4 / Phase 11 item 8)
    env = dict(python=platform.python_version(), platform=platform.platform(), seed=C.SEED,
               date=str(datetime.datetime.now()))
    for pkg in ("numpy", "pandas", "scipy", "sklearn", "torch", "statsmodels", "lifelines"):
        try:
            env[pkg] = __import__(pkg).__version__
        except Exception:
            env[pkg] = "not installed"
    json.dump(env, open(C.OUT / "environment_v3.json", "w"), indent=2)

    # --- pre-registration template (Step 0.3)
    pr = C.OUT / "PREREGISTRATION_v3.md"
    if not pr.exists():
        pr.write_text(f"""# Pre-registration (v3) -- written {datetime.date.today()} BEFORE training v3 models

NOTE: v2 test results were already seen. v3 design choices were therefore made on cross-validation /
validation data only; state this honestly in the paper.

1. Primary accuracy endpoint: out-of-fold MAE on CN scans (cross-fitted bias correction), with subject-cluster bootstrap CI.
2. Primary reliability endpoint: pooled within-session sigma_SEM and MDC, subject-cluster bootstrap CI (1000 resamples).
3. Fixed model set (do not change after seeing survival results): m0 (v2-style), m1 (v3 primary), m2 (m1 + consistency loss). Ensemble = seeds within fold.
4. Baseline window: first k=3 sessions, all CDR 0, >=1 later session. Minimum window span: ____ years (primary) / sensitivity ____.
5. Primary clinical endpoint rule: >=30 conversion events -> survival primary; otherwise continuous (CDR-SB/MMSE slope) primary.
6. Primary comparison: paired out-of-bag C-index difference, shrunk-slope vs raw-slope Cox model (95% bootstrap CI).
7. Covariates in floor model: baseline age, sex, baseline BAG.
8. Exclusion rules (image QC only, never error-based): ____
9. Known limitations to state: few converters, interval censoring, single cohort, affine-only registration.
""")
    print("\nFrozen folds saved to", C.FOLDS_CSV, "-- never regenerate.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
