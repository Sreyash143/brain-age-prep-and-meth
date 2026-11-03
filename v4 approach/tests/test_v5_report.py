"""Synthetic-data tests for v5_report.py (no torch, no OASIS data)."""
import numpy as np
import pandas as pd
import pytest

import v5_report as R


def make_raw(n_subj=120, seed=0, noise=3.0, shrink=0.8, scans=(1, 3)):
    """Synthetic out-of-fold predictions: pred = shrink*age + c + noise, folds assigned PER SUBJECT."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subj):
        age0 = rng.uniform(55, 90)
        fold = s % 5
        for v in range(rng.integers(scans[0], scans[1] + 1)):
            age = age0 + v * 1.5
            for run in (1, 2):
                rows.append(dict(subject_id=f"S{s:03d}", day=int(v * 500), run=run, mri_label=f"S{s:03d}_MR_d{v*500}",
                                 file_path=f"/x/S{s:03d}_{v}_{run}.npz", age=age, CDRTOT=0.0, fold=fold,
                                 sex="F" if s % 2 else "M",
                                 predicted_age=shrink * age + (1 - shrink) * 72 + rng.normal(0, noise)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- validation
def test_validate_clean_file_passes():
    res = R.validate(make_raw(), "t")
    assert all(r["status"] in ("PASS", "WARN") for r in res)


def test_validate_detects_subject_in_two_folds():
    df = make_raw()
    df.loc[df.subject_id == "S000", "fold"] = [0] + [1] * ((df.subject_id == "S000").sum() - 1)
    res = {r["check"]: r["status"] for r in R.validate(df, "t")}
    assert res["every subject belongs to exactly one fold (no repeat-scan leakage)"] == "FAIL"


def test_validate_detects_duplicate_rows_and_missing_columns():
    df = make_raw()
    dup = pd.concat([df, df.iloc[:2]])
    assert {r["check"]: r["status"] for r in R.validate(dup, "t")}["file_path is unique (one row per scan)"] == "FAIL"
    assert R.validate(df.drop(columns=["age"]), "t")[0]["status"] == "FAIL"


def test_validate_reports_key_collisions_as_warning():
    df = make_raw()
    df.loc[1, ["subject_id", "day", "run"]] = df.loc[0, ["subject_id", "day", "run"]].values
    st = {r["check"]: r["status"] for r in R.validate(df, "t")}
    assert st["(subject,day,run) key collisions [informational]"] == "WARN"


# ---------------------------------------------------------------- bias correction
def test_crossfit_never_uses_heldout_labels():
    """Changing the held-out fold's labels must not change ITS correction parameters."""
    df = make_raw()
    _, p0 = R.crossfit_correct(df)
    df2 = df.copy()
    m = df2.fold == 2
    df2.loc[m, "age"] = df2.loc[m, "age"] + 25          # corrupt ONLY fold-2 labels
    _, p1 = R.crossfit_correct(df2)
    assert p0[2] == p1[2]                                # fold 2's own params unchanged
    assert p0[0] != p1[0]                                # other folds legitimately use fold 2


def test_crossfit_inverts_known_shrinkage():
    df = make_raw(n_subj=400, noise=0.5, shrink=0.8)
    out, params = R.crossfit_correct(df)
    assert all(abs(p["slope"] - 0.8) < 0.03 for p in params.values())
    slope = np.polyfit(out.age, out.brain_age - out.age, 1)[0]
    assert abs(slope) < 0.03                              # age-bias removed


def test_correction_can_raise_mae_and_both_are_reported_separately():
    """Documents V4's behaviour: inverting a shrinkage slope amplifies noise -> corrected MAE > raw MAE."""
    df = make_raw(n_subj=300, noise=3.0, shrink=0.8)
    out, _ = R.crossfit_correct(df)
    raw = (out.predicted_age - out.age).abs().mean()
    cor = (out.brain_age - out.age).abs().mean()
    assert cor > raw
    m = R.err_metrics(out, "predicted_age", 200), R.err_metrics(out, "brain_age", 200)
    assert m[0]["MAE_pooled"] == pytest.approx(raw) and m[1]["MAE_pooled"] == pytest.approx(cor)


# ---------------------------------------------------------------- metrics definitions
def test_r2_is_sse_over_sst_not_squared_correlation():
    df = make_raw(n_subj=200, shrink=0.8)
    df["brain_age"] = df.predicted_age                       # raw column reused
    m = R.err_metrics(df, "brain_age", 100)
    e = df.brain_age - df.age
    assert m["R2_SSE_over_SST"] == pytest.approx(1 - (e ** 2).sum() / ((df.age - df.age.mean()) ** 2).sum())
    assert m["R2_SSE_over_SST"] != pytest.approx(m["pearson_r"] ** 2, abs=1e-3)   # biased preds => differs


def test_fold_mean_mae_differs_from_pooled_when_folds_unbalanced():
    df = make_raw(n_subj=100)
    df = df[~((df.fold == 0) & (df.index % 2 == 0))]          # unbalance fold 0
    df["brain_age"] = df.predicted_age
    m = R.err_metrics(df, "brain_age", 100)
    assert m["MAE_fold_mean"] == pytest.approx(np.mean(list(m["MAE_by_fold"].values())))


def test_cluster_bootstrap_is_wider_than_naive_when_scans_are_correlated():
    """Strongly subject-correlated errors: a scan-level bootstrap would be too narrow; cluster CI must be wider."""
    rng = np.random.default_rng(1)
    subj = np.repeat(np.arange(60), 8)
    err = np.repeat(rng.normal(3, 2, 60), 8) + rng.normal(0, .1, 480)
    s, c, _ = R.subject_sums(pd.Series(subj), np.abs(err))
    ci_cluster, _ = R.boot_ratio_ci(s, c, 2000)
    s2, c2, _ = R.subject_sums(pd.Series(np.arange(480)), np.abs(err))
    ci_naive, _ = R.boot_ratio_ci(s2, c2, 2000)
    assert (ci_cluster[1] - ci_cluster[0]) > 1.8 * (ci_naive[1] - ci_naive[0])


def test_paired_delta_recovers_known_difference():
    base = make_raw(n_subj=250, seed=3, noise=3.0)
    base, _ = R.crossfit_correct(base)
    other = base.copy()
    other["predicted_age"] = other.age + (other.predicted_age - other.age) * 0.8    # 20% smaller errors
    a, b = other.set_index("file_path"), base.set_index("file_path")
    a["err"], b["err"] = (a.predicted_age - a.age).abs(), (b.predicted_age - b.age).abs()
    d = R.paired_delta(a, b.loc[a.index], 2000)
    assert d["delta_MAE"] < 0 and d["CI95"][1] < 0 and d["p_delta_lt_0"] > 0.99
    assert d["n_subjects"] == 250


# ---------------------------------------------------------------- QC rules
def test_qc_rule_matches_v4_semantics():
    qc = pd.DataFrame(dict(file_path=list("abcd"), qc_flag=[1, 1, 0, 0],
                           qc_manual_review=["", "LOOKS-OK", "CONFIRMED-FAILURE", ""]))
    assert R.qc_bad_paths(qc) == {"a", "c"}                 # b overridden to OK, c confirmed failure
    assert R.qc_bad_paths(qc.drop(columns="qc_manual_review")) == {"a", "b"}


def test_empty_qc_flag_column_treated_as_good():
    assert R.qc_bad_paths(pd.DataFrame(dict(file_path=["a"]))) == set()


# ---------------------------------------------------------------- end-to-end
def _write(tmp_path, tag, df):
    d = tmp_path / "p"
    d.mkdir(exist_ok=True)
    df.to_csv(d / f"raw_{tag}.csv", index=False)
    return d


def test_end_to_end_and_comparable_cohort_enforced(tmp_path):
    a = make_raw(n_subj=100, seed=1)
    b = a.copy()
    b["predicted_age"] = b.age + (b.predicted_age - b.age) * 0.9
    _write(tmp_path, "A", a)
    d = _write(tmp_path, "B", b)
    qc = pd.DataFrame(dict(file_path=a.file_path, qc_flag=0))
    qc.to_csv(tmp_path / "qc.csv", index=False)
    rc = R.main(["--tags", "A,B", "--baseline", "A", "--pred_dir", str(d), "--qc_csv", str(tmp_path / "qc.csv"),
                 "--out_dir", str(tmp_path / "o"), "--boot", "300"])
    assert rc == 0 and (tmp_path / "o" / "comparison.json").exists()
    # now make B cover a different scan set -> must FAIL (no silent cross-cohort comparison)
    _write(tmp_path, "B", b.iloc[:-10])
    rc = R.main(["--tags", "A,B", "--baseline", "A", "--pred_dir", str(d), "--qc_csv", str(tmp_path / "qc.csv"),
                 "--out_dir", str(tmp_path / "o2"), "--boot", "300"])
    assert rc == 1 and not (tmp_path / "o2" / "comparison.json").exists()


def test_fold_subset_still_fits_correction_on_all_other_folds(tmp_path):
    a = make_raw(n_subj=100, seed=5)
    d = _write(tmp_path, "A", a)
    pd.DataFrame(dict(file_path=a.file_path, qc_flag=0)).to_csv(tmp_path / "qc.csv", index=False)
    R.main(["--tags", "A", "--baseline", "A", "--pred_dir", str(d), "--qc_csv", str(tmp_path / "qc.csv"),
            "--out_dir", str(tmp_path / "o"), "--boot", "100", "--folds", "0"])
    import json
    j = json.load(open(tmp_path / "o" / "comparison.json"))
    r = j["populations"]["qc_passed"]["A"]
    assert set(r["raw"]["MAE_by_fold"]) == {"0"} and set(r["bias_params_per_fold"]) == {"0", "1", "2", "3", "4"}


def test_empty_population_does_not_crash_validation():
    df = make_raw().iloc[0:0]
    res = R.validate(df, "t")
    assert any(r["status"] == "FAIL" for r in res)          # folds 0..4 missing is flagged
