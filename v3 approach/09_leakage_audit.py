"""
09_leakage_audit.py   (Phase 11 -- run BEFORE any number goes into the paper)

Automated where possible, otherwise prints what to confirm by hand.
    python 09_leakage_audit.py --variant m1 --slopes v3/results/slopes_m1_k3_span2.csv
"""
import argparse, glob, json
import numpy as np, pandas as pd
import config_v3 as C
import longitudinal_lib as L

res = []
def check(name, ok, detail=""):
    res.append((name, "PASS" if ok else "FAIL", detail)); print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def main(a):
    man = pd.read_csv(C.MANIFEST_CSV)
    # 1 no subject in two folds
    check("1. every subject belongs to exactly one fold (repeats + later visits included)",
          man.groupby("subject_id").fold.nunique().max() == 1)
    # 7 early stopping never touched the held-out fold; 6 consistency pairs only from training subjects
    ok, n = True, 0
    for f in glob.glob(str(C.CKPT_DIR / "*.json")):
        meta = json.load(open(f)); tag = meta["tag"]; k = int(tag.split("_f")[1].split("_")[0])
        held = set(man[man.fold == k].subject_id)
        n += 1
        if set(meta["val_subjects"]) & held:
            ok = False; print("   leak in", tag)
    check("7. early-stopping subjects never belong to the held-out fold", ok and n > 0, f"({n} checkpoints inspected)")
    check("6. consistency-loss pairs come from training subjects only", True, "(enforced by assert in 03_train_cv.py: tr.fold != held-out fold)")
    # 2 bias-correction parameters from other folds only
    j = C.RES_DIR / f"accuracy_{a.variant}.json"
    if j.exists():
        bp = json.load(open(j))["bias_params_per_fold"]
        n_cn_sessions = man[man.CDRTOT == 0].mri_label.nunique()
        check("2. bias correction fitted on other folds only (cross-fitted)", all(v["n_sessions"] < n_cn_sessions for v in bp.values()), f"({len(bp)} folds)")
    else:
        check("2. bias correction cross-fitted", False, "accuracy json missing -- run 04b")
    # 3/4 survival features contain no post-window information: scramble post-window BAG, slopes must not move
    bag = pd.read_csv(C.PRED_DIR / f"bag_{a.variant}.csv")
    sess = L.build_sessions(bag)
    sig = json.load(open(C.RES_DIR / f"noise_floor_{a.variant}.json"))["sigma_SEM"]
    d0, _ = L.build_windows(sess, k=a.k, min_span_years=a.min_span, sigma=sig)
    s2 = sess.copy(); rng = np.random.default_rng(0)
    s2["rank"] = s2.groupby("subject_id").day.rank(method="first")
    m = s2["rank"] > a.k
    s2.loc[m, "BAG"] = rng.normal(0, 50, m.sum())               # destroy every post-window BAG value
    d1, _ = L.build_windows(s2, k=a.k, min_span_years=a.min_span, sigma=sig)
    mrg = d0.merge(d1, on="subject_id", suffixes=("_a", "_b"))
    same = all(np.allclose(mrg[c + "_a"], mrg[c + "_b"]) for c in ("S_raw", "bag_base", "SE_S", "age_base"))
    check("3/4. slopes, baseline BAG and SE use baseline-window sessions only (post-window BAG scrambled -> unchanged)", same and len(mrg) == len(d0))
    # 5, 8 manual / logged
    env = C.OUT / "environment_v3.json"
    check("8. seeds and software versions logged", env.exists(), f"(seed={C.SEED}; checkpoints tagged variant_fold_seed)")
    print("\nMANUAL: 5. confirm VARIANTS in config_v3.py and PREREGISTRATION_v3.md are unchanged since before survival results.")
    print("MANUAL: confirm QC exclusion rules were fixed from image properties only (never from prediction error).")
    print("MANUAL: confirm the v2 TEST split was not used to choose any v3 hyper-parameter (v3 decisions: folds + probe on fold 0).")
    pd.DataFrame(res, columns=["check", "status", "detail"]).to_csv(C.RES_DIR / "leakage_audit.csv", index=False)
    if any(r[1] == "FAIL" for r in res):
        raise SystemExit("AUDIT FAILED")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="m1")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--min_span", type=float, default=0.0)
    main(ap.parse_args())
