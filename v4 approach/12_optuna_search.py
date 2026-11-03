# -*- coding: utf-8 -*-
"""
12_optuna_search.py -- V4 Phase 4.

Nested design, exactly as v4_approach.txt specifies:
  - Frozen subject-level OUTER folds (config_v3's 5 folds) -- each one reserved
    for final out-of-fold evaluation and NEVER seen by Optuna.
  - INNER search: for a given outer fold k, Optuna samples hyperparameters and
    trains only on subjects from the other folds, with its own inner
    subject-level train/val split (same split function as 11_train_experiment,
    different seed per trial so trials aren't all evaluated on the same slice).
  - Best trial's hyperparameters are then handed to 11_train_experiment.py to
    retrain on the full permitted-training set and produce the real held-out
    prediction for fold k, via 10_eval_canonical.py. This script does NOT
    itself produce the number that goes in Table A -- it only picks the config.

Two modes:
    --mode debug : ~30-50 trials on ONE outer fold (default: fold 0), to debug
                   the search space before spending the full budget. This is
                   the default, matching "for a practical first run, use about
                   30-50 trials on a single outer fold to debug the search."
    --mode full  : repeats the (smaller) search across all 5 outer folds.

Requires `pip install optuna`.

Usage:
    python 12_optuna_search.py --base_experiment v4_sfcn_baseline --mode debug
    python 12_optuna_search.py --base_experiment v4_sfcn_baseline --mode full --n_trials 20
"""
import argparse
import json

import numpy as np
import pandas as pd
import optuna
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

import config_v3 as C3
import config_v4 as C4
import core_v3 as K
import core_v4 as K4


def sample_params(trial, space):
    out = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "loguniform":
            out[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif kind == "uniform":
            out[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "categorical":
            out[name] = trial.suggest_categorical(name, spec[1])
        else:
            raise ValueError(kind)
    return out


def split_train_val(df_train_cn, seed, frac):
    subj = np.array(sorted(df_train_cn.subject_id.unique()))
    rng = np.random.default_rng(seed)
    val_s = set(rng.choice(subj, size=max(5, int(len(subj) * frac)), replace=False))
    is_val = df_train_cn.subject_id.isin(val_s)
    return df_train_cn[~is_val].reset_index(drop=True), df_train_cn[is_val].reset_index(drop=True)


@torch.no_grad()
def quick_eval(model, loader, cfg, centers, dev, cv):
    model.eval()
    err, n = 0.0, 0
    for x, age in loader:
        x, age = x.to(dev, non_blocking=True), age.to(dev)
        x = K.to_input(x, cfg["input_mode"], cv)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            lp = model(x)
        err += (K.expected_age(lp, centers) - age).abs().sum().item()
        n += len(age)
    return err / max(n, 1)


def objective(trial, base_cfg, outer_fold, man, store, bins, dev, args):
    sampled = sample_params(trial, C4.OPTUNA_SEARCH_SPACE)
    cfg = dict(base_cfg, **sampled)

    cn_train_pool = man[(man.fold != outer_fold) & (man.CDRTOT == 0)]
    tr, va = split_train_val(cn_train_pool, C3.SEED + outer_fold + trial.number, C3.VAL_FRACTION)
    assert not (set(tr.subject_id) & set(va.subject_id))
    assert (tr.fold != outer_fold).all() and (va.fold != outer_fold).all(), "outer fold leaked into inner search"
    store.add(pd.concat([tr, va]).file_path)

    centers = K.bin_centers(bins, dev)
    model, trainable = K4.build_model(cfg, bins, dev)
    opt = K4.build_optimizer(model, trainable, cfg)

    train_ds, val_ds = K.ScanDS(store, tr), K.ScanDS(store, va)
    if cfg.get("reweight_tail"):
        sampler = WeightedRandomSampler(K.tail_weights(tr.age.values), num_samples=len(tr), replacement=True)
        tl = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler, num_workers=2, drop_last=False)
    else:
        tl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=2, drop_last=False)
    vl = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

    best = 1e9
    for ep in range(args.trial_epochs):
        model.train()
        for x, age in tl:
            x, age = x.to(dev, non_blocking=True), age.to(dev)
            x = K.gpu_augment(K.to_input(x, cfg["input_mode"], args.cv))
            y = K.soft_labels(age, bins, dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                lp = model(x)
            ea = K.expected_age(lp, centers)
            loss = K4.compute_loss(lp, y, ea, age, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        val_mae = quick_eval(model, vl, cfg, centers, dev, args.cv)
        best = min(best, val_mae)
        trial.report(val_mae, ep)
        if trial.should_prune():
            del model, opt
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()

    del model, opt
    torch.cuda.empty_cache()
    return best


def run_fold_search(base_cfg, outer_fold, man, bins, dev, args):
    store = K.ScanStore()
    study = optuna.create_study(direction="minimize",
                                 pruner=optuna.pruners.MedianPruner(n_warmup_steps=3))
    study.optimize(lambda t: objective(t, base_cfg, outer_fold, man, store, bins, dev, args),
                    n_trials=args.n_trials, gc_after_trial=True)

    best = dict(base_cfg, **study.best_params)
    out = dict(outer_fold=outer_fold, n_trials=args.n_trials, trial_epochs=args.trial_epochs,
               best_value_val_mae=study.best_value, best_params=study.best_params, best_full_config=best,
               all_trials=[dict(number=t.number, value=t.value, params=t.params, state=str(t.state))
                           for t in study.trials])
    path = C4.OPTUNA_DIR / f"optuna_{args.base_experiment}_fold{outer_fold}.json"
    json.dump(out, open(path, "w"), indent=2, default=str)
    print(f"\nFold {outer_fold}: best val MAE {study.best_value:.3f} | best params {study.best_params}")
    print(f"Saved {path}")
    print(f"Next: add this config to config_v4.EXPERIMENTS (e.g. '{args.base_experiment}_optuna_f{outer_fold}') "
          f"and run 11_train_experiment.py --experiment <name> --folds {outer_fold} --force "
          f"to retrain on the full permitted set, THEN 10_eval_canonical.py to get the real held-out number. "
          f"This script's val MAE is an inner-search number -- it must never go into Table A directly.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_experiment", required=True, choices=list(C4.EXPERIMENTS),
                     help="fixed fields (arch, activation unless searched, loss type, etc.) come from this row; "
                          "OPTUNA_SEARCH_SPACE fields get overridden by the search")
    ap.add_argument("--mode", default="debug", choices=["debug", "full"])
    ap.add_argument("--n_trials", type=int, default=None)
    ap.add_argument("--trial_epochs", type=int, default=20, help="inner-search training budget; final models use the longer schedule")
    ap.add_argument("--cv", type=float, default=C3.MEANNORM_CV)
    args = ap.parse_args()
    if args.n_trials is None:
        args.n_trials = C4.OPTUNA_N_TRIALS_DEBUG if args.mode == "debug" else C4.OPTUNA_N_TRIALS_FULL

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    man = pd.read_csv(C3.MANIFEST_CSV)
    bins = K.load_bins(C3.BINS_JSON)
    qc = pd.read_csv(C3.QC_CSV) if C3.QC_CSV.exists() else None
    qc_bad = K4.qc_bad_table(qc)
    if qc_bad is not None:
        man = man.merge(qc_bad, on="file_path", how="left")
        man["qc_bad"] = man["qc_bad"].fillna(False).astype(bool)
        man = man[~man.qc_bad].copy()

    base_cfg = C4.EXPERIMENTS[args.base_experiment]
    folds = [C4.OPTUNA_DEBUG_FOLD] if args.mode == "debug" else list(range(C3.N_FOLDS))
    for k in folds:
        run_fold_search(base_cfg, k, man, bins, dev, args)


if __name__ == "__main__":
    main()
