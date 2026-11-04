# -*- coding: utf-8 -*-
"""
v5_optuna.py -- hyper-parameter search with a FIXED validation protocol (round 2, only if round 1 justifies it).

Fixes two protocol defects found in V4's 12_optuna_search.py (see report):
  1. V4 re-drew the inner train/val subjects for every trial (seed = SEED + fold + trial.number), so trial
     scores were measured on different validation sets and the 'best' trial was partly the luckiest split.
     Here every trial uses the SAME inner split (v5_train.train_one -> V4's split_train_val, seed SEED+fold).
  2. V4 used MedianPruner(n_warmup_steps=3), which systematically kills slow-converging (low-LR) configs.
     Here pruning is OFF by default (NopPruner); --prune enables a gentle Hyperband for budget-limited runs.
  Also: trials run the SAME training loop as final models (warm-up, ReduceLROnPlateau, EMA), the study is stored in
  SQLite so it can be interrupted and resumed, the sampler is seeded, and failures are stored with their reason.

NESTING CAVEAT (important):  a config tuned with outer fold k's TRAINING folds may only be evaluated on outer fold k.
Evaluating it on the other folds would reuse subjects that influenced the search.  To report a 5-fold number for a
tuned config, run --outer_fold k for each k (5x the cost) and evaluate each tuned config on its own outer fold.

Usage:
    python v5_optuna.py --experiment v5_z_khuber --outer_fold 0 --n_trials 25 --trial_epochs 20
    python v5_train.py --experiment v5_z_khuber --name v5_z_khuber_opt_f0 --folds 0 --seeds 2 --set <printed params>
"""
import argparse
import json

import optuna
import torch

import config_v3 as C3
import config_v5 as C5
import core_v3 as K
import v5_train as T5


def sample(trial, space):
    out = {}
    for name, spec in space.items():
        if spec[0] == "loguniform":
            out[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif spec[0] == "uniform":
            out[name] = trial.suggest_float(name, spec[1], spec[2])
        elif spec[0] == "categorical":
            out[name] = trial.suggest_categorical(name, spec[1])
        else:
            raise ValueError(spec[0])
    return out


def make_objective(base_cfg, outer_fold, man, store, bins, dev, args, build_model_fn=None):
    def objective(trial):
        cfg = dict(base_cfg, **sample(trial, C5.OPTUNA_SPACE))
        s = T5.train_one(f"optuna_t{trial.number}", cfg, outer_fold, args.seed, man, store, bins, dev,
                         epochs=args.trial_epochs, patience=args.trial_epochs + 1,     # no early stop inside a trial
                         workers=args.workers, cv=args.cv, save=False, build_model_fn=build_model_fn)
        trial.set_user_attr("best_val_epoch", s["best_val_epoch"])
        trial.set_user_attr("runtime_seconds", s["runtime_seconds"])
        return s["best_val_mae_raw"]          # RAW-weights validation MAE: the same selection metric as final training
    return objective


def run_study(base_cfg, outer_fold, man, store, bins, dev, args, build_model_fn=None):
    C5.OPTUNA_DIR.mkdir(parents=True, exist_ok=True)       # config_v5 only creates it at import time (cwd-dependent)
    pruner = optuna.pruners.HyperbandPruner() if args.prune else optuna.pruners.NopPruner()
    study = optuna.create_study(
        study_name=f"{args.experiment}_outer{outer_fold}", direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=C3.SEED + outer_fold), pruner=pruner,
        storage=f"sqlite:///{C5.OPTUNA_DIR / (args.experiment + f'_outer{outer_fold}.db')}", load_if_exists=True)
    remaining = max(0, args.n_trials - len([t for t in study.trials if t.state.is_finished()]))
    study.optimize(make_objective(base_cfg, outer_fold, man, store, bins, dev, args, build_model_fn),
                   n_trials=remaining, catch=(RuntimeError,))      # a failed trial (e.g. OOM) is stored, search continues
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    out = dict(experiment=args.experiment, outer_fold=outer_fold, n_trials_budget=args.n_trials,
               trial_epochs=args.trial_epochs, inner_validation="fixed (same subjects for every trial)",
               best_value_val_mae=study.best_value if done else None, best_params=study.best_params if done else None,
               trials=[dict(number=t.number, value=t.value, params=t.params, state=str(t.state), user_attrs=t.user_attrs)
                       for t in study.trials])
    path = C5.OPTUNA_DIR / f"optuna_{args.experiment}_outer{outer_fold}.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print("saved", path)
    if done:
        sets = " ".join(f"--set {k}={json.dumps(v)}" for k, v in study.best_params.items())
        print(f"best inner-val MAE {study.best_value:.3f}  (NOT a held-out result)\nretrain: python v5_train.py --experiment "
              f"{args.experiment} --name {args.experiment}_opt_f{outer_fold} --folds {outer_fold} --seeds 2 {sets}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="v5_z_khuber", choices=list(C5.EXPERIMENTS))
    ap.add_argument("--outer_fold", type=int, default=0)
    ap.add_argument("--n_trials", type=int, default=25, help="total budget, fixed BEFORE starting; resumes if the DB exists")
    ap.add_argument("--trial_epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cv", type=float, default=C3.MEANNORM_CV)
    ap.add_argument("--prune", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    man, bins = T5.load_manifest_for_training(), K.load_bins(C3.BINS_JSON)
    run_study(dict(C5.EXPERIMENTS[args.experiment]), args.outer_fold, man, K.ScanStore(), bins, dev, args)


if __name__ == "__main__":
    main()
