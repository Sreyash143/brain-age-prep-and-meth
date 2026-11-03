"""Tests for the fixed-protocol Optuna wrapper (tiny stand-in network, CPU)."""
import importlib
import json
from argparse import Namespace

import pytest

torch = pytest.importorskip("torch")
optuna = pytest.importorskip("optuna")
pytest.importorskip("core_v3")

from test_v5_train import BINS, FakeStore, Tiny, cfg, manifest   # noqa: E402


@pytest.fixture()
def V(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return {n: importlib.import_module(n) for n in ("config_v3", "config_v4", "config_v5", "v5_train", "v5_optuna")}


def test_every_trial_uses_the_same_validation_subjects_and_study_is_resumable(V, tmp_path, monkeypatch):
    O, C5 = V["v5_optuna"], V["config_v5"]
    seen = []
    T5 = V["v5_train"]
    real = T5.T.split_train_val

    def spy(pool, seed, frac):
        a, b = real(pool, seed, frac)
        seen.append(tuple(sorted(b.subject_id.unique())))
        return a, b

    monkeypatch.setattr(T5.T, "split_train_val", spy)
    monkeypatch.setattr(C5, "OPTUNA_SPACE", dict(lr_head=("loguniform", 1e-3, 1e-2), batch_size=("categorical", [2, 4])))
    monkeypatch.setattr(T5.K4, "build_model", lambda c, b, d: (Tiny().to(d), [n for n, _ in Tiny().named_parameters()]))
    args = Namespace(experiment="v5_z_khuber", n_trials=3, trial_epochs=2, seed=0, workers=0, cv=0.35, prune=False)
    base = cfg(V, ema_decay=0.9)
    out = O.run_study(base, 0, manifest(), FakeStore(), BINS, torch.device("cpu"), args)
    assert len(out["trials"]) == 3 and len(set(seen)) == 1, "validation subjects differed between trials"
    assert all(t["state"].endswith("COMPLETE") for t in out["trials"])
    assert not any((tmp_path / "v5" / "checkpoints").glob("optuna_*")), "trials must not write checkpoints"
    args.n_trials = 4                                           # resume: only ONE additional trial may run
    out2 = O.run_study(base, 0, manifest(), FakeStore(), BINS, torch.device("cpu"), args)
    assert len(out2["trials"]) == 4
    assert json.loads((C5.OPTUNA_DIR / "optuna_v5_z_khuber_outer0.json").read_text())["inner_validation"].startswith("fixed")
