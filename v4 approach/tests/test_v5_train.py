"""CPU smoke tests for v5_train.py using a tiny stand-in network and fake scan store.
Needs torch and the V4 project files on the path (see conftest.py).  Does NOT need OASIS data or the UKB checkpoint."""
import importlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F

pytest.importorskip("core_v3", reason="V4 project files not on sys.path (set V4_PROJECT_DIR)")

BINS = dict(bin_min=38.0, bin_max=100.0, bin_step=1.0, sigma=1.0, n_bins=62)


class Tiny(nn.Module):
    def __init__(self, n=62, boom_after=None):
        super().__init__()
        self.c, self.bn, self.fc = nn.Conv3d(1, 4, 3, padding=1), nn.BatchNorm3d(4), nn.Linear(4, n)
        self.calls, self.boom_after = 0, boom_after

    def forward(self, x):
        self.calls += 1
        if self.boom_after and self.calls > self.boom_after:
            raise RuntimeError("injected failure")
        return F.log_softmax(self.fc(F.relu(self.bn(self.c(x))).mean((2, 3, 4))), 1)


class FakeStore:
    arr = {}

    def add(self, paths, desc=""):
        pass

    def get(self, p):
        g = torch.Generator().manual_seed(abs(hash(p)) % (2 ** 31))
        return torch.randn(1, 8, 8, 8, generator=g)


@pytest.fixture()
def V(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                      # config_v4 / config_v5 create ./v4 ./v5 at import
    mods = {n: importlib.import_module(n) for n in ("config_v3", "config_v4", "config_v5", "v5_train")}
    return mods


def cfg(V, **kw):
    c = dict(arch="tiny", activation="relu", input_mode="zscore", unfreeze="full", llrd=1.0, lr_head=1e-2,
             lr_backbone=1e-2, optimizer="adamw", weight_decay=1e-4, dropout=0.0, loss="kl_huber", l1_weight=0.0,
             huber_delta=1.0, reweight_tail=True, consistency=0.0, batch_size=4, ema_decay=0.9)
    c.update(kw)
    return c


def manifest(n_subj=60, scans=2):
    rows = []
    for s in range(n_subj):
        for v in range(scans):
            rows.append(dict(subject_id=f"S{s:03d}", fold=s % 5, CDRTOT=0.0, age=60 + (s * 7 % 30) + v,
                             file_path=f"/x/{s}_{v}.npz", mri_label=f"S{s}_{v}"))
    return pd.DataFrame(rows)


def build(c, bins, dev, boom_after=None):
    m = Tiny(boom_after=boom_after).to(dev)
    return m, [n for n, _ in m.named_parameters()]


def run(V, tmp_path, name="exp", fold=0, seed=0, epochs=3, patience=50, boom_after=None, **cfg_kw):
    T = V["v5_train"]
    return T.train_one(name, cfg(V, **cfg_kw), fold, seed, manifest(), FakeStore(), BINS, torch.device("cpu"),
                       epochs=epochs, patience=patience, workers=0, cv=0.35, ckpt_dir=tmp_path / "ck",
                       build_model_fn=lambda c, b, d: build(c, b, d, boom_after))


def test_ema_update_semantics(V):
    T = V["v5_train"]
    m = Tiny()
    ema = T.EMA(m, 0.9)
    before = [p.clone() for p in ema.model.parameters()]
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    ema.update(m, step=10_000)                               # warm-up factor ~1 -> plain decay 0.9
    for e, b, p in zip(ema.model.parameters(), before, m.parameters()):
        assert torch.allclose(e, 0.9 * b + 0.1 * p.detach(), atol=1e-6)
    assert not any(p.requires_grad for p in ema.model.parameters())
    # BN buffers are copied from the live model
    for eb, b in zip(ema.model.buffers(), m.buffers()):
        assert torch.equal(eb, b)


def test_train_one_writes_expected_files_and_registry(V, tmp_path):
    s = run(V, tmp_path)
    ck = tmp_path / "ck"
    for f in ("exp_f0_s0.pt", "exp_f0_s0.ema.pt", "exp_f0_s0.json", "exp_f0_s0.resume.pt"):
        assert (ck / f).exists(), f
    meta = json.load(open(ck / "exp_f0_s0.json"))
    held = set(manifest().query("fold == 0").subject_id)
    assert not (set(meta["val_subjects"]) & held), "held-out subjects used for early stopping"
    assert meta["completed"] and meta["best_val_mae_raw"] == s["best_val_mae_raw"]
    reg = pd.read_csv(V["config_v5"].REGISTRY_CSV)
    assert len(reg) == 1 and reg.status[0] == "completed" and reg.seed[0] == 0 and reg.fold[0] == 0
    assert json.loads(reg.hyperparameters_json[0])["loss"] == "kl_huber"
    sd = torch.load(ck / "exp_f0_s0.ema.pt", map_location="cpu")        # plain state_dict, loadable
    Tiny().load_state_dict(sd)


def test_completed_run_is_skipped_not_overwritten(V, tmp_path):
    run(V, tmp_path)
    mtime = (tmp_path / "ck" / "exp_f0_s0.pt").stat().st_mtime_ns
    assert run(V, tmp_path)["status"] == "skipped"
    assert (tmp_path / "ck" / "exp_f0_s0.pt").stat().st_mtime_ns == mtime


def test_failure_is_recorded_and_run_can_resume(V, tmp_path):
    # Tiny.forward is called for train batches AND val batches each epoch; fail during epoch 2
    with pytest.raises(RuntimeError, match="injected"):
        run(V, tmp_path, epochs=4, boom_after=40)
    reg = pd.read_csv(V["config_v5"].REGISTRY_CSV)
    assert reg.status.iloc[-1] == "failed" and "injected" in reg.failure_reason.iloc[-1]
    assert not (tmp_path / "ck" / "exp_f0_s0.json").exists()            # no completion marker after failure
    st = torch.load(tmp_path / "ck" / "exp_f0_s0.resume.pt", map_location="cpu", weights_only=False)
    done = st["next_epoch"]
    assert 1 <= done < 4
    s = run(V, tmp_path, epochs=4)                                       # resume, no failure this time
    meta = json.load(open(tmp_path / "ck" / "exp_f0_s0.json"))
    assert s["status"] == "completed" and [h["epoch"] for h in meta["history"]] == [1, 2, 3, 4]


def test_ema_disabled_creates_no_ema_checkpoint(V, tmp_path):
    run(V, tmp_path, ema_decay=None)
    assert not (tmp_path / "ck" / "exp_f0_s0.ema.pt").exists()


def test_val_split_identical_across_seeds_and_matches_v4_function(V, tmp_path):
    """Same inner split for every seed/experiment => selection noise is not a confound between runs."""
    T4 = importlib.import_module("11_train_experiment")
    C3 = V["config_v3"]
    pool = manifest().query("fold != 0 and CDRTOT == 0")
    a = T4.split_train_val(pool, C3.SEED + 0, C3.VAL_FRACTION)[1].subject_id.unique()
    run(V, tmp_path, seed=0)
    run(V, tmp_path, seed=1)
    for s in (0, 1):
        meta = json.load(open(tmp_path / "ck" / f"exp_f0_s{s}.json"))
        assert sorted(a) == meta["val_subjects"]


def test_cli_rejects_v4_name_collision_and_set_without_name(V, monkeypatch):
    T = V["v5_train"]
    for argv in (["--experiment", "v5_z_khuber", "--name", "v4_sfcn_baseline"],
                 ["--experiment", "v5_z_khuber", "--set", "lr_head=1e-3"]):
        monkeypatch.setattr("sys.argv", ["v5_train.py"] + argv)
        with pytest.raises(SystemExit) as e:
            T.main()
        assert e.value.code == 2


def test_v5_output_tree_is_isolated_from_v3_v4(V):
    C5, C4, C3 = V["config_v5"], V["config_v4"], V["config_v3"]
    for p in (C5.CKPT_DIR, C5.PRED_DIR, C5.RES_DIR, C5.OPTUNA_DIR, C5.REGISTRY_CSV):
        assert not str(p).startswith((str(C4.OUT), str(C3.OUT)))


def test_v5_experiments_differ_from_best_v4_in_the_intended_fields_only(V):
    C5, C4 = V["config_v5"], V["config_v4"]
    base = C4.EXPERIMENTS[C5.BASE_EXPERIMENT]
    diff = {k for k in C5.EXPERIMENTS["v5_z_khuber"] if C5.EXPERIMENTS["v5_z_khuber"][k] != base.get(k)}
    assert diff == {"loss", "l1_weight", "ema_decay", "description"}
    diff = {k for k in C5.EXPERIMENTS["v5_z_ctl_ema"] if C5.EXPERIMENTS["v5_z_ctl_ema"][k] != base.get(k)}
    assert diff == {"ema_decay", "description"}


def test_strict_checkpoint_regex_excludes_resume_files_unlike_v4_glob(V, tmp_path, monkeypatch):
    """Regression for the latent V4 bug: glob('<exp>_f0_s*.pt') also matches '<tag>.resume.pt'."""
    P = importlib.import_module("v5_predict")
    ck = tmp_path / "ck"
    ck.mkdir(exist_ok=True)
    for n in ("e_f0_s0.pt", "e_f0_s1.pt", "e_f0_s0.resume.pt", "e_f0_s1.resume.pt", "e_f0_s0.ema.pt", "e_f1_s0.pt"):
        (ck / n).touch()
    monkeypatch.setattr(V["config_v5"], "CKPT_DIR", ck)
    assert len(list(ck.glob("e_f0_s*.pt"))) == 5                         # V4 pattern: wrongly picks resume + ema files
    assert [p.name for p in P.ckpt_paths("e", 0, "raw")] == ["e_f0_s0.pt", "e_f0_s1.pt"]
    assert [p.name for p in P.ckpt_paths("e", 0, "ema")] == ["e_f0_s0.ema.pt"]
