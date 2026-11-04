"""Tests for V5.2 subject-balanced sampling and the resume/interrupt hardening (synthetic data only)."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("core_v3")

from test_v5_train import BINS, FakeStore, V, build, cfg, manifest, run, Tiny   # noqa: E402,F401


def uneven_manifest():
    rows = []
    for s, n in enumerate([1, 1, 2, 5, 10, 3]):
        for v in range(n):
            rows.append(dict(subject_id=f"S{s}", fold=s % 5, CDRTOT=0.0, age=60 + s, file_path=f"/x/{s}_{v}"))
    return pd.DataFrame(rows)


def test_weights_give_every_subject_equal_total_mass_and_mean_one(V):
    T = V["v5_train"]
    df = uneven_manifest()
    w = T.subject_balanced_weights(df.subject_id.values)
    assert len(w) == len(df) and abs(w.mean() - 1) < 1e-12
    mass = pd.Series(w).groupby(df.subject_id.values).sum()
    assert np.allclose(mass / mass.sum(), 1 / df.subject_id.nunique())          # repeated scans do NOT multiply a subject's share
    one, ten = w[df.subject_id == "S0"][0], w[df.subject_id == "S4"][0]
    assert one / ten == pytest.approx(10.0)                                     # w proportional to 1/n_i


def test_weights_depend_only_on_the_given_training_split(V):
    T = V["v5_train"]
    df = uneven_manifest()
    sub = df[df.subject_id != "S4"]
    w_full = T.subject_balanced_weights(df.subject_id.values)
    w_sub = T.subject_balanced_weights(sub.subject_id.values)
    assert len(w_sub) == len(sub)
    assert not np.allclose(w_full[df.subject_id != "S4"], w_sub)                # removing subjects changes the normalisation


def test_empty_split_raises(V):
    with pytest.raises(ValueError):
        V["v5_train"].subject_balanced_weights([])


def test_flags_mutually_exclusive(V):
    with pytest.raises(ValueError, match="mutually exclusive"):
        V["v5_train"].check_sampling_cfg(dict(subject_balanced=True, reweight_tail=True))
    V["v5_train"].check_sampling_cfg(dict(subject_balanced=True, reweight_tail=False))


def test_sampler_length_equals_scans_and_draws_are_subject_balanced(V):
    T = V["v5_train"]
    df = uneven_manifest()
    va = df.iloc[:1]
    torch.manual_seed(0)
    tl, _ = T.make_loaders(cfg(V, reweight_tail=False, subject_balanced=True, batch_size=4), df, va, FakeStore(), 0, torch.device("cpu"))
    assert tl.sampler.num_samples == len(df) and tl.sampler.replacement
    idx = np.concatenate([np.fromiter(iter(tl.sampler), int) for _ in range(3000)])
    share = pd.Series(df.subject_id.values[idx]).value_counts(normalize=True)
    assert np.allclose(share.values, 1 / df.subject_id.nunique(), atol=0.01)     # each subject ~1/6 of draws


def test_v5_2_config_differs_from_v5_1_only_by_subject_balanced(V):
    C5 = V["config_v5"]
    v51 = dict(C5.EXPERIMENTS["v5_z_khuber"], reweight_tail=False)               # what `--set reweight_tail=false` produces
    v52 = C5.EXPERIMENTS["v5_2_subject_balanced"]
    diff = {k for k in set(v51) | set(v52) if v51.get(k) != v52.get(k) and k != "description"}
    assert diff == {"subject_balanced"}
    assert v52["reweight_tail"] is False and v52["ema_decay"] == 0.999


def test_training_with_subject_balanced_runs_and_registers_flag(V, tmp_path):
    s = run(V, tmp_path, name="v5_2_subject_balanced", reweight_tail=False, subject_balanced=True)
    assert s["status"] == "completed"
    reg = pd.read_csv(V["config_v5"].REGISTRY_CSV)
    import json
    hp = json.loads(reg.hyperparameters_json.iloc[-1])
    assert hp["subject_balanced"] is True and hp["reweight_tail"] is False
    assert reg.checkpoint_raw.iloc[-1].endswith("v5_2_subject_balanced_f0_s0.pt")


def test_resume_refuses_config_change(V, tmp_path):
    with pytest.raises(RuntimeError, match="injected"):
        run(V, tmp_path, epochs=4, boom_after=40, reweight_tail=False, subject_balanced=True)
    with pytest.raises(RuntimeError, match="different config"):
        run(V, tmp_path, epochs=4, reweight_tail=False, subject_balanced=False)   # silently changing sampler must not resume
    s = run(V, tmp_path, epochs=4, reweight_tail=False, subject_balanced=True)     # same config resumes
    assert s["status"] == "completed"


def test_systemexit_is_not_recorded_as_completed(V, tmp_path, monkeypatch):
    T = V["v5_train"]
    real = T.K.gpu_augment
    n = {"c": 0}

    def boom(x):
        n["c"] += 1
        if n["c"] > 3:
            raise SystemExit(15)                                                  # what a SIGTERM handler would raise
        return real(x)

    monkeypatch.setattr(T.K, "gpu_augment", boom)
    with pytest.raises(SystemExit):
        run(V, tmp_path)
    reg = pd.read_csv(V["config_v5"].REGISTRY_CSV)
    assert reg.status.iloc[-1] == "failed" and "SystemExit" in reg.failure_reason.iloc[-1]


def test_keyboard_interrupt_recorded_as_interrupted(V, tmp_path, monkeypatch):
    T = V["v5_train"]
    real = T.K.gpu_augment
    n = {"c": 0}

    def boom(x):
        n["c"] += 1
        if n["c"] > 3:
            raise KeyboardInterrupt
        return real(x)

    monkeypatch.setattr(T.K, "gpu_augment", boom)
    with pytest.raises(KeyboardInterrupt):
        run(V, tmp_path)
    assert pd.read_csv(V["config_v5"].REGISTRY_CSV).status.iloc[-1] == "interrupted"
