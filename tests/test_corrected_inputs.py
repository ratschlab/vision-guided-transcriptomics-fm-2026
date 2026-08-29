"""Batch-corrected gene inputs: the arm that fits guidance on a corrected source.

What is pinned is the plumbing that makes it comparable with the uncorrected arm:
the inner model is the ordinary one, the correction is computed once over every
spot, and the paired delta is taken against the corrected baseline.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from vgtfm.config import load_config
from vgtfm.degraded import Incomplete
from vgtfm.diagnostics import integration
from vgtfm.models import corrected
from vgtfm.models.base import Inputs
from vgtfm.models.train import embedding_path

from test_stages import FIT_SLIDES, synthetic_table  # noqa: F401


# ── name parsing ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,want",
    [
        ("harmony_ae", ("harmony", "ae")),
        ("combat_pca", ("combat", "pca")),
        ("harmony_cdann", ("harmony", "cdann")),
        # Ordinary models whose names contain an underscore must parse as
        # themselves; `hvg_pca` and the oracles are the ones that could collide.
        ("pca", None),
        ("ae", None),
        ("hvg_pca", None),
        ("pca_oracle", None),
        ("pca_oracle_matched", None),
        ("harmony", None),
    ],
)
def test_only_a_correction_prefix_makes_a_corrected_name(name, want):
    assert corrected.parse(name) == want


def test_a_corrected_arm_is_compared_with_its_own_corrected_baseline():
    """The delta has to isolate guidance, not guidance plus the correction."""
    from vgtfm.evaluate.run_eval import _reference_model

    cfg = load_config(None, {"models.names": "pca,harmony_pca,harmony_ae"})
    assert _reference_model("harmony_ae", cfg) == "harmony_pca"
    assert _reference_model("ae", cfg) == "pca"
    # The corrected baseline is its own arm's reference and gets no self-delta.
    assert corrected.reference_for("harmony_pca") is None


def test_a_corrected_arm_falls_back_to_pca_when_its_baseline_is_not_in_the_run():
    """Better a two-change delta, labelled as one, than no delta at all."""
    from vgtfm.evaluate.run_eval import _reference_model

    cfg = load_config(None, {"models.names": "pca,harmony_ae"})
    assert _reference_model("harmony_ae", cfg) == "pca"


@pytest.mark.parametrize(
    "name,match",
    [
        ("harmony_hvg_pca", "raw counts"),
        ("combat_pca_oracle", "H&E"),
        ("harmony_combat_ae", "stacked"),
    ],
)
def test_an_inner_model_that_cannot_read_corrected_genes_is_refused_before_fitting(name, match):
    with pytest.raises(Incomplete, match=match):
        corrected.validate(["pca", name])


def test_the_registry_refuses_a_corrected_name_rather_than_guessing():
    from vgtfm.models.base import build

    with pytest.raises(SystemExit, match="not built through the registry"):
        build("harmony_ae", load_config(None, {}), 42)


# ── the correction itself ────────────────────────────────────────────


@pytest.fixture
def cfg(tmp_path):
    return load_config(
        None,
        {
            "paths.artifact_root": str(tmp_path),
            "run_name": "t",
            "seeds": "42",
            "models.names": "pca,harmony_pca",
            "models.pca_components": "4",
            "folds.levels": "cross_donor",
            "eval.protocols": "heldout_donor",
            "eval.knn_k": "3",
            "eval.bootstrap_n": "25",
            "perf.device": "cpu",
        },
    )


def test_the_correction_is_computed_once_and_reused_across_models(cfg, monkeypatch):
    """Harmony is the expensive part of the arm; two models must not pay it twice."""
    calls = []

    def stub(X, batch, *, seed=42):
        calls.append(seed)
        return np.asarray(X, dtype=np.float32) + 1.0

    monkeypatch.setattr(integration, "correct_harmony", stub)

    table = synthetic_table()
    rows = np.arange(len(FIT_SLIDES) * 30)
    a = corrected.corrected_gene(cfg, table, "harmony", 42, rows)
    b = corrected.corrected_gene(cfg, table, "harmony", 42, rows)

    assert calls == [42], "the second call should have read the cache"
    assert np.array_equal(a, b)
    assert np.allclose(a, table.gene + 1.0)


def test_a_cache_built_from_other_features_is_not_reused(cfg, monkeypatch):
    """A changed substrate or data filter must recompute, not silently carry over."""
    calls = []

    def stub(X, batch, *, seed=42):
        calls.append(X.shape)
        return np.asarray(X, dtype=np.float32) + 1.0

    monkeypatch.setattr(integration, "correct_harmony", stub)

    table = synthetic_table()
    rows = np.arange(len(FIT_SLIDES) * 30)
    corrected.corrected_gene(cfg, table, "harmony", 42, rows)

    other = synthetic_table(seed=7)
    Z = corrected.corrected_gene(cfg, other, "harmony", 42, rows)

    assert len(calls) == 2
    assert np.allclose(Z, other.gene + 1.0)


def test_combat_is_cached_once_for_every_seed_and_harmony_per_seed(cfg):
    """ComBat is closed-form; Harmony converges somewhere the seed decides."""
    h42 = corrected._cache_path(cfg, "harmony", 42)
    h43 = corrected._cache_path(cfg, "harmony", 43)
    c42 = corrected._cache_path(cfg, "combat", 42)
    c43 = corrected._cache_path(cfg, "combat", 43)
    assert h42 != h43
    assert c42 == c43


def test_the_correction_leaves_every_block_but_the_genes_alone(cfg, monkeypatch):
    """`patch` is the supervision target and must reach the AE uncorrected."""
    monkeypatch.setattr(
        integration, "correct_harmony", lambda X, batch, *, seed=42: np.asarray(X) + 1.0
    )
    table = synthetic_table()
    inputs = Inputs.from_table(table)
    rows = np.arange(len(FIT_SLIDES) * 30)

    out = corrected.corrected_inputs(cfg, table, inputs, "harmony", 42, rows)

    assert np.allclose(out.gene, inputs.gene + 1.0)
    assert out.patch is inputs.patch
    assert np.array_equal(out.sample_id, inputs.sample_id)
    assert np.array_equal(out.annotation, inputs.annotation)


def test_a_correction_that_changes_the_width_is_refused(cfg, monkeypatch):
    """The models downstream address spots by row and assume the width holds."""
    monkeypatch.setattr(
        integration, "correct_harmony", lambda X, batch, *, seed=42: np.asarray(X)[:, :2]
    )
    table = synthetic_table()
    with pytest.raises(Incomplete, match="width is unchanged"):
        corrected.corrected_gene(cfg, table, "harmony", 42, np.arange(60))


def test_correction_input_dim_reduces_before_correcting(cfg, monkeypatch):
    """The cheap path: correct in a PCA basis rather than at full gene width."""
    seen = []
    monkeypatch.setattr(
        integration,
        "correct_harmony",
        lambda X, batch, *, seed=42: (seen.append(X.shape), np.asarray(X))[1],
    )
    cfg.models.correction_input_dim = 3
    table = synthetic_table()
    Z = corrected.corrected_gene(cfg, table, "harmony", 42, np.arange(60))

    assert seen == [(table.n, 3)]
    assert Z.shape == (table.n, 3)


# ── through the stages ───────────────────────────────────────────────


def test_the_train_stage_fits_the_inner_model_on_corrected_genes(cfg, monkeypatch):
    from vgtfm.data import tables as tables_mod
    from vgtfm.models import train

    table = synthetic_table()
    monkeypatch.setattr(tables_mod, "load", lambda cfg, substrate=None: table)
    # A correction big enough that the two PCA fits cannot come out identical.
    monkeypatch.setattr(
        integration,
        "correct_harmony",
        lambda X, batch, *, seed=42: np.asarray(X)[:, ::-1].copy(),
    )

    train.run(cfg)

    plain = np.load(embedding_path(cfg, "pca", 42))
    corr = np.load(embedding_path(cfg, "harmony_pca", 42))
    assert plain.shape == corr.shape == (table.n, 4)
    assert not np.allclose(plain, corr)

    summary = pd.read_csv(cfg.out_dir / "train" / "train_summary.csv")
    got = summary.set_index("model")["correction"].to_dict()
    assert got == {"pca": "none", "harmony_pca": "harmony"}

    state = json.loads(
        (cfg.out_dir / "train" / "history" / "harmony_pca__seed-42.json").read_text()
    )
    # The file is keyed by the outer name; the inner model only knows its own.
    assert state["model"] == "harmony_pca" and state["correction"] == "harmony"
    assert state["name"] == "pca"


def test_eval_reports_the_guided_arm_against_the_corrected_baseline(cfg, monkeypatch):
    pytest.importorskip("torch")
    from vgtfm.data import tables as tables_mod
    from vgtfm.evaluate import run_eval
    from vgtfm.models import train

    table = synthetic_table()
    monkeypatch.setattr(tables_mod, "load", lambda cfg, substrate=None: table)
    monkeypatch.setattr(tables_mod, "load_meta", lambda cfg, substrate=None: table.meta)
    monkeypatch.setattr(
        integration,
        "correct_harmony",
        lambda X, batch, *, seed=42: np.asarray(X)[:, ::-1].copy(),
    )
    cfg.models.names = ("pca", "ae", "harmony_pca", "harmony_ae")
    cfg.models.ae.latent_dim = 4
    cfg.models.ae.num_epochs = 2
    cfg.models.ae.batch_size = 32

    train.run(cfg)
    run_eval.run(cfg)

    deltas = pd.read_csv(cfg.out_dir / "eval" / "deltas.csv")
    refs = deltas.drop_duplicates("model").set_index("model")["reference"].to_dict()
    assert refs["harmony_ae"] == "harmony_pca"
    assert refs["ae"] == "pca"
    assert refs["harmony_pca"] == "pca"


def test_a_per_fold_refit_of_a_corrected_arm_is_refused(cfg):
    from vgtfm.evaluate.run_eval import _refuse_refit_of_corrected

    _refuse_refit_of_corrected("pca")  # no-op for an ordinary model
    with pytest.raises(Incomplete, match="no per-fold form"):
        _refuse_refit_of_corrected("harmony_ae")
