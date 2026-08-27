"""Evaluation protocols and the row-building that turns predictions into results.

These tests encode what each protocol is *supposed* to leak. A protocol that trains
on the slides it scores still produces a plausible macro-F1, so the leakage
properties are asserted directly rather than inferred from a reasonable number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vgtfm.config import load_config
from vgtfm.data.folds import FoldSpec
from vgtfm.evaluate import protocol as proto
from vgtfm.evaluate.reporting import metric_rows, paired_delta_rows, pool, score_vocabulary, scopes


@pytest.fixture
def cfg():
    return load_config(
        None, {"eval.knn_k": 1, "eval.max_train_samples": 0, "eval.standardize": "false"}
    )


def cohort(
    *,
    n_per_slide=40,
    slides=(
        ("A-1-1", "A", "mel"),
        ("A-1-2", "A", "mel"),
        ("B-1-1", "B", "mel"),
        ("B-1-2", "B", "mel"),
    ),
    unlabeled_every=0,
    seed=0,
):
    """A small annotated cohort: 2 donors x 2 replicate slides, 2 classes."""
    rng = np.random.default_rng(seed)
    rows, Z = [], []
    for sid, donor, tissue in slides:
        for i in range(n_per_slide):
            cls = "TUM" if i % 2 == 0 else "STR"
            if unlabeled_every and i % unlabeled_every == 0:
                cls = "UNASSIGNED"
            rows.append({"sample_id": sid, "donor": donor, "tissue": tissue, "annotation": cls})
            Z.append(rng.standard_normal(3))
    return np.asarray(Z, dtype=np.float32), pd.DataFrame(rows)


def separable(meta, *, by="annotation", scale=10.0):
    """Features that make *by* trivially predictable by a 1-NN probe."""
    codes = pd.factorize(meta[by])[0]
    return np.eye(codes.max() + 1, dtype=np.float32)[codes] * scale


# ── heldout_donor ────────────────────────────────────────────────────


def test_heldout_donor_scores_only_the_held_out_slides(cfg):
    Z, meta = cohort()
    fold = FoldSpec("f", "cross_donor", ("A-1-1", "A-1-2"), ("B-1-1", "B-1-2"))
    pred = proto.heldout_donor(Z, meta, fold, cfg=cfg)
    assert set(pred["sample_id"]) == {"B-1-1", "B-1-2"}
    assert set(pred["donor"]) == {"B"}


def test_heldout_donor_ignores_spots_outside_the_fold(cfg):
    """The probe must be a function of the fold's slides alone.

    Corrupting every slide that the fold does not name must not move a single
    prediction; if it does, rows are leaking in from outside the split.
    """
    Z, meta = cohort()
    Z = separable(meta)
    fold = FoldSpec("f", "cross_donor", ("A-1-1",), ("B-1-1",))
    base = proto.heldout_donor(Z, meta, fold, cfg=cfg)

    outside = ~meta["sample_id"].isin(["A-1-1", "B-1-1"]).to_numpy()
    corrupted = Z.copy()
    corrupted[outside] = 999.0
    after = proto.heldout_donor(corrupted, meta, fold, cfg=cfg)
    assert np.array_equal(base["y_pred"], after["y_pred"])


def test_heldout_donor_returns_nothing_when_a_side_is_unannotated(cfg):
    Z, meta = cohort()
    meta.loc[meta.sample_id == "B-1-1", "annotation"] = "UNASSIGNED"
    fold = FoldSpec("f", "cross_donor", ("A-1-1",), ("B-1-1",))
    assert proto.heldout_donor(Z, meta, fold, cfg=cfg) == {}


def test_protocols_never_score_an_unannotated_spot(cfg):
    Z, meta = cohort(unlabeled_every=4)
    fold = FoldSpec("f", "cross_donor", ("A-1-1", "A-1-2"), ("B-1-1", "B-1-2"))
    for pred in (
        proto.heldout_donor(Z, meta, fold, cfg=cfg),
        proto.loso_donor(Z, meta, cfg=cfg),
        proto.pooled_loso(Z, meta, cfg=cfg),
    ):
        assert len(pred["y_true"]) > 0
        assert "UNASSIGNED" not in set(pred["y_true"])


# ── what each protocol leaks ─────────────────────────────────────────


def test_pooled_loso_leaks_the_technical_replicate(cfg):
    """The documented weakness of ``pooled_loso``, demonstrated.

    Features here carry a donor-specific code for the class, so a probe can only
    score above chance if a slide from the *same donor* is in its training set.
    Holding out a slide leaves that donor's replicate behind; holding out the donor
    does not.
    """
    Z, meta = cohort()
    codes = pd.factorize(meta["donor"] + "|" + meta["annotation"])[0]
    Z = np.eye(codes.max() + 1, dtype=np.float32)[codes] * 10.0

    slide_out = proto.pooled_loso(Z, meta, cfg=cfg)
    donor_out = proto.loso_donor(Z, meta, cfg=cfg)

    slide_acc = float((slide_out["y_true"] == slide_out["y_pred"]).mean())
    donor_acc = float((donor_out["y_true"] == donor_out["y_pred"]).mean())
    assert slide_acc == pytest.approx(1.0)  # the replicate gives it away
    assert donor_acc <= 0.6  # chance, for two balanced classes


def test_legacy_protocol_is_the_same_computation_for_every_fold(cfg):
    """``pooled_loso_on_fold`` depends only on the *set* of slides in the fold.

    With the representation held fixed, rearranging which slides a fold calls train
    and which it calls eval returns bit-identical predictions, because the protocol
    pools them and never separates the two.
    """
    Z, meta = cohort()
    Z = separable(meta)
    slides = ("A-1-1", "A-1-2", "B-1-1", "B-1-2")
    a = proto.pooled_loso_on_fold(
        Z, meta, FoldSpec("a", "cross_donor", slides[:2], slides[2:]), cfg=cfg
    )
    b = proto.pooled_loso_on_fold(
        Z, meta, FoldSpec("b", "cross_donor", slides[2:], slides[:2]), cfg=cfg
    )
    c = proto.pooled_loso_on_fold(
        Z, meta, FoldSpec("c", "cross_donor", slides[1:], slides[:1]), cfg=cfg
    )

    assert np.array_equal(a["y_pred"], b["y_pred"])
    assert np.array_equal(a["y_true"], b["y_true"])
    assert np.array_equal(a["y_pred"], c["y_pred"])


def test_heldout_donor_does_depend_on_the_split(cfg):
    """The contrast to the test above: the corrected protocol is not fold-blind."""
    Z, meta = cohort()
    Z = separable(meta)
    slides = ("A-1-1", "A-1-2", "B-1-1", "B-1-2")
    a = proto.heldout_donor(Z, meta, FoldSpec("a", "l", slides[:2], slides[2:]), cfg=cfg)
    b = proto.heldout_donor(Z, meta, FoldSpec("b", "l", slides[2:], slides[:2]), cfg=cfg)
    assert set(a["sample_id"]).isdisjoint(set(b["sample_id"]))


def test_leave_one_out_stays_inside_a_tissue(cfg):
    """A tissue with a single group cannot be scored and must be skipped, not pooled."""
    Z, meta = cohort(slides=(("K1", "K1", "kidney"), ("L1", "L1", "lung"), ("L2", "L2", "lung")))
    pred = proto.loso_donor(Z, meta, cfg=cfg)
    assert set(pred["tissue"]) == {"lung"}  # kidney has one donor -> skipped


# ── dispatch ─────────────────────────────────────────────────────────


def test_fold_protocols_require_a_fold(cfg):
    Z, meta = cohort()
    for name in ("heldout_donor", "pooled_loso_on_fold"):
        assert proto.is_fold_protocol(name)
        with pytest.raises(ValueError):
            proto.run(name, Z, meta, cfg=cfg, fold=None)


def test_unknown_protocol_is_rejected(cfg):
    Z, meta = cohort()
    with pytest.raises(SystemExit):
        proto.run("loso_spot", Z, meta, cfg=cfg)


def test_majority_baseline_ignores_the_embedding(cfg):
    """The chance floor must be identical for any features whatsoever."""
    Z, meta = cohort()
    fold = FoldSpec("f", "cross_donor", ("A-1-1", "A-1-2"), ("B-1-1", "B-1-2"))
    a = proto.heldout_donor(Z, meta, fold, cfg=cfg, majority=True)
    b = proto.heldout_donor(separable(meta), meta, fold, cfg=cfg, majority=True)
    assert np.array_equal(a["y_pred"], b["y_pred"])
    assert len(set(a["y_pred"])) == 1


# ── reporting ────────────────────────────────────────────────────────


def _pred(y_true, y_pred, donor, tissue, sample_id=None):
    n = len(y_true)
    return {
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
        "donor": np.array(donor),
        "tissue": np.array(tissue),
        "sample_id": np.array(sample_id if sample_id is not None else donor),
        "n_train": n,
    }


def test_scopes_cover_the_global_pool_and_each_tissue():
    pred = _pred(
        ["A"] * 4, ["A"] * 4, ["d1", "d1", "d2", "d2"], ["kidney", "kidney", "lung", "lung"]
    )
    got = {name: int(mask.sum()) for name, mask in scopes(pred)}
    assert got == {"global": 4, "kidney": 2, "lung": 2}


def test_score_vocabulary_does_not_depend_on_the_predictions():
    """The macro denominator is fixed by the protocol, never by the model."""
    truth = ["A", "A", "B", "B"]
    good = _pred(truth, ["A", "A", "B", "B"], ["d1"] * 4, ["mel"] * 4)
    bad = _pred(truth, ["C", "C", "C", "C"], ["d1"] * 4, ["mel"] * 4)
    assert score_vocabulary(good) == score_vocabulary(bad)
    assert score_vocabulary(good)["mel"] == ["A", "B"]
    # The global scope names each class by the organ it was scored in.
    assert score_vocabulary(good)["global"] == ["mel | A", "mel | B"]


def test_metric_rows_average_over_the_shared_denominator():
    """A model is not rewarded for a fold that happens to contain fewer classes."""
    pred = _pred(["A", "A", "B", "B"], ["A", "A", "B", "B"], ["d1", "d1", "d2", "d2"], ["mel"] * 4)
    vocab = {"global": ["mel | A", "mel | B", "mel | C"], "mel": ["A", "B", "C"]}
    macro, per_class = metric_rows(pred, ["A", "B", "C"], {"model": "m"}, vocab)
    glob = next(r for r in macro if r["scope"] == "global")
    assert glob["f1_score"] == pytest.approx(2 / 3)  # C absent but still averaged
    assert glob["n_donors"] == 2
    assert {r["class"] for r in per_class if r["scope"] == "global"} == {
        "mel | A",
        "mel | B",
        "mel | C",
    }


def test_one_tissue_scores_identically_under_either_label_space():
    """Qualifying by organ must be a renaming, not a change of number, when there
    is only one organ to qualify with."""
    truth = ["A", "A", "B", "B", "C", "C"]
    guess = ["A", "B", "B", "B", "C", "A"]
    pred = _pred(truth, guess, ["d1", "d1", "d2", "d2", "d3", "d3"], ["mel"] * 6)
    macro, _ = metric_rows(pred, ["A", "B", "C"], {}, score_vocabulary(pred))
    by_scope = {r["scope"]: r["f1_score"] for r in macro}
    assert by_scope["global"] == pytest.approx(by_scope["mel"])


def test_the_global_macro_scores_each_organ_on_its_own_classes():
    """Skin and lung share the name ``Tumor`` and nothing else. Pooling them into
    one cell weights it by whichever organ brought more spots, and averages every
    fold against classes it cannot contain."""
    # Perfect in lung, chance in skin. Two organs, one shared class name.
    truth = ["Tumor", "TLS"] * 2 + ["Tumor", "Stroma"] * 2
    guess = ["Tumor", "TLS"] * 2 + ["Tumor", "Tumor"] * 2
    tissue = ["lung"] * 4 + ["skin"] * 4
    pred = _pred(truth, guess, ["l1", "l1", "l2", "l2", "s1", "s1", "s2", "s2"], tissue)
    vocab = score_vocabulary(pred)

    # Four cells, not three classes: Tumor is asked twice because it is two
    # questions.
    assert vocab["global"] == ["lung | TLS", "lung | Tumor", "skin | Stroma", "skin | Tumor"]

    macro, _ = metric_rows(pred, ["TLS", "Stroma", "Tumor"], {}, vocab)
    by_scope = {r["scope"]: r for r in macro}
    assert by_scope["lung"]["f1_score"] == pytest.approx(1.0)
    assert by_scope["global"]["n_classes_scored"] == 4
    # The global macro is the mean of the four cells, so a perfect lung cannot be
    # diluted by skin's spot count, nor skin flattered by lung's.
    cells = [
        1.0,  # lung | TLS
        1.0,  # lung | Tumor
        0.0,  # skin | Stroma  — never predicted
        2 / 3,
    ]  # skin | Tumor   — precision 1/2, recall 1
    assert by_scope["global"]["f1_score"] == pytest.approx(np.mean(cells))


def test_a_fold_is_never_scored_against_another_organs_classes():
    """The pooled vocabulary names cells a single-organ fold cannot contain; they
    must not enter its denominator as automatic zeros."""
    pooled = _pred(
        ["Tumor", "TLS", "Tumor", "Stroma"],
        ["Tumor", "TLS", "Tumor", "Stroma"],
        ["l1", "l1", "s1", "s1"],
        ["lung", "lung", "skin", "skin"],
    )
    vocab = score_vocabulary(pooled)
    assert len(vocab["global"]) == 4

    skin_fold = _pred(["Tumor", "Stroma"], ["Tumor", "Stroma"], ["s1", "s1"], ["skin", "skin"])
    macro, _ = metric_rows(skin_fold, ["TLS", "Stroma", "Tumor"], {}, vocab)
    glob = next(r for r in macro if r["scope"] == "global")
    assert glob["n_classes_scored"] == 2  # not 4
    assert glob["f1_score"] == pytest.approx(1.0)  # a perfect fold reads as one


def test_pool_concatenates_folds_and_averages_the_training_size():
    a = _pred(["A"], ["A"], ["d1"], ["mel"])
    b = _pred(["B", "B"], ["B", "B"], ["d2", "d2"], ["mel", "mel"])
    a["n_train"], b["n_train"] = 10, 20
    out = pool([a, b, {}])
    assert out["y_true"].tolist() == ["A", "B", "B"]
    assert out["n_train"] == 15
    assert pool([{}]) == {}


def test_paired_delta_sign_follows_the_argument_order():
    """``paired_delta_rows(pred, a, b)`` reports ``metric(a) - metric(b)``."""
    truth = np.array(["A", "B"] * 20)
    donors = np.repeat([f"d{i}" for i in range(4)], 10)
    pred = _pred(truth, truth, donors, ["mel"] * 40)
    wrong = np.where(truth == "A", "B", "A")
    cfg = load_config(None, {"eval.bootstrap_n": 50})

    better = paired_delta_rows(pred, truth, wrong, {}, ["A", "B"], cfg, {})
    worse = paired_delta_rows(pred, wrong, truth, {}, ["A", "B"], cfg, {})
    macro_better = next(r for r in better if r["scope"] == "global")
    macro_worse = next(r for r in worse if r["scope"] == "global")
    assert macro_better["delta"] > 0
    assert macro_worse["delta"] == pytest.approx(-macro_better["delta"])


def test_ablation_deltas_are_matched_minus_shuffled():
    """``ablate`` compares against the matched condition, not against PCA.

    The sign convention is the opposite of ``eval``'s and is easy to invert by
    accident: a positive delta must mean the real correspondence helped.
    """
    from vgtfm.ablations.run_ablation import REFERENCE_CONDITION, _paired_deltas

    truth = np.array(["A", "B"] * 20)
    donors = np.repeat([f"d{i}" for i in range(4)], 10)
    wrong = np.where(truth == "A", "B", "A")

    matched = _pred(truth, truth, donors, ["mel"] * 40)  # perfect
    shuffled = _pred(truth, wrong, donors, ["mel"] * 40)  # useless
    preds = {
        (42, REFERENCE_CONDITION, "pooled_loso", "all"): matched,
        (42, "global", "pooled_loso", "all"): shuffled,
    }

    cfg = load_config(None, {"eval.bootstrap_n": 50})
    rows = _paired_deltas(preds, {}, ["A", "B"], cfg)
    macro = next(r for r in rows if r["class"] == "macro" and r["scope"] == "global")

    assert macro["condition"] == "global"
    assert macro["reference"] == REFERENCE_CONDITION
    assert macro["delta"] == pytest.approx(1.0)  # matched wins outright
    # The reference condition itself is never compared against itself.
    assert all(r["condition"] != REFERENCE_CONDITION for r in rows)


def test_ablation_deltas_skip_conditions_that_scored_different_rows():
    """Unaligned predictions cannot be paired, and must be dropped, not compared."""
    from vgtfm.ablations.run_ablation import REFERENCE_CONDITION, _paired_deltas

    matched = _pred(["A", "B"], ["A", "B"], ["d1", "d2"], ["mel"] * 2)
    other = _pred(["B", "A"], ["B", "A"], ["d1", "d2"], ["mel"] * 2)
    preds = {
        (42, REFERENCE_CONDITION, "pooled_loso", "all"): matched,
        (42, "global", "pooled_loso", "all"): other,
    }
    assert _paired_deltas(preds, {}, ["A", "B"], load_config(None)) == []
