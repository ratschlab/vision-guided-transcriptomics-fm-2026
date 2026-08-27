"""Labels, probe metrics, subsampling and the donor bootstrap.

Concentrated on the statistics rather than the plumbing: a broken dataloader fails
loudly, whereas a macro-F1 averaged over the wrong denominator, a bootstrap that
resamples spots instead of donors, or a label mask that admits unannotated spots
all produce plausible numbers.

Folds live in ``test_folds.py``, models in ``test_models.py``, diagnostics in
``test_diagnostics.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from vgtfm.ablations.patch_shuffle import PATCH_TRANSFORMS, apply_patch_transform
from vgtfm.evaluate import bootstrap as boot
from vgtfm.evaluate.probes import (
    LabelCodec,
    classification_metrics,
    confusion_counts,
    knn_predict,
    majority_predict,
    sklearn_reference_metrics,
    standardize,
    stratified_subsample,
)
from vgtfm.labels import BELOW_FLOOR, UNLABELED, class_floor_mask, labeled_mask, real_classes


# ── labels ───────────────────────────────────────────────────────────


def test_labeled_mask_excludes_unannotated_spots():
    labels = np.array(["Tumor", "UNASSIGNED", None, "", "Stroma", "unassigned"], dtype=object)
    assert labeled_mask(labels).tolist() == [True, False, False, False, True, False]


@pytest.mark.parametrize("value", sorted(UNLABELED))
def test_every_unlabelled_spelling_is_rejected_case_and_space_insensitively(value):
    """Cohorts spell "no label here" differently; the rule lives in one place."""
    assert not labeled_mask(np.array([f"  {value.lower()}  "], dtype=object))[0]


def test_real_classes_are_sorted_deduplicated_and_label_only():
    labels = np.array(["TUM", "STR", "TUM", "UNASSIGNED", "Mixed"], dtype=object)
    assert real_classes(labels) == ["STR", "TUM"]


def test_the_label_mask_is_what_decides_the_class_vocabulary():
    """Dropping the mask would admit UNASSIGNED as if it were a pathology class."""
    labels = np.array(["TUM", "UNASSIGNED", "STR"], dtype=object)
    assert len(real_classes(labels)) == int(labeled_mask(labels).sum())


# ── class support floor ──────────────────────────────────────────────


def _floor_fixture():
    """Two organs. `Tumor` clears both bounds in each; `LN` is short on spots and
    `Pigment` has enough spots but sits on one slide."""
    ann = np.array(["Tumor"] * 4 + ["LN"] + ["Tumor"] * 3 + ["Pigment"] * 3, dtype=object)
    tis = np.array(["lung"] * 5 + ["skin"] * 6, dtype=object)
    sid = np.array(["L1", "L1", "L2", "L2", "L1", "S1", "S1", "S2", "S1", "S1", "S1"], dtype=object)
    return ann, tis, sid


def test_class_floor_drops_below_either_bound_and_keeps_the_rest():
    ann, tis, sid = _floor_fixture()
    drop, dropped = class_floor_mask(ann, tis, sid, min_spots=3, min_slides=2)
    names = {(d["tissue"], d["annotation"]) for d in dropped}
    assert names == {("lung", "LN"), ("skin", "Pigment")}
    assert ann[drop].tolist() == ["LN", "Pigment", "Pigment", "Pigment"]
    assert set(ann[~drop]) == {"Tumor"}


def test_class_floor_reports_which_bound_failed():
    ann, tis, sid = _floor_fixture()
    _, dropped = class_floor_mask(ann, tis, sid, min_spots=3, min_slides=2)
    why = {d["annotation"]: d["reason"] for d in dropped}
    assert why == {"LN": "spots", "Pigment": "slides"}


def test_class_floor_counts_within_tissue_not_across_the_cohort():
    """A class pooled over two organs can clear a bound neither organ clears."""
    ann = np.array(["TLS"] * 4, dtype=object)
    tis = np.array(["lung", "lung", "kidney", "kidney"], dtype=object)
    sid = np.array(["L1", "L2", "K1", "K2"], dtype=object)
    drop, dropped = class_floor_mask(ann, tis, sid, min_spots=3, min_slides=2)
    assert drop.all() and len(dropped) == 2


def test_class_floor_ignores_unlabelled_rows():
    ann = np.array(["Tumor", "Tumor", "UNASSIGNED", "UNASSIGNED"], dtype=object)
    tis = np.array(["skin"] * 4, dtype=object)
    sid = np.array(["S1", "S2", "S1", "S2"], dtype=object)
    drop, dropped = class_floor_mask(ann, tis, sid, min_spots=2, min_slides=2)
    assert not drop.any() and dropped == []


@pytest.mark.parametrize(
    "min_spots, min_slides, expected",
    [
        (0, 0, set()),  # floor off entirely
        (0, 2, {"LN", "Pigment"}),  # slide bound alone: both sit on one slide
        (3, 0, {"LN"}),  # spot bound alone: only LN is short
        (3, 2, {"LN", "Pigment"}),  # both
    ],
)
def test_each_class_floor_bound_acts_independently(min_spots, min_slides, expected):
    """0 switches one bound off without disabling the other."""
    ann, tis, sid = _floor_fixture()
    _, dropped = class_floor_mask(ann, tis, sid, min_spots=min_spots, min_slides=min_slides)
    assert {str(d["annotation"]) for d in dropped} == expected


def test_the_floor_sentinel_reads_as_unlabelled():
    """Demotion has to reuse the unlabelled rule, or metrics would score it."""
    assert not labeled_mask(np.array([BELOW_FLOOR], dtype=object))[0]
    assert real_classes(np.array(["Tumor", BELOW_FLOOR], dtype=object)) == ["Tumor"]


# ── metrics ──────────────────────────────────────────────────────────


def test_classification_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    classes = ["A", "B", "C", "D"]
    for _ in range(50):
        n = int(rng.integers(30, 300))
        y_true = rng.choice(classes[:3], size=n)
        y_pred = rng.choice(classes, size=n)  # may predict an absent class
        mine = classification_metrics(y_true, y_pred, classes)
        ref = sklearn_reference_metrics(y_true, y_pred, classes)
        for key in ("accuracy", "precision", "recall", "f1_score"):
            assert mine[key] == pytest.approx(ref[key], abs=1e-12)
        for c in classes:
            for key in ("precision", "recall", "f1_score", "support"):
                assert mine["per_class"][c][key] == pytest.approx(
                    ref["per_class"][c][key], abs=1e-12
                )


def test_score_subset_changes_only_the_macro_denominator():
    y_true = np.array(["A"] * 10 + ["B"] * 10)
    y_pred = np.array(["A"] * 10 + ["B"] * 10)
    full = classification_metrics(y_true, y_pred, ["A", "B", "C", "D"])
    subset = classification_metrics(y_true, y_pred, ["A", "B", "C", "D"], score_classes=["A", "B"])
    # Averaging over two absent classes halves a perfect score.
    assert full["f1_score"] == pytest.approx(0.5)
    assert subset["f1_score"] == pytest.approx(1.0)
    assert full["per_class"] == subset["per_class"]  # only the average moved


def test_out_of_subset_prediction_still_counts_as_an_error():
    """A prediction outside the scored classes must not be silently dropped."""
    y_true = np.array(["A", "A", "B", "B"])
    y_pred = np.array(["A", "C", "B", "B"])
    m = classification_metrics(y_true, y_pred, ["A", "B", "C"], score_classes=["A", "B"])
    assert m["per_class"]["A"]["recall"] == pytest.approx(0.5)
    assert m["n_test"] == 4


def test_metrics_of_a_perfect_and_of_a_hopeless_classifier():
    y = np.array(["A", "B", "A", "B"])
    assert classification_metrics(y, y, ["A", "B"])["f1_score"] == pytest.approx(1.0)
    flipped = np.where(y == "A", "B", "A")
    assert classification_metrics(y, flipped, ["A", "B"])["f1_score"] == 0.0


def test_metrics_infer_the_vocabulary_when_none_is_given():
    m = classification_metrics(["A", "B"], ["A", "A"])
    assert set(m["per_class"]) == {"A", "B"}


def test_label_codec_marks_unknown_labels():
    codec = LabelCodec(["A", "B"])
    assert codec.encode(["A", "B", "Z"]).tolist() == [0, 1, -1]


def test_label_codec_sorts_and_deduplicates_its_vocabulary():
    codec = LabelCodec(["B", "A", "B"])
    assert codec.classes.tolist() == ["A", "B"] and len(codec) == 2
    assert codec.subset_index(["B", "missing"]).tolist() == [1]


def test_confusion_counts_drop_rows_outside_the_vocabulary():
    """Encoding gives -1 to unknown labels; those rows cannot enter the matrix."""
    codec = LabelCodec(["A", "B"])
    cm = confusion_counts(codec.encode(["A", "A", "Z"]), codec.encode(["A", "B", "A"]), len(codec))
    assert cm.tolist() == [[1, 1], [0, 0]]
    assert cm.sum() == 2


def test_majority_predicts_the_most_frequent_training_class():
    train_y = np.array(["A"] * 3 + ["B"] * 7)
    assert set(majority_predict(train_y, 5)) == {"B"}
    assert len(majority_predict(train_y, 5)) == 5


# ── probe preprocessing ──────────────────────────────────────────────


def test_stratified_subsample_preserves_class_shares_and_size():
    rng = np.random.default_rng(0)
    labels = np.array(["A"] * 900 + ["B"] * 90 + ["C"] * 10)
    X = rng.standard_normal((1000, 4))
    Xs, ys = stratified_subsample(X, labels, labels=labels, max_samples=100, random_seed=42)
    assert len(Xs) == len(ys) == 100
    # The rare class survives rather than being sampled away.
    assert (ys == "C").sum() >= 1
    assert (ys == "A").sum() == pytest.approx(90, abs=2)


def test_stratified_subsample_keeps_every_array_row_aligned():
    """Several arrays are subsampled together; misalignment would scramble labels."""
    labels = np.array(["A"] * 60 + ["B"] * 40)
    ids = np.arange(100)
    out_ids, out_labels = stratified_subsample(
        ids, labels, labels=labels, max_samples=30, random_seed=0
    )
    assert np.array_equal(labels[out_ids], out_labels)


def test_stratified_subsample_is_a_no_op_below_the_cap():
    X = np.arange(10).reshape(10, 1)
    labels = np.array(["A"] * 10)
    assert stratified_subsample(X, labels=labels, max_samples=50)[0] is X
    assert stratified_subsample(X, labels=labels, max_samples=None)[0] is X


def test_stratified_subsample_is_deterministic():
    labels = np.array(["A"] * 50 + ["B"] * 50)
    X = np.arange(100).reshape(100, 1).astype(float)
    a = stratified_subsample(X, labels=labels, max_samples=30, random_seed=42)[0]
    b = stratified_subsample(X, labels=labels, max_samples=30, random_seed=42)[0]
    assert np.array_equal(a, b)


def test_standardize_uses_training_statistics_only():
    train = np.array([[0.0], [2.0]])
    test = np.array([[4.0]])
    tr, te = standardize(train, test)
    assert tr.mean() == pytest.approx(0.0)
    assert te[0, 0] == pytest.approx(3.0)  # (4 - 1) / 1


def test_standardize_accepts_several_test_blocks():
    train = np.array([[0.0], [2.0]])
    tr, a, b = standardize(train, np.array([[4.0]]), np.array([[-2.0]]))
    assert a[0, 0] == pytest.approx(3.0) and b[0, 0] == pytest.approx(-3.0)


def test_knn_caps_k_at_the_training_set_size():
    """A fold can be smaller than k; that must degrade, not raise."""
    X = np.array([[0.0], [1.0]])
    y = np.array(["A", "B"])
    pred = knn_predict(X, y, np.array([[0.1]]), k=5)
    assert pred.tolist() == ["A"]


# ── bootstrap ────────────────────────────────────────────────────────


def test_donor_bootstrap_point_estimate_is_the_observed_value():
    rng = np.random.default_rng(0)
    y_true = rng.choice(["A", "B"], size=400)
    y_pred = np.where(rng.random(400) < 0.8, y_true, np.where(y_true == "A", "B", "A"))
    donors = rng.choice([f"d{i}" for i in range(8)], size=400)
    res = boot.donor_bootstrap(y_true, y_pred, donors, classes=["A", "B"], n_boot=200)
    direct = classification_metrics(y_true, y_pred, ["A", "B"])
    assert res["point"]["f1_score"] == pytest.approx(direct["f1_score"])
    lo, hi = res["ci"]["f1_score"]
    assert lo <= direct["f1_score"] <= hi
    assert res["n_donors"] == 8


def test_donor_bootstrap_is_wider_than_a_spot_level_one():
    """A donor effect must widen the interval; this is the whole reason for it."""
    rng = np.random.default_rng(1)
    y_true, y_pred, donors = [], [], []
    for d in range(6):
        acc = 0.5 + 0.4 * rng.random()  # per-donor difficulty
        truth = rng.choice(["A", "B"], size=300)
        pred = np.where(rng.random(300) < acc, truth, np.where(truth == "A", "B", "A"))
        y_true.append(truth)
        y_pred.append(pred)
        donors.append(np.full(300, f"d{d}"))
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    donors = np.concatenate(donors)

    by_donor = boot.donor_bootstrap(y_true, y_pred, donors, classes=["A", "B"], n_boot=400)
    by_spot = boot.donor_bootstrap(
        y_true, y_pred, np.arange(len(y_true)).astype(str), classes=["A", "B"], n_boot=400
    )
    donor_width = by_donor["ci"]["f1_score"][1] - by_donor["ci"]["f1_score"][0]
    spot_width = by_spot["ci"]["f1_score"][1] - by_spot["ci"]["f1_score"][0]
    assert donor_width > 2 * spot_width


def test_donor_bootstrap_resamples_whole_donors():
    """Every replicate must take all of a donor's spots or none of them.

    With one donor scored perfectly and the other not at all, resampling donors
    can only ever produce a handful of distinct values; resampling spots would
    produce a smooth distribution around the middle.
    """
    y_true = np.array(["A"] * 100 + ["B"] * 100)
    y_pred = np.concatenate([np.full(100, "A"), np.full(100, "A")])
    donors = np.array(["good"] * 100 + ["bad"] * 100)
    res = boot.donor_bootstrap(y_true, y_pred, donors, classes=["A", "B"], n_boot=300, seed=0)
    lo, hi = res["ci"]["accuracy"]
    assert {round(lo, 3), round(hi, 3)} <= {0.0, 0.5, 1.0}


def test_bootstrap_reports_no_interval_when_there_is_one_donor():
    """One patient cannot support an interval, and must not fake one."""
    y = np.array(["A", "B", "A", "B"])
    res = boot.donor_bootstrap(y, y, np.full(4, "d1"), classes=["A", "B"], n_boot=100)
    assert res["ci"] == {} and res["per_class_ci"] == {}
    assert res["n_donors"] == 1
    assert res["point"]["f1_score"] == pytest.approx(1.0)  # point estimate survives


def test_bootstrap_can_be_switched_off():
    y = np.array(["A", "B"] * 10)
    res = boot.donor_bootstrap(y, y, np.repeat(["d1", "d2"], 10), classes=["A", "B"], n_boot=0)
    assert res["ci"] == {} and res["n_boot"] == 0


def test_paired_delta_is_zero_and_tight_for_identical_predictions():
    rng = np.random.default_rng(2)
    y_true = rng.choice(["A", "B"], size=300)
    donors = rng.choice(["d0", "d1", "d2", "d3"], size=300)
    res = boot.paired_delta_ci(y_true, y_true, y_true, donors, classes=["A", "B"], n_boot=100)
    assert res["delta_f1_score"] == pytest.approx(0.0)
    assert res["ci"]["delta_f1_score"] == (pytest.approx(0.0), pytest.approx(0.0))
    # Every replicate is exactly zero, so both tails are 1.0. The doubled minimum
    # would be 2.0; a p-value is clamped to a probability.
    assert res["p_two_sided"] == pytest.approx(1.0)  # never resolves a sign


def test_paired_delta_detects_a_real_difference():
    rng = np.random.default_rng(3)
    y_true = rng.choice(["A", "B"], size=600)
    donors = rng.choice([f"d{i}" for i in range(6)], size=600)
    good = np.where(rng.random(600) < 0.9, y_true, np.where(y_true == "A", "B", "A"))
    bad = np.where(rng.random(600) < 0.6, y_true, np.where(y_true == "A", "B", "A"))
    res = boot.paired_delta_ci(y_true, good, bad, donors, classes=["A", "B"], n_boot=400)
    assert res["delta_f1_score"] > 0.1
    assert res["ci"]["delta_f1_score"][0] > 0  # interval excludes zero
    assert res["p_two_sided"] < 0.05


def test_paired_delta_is_antisymmetric():
    rng = np.random.default_rng(4)
    y_true = rng.choice(["A", "B"], size=200)
    donors = rng.choice([f"d{i}" for i in range(4)], size=200)
    a = rng.choice(["A", "B"], size=200)
    b = rng.choice(["A", "B"], size=200)
    ab = boot.paired_delta_ci(y_true, a, b, donors, classes=["A", "B"], n_boot=100)
    ba = boot.paired_delta_ci(y_true, b, a, donors, classes=["A", "B"], n_boot=100)
    assert ab["delta_f1_score"] == pytest.approx(-ba["delta_f1_score"])


def test_pool_replicates_spans_the_runs_it_pools():
    """A mixture quantile lies between the components' — never outside them."""
    a = np.linspace(0.0, 1.0, 500)
    b = np.linspace(0.5, 1.5, 500)
    lo, hi, p, n = boot.pool_replicates([a, b])
    assert n == 2
    assert min(np.percentile(a, 2.5), np.percentile(b, 2.5)) <= lo
    assert lo <= max(np.percentile(a, 2.5), np.percentile(b, 2.5))
    assert min(np.percentile(a, 97.5), np.percentile(b, 97.5)) <= hi
    assert hi <= max(np.percentile(a, 97.5), np.percentile(b, 97.5))


def test_pool_replicates_widens_when_the_runs_disagree():
    """The point of pooling: a run that lands elsewhere widens the interval."""
    agree = boot.pool_replicates([np.linspace(0.1, 0.2, 400)] * 3)
    differ = boot.pool_replicates(
        [np.linspace(0.1, 0.2, 400), np.linspace(0.1, 0.2, 400), np.linspace(0.4, 0.5, 400)]
    )
    assert (differ[1] - differ[0]) > (agree[1] - agree[0])


def test_pool_replicates_of_nothing_is_not_an_interval():
    lo, hi, p, n = boot.pool_replicates([])
    assert n == 0 and lo != lo and hi != hi and p != p


def test_pool_replicates_ignores_runs_that_produced_no_draws():
    """A scope with one donor yields no draws; it must not sink the others."""
    lo, hi, _p, n = boot.pool_replicates([np.linspace(0.0, 1.0, 100), None, np.array([])])
    assert n == 1 and lo == pytest.approx(np.percentile(np.linspace(0, 1, 100), 2.5))


def test_paired_delta_requires_aligned_predictions():
    """Unaligned rows would compare two models on different spots."""
    with pytest.raises(ValueError, match="aligned"):
        boot.paired_delta_ci(
            np.array(["A", "B"]), np.array(["A", "B"]), np.array(["A"]), np.array(["d1", "d1"])
        )


# ── organ-balanced (two-level) bootstrap ─────────────────────────────


def _two_organ_pred(seed=0):
    """Two organs, 3 donors each, with different class vocabularies per organ."""
    rng = np.random.default_rng(seed)
    y_true, y_pred, donors, tissues = [], [], [], []
    for organ, classes, acc in (("lung", ["NOR", "TUM"], 0.8), ("skin", ["Stroma", "Tumor"], 0.6)):
        for d in range(3):
            t = rng.choice(classes, size=60)
            p = np.where(rng.random(60) < acc, t, rng.choice(classes, size=60))
            y_true += list(t)
            y_pred += list(p)
            donors += [f"{organ}{d}"] * 60
            tissues += [organ] * 60
    return (np.array(y_true), np.array(y_pred), np.array(donors), np.array(tissues))


def _vocab():
    return {"lung": ["NOR", "TUM"], "skin": ["Stroma", "Tumor"]}


def _organ_f1(y_true, y_pred, tissues, organ, classes):
    m = tissues == organ
    return classification_metrics(y_true[m], y_pred[m], classes, score_classes=_vocab()[organ])[
        "f1_score"
    ]


def test_organ_balanced_point_is_the_unweighted_mean_of_the_organ_scores():
    yt, yp, dn, ts = _two_organ_pred()
    classes = sorted(set(yt) | set(yp))
    res = boot.tissue_balanced_bootstrap(
        yt, yp, dn, ts, classes=classes, score_classes_by_tissue=_vocab(), n_boot=0
    )
    expected = np.mean([_organ_f1(yt, yp, ts, o, classes) for o in ("lung", "skin")])
    assert res["point"] == pytest.approx(expected, abs=1e-12)
    assert res["n_organs"] == 2 and res["n_donors"] == 6


def test_organ_balanced_ignores_how_many_donors_each_organ_has():
    """Equal weight per organ is the point: duplicating one organ's donors must
    move the score only through that organ's own value, not by out-voting."""
    yt, yp, dn, ts = _two_organ_pred()
    classes = sorted(set(yt) | set(yp))
    base = boot.tissue_balanced_bootstrap(
        yt, yp, dn, ts, classes=classes, score_classes_by_tissue=_vocab(), n_boot=0
    )
    keep = ts == "skin"  # duplicate every skin donor
    yt2, yp2 = np.r_[yt, yt[keep]], np.r_[yp, yp[keep]]
    dn2 = np.r_[dn, np.char.add(dn[keep], "_copy")]
    ts2 = np.r_[ts, ts[keep]]
    dup = boot.tissue_balanced_bootstrap(
        yt2, yp2, dn2, ts2, classes=classes, score_classes_by_tissue=_vocab(), n_boot=0
    )
    assert dup["point"] == pytest.approx(base["point"], abs=1e-12)
    assert dup["n_donors"] == 9  # 6 + skin's 3 again; the count grows


def test_organ_balanced_interval_brackets_the_point():
    yt, yp, dn, ts = _two_organ_pred()
    classes = sorted(set(yt) | set(yp))
    res = boot.tissue_balanced_bootstrap(
        yt,
        yp,
        dn,
        ts,
        classes=classes,
        score_classes_by_tissue=_vocab(),
        n_boot=400,
        return_draws=True,
    )
    lo, hi = res["ci"]
    assert lo <= res["point"] <= hi
    assert len(res["draws"]) == 400


def test_organ_balanced_resamples_organs_as_well_as_donors():
    """Two organs that disagree must produce draws at *both* organ means, which a
    donors-only resample could never reach."""
    yt, yp, dn, ts = _two_organ_pred()
    classes = sorted(set(yt) | set(yp))
    res = boot.tissue_balanced_bootstrap(
        yt,
        yp,
        dn,
        ts,
        classes=classes,
        score_classes_by_tissue=_vocab(),
        n_boot=2000,
        seed=7,
        return_draws=True,
    )
    per_organ = [_organ_f1(yt, yp, ts, o, classes) for o in ("lung", "skin")]
    spread = abs(per_organ[0] - per_organ[1])
    # Drawing (lung, lung) or (skin, skin) pushes the mean toward one organ's value.
    assert res["draws"].max() > res["point"] + spread / 4
    assert res["draws"].min() < res["point"] - spread / 4


def test_organ_balanced_needs_at_least_two_organs():
    yt, yp, dn, ts = _two_organ_pred()
    one = ts == "lung"
    res = boot.tissue_balanced_bootstrap(
        yt[one],
        yp[one],
        dn[one],
        ts[one],
        classes=sorted(set(yt[one]) | set(yp[one])),
        score_classes_by_tissue={"lung": _vocab()["lung"]},
        n_boot=10,
    )
    assert res == {}


def test_organ_balanced_boot_row_uses_the_same_columns_as_every_other_boot_row():
    """Both row builders feed one ``bootstrap.csv``. A second name for the point
    estimate does not raise — it lands as a NaN in the column the readers use, and
    the interval then brackets nothing."""
    from types import SimpleNamespace

    from vgtfm.evaluate import reporting as rep

    yt, yp, dn, ts = _two_organ_pred()
    pred = {"y_true": yt, "y_pred": yp, "donor": dn, "tissue": ts, "sample_id": dn}
    classes = sorted(set(yt) | set(yp))
    vocab = {
        "global": [f"{t}{rep.TISSUE_SEP}{c}" for t in ("lung", "skin") for c in classes],
        **_vocab(),
    }
    cfg = SimpleNamespace(
        eval=SimpleNamespace(bootstrap_n=200, bootstrap_seed=42, bootstrap_ci=0.95),
        folds=SimpleNamespace(seed=42),
    )
    base = {"model": "pca", "protocol": "heldout_donor", "level": "cross_donor"}

    plain, _ = rep.bootstrap_rows(pred, classes, base, cfg, vocab, return_draws=True)
    _, organ_row, _ = rep.tissue_balanced_rows(pred, classes, base, cfg, vocab)

    assert organ_row is not None
    assert set(plain[0]) - set(organ_row) == set()
    assert organ_row["ci_lo"] <= organ_row["value"] <= organ_row["ci_hi"]


# ── patch transforms ─────────────────────────────────────────────────


def test_patch_transforms_break_correspondence_as_advertised():
    rng = np.random.default_rng(0)
    patch = rng.standard_normal((200, 8)).astype(np.float32)
    slides = np.repeat(["s1", "s2", "s3", "s4"], 50)

    for name in PATCH_TRANSFORMS:
        out, perm = apply_patch_transform(name, patch, slides, np.random.default_rng(42))
        assert out.shape == patch.shape
        fixed = float(np.mean(perm == np.arange(len(perm))))
        if name in ("none", "gaussian"):
            assert fixed == 1.0
        else:
            assert fixed < 0.1


def test_the_identity_transform_returns_the_patches_untouched():
    rng = np.random.default_rng(0)
    patch = rng.standard_normal((50, 4)).astype(np.float32)
    out, perm = apply_patch_transform("none", patch, np.zeros(50), rng)
    assert out is patch
    assert np.array_equal(perm, np.arange(50))


def test_shuffling_preserves_the_marginal_distribution_of_the_target():
    """Only the pairing may change: same patches, reordered."""
    rng = np.random.default_rng(0)
    patch = rng.standard_normal((120, 6)).astype(np.float32)
    slides = np.repeat(["s1", "s2", "s3"], 40)
    for name in ("within-sample", "global"):
        out, _ = apply_patch_transform(name, patch, slides, np.random.default_rng(1))
        assert np.allclose(np.sort(out, axis=0), np.sort(patch, axis=0))


def test_within_sample_shuffle_stays_inside_each_slide():
    rng = np.random.default_rng(0)
    patch = rng.standard_normal((200, 4)).astype(np.float32)
    slides = np.repeat(["s1", "s2", "s3", "s4"], 50)
    _out, perm = apply_patch_transform("within-sample", patch, slides, np.random.default_rng(0))
    assert np.array_equal(slides[perm], slides)
    assert sorted(perm.tolist()) == list(range(200))  # a true permutation


def test_global_shuffle_crosses_slide_boundaries():
    """The stronger control: most spots get a patch from a different slide."""
    rng = np.random.default_rng(0)
    patch = rng.standard_normal((200, 4)).astype(np.float32)
    slides = np.repeat(["s1", "s2", "s3", "s4"], 50)
    _out, perm = apply_patch_transform("global", patch, slides, np.random.default_rng(0))
    assert float(np.mean(slides[perm] == slides)) < 0.5


def test_gaussian_control_matches_the_per_dimension_moments():
    rng = np.random.default_rng(0)
    patch = (rng.standard_normal((5000, 3)) * [1.0, 5.0, 0.2] + [0.0, 3.0, -1.0]).astype(np.float32)
    out, _ = apply_patch_transform(
        "gaussian", patch, np.zeros(5000, dtype=int), np.random.default_rng(0)
    )
    assert np.allclose(out.mean(axis=0), patch.mean(axis=0), atol=0.15)
    assert np.allclose(out.std(axis=0), patch.std(axis=0), rtol=0.1)


def test_gaussian_control_keeps_no_structure_at_all():
    """The floor condition: correlations between dimensions must be destroyed."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal((4000, 1))
    patch = np.hstack([base, base * 2.0 + 0.01 * rng.standard_normal((4000, 1))]).astype(np.float32)
    assert abs(np.corrcoef(patch.T)[0, 1]) > 0.99
    out, _ = apply_patch_transform(
        "gaussian", patch, np.zeros(4000, dtype=int), np.random.default_rng(0)
    )
    assert abs(np.corrcoef(out.T)[0, 1]) < 0.1


def test_an_unknown_transform_names_the_valid_ones():
    with pytest.raises(ValueError, match="Unknown patch transform"):
        apply_patch_transform("rotate", np.zeros((4, 2)), np.zeros(4), np.random.default_rng(0))


# ── label canonicalisation ───────────────────────────────────────────


def test_tum_and_tumor_are_one_class():
    """TuPro and USZ annotated independently; the pooled average must not
    treat their two names for the same class as two classes."""
    from vgtfm.labels import canonical, canonical_labels

    assert canonical("TUM") == "Tumor"
    assert canonical("  tum ") == "Tumor"
    assert canonical("Tumor") == "Tumor"
    assert canonical_labels(np.array(["TUM", "Tumor", "TLS"], dtype=object)).tolist() == [
        "Tumor",
        "Tumor",
        "TLS",
    ]


def test_normal_is_not_merged_across_cohorts():
    """TuPro's `Normal lymphoid tissue` is lymphoid tissue; USZ's `NOR` is normal
    parenchyma, and USZ's lymphoid equivalent is `TLS`. Merging them on name
    similarity would assert a biological identity that does not hold.
    """
    from vgtfm.labels import canonical

    assert canonical("NOR") == "NOR"
    assert canonical("Normal lymphoid tissue") == "Normal lymphoid tissue"
    assert canonical("TLS") == "TLS"
    assert canonical("LN") == "LN"


def test_canonicalisation_leaves_unlabelled_values_alone():
    """Canonicalising must not turn "no label" into a class, or vice versa."""
    from vgtfm.labels import canonical, canonical_labels

    for value in ("UNASSIGNED", "", "nan", "Mixed", None):
        assert not labeled_mask(np.array([canonical(value)], dtype=object))[0]
    arr = np.array(["TUM", "UNASSIGNED", None], dtype=object)
    assert labeled_mask(canonical_labels(arr)).tolist() == [True, False, False]


def test_canonicalisation_is_idempotent():
    """It runs at cache build; a migrated cache must not shift again on reuse."""
    from vgtfm.labels import canonical_labels

    once = canonical_labels(np.array(["TUM", "Tumor", "NOR", ""], dtype=object))
    assert canonical_labels(once).tolist() == once.tolist()
