"""Diagnostics: effective rank, variance decomposition, CCA and its controls.

Each statistic is checked against something independent of its own implementation —
an analytic value, a brute-force computation, or a planted signal whose answer is
known in advance — because none of them fails loudly when wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from vgtfm.diagnostics import scib
from vgtfm.diagnostics.cca import cca, leave_one_slide_out, permute_within, spectrum_with_nulls
from vgtfm.diagnostics.effective_rank import (
    bootstrap_effective_rank,
    control_matrices,
    mean_pairwise_cosine,
)
from vgtfm.diagnostics.effective_rank import run as effective_rank_run
from vgtfm.diagnostics.variance import decompose, decompose_subsampled
from vgtfm.diagnostics.variance import run as variance_run
from vgtfm.models.nn import effective_rank


# ── effective rank ───────────────────────────────────────────────────


def test_effective_rank_is_invariant_to_rotation_and_scale():
    """It reads the shape of the spectrum, not the basis it is expressed in."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 12)) * np.array([5.0, 3.0, 1.0] * 4)
    Q = np.linalg.qr(rng.standard_normal((12, 12)))[0]
    assert effective_rank(X @ Q) == pytest.approx(effective_rank(X), rel=1e-9)
    assert effective_rank(X * 7.0) == pytest.approx(effective_rank(X), rel=1e-9)


def test_effective_rank_is_bounded_by_the_dimension():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((4000, 16))
    assert effective_rank(noise) <= 16.0 + 1e-9
    assert effective_rank(noise) > 15.0  # i.i.d. noise nearly saturates


def test_effective_rank_of_too_few_rows_is_zero():
    assert effective_rank(np.ones((1, 5))) == 0.0
    assert effective_rank(np.zeros((10, 5))) == 0.0


def test_bootstrap_effective_rank_reports_a_stable_mean_and_spread():
    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.standard_normal((40, 6)))[0]
    X = rng.standard_normal((3000, 6)) @ basis.T
    out = bootstrap_effective_rank(X, n_samples=800, n_iters=5, seed=0)
    assert out["eff_rank"] == pytest.approx(6.0, rel=0.05)
    assert out["eff_rank_min"] <= out["eff_rank"] <= out["eff_rank_max"]
    assert out["pct_of_dim"] == pytest.approx(100.0 * out["eff_rank"] / 40)
    assert out["n_samples"] == 800


def test_control_matrices_bracket_the_observed_scale():
    """Gaussian saturates, column-shuffling keeps marginals but kills correlation."""
    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.standard_normal((24, 3)))[0]
    X = (rng.standard_normal((600, 3)) @ basis.T).astype(np.float32)

    controls = control_matrices(X, seed=0, pca_components=8)
    assert set(controls) == {"gaussian", "column_shuffled", "pca8"}

    # Column shuffling must preserve every column's multiset of values exactly.
    for j in range(X.shape[1]):
        assert np.allclose(np.sort(controls["column_shuffled"][:, j]), np.sort(X[:, j]))
    # ...while destroying the low-rank structure it had.
    assert effective_rank(controls["column_shuffled"]) > 3 * effective_rank(X)
    assert effective_rank(controls["gaussian"]) > effective_rank(controls["column_shuffled"])


def test_mean_pairwise_cosine_matches_a_brute_force_average():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 5)) + 3.0  # off-centre: cosines are high
    got = mean_pairwise_cosine(X, n_samples=60, seed=0)

    unit = X / np.linalg.norm(X, axis=1, keepdims=True)
    pairs = [float(unit[i] @ unit[j]) for i in range(len(X)) for j in range(len(X)) if i != j]
    assert got == pytest.approx(float(np.mean(pairs)), abs=1e-10)


def test_mean_pairwise_cosine_detects_collapse():
    rng = np.random.default_rng(0)
    direction = rng.standard_normal(16)
    collapsed = np.abs(rng.standard_normal((200, 1))) * direction
    assert mean_pairwise_cosine(collapsed, seed=0) == pytest.approx(1.0, abs=1e-6)


# ── variance decomposition ───────────────────────────────────────────


def test_variance_decomposition_matches_a_one_way_anova_eta_squared():
    rng = np.random.default_rng(0)
    groups = np.repeat(["a", "b", "c", "d"], 50)
    X = rng.standard_normal((200, 1)) + np.array(
        [[{"a": 0, "b": 2, "c": -1, "d": 4}[g]] for g in groups]
    )
    res = decompose(X, groups)

    x = X[:, 0]
    ss_between = sum(
        int((groups == g).sum()) * (x[groups == g].mean() - x.mean()) ** 2
        for g in np.unique(groups)
    )
    assert res["between_ss"] == pytest.approx(ss_between)
    assert res["between_fraction"] == pytest.approx(ss_between / ((x - x.mean()) ** 2).sum())


def test_variance_decomposition_is_invariant_to_rotation():
    """A rotation cannot change how much variance a grouping explains."""
    rng = np.random.default_rng(0)
    groups = np.repeat(["a", "b"], 100)
    X = rng.standard_normal((200, 6)) + (groups == "a")[:, None] * 2.0
    Q = np.linalg.qr(rng.standard_normal((6, 6)))[0]
    assert decompose(X @ Q, groups)["between_fraction"] == pytest.approx(
        decompose(X, groups)["between_fraction"]
    )


def test_variance_decomposition_of_a_single_group_is_all_within():
    rng = np.random.default_rng(0)
    res = decompose(rng.standard_normal((50, 3)), np.repeat("only", 50))
    assert res["between_fraction"] == pytest.approx(0.0, abs=1e-12)
    assert res["n_groups"] == 1


def test_variance_decomposition_handles_too_few_rows():
    res = decompose(np.zeros((1, 3)), np.array(["a"]))
    assert res["n_groups"] == 0 and np.isnan(res["between_fraction"])


def test_subsampled_decomposition_records_the_original_size():
    rng = np.random.default_rng(0)
    groups = np.repeat(["a", "b"], 500)
    X = rng.standard_normal((1000, 2)) + (groups == "a")[:, None] * 3.0
    full = decompose(X, groups)
    sub = decompose_subsampled(X, groups, max_spots=400, seed=0)
    assert sub["subsampled_from"] == 1000
    assert sub["n"] == 400
    assert sub["between_fraction"] == pytest.approx(full["between_fraction"], abs=0.05)
    assert "subsampled_from" not in decompose_subsampled(X, groups, max_spots=5000)


# ── CCA ──────────────────────────────────────────────────────────────


def test_cca_recovers_a_planted_canonical_correlation():
    rng = np.random.default_rng(0)
    n = 4000
    shared = rng.standard_normal(n)
    X = np.column_stack([shared + 0.5 * rng.standard_normal(n), rng.standard_normal(n)])
    # y1 shares `shared` with x1 at a known correlation; y2 is pure noise.
    Y = np.column_stack([shared + 1.5 * rng.standard_normal(n), rng.standard_normal(n)])

    expected = (1 / np.sqrt(1 + 0.25)) * (1 / np.sqrt(1 + 2.25))
    rho = cca(X, Y, ridge=1e-8)["rho"]
    assert rho[0] == pytest.approx(expected, abs=0.03)
    assert rho[1] < 0.1  # nothing else is shared


def test_cca_is_invariant_to_invertible_linear_maps_of_either_view():
    """Canonical correlations are basis-free; that is the point of the whitening."""
    rng = np.random.default_rng(1)
    shared = rng.standard_normal((500, 2))
    X = np.hstack([shared, rng.standard_normal((500, 2))])
    Y = np.hstack([shared * 0.7, rng.standard_normal((500, 2))])

    A = rng.standard_normal((4, 4))
    B = rng.standard_normal((4, 4))
    base = cca(X, Y, ridge=1e-10)["rho"]
    mapped = cca(X @ A, Y @ B, ridge=1e-10)["rho"]
    assert np.allclose(base, mapped, atol=1e-6)


def test_cca_of_identical_views_is_perfectly_correlated():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 4))
    assert np.allclose(cca(X, X.copy(), ridge=1e-10)["rho"], 1.0, atol=1e-6)


def test_cca_correlations_are_bounded():
    rng = np.random.default_rng(0)
    rho = cca(rng.standard_normal((80, 20)), rng.standard_normal((80, 20)))["rho"]
    assert rho.min() >= 0.0 and rho.max() <= 1.0


def test_permute_within_keeps_every_row_inside_its_group():
    rng = np.random.default_rng(0)
    groups = np.repeat(["s1", "s2", "s3"], 40)
    perm = permute_within(groups, rng)
    assert np.array_equal(groups[perm], groups)
    assert sorted(perm.tolist()) == list(range(len(groups)))
    assert float(np.mean(perm == np.arange(len(perm)))) < 0.2


def test_within_slide_null_absorbs_coupling_that_is_only_slide_identity():
    """Coupling built entirely from slide means must not survive as spot signal.

    Both views get the same per-slide offset and independent noise. The
    within-slide permutation preserves that offset, so the null reproduces the
    observed spectrum and the spot-level fraction collapses toward zero.
    """
    rng = np.random.default_rng(0)
    slides = np.repeat([f"s{i}" for i in range(6)], 100)
    offset = {s: rng.standard_normal(3) * 6.0 for s in np.unique(slides)}
    shift = np.array([offset[s] for s in slides])
    X = shift + rng.standard_normal((600, 3))
    Y = shift + rng.standard_normal((600, 3))

    spec = spectrum_with_nulls(X, Y, slides, n_permutations=12, seed=0)
    assert spec["rho1_observed"] > 0.8  # looks impressive
    assert spec["spot_level_fraction_mean_null"] < 0.1  # and is entirely batch


def test_within_slide_null_leaves_genuine_spot_level_coupling_standing():
    """The complement: real per-spot coupling must clear its own null."""
    rng = np.random.default_rng(0)
    slides = np.repeat([f"s{i}" for i in range(6)], 100)
    shared = rng.standard_normal((600, 3))
    X = shared + 0.3 * rng.standard_normal((600, 3))
    Y = shared + 0.3 * rng.standard_normal((600, 3))

    spec = spectrum_with_nulls(X, Y, slides, n_permutations=12, seed=0)
    assert spec["rho1_observed"] > spec["rho1_null_within_q95"] + 0.3
    assert spec["spot_level_fraction_mean_null"] > 0.8


def test_leave_one_slide_out_separates_paired_from_unpaired_donors():
    """Patient leakage, planted and then measured.

    Donor ``p`` contributes two slides and its cross-modal map is donor-specific;
    the singleton donors get their own maps. Holding out one of ``p``'s slides
    leaves the other behind, so the projection transfers; holding out a singleton
    leaves nothing behind and it does not.
    """
    rng = np.random.default_rng(0)
    slides, donors, Xs, Ys = [], [], [], []
    donor_map = {}
    for slide, donor in [("p-1", "p"), ("p-2", "p"), ("q-1", "q"), ("r-1", "r")]:
        if donor not in donor_map:
            donor_map[donor] = np.linalg.qr(rng.standard_normal((4, 4)))[0]
        shared = rng.standard_normal((150, 4))
        Xs.append(shared + 0.05 * rng.standard_normal((150, 4)))
        Ys.append(shared @ donor_map[donor] + 0.05 * rng.standard_normal((150, 4)))
        slides += [slide] * 150
        donors += [donor] * 150

    out = leave_one_slide_out(
        np.vstack(Xs), np.vstack(Ys), np.array(slides), np.array(donors), ridge=1e-6
    )
    assert out["n_paired"] == 2 and out["n_unpaired"] == 2
    assert out["patient_leakage_gap"] > 0.2
    assert out["rho1_test_paired_mean"] > out["rho1_test_unpaired_mean"]


def test_leave_one_slide_out_skips_slides_that_are_too_small():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 3))
    slides = np.array(["big"] * 55 + ["tiny"] * 5)
    out = leave_one_slide_out(X, rng.standard_normal((60, 3)), slides, slides)
    assert out["per_slide"] == [] and np.isnan(out["rho1_test_mean"])


# ── scIB panel ───────────────────────────────────────────────────────


def test_scib_panels_are_split_and_the_composite_is_never_reported():
    """A weighted total lets a method trade biology for batch mixing; it is dropped."""
    metrics = {
        "ilisi": 0.04,
        "kbet": 0.2,
        "kmeans_nmi": 0.5,
        "clisi": 0.9,
        "total": 0.42,
        "batch_correction": 0.3,
        "bio_conservation": 0.6,
        "some_new_metric": 0.1,
        "n_spots": 100,
        "n_batches": 4,
    }
    batch, bio, other = scib.split_panels(metrics)

    assert set(batch) == {"ilisi", "kbet"}
    assert set(bio) == {"kmeans_nmi", "clisi"}
    assert set(other) == {"some_new_metric"}  # unknown, kept and flagged
    # No composite reaches any panel, whoever calls this.
    assert not (set(batch) | set(bio) | set(other)) & set(scib.COMPOSITE_METRICS)


def test_scib_metric_names_do_not_overlap_between_panels():
    assert not set(scib.BATCH_METRICS) & set(scib.BIO_METRICS)


# ── the stage wrappers ───────────────────────────────────────────────
#
# Every statistic above is checked against something independent, but the two
# functions that choose *which rows* to hand them were not, and that is where the
# `diagnose` stage was broken: the split's row indices and the accumulator for the
# output records shared the name `rows`, so each statistic was fed the empty
# selection `X[[]]`. Effective rank died on the cluster inside PCA ("Found array
# with 0 sample(s)"); the variance decomposition returned an empty frame without a
# word. Row counts are what the bug got wrong, so row counts are what these check.


def diagnostics_table(n_train=60, n_test=30, seed=0):
    """A cohort with two splits, three slides and two donors.

    The ids encode the donor, because the registry's `donor_pattern` is what the
    `patient` grouping reads: `A-3-1` labelled donor `B` would be a fixture whose
    two columns disagree.
    """
    import pandas as pd

    from vgtfm.data.tables import SpotTable

    rng = np.random.default_rng(seed)
    n = n_train + n_test
    meta = pd.DataFrame(
        {
            "split": ["train"] * n_train + ["test"] * n_test,
            "sample_id": [("A-1-1", "A-1-2", "B-1-1")[i % 3] for i in range(n)],
            "donor": ["A" if i % 3 < 2 else "B" for i in range(n)],
            "tissue": ["mel"] * n,  # constant: never a grouping
            "dataset_id": ["10x_TuPro"] * n,  # constant: never a grouping
        }
    )
    return SpotTable(
        gene=rng.standard_normal((n, 8)).astype(np.float32),
        patch=rng.standard_normal((n, 5)).astype(np.float32),
        meta=meta,
        substrate="geneformer",
    )


def test_the_patient_grouping_collapses_a_cohorts_multi_slide_patients():
    """The defect this grouping replaces: `SpotTable.donor` parses TuPro ids and
    falls back to the slide id, so on the train split — where the multi-slide
    patients actually live — every slide was its own patient and the between-donor
    fraction was the between-slide one by construction."""
    from vgtfm.data.tables import load_registry
    from vgtfm.diagnostics.variance import _grouping_values

    cfg = diagnostics_config("all")
    slides = ["P10-B1", "P10-B2", "P10-T1", "P11-B1", "P11-B2", "KC1"]
    datasets = ["LUNG_CANCER_LUAD"] * 3 + ["LUNG_CANCER_LUSC"] * 2 + ["TLS_VISIUM_USZ_kidney"]
    table = _slide_table(slides, datasets)
    rows = np.arange(len(slides))

    assert list(_grouping_values(cfg, table, rows, "sample_id")) == slides
    # Three patients where the slide-level key would have found six.
    assert list(_grouping_values(cfg, table, rows, "patient")) == [
        "P10",
        "P10",
        "P10",
        "P11",
        "P11",
        "KC1",
    ]
    # ...and the key really is the registry's, not a second heuristic.
    assert list(load_registry(cfg.paths.datasets_json).patient_keys(slides, datasets)) == [
        "P10",
        "P10",
        "P10",
        "P11",
        "P11",
        "KC1",
    ]


def test_a_cohort_with_one_slide_per_patient_is_left_alone():
    """USZ and MOSAIC set `donor_pattern: null`; every slide really is a patient."""
    from vgtfm.diagnostics.variance import _grouping_values

    slides = ["KC1", "KC3", "LC1"]
    table = _slide_table(slides, ["TLS_VISIUM_USZ_kidney"] * 2 + ["TLS_VISIUM_USZ_lung"])
    got = _grouping_values(diagnostics_config("all"), table, np.arange(len(slides)), "patient")
    assert list(got) == slides


def _slide_table(slides, datasets):
    import pandas as pd

    from vgtfm.data.tables import SpotTable

    meta = pd.DataFrame(
        {
            "split": ["train"] * len(slides),
            "sample_id": slides,
            "donor": slides,
            "tissue": ["lung"] * len(slides),
            "dataset_id": datasets,
        }
    )
    n = len(slides)
    return SpotTable(
        gene=np.zeros((n, 3), dtype=np.float32),
        patch=np.zeros((n, 3), dtype=np.float32),
        meta=meta,
        substrate="geneformer",
    )


def diagnostics_config(split="train", **overrides):
    from vgtfm.config import Config

    cfg = Config()
    cfg.diagnostics.split = split
    cfg.diagnostics.n_samples = 40
    cfg.diagnostics.n_iters = 2
    cfg.diagnostics.include_controls = False
    for k, v in overrides.items():
        setattr(cfg.diagnostics, k, v)
    return cfg


SPLIT_SIZES = [("train", 60), ("test", 30), ("all", 90)]


@pytest.mark.parametrize("split,n_rows", SPLIT_SIZES)
def test_effective_rank_scores_the_rows_of_the_requested_split(split, n_rows):
    table = diagnostics_table()
    er = effective_rank_run(diagnostics_config(split), table)

    observed = er[er.variant == "observed"]
    assert list(observed.column) == ["gene_features", "patch_features"]
    assert list(observed.dim) == [table.gene_dim, table.patch_dim]
    assert set(observed.n_total) == {n_rows}


def test_effective_rank_builds_its_controls_from_the_split_too():
    """The exact path that crashed on the cluster: a PCA of the selected rows."""
    er = effective_rank_run(diagnostics_config(include_controls=True), diagnostics_table())

    assert {"observed", "gaussian", "column_shuffled"} <= set(er.variant)
    assert any(v.startswith("pca") for v in er.variant)
    assert er.n_total.min() > 0


@pytest.mark.parametrize("split,n_rows", SPLIT_SIZES)
def test_variance_decomposition_runs_on_the_rows_of_the_requested_split(split, n_rows):
    var = variance_run(diagnostics_config(split), diagnostics_table())

    assert not var.empty, "the shadowed `rows` made every grouping look constant"
    assert set(var.grouping) == {"sample_id", "patient"}
    # Three slides, two patients — so the two groupings are genuinely different
    # numbers, which is the whole point of reporting both.
    n_groups = dict(zip(var.grouping, var.n_groups))
    assert (n_groups["sample_id"], n_groups["patient"]) == (3, 2)
    assert set(var.n) == {n_rows}
    assert set(var.column) == {"gene_features", "patch_features"}


def test_a_split_that_selects_no_spots_is_refused_before_any_statistic_runs():
    with pytest.raises(SystemExit, match="selects no spots"):
        effective_rank_run(diagnostics_config("validation"), diagnostics_table())


# ── surviving a native crash in an integration method ────────────────


def test_a_correction_that_segfaults_is_reported_not_fatal():
    """`except Exception` cannot see a signal — the process is simply gone — so a
    correction that can fault in compiled code has to run somewhere else, or it
    takes every method already scored down with it."""
    import faulthandler
    import os
    import signal

    from vgtfm.diagnostics.integration import NativeCrash, _isolated

    def dies():
        # pytest installs a fault handler that would dump the child's C stack into
        # the test log. The crash is the point here, not its traceback.
        faulthandler.disable()
        os.kill(os.getpid(), signal.SIGILL)

    with pytest.raises(NativeCrash, match=f"killed by signal {signal.SIGILL}"):
        _isolated(dies, name="bbknn")


def test_an_isolated_correction_returns_its_matrix_unchanged():
    from vgtfm.diagnostics.integration import _isolated

    X = np.arange(24, dtype=np.float32).reshape(6, 4)
    assert np.array_equal(_isolated(lambda: X * 2.0, name="harmony"), X * 2.0)


def test_an_isolated_graph_only_method_still_returns_none():
    """BBKNN's `None` means "no embedding to probe" and must survive the round
    trip as `None`, not as an empty array."""
    from vgtfm.diagnostics.integration import _isolated

    assert _isolated(lambda: None, name="bbknn") is None


def test_a_correction_that_hangs_is_killed_and_recorded():
    """The failure that actually happened: forking after JAX and OpenMP have started
    threads can deadlock the child rather than fault it. Without a deadline the
    stage waits until SLURM kills the job, discarding every method already scored."""
    import time

    from vgtfm.diagnostics.integration import NativeCrash, _isolated

    with pytest.raises(NativeCrash, match="timed out after 1s and was killed"):
        _isolated(lambda: time.sleep(120), name="bbknn", timeout_s=1.0)


def test_a_hang_is_a_recorded_row_like_any_other_crash():
    """...and, like a signal, it must become a value so the comparison survives."""
    import time

    from vgtfm.diagnostics.integration import _correct_or_crash

    Z, crash = _correct_or_crash(lambda: time.sleep(120), name="bbknn", isolate=True, timeout_s=1.0)
    assert Z is None and "timed out" in crash


def test_a_timeout_leaves_no_child_behind():
    """A deadlocked child that outlived the stage would keep its share of the node."""
    import multiprocessing
    import time

    from vgtfm.diagnostics.integration import NativeCrash, _isolated

    before = set(multiprocessing.active_children())
    with pytest.raises(NativeCrash):
        _isolated(lambda: time.sleep(120), name="bbknn", timeout_s=1.0)
    assert not {p for p in multiprocessing.active_children() if p.is_alive()} - before


def test_no_deadline_means_no_deadline():
    """`isolate_timeout_s = 0` restores the unbounded wait, for a method known to be
    slow rather than stuck."""
    from vgtfm.diagnostics.integration import _isolated

    X = np.arange(6, dtype=np.float32).reshape(3, 2)
    assert np.array_equal(_isolated(lambda: X, name="harmony", timeout_s=0.0), X)


def test_only_a_native_crash_becomes_a_value():
    """An ordinary exception must still refuse on the spot."""
    from vgtfm.diagnostics.integration import _correct_or_crash

    Z, crash = _correct_or_crash(
        lambda: np.zeros((2, 2), dtype=np.float32), name="combat", isolate=False
    )
    assert crash is None and Z.shape == (2, 2)

    with pytest.raises(ValueError):
        _correct_or_crash(_raise_value_error, name="combat", isolate=False)


def _raise_value_error():
    raise ValueError("combat failed on its own terms")


def test_the_panel_drops_composites_and_normalises_the_metric_names(monkeypatch):
    """scib-metrics returns display names with spaces and a weighted total. The
    total must never reach a panel — it lets a method trade biology for batch
    mixing and report the trade as an improvement."""
    import sys
    import types as _types

    import pandas as pd

    class _Bench:
        def __init__(self, *a, **k):
            pass

        def benchmark(self):
            pass

        def get_results(self, min_max_scale=False):
            assert min_max_scale is False, "scaled values are only relative"
            return pd.DataFrame(
                [
                    {
                        "iLISI": 0.2,
                        "KBET": 0.4,
                        "Graph connectivity": 0.9,
                        "KMeans NMI": 0.7,
                        "cLISI": 0.99,
                        "Batch correction": 0.5,
                        "Bio conservation": 0.8,
                        "Total": 0.68,
                        "Brand new metric": 0.1,
                    }
                ]
            )

    module = _types.ModuleType("scib_metrics.benchmark")
    module.Benchmarker = _Bench
    module.BatchCorrection = lambda *a, **k: None
    module.BioConservation = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "scib_metrics.benchmark", module)

    rng = np.random.default_rng(0)
    labels = np.array(["Tumor", "Stroma"] * 10)
    out = scib.benchmark(
        rng.standard_normal((20, 4)), np.array(["s1", "s2"] * 10), labels, max_spots=20
    )

    assert set(out) & set(scib.COMPOSITE_METRICS) == set()
    assert out["ilisi"] == 0.2 and out["kmeans_nmi"] == 0.7
    assert out["graph_connectivity"] == 0.9, "spaces become underscores"
    assert out["n_spots"] == 20 and out["n_batches"] == 2

    batch, bio, other = scib.split_panels(out)
    assert set(batch) == {"ilisi", "kbet", "graph_connectivity"}
    assert set(bio) == {"kmeans_nmi", "clisi"}
    assert set(other) == {"brand_new_metric"}, "a new metric is reported, not dropped"


def test_the_panel_says_so_when_there_is_nothing_annotated_to_score():
    labels = np.array(["UNASSIGNED"] * 8)
    out = scib.benchmark(np.zeros((8, 3), dtype=np.float32), np.array(["s1"] * 8), labels)
    assert out == {"error": "no annotated spots"}


# ── scoring a correction that returns a graph instead of an embedding ──


def _graph_fixture(n_batches=40, per_batch=16, dim=8, seed=3):
    """Enough batches that BBKNN's `neighbors_within_batch * n_batches` clears the
    k=90 the panel scores iLISI and cLISI at."""
    rng = np.random.default_rng(seed)
    n = n_batches * per_batch
    X = rng.standard_normal((n, dim)).astype(np.float32)
    batch = np.array([f"s{i % n_batches}" for i in range(n)])
    cls = np.array(["TUM" if i % 2 else "STR" for i in range(n)])
    return X, batch, cls


def test_bbknn_returns_each_spot_as_its_own_first_neighbour_then_the_rest_in_order():
    """Every metric below reads "the k nearest neighbours" off the front of each row,
    so an unsorted row silently turns that into "k arbitrary neighbours"."""
    from vgtfm.diagnostics.integration import correct_bbknn

    X, batch, _ = _graph_fixture(n_batches=6, per_batch=10)
    g = correct_bbknn(X, batch)

    assert g.indices.shape == g.distances.shape == (len(X), g.k)
    assert np.array_equal(g.indices[:, 0], np.arange(len(X)))
    assert np.all(np.diff(g.distances, axis=1) >= 0)


def test_bbknn_is_given_every_component_and_not_just_the_first_fifty():
    """bbknn defaults to `n_pcs=50` and truncates in silence. Every other method in
    the comparison corrects all `models.pca_components`, so the default would score
    BBKNN on 50 features against the rest on 128 and read the gap as a method effect.
    """
    from vgtfm.diagnostics.integration import correct_bbknn

    rng = np.random.default_rng(1)
    n = 96
    X = np.zeros((n, 60), dtype=np.float32)
    X[:, 50:] = rng.standard_normal((n, 10))  # every informative column is past 50
    batch = np.array([f"s{i % 6}" for i in range(n)])

    # Truncated at 50 PCs every spot is identical and every distance is zero.
    assert correct_bbknn(X, batch).distances[:, 1:].max() > 0


def test_the_graph_panel_scores_the_graph_metrics_and_leaves_the_rest_absent():
    """Absent, not zero: a method with no coordinates has no silhouette, and filling
    one in would let it be averaged into a comparison it never took part in."""
    from vgtfm.diagnostics.integration import correct_bbknn

    X, batch, cls = _graph_fixture()
    m = scib.benchmark_graph(correct_bbknn(X, batch), batch, cls)
    b, bio, other = scib.split_panels(m)

    assert set(b) == {"ilisi", "kbet", "graph_connectivity"}
    assert set(bio) == {"clisi"}
    assert not other
    assert all(v is not None and 0.0 <= v <= 1.0 for v in {**b, **bio}.values())
    assert m["n_spots"] == len(X) and m["n_batches"] == len(np.unique(batch))


def test_a_graph_narrower_than_the_panel_scores_at_is_refused():
    """Truncating to k is what makes a supplied graph comparable to one the
    Benchmarker built for itself; a graph that cannot reach k is not comparable."""
    from vgtfm.diagnostics.integration import NeighborGraph

    n, k = 60, 10
    g = NeighborGraph(
        indices=np.tile(np.arange(k), (n, 1)), distances=np.tile(np.arange(k, dtype=float), (n, 1))
    )
    with pytest.raises(SystemExit, match="neighbours per spot"):
        scib.benchmark_graph(g, np.array(["a", "b"] * (n // 2)), np.array(["x"] * n))


def test_the_graph_panel_refuses_labels_that_do_not_match_the_graph():
    """The graph is built on a subsample; scoring it against the full table's labels
    would silently pair each spot with someone else's annotation."""
    from vgtfm.diagnostics.integration import NeighborGraph

    n = 120
    g = NeighborGraph(
        indices=np.tile(np.arange(95), (n, 1)),
        distances=np.tile(np.arange(95, dtype=float), (n, 1)),
    )
    with pytest.raises(SystemExit, match="rows but"):
        scib.benchmark_graph(g, np.array(["a"] * 4), np.array(["x"] * 4))


def test_scoring_rows_picks_annotated_spots_only_and_picks_them_reproducibly():
    batch = np.array([f"s{i % 3}" for i in range(300)])
    cls = np.array([("TUM", "STR", "UNASSIGNED")[i % 3] for i in range(300)])

    rows = scib.scoring_rows(batch, cls, max_spots=50, seed=42)

    assert len(rows) == 50
    assert set(cls[rows]) == {"TUM", "STR"}
    assert np.array_equal(rows, scib.scoring_rows(batch, cls, max_spots=50, seed=42))


def test_a_graph_only_method_is_fitted_on_exactly_the_spots_the_panel_scores():
    """A corrected matrix can be subsampled after the fact; a neighbour graph cannot,
    so the two paths have to agree on the rows or they are scoring different cohorts.
    """
    batch = np.array([f"s{i % 5}" for i in range(400)])
    cls = np.array([("TUM", "STR", "UNASSIGNED")[i % 3] for i in range(400)])
    embedding = np.arange(400, dtype=np.float32).reshape(400, 1)

    rows = scib.scoring_rows(batch, cls, max_spots=60, seed=7)
    # `benchmark` reaches its subsample through the same helper, so the rows it keeps
    # are these ones — recorded here as the row count it reports back.
    assert scib.benchmark(np.repeat(embedding, 4, axis=1), batch, cls, max_spots=60, seed=7)[
        "n_spots"
    ] == len(rows)


def test_a_graph_only_method_cannot_be_isolated():
    """`_isolated` carries an array or None back from its child; a NeighborGraph
    would go through `np.asarray` and come back as a mangled object array."""
    from vgtfm.diagnostics import integration

    cfg = diagnostics_config()
    cfg.diagnostics.isolate_methods = ("bbknn",)
    with pytest.raises(SystemExit, match="isolate_methods"):
        integration.run(cfg)


def test_the_graph_plan_widens_the_neighbourhood_until_the_panel_can_score_it():
    """BBKNN's width is `neighbors_within_batch * n_batches`. On this cohort only ~24
    slides carry annotated spots, so its default of 3 yields 72 — short of the k=90
    the panel scores iLISI at, and not comparable with the other methods' graphs."""
    from vgtfm.diagnostics.integration import _graph_plan

    batch = np.array([f"s{i % 24}" for i in range(24 * 400)])
    rows, per_batch = _graph_plan(np.arange(len(batch)), batch, min_neighbors=90)

    assert per_batch == 4 and per_batch * 24 >= 90
    assert len(rows) == len(batch)


def test_the_graph_plan_keeps_bbknns_own_default_when_the_batches_are_plentiful():
    from vgtfm.diagnostics.integration import BBKNN_NEIGHBORS_WITHIN_BATCH, _graph_plan

    batch = np.array([f"s{i % 60}" for i in range(60 * 20)])
    _, per_batch = _graph_plan(np.arange(len(batch)), batch, min_neighbors=90)

    assert per_batch == BBKNN_NEIGHBORS_WITHIN_BATCH


def test_the_graph_plan_drops_a_batch_too_small_to_supply_its_neighbours():
    """BBKNN refuses outright if any batch is smaller than `neighbors_within_batch`,
    and the scored spots are a class-stratified draw, not a slide-stratified one."""
    from vgtfm.diagnostics.integration import _graph_plan

    batch = np.array([f"s{i % 24}" for i in range(24 * 400)])
    batch = np.concatenate([batch, np.array(["tiny", "tiny"])])
    rows, per_batch = _graph_plan(np.arange(len(batch)), batch, min_neighbors=90)

    assert "tiny" not in set(batch[rows])
    assert len(rows) == 24 * 400 and per_batch == 4


def test_a_cohort_that_cannot_reach_the_panels_k_is_refused_rather_than_scored():
    from vgtfm.diagnostics.integration import _graph_plan

    batch = np.array([f"s{i}" for i in range(4)])  # one spot per batch
    with pytest.raises(SystemExit, match="neighbours per spot"):
        _graph_plan(np.arange(len(batch)), batch, min_neighbors=90)


# ── the integrate stage's probe column ───────────────────────────────


def _probe_pred(y_pred=None):
    """One pooled prediction vector, in the shape `reporting.pool` returns.

    Two organs sharing a class name, which is the case the pooled `global` scope
    exists for: `Tumor` in skin and `Tumor` in lung are separate cells.
    """
    tissue = np.array(["skin"] * 8 + ["lung"] * 8)
    y_true = np.array(["Tumor", "Stroma"] * 4 + ["Tumor", "TLS"] * 4)
    donor = np.array([f"d{i // 2}" for i in range(16)])
    return {
        "y_true": y_true,
        "y_pred": np.array(y_true if y_pred is None else y_pred),
        "donor": donor,
        "sample_id": np.array([f"s{i // 4}" for i in range(16)]),
        "tissue": tissue,
        "n_train": 100,
    }


def _probe_config(n_boot=64):
    from vgtfm.config import Config

    cfg = Config()
    cfg.eval.bootstrap_n = n_boot
    cfg.eval.bootstrap_seed = 7
    return cfg


def test_the_probe_counts_a_prediction_outside_the_held_out_vocabulary():
    """The bug this stage carried until the intervals went in: scoring against a
    vocabulary taken from `y_true` alone leaves no column for a class the probe
    predicted but the held-out slides do not carry, so `confusion_counts` drops
    those rows instead of counting them as errors.

    Here that is the difference between a wrong answer and a perfect score. Two of
    the four skin `Stroma` spots are predicted `Necrosis` — a class the cohort has
    and these rows do not. Dropping them leaves `Stroma` with two spots, both
    correct, and the naive macro-F1 is 1.0 despite two errors; counting them gives
    the class a recall of 0.5.
    """
    from vgtfm.evaluate.probes import classification_metrics

    from vgtfm.diagnostics.integration import _probe_scores

    pred = _probe_pred()
    skin_stroma = np.flatnonzero((pred["tissue"] == "skin") & (pred["y_true"] == "Stroma"))
    # `astype(object)` first: the true labels are `<U6`, and assigning a longer
    # class name into that array truncates it to "Necros" instead of failing.
    y_pred = pred["y_true"].astype(object)
    y_pred[skin_stroma[:2]] = "Necrosis"
    wrong = _probe_pred(y_pred)
    classes = ["Necrosis", "Stroma", "TLS", "Tumor"]

    naive = classification_metrics(
        wrong["y_true"], wrong["y_pred"], sorted(set(wrong["y_true"].tolist()))
    )
    assert naive["f1_score"] == 1.0  # what this stage used to print

    scored, *_ = _probe_scores(
        _probe_config(0), {"cross_donor": wrong}, classes, method="harmony", seed=42
    )
    # (1 + 2/3 + 1 + 1) / 4 over skin|Tumor, skin|Stroma, lung|Tumor, lung|TLS.
    assert scored["f1/global/cross_donor"] == pytest.approx(0.9166667)


def test_the_probe_carries_a_donor_interval_and_its_donor_count():
    from vgtfm.diagnostics.integration import _probe_scores

    cols, *_ = _probe_scores(
        _probe_config(),
        {"cross_donor": _probe_pred()},
        ["Stroma", "TLS", "Tumor"],
        method="none",
        seed=42,
    )

    assert cols["f1/global/cross_donor_n_donors"] == 8
    lo, hi = cols["f1/global/cross_donor_lo"], cols["f1/global/cross_donor_hi"]
    assert lo <= cols["f1/global/cross_donor"] <= hi

    # Each organ is scored on its own, over its own donors: the pooled view is
    # one row of the table and not the only one.
    assert cols["f1/skin/cross_donor_n_donors"] == 4
    assert cols["f1/lung/cross_donor_n_donors"] == 4


def test_the_paired_delta_is_the_difference_of_the_two_point_estimates():
    """The delta is bootstrapped on a shared donor resample, but its point estimate
    is the observed difference — so it has to agree with the two absolute columns,
    which is what lets the table print them side by side."""
    from vgtfm.diagnostics.integration import _probe_scores

    ref = _probe_pred()
    y_pred = np.where(ref["y_true"] == "TLS", "Tumor", ref["y_true"])
    worse = {"cross_donor": _probe_pred(y_pred)}
    classes = ["Stroma", "TLS", "Tumor"]
    cfg = _probe_config()

    base, no_deltas, *_ = _probe_scores(cfg, {"cross_donor": ref}, classes, method="none", seed=42)
    cols, deltas, *_ = _probe_scores(
        cfg, worse, classes, method="harmony", seed=42, reference={"cross_donor": ref}
    )

    assert no_deltas == []
    by_scope = {d["scope"]: d for d in deltas}
    assert set(by_scope) == {"global", "skin", "lung"}
    assert all(d["method"] == "harmony" for d in deltas)
    for scope, d in by_scope.items():
        assert d["delta_f1"] == pytest.approx(
            cols[f"f1/{scope}/cross_donor"] - base[f"f1/{scope}/cross_donor"]
        )
    assert by_scope["global"]["delta_f1"] < 0
    assert by_scope["global"]["n_donors"] == 8
    # Only lung carries TLS, so only lung pays for predicting it as Tumor.
    assert by_scope["lung"]["delta_f1"] < 0
    assert by_scope["skin"]["delta_f1"] == pytest.approx(0.0)


def test_a_reference_describing_other_rows_is_refused_rather_than_differenced():
    from vgtfm.diagnostics.integration import _probe_scores

    other = _probe_pred()
    other["y_true"] = np.roll(other["y_true"], 1)
    with pytest.raises(SystemExit, match="different rows"):
        _probe_scores(
            _probe_config(0),
            {"cross_donor": _probe_pred()},
            ["Stroma", "TLS", "Tumor"],
            method="harmony",
            seed=42,
            reference={"cross_donor": other},
        )


# ── the uncorrected reference is the trained `pca` baseline ──────────────


def _integrate_config(tmp_path, **overrides):
    from vgtfm.config import Config

    cfg = Config()
    cfg.run_name = "t"
    cfg.paths.artifact_root = str(tmp_path)
    for k, v in overrides.items():
        setattr(cfg.diagnostics, k, v)
    return cfg


def test_the_uncorrected_row_reuses_the_trained_pca_embedding(tmp_path):
    """`integrate` corrects the matrix `eval` scores, not a fresh PCA of the genes.

    The two are the same width and look interchangeable, and are not: `train` fits
    the baseline on `train.fit_split`, which carries none of the evaluated donors,
    while a PCA fitted inside this stage sees every spot in the cohort. Recomputing
    is what made the `none` row disagree with Table 1's PCA row.
    """
    from vgtfm.diagnostics.integration import _reference_embedding
    from vgtfm.models.train import embedding_path

    table = diagnostics_table()
    cfg = _integrate_config(tmp_path)
    n = len(table.gene)
    # Deliberately not a PCA of anything: a recomputed basis cannot reproduce it.
    saved = np.arange(n * 4, dtype=np.float32).reshape(n, 4)
    np.save(embedding_path(cfg, "pca", cfg.diagnostics.seed), saved)

    assert np.array_equal(_reference_embedding(cfg, table, cfg.diagnostics.seed), saved)


def test_integrate_refuses_when_the_pca_baseline_has_not_been_trained(tmp_path):
    from vgtfm.diagnostics.integration import _reference_embedding

    with pytest.raises(SystemExit, match="run `python run.py train`"):
        _reference_embedding(_integrate_config(tmp_path), diagnostics_table(), 42)


def test_integrate_refuses_a_reference_embedding_of_the_wrong_length(tmp_path):
    """It is addressed by row against the table, so a length mismatch would pair
    every spot with another spot's features rather than fail."""
    from vgtfm.diagnostics.integration import _reference_embedding
    from vgtfm.models.train import embedding_path

    table = diagnostics_table()
    cfg = _integrate_config(tmp_path)
    np.save(
        embedding_path(cfg, "pca", cfg.diagnostics.seed),
        np.zeros((len(table.gene) - 1, 4), dtype=np.float32),
    )

    with pytest.raises(SystemExit, match="rows but the cohort has"):
        _reference_embedding(cfg, table, cfg.diagnostics.seed)


# ── the reference row and the corrected rows describe one matrix ─────────


def _cached_prediction(
    cfg, *levels: str, fingerprint: str, n: int = 6, refit: bool = False, index_levels=None
):
    """`eval`'s prediction cache, as `_stored_predictions` expects to find it.

    The index goes with the files: it is `eval`'s own statement of what it scored,
    and what tells a level with nothing to reuse apart from a level whose file has
    gone missing. `index_levels` overrides it, for the second case.
    """
    import json

    from vgtfm.evaluate.run_eval import REFERENCE, prediction_path

    seed = cfg.diagnostics.seed
    lab = np.array(["TUM", "STR"] * (n // 2))
    for level in levels:
        np.savez_compressed(
            prediction_path(cfg, REFERENCE, seed, "heldout_donor", level),
            y_true=lab,
            y_pred=lab,
            donor=np.array(["d1"] * n),
            sample_id=np.array(["s1"] * n),
            tissue=np.array(["mel"] * n),
            embedding_fingerprint=np.array(fingerprint),
            refit_per_fold=np.array(refit),
        )
    listed = levels if index_levels is None else index_levels
    (cfg.sub("eval", "predictions") / "index.json").write_text(
        json.dumps(
            [
                {
                    "model": REFERENCE,
                    "seed": seed,
                    "protocol": "heldout_donor",
                    "level": level,
                    "embedding_fingerprint": fingerprint,
                    "refit_per_fold": refit,
                }
                for level in listed
            ]
        )
    )


def test_the_cached_reference_must_come_from_the_matrix_being_corrected(tmp_path):
    """The failure this rules out is worse than the one it replaces.

    Reusing `eval`'s score while Harmony and ComBat correct a *different* matrix
    would leave every row looking consistent while each delta compared a corrected
    embedding against a reference that is not its uncorrected form. One stage rerun
    without the other is otherwise invisible here.
    """
    from vgtfm.diagnostics.integration import _stored_predictions
    from vgtfm.provenance import array_fingerprint

    cfg = _integrate_config(tmp_path)
    X = np.arange(24, dtype=np.float32).reshape(6, 4)
    _cached_prediction(cfg, "cross_donor", fingerprint=array_fingerprint(X))

    # The matrix that produced the cache: accepted, and handed back unchanged.
    got = _stored_predictions(cfg, {"cross_donor": []}, X, cfg.diagnostics.seed)
    assert list(got) == ["cross_donor"]
    assert got["cross_donor"]["y_true"].tolist() == ["TUM", "STR"] * 3

    # A different matrix: refused, however slight the difference.
    with pytest.raises(SystemExit, match="was produced from embedding"):
        _stored_predictions(cfg, {"cross_donor": []}, X * 1.001, cfg.diagnostics.seed)


def test_a_reference_refitted_per_fold_is_refused(tmp_path):
    """`refit_per_fold` predictions come from a representation refitted inside each
    fold, so they are not the frozen embedding's row at all."""
    from vgtfm.diagnostics.integration import _stored_predictions
    from vgtfm.provenance import array_fingerprint

    cfg = _integrate_config(tmp_path)
    X = np.arange(24, dtype=np.float32).reshape(6, 4)
    _cached_prediction(cfg, "cross_donor", fingerprint=array_fingerprint(X), refit=True)

    with pytest.raises(SystemExit, match="refit_per_fold"):
        _stored_predictions(cfg, {"cross_donor": []}, X, cfg.diagnostics.seed)


def test_a_reference_predating_the_fingerprint_is_refused(tmp_path):
    """Silently skipping the check on an old file is how it would stop being one."""
    from vgtfm.diagnostics.integration import _stored_predictions

    cfg = _integrate_config(tmp_path)
    X = np.arange(24, dtype=np.float32).reshape(6, 4)
    _cached_prediction(cfg, "cross_donor", fingerprint="")

    with pytest.raises(SystemExit, match="no embedding fingerprint"):
        _stored_predictions(cfg, {"cross_donor": []}, X, cfg.diagnostics.seed)


def test_the_two_stages_fingerprint_one_reading_of_one_file(tmp_path):
    """`eval` and `integrate` must load an embedding identically, not just from the
    same path.

    `array_fingerprint` hashes the dtype with the bytes, so a `float32` view of a
    `float64` file is a different fingerprint. `integrate` would then refuse with
    "`train` and `eval` are out of step" over a pair of files that are perfectly in
    step, and no rerun could clear it. Today only `pca` is read this way and it is
    saved as float32; the test is written against a float64 file because that is the
    case a cast at one call site would silently break.
    """
    from vgtfm.diagnostics.integration import _reference_embedding
    from vgtfm.evaluate.run_eval import _load_embedding
    from vgtfm.models.train import embedding_path
    from vgtfm.provenance import array_fingerprint

    table = diagnostics_table()
    cfg = _integrate_config(tmp_path)
    n = len(table.gene)
    saved = np.linspace(0, 1, n * 4, dtype=np.float64).reshape(n, 4)
    np.save(embedding_path(cfg, "pca", cfg.diagnostics.seed), saved)

    scored = _load_embedding(cfg, "pca", cfg.diagnostics.seed, n)
    corrected = _reference_embedding(cfg, table, cfg.diagnostics.seed)

    assert scored.dtype == corrected.dtype == saved.dtype
    assert array_fingerprint(scored) == array_fingerprint(corrected)


def test_the_scib_panel_is_named_at_the_seed_integrate_corrects(tmp_path):
    """`diagnose` and `integrate` score the same panel on the same `pca` embedding,
    and `vgtfm.results` checks the two against each other keyed by seed. Naming
    different seeds does not produce a conflict — it produces no comparison, and the
    check then passes by never running.
    """
    from vgtfm.diagnostics.report import _learned_embeddings
    from vgtfm.models.train import embedding_path

    cfg = _integrate_config(tmp_path)
    cfg.seeds = (43, 42)  # seeds[0] is deliberately not the one
    cfg.models.names = ("pca",)
    cfg.diagnostics.seed = 42
    np.save(embedding_path(cfg, "pca", 42), np.zeros((6, 4), dtype=np.float32))

    assert list(_learned_embeddings(cfg, 6)) == ["pca (seed 42)"]


def test_a_level_eval_scored_nothing_at_is_skipped_not_refused(tmp_path):
    """The probe below drops a level whose folds all come up empty, so refusing here
    on a level that simply has no prediction to reuse would fail a run that is not
    broken. `eval`'s index is what says which of the two a missing file is."""
    from vgtfm.diagnostics.integration import _stored_predictions
    from vgtfm.provenance import array_fingerprint

    cfg = _integrate_config(tmp_path)
    X = np.arange(24, dtype=np.float32).reshape(6, 4)
    _cached_prediction(cfg, "cross_donor", fingerprint=array_fingerprint(X))

    got = _stored_predictions(cfg, {"cross_donor": [], "cross_region": []}, X, cfg.diagnostics.seed)
    assert list(got) == ["cross_donor"]


def test_a_level_eval_did_score_whose_file_is_gone_is_still_refused(tmp_path):
    """An incomplete cache is a different problem from an unscored level, and it has
    a different fix."""
    from vgtfm.diagnostics.integration import _stored_predictions
    from vgtfm.provenance import array_fingerprint

    cfg = _integrate_config(tmp_path)
    X = np.arange(24, dtype=np.float32).reshape(6, 4)
    _cached_prediction(
        cfg,
        "cross_donor",
        fingerprint=array_fingerprint(X),
        index_levels=("cross_donor", "cross_region"),
    )

    with pytest.raises(SystemExit, match="lists it but"):
        _stored_predictions(cfg, {"cross_donor": [], "cross_region": []}, X, cfg.diagnostics.seed)


def test_integrate_refuses_when_eval_has_not_cached_any_prediction(tmp_path):
    from vgtfm.diagnostics.integration import _stored_predictions

    cfg = _integrate_config(tmp_path)
    with pytest.raises(SystemExit, match="run `python run.py eval`"):
        _stored_predictions(
            cfg, {"cross_donor": []}, np.zeros((6, 4), np.float32), cfg.diagnostics.seed
        )


def test_integrate_names_the_protocol_when_eval_ran_without_it(tmp_path):
    """`eval.protocols` without `heldout_donor` leaves a full prediction cache with
    nothing in it this stage can read. "run eval first" would be the wrong advice."""
    import json

    from vgtfm.diagnostics.integration import _stored_predictions

    cfg = _integrate_config(tmp_path)
    (cfg.sub("eval", "predictions") / "index.json").write_text(
        json.dumps(
            [
                {
                    "model": "pca",
                    "seed": cfg.diagnostics.seed,
                    "protocol": "pooled_loso",
                    "level": "all",
                }
            ]
        )
    )

    with pytest.raises(SystemExit, match="eval.protocols"):
        _stored_predictions(
            cfg, {"cross_donor": []}, np.zeros((6, 4), np.float32), cfg.diagnostics.seed
        )


# ── the fits these stages score are the fits `eval` scores ───────────


def test_the_diagnostics_score_every_seed_eval_scores(tmp_path):
    """Not one fit named by `diagnostics.seed`. `eval` averages three replicates of
    each model, so a panel or an integration row scored on a single fit is a
    different experiment and cannot be read beside Table 1's."""
    from vgtfm.diagnostics import model_seeds

    cfg = _integrate_config(tmp_path)
    cfg.seeds = (42, 43, 44)
    assert model_seeds(cfg) == (42, 43, 44)

    # Narrowing is expressible, for a run that would rather have the wall time.
    cfg.diagnostics.model_seeds = (42, 44)
    assert model_seeds(cfg) == (42, 44)

    # But not to a fit `train` never produced: there is no embedding to score, and
    # the failure would otherwise be a missing-file error one stage later.
    cfg.diagnostics.model_seeds = (42, 99)
    with pytest.raises(SystemExit, match="99"):
        model_seeds(cfg)


def test_the_scib_panel_covers_every_fit(tmp_path):
    from vgtfm.diagnostics.report import _learned_embeddings
    from vgtfm.models.train import embedding_path

    cfg = _integrate_config(tmp_path)
    cfg.seeds = (42, 43)
    cfg.models.names = ("pca",)
    for seed in cfg.seeds:
        np.save(embedding_path(cfg, "pca", seed), np.zeros((6, 4), dtype=np.float32))

    assert sorted(_learned_embeddings(cfg, 6)) == ["pca (seed 42)", "pca (seed 43)"]


def test_the_algorithm_seed_does_not_follow_the_fit(tmp_path):
    """`diagnostics.seed` picks which spots a diagnostic subsamples, and must not
    move between fits: redrawing per seed would confound the spread across fits with
    the spread across subsamples. Only *which fit* comes from `seeds`."""
    from vgtfm.diagnostics.scib import scoring_rows

    batch = np.array([f"s{i % 4}" for i in range(400)])
    label = np.array(["TUM", "STR"] * 200)
    a = scoring_rows(batch, label, max_spots=50, seed=42)
    b = scoring_rows(batch, label, max_spots=50, seed=42)
    c = scoring_rows(batch, label, max_spots=50, seed=43)

    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_the_probe_bracket_is_pooled_over_the_fits_not_taken_from_one(tmp_path):
    """The wide table folds the scope and level into the column name, so
    `reporting.attach_pooled` cannot match on them and `_attach_pooled_probe` writes
    the merge out. The columns must be constant within a method: that is what lets
    the table layer collapse the seeds with a plain groupby-mean.
    """
    from vgtfm.diagnostics.integration import _attach_pooled_probe

    cfg = _integrate_config(tmp_path)
    key = "f1/kidney/cross_donor"
    rows = [
        {"method": "none", "seed": s, key: 0.4 + 0.01 * i, f"{key}_lo": 0.3, f"{key}_hi": 0.5}
        for i, s in enumerate((42, 43, 44))
    ]
    # Three seeds' replicate draws, deliberately far apart: a bracket taken from one
    # of them cannot cover the other two.
    draws = {
        ("none", "cross_donor", "kidney"): [
            np.full(200, 0.20),
            np.full(200, 0.45),
            np.full(200, 0.80),
        ]
    }

    _attach_pooled_probe(rows, draws, cfg)

    pooled = {(r[f"{key}_lo_pooled"], r[f"{key}_hi_pooled"]) for r in rows}
    assert len(pooled) == 1, "the pooled bracket must not vary within a method"
    lo, hi = pooled.pop()
    assert lo == pytest.approx(0.20) and hi == pytest.approx(0.80)
    assert {r[f"{key}_n_seeds"] for r in rows} == {3}

    # A method with no probe cell gets no columns rather than empty ones.
    bbknn = [{"method": "bbknn", "seed": 42, "note": "graph-only"}]
    _attach_pooled_probe(bbknn, draws, cfg)
    assert set(bbknn[0]) == {"method", "seed", "note"}


def test_the_scvi_manager_stores_this_stage_clears_still_exist():
    """`_scvi_embedding` empties two dicts that live on `scvi.model.SCVI` itself.

    They are private, so a scvi-tools upgrade may rename them — and the failure that
    causes is invisible: the fit still runs, but each seed's AnnData stays pinned
    behind the class, and the stage dies of an OOM three hours in on a machine this
    suite does not run on. Naming them here turns that into a local red test.
    """
    scvi = pytest.importorskip("scvi")

    for name in ("_setup_adata_manager_store", "_per_instance_manager_store"):
        store = getattr(scvi.model.SCVI, name, None)
        assert store is not None, f"scvi.model.SCVI.{name} is gone — see _scvi_embedding"
        assert hasattr(store, "clear"), f"scvi.model.SCVI.{name} is no longer a mapping"


def _panel_frame(fingerprint, *, seed=42, representation=None, **metrics):
    """One `scib_panel.csv` row, shaped as `diagnose` writes it."""
    import pandas as pd

    row = {
        "substrate": "geneformer",
        "representation": representation or f"pca (seed {seed})",
        "batch/ilisi": 0.11,
        "batch/kbet": 0.22,
        "batch/graph_connectivity": 0.33,
        "bio/clisi": 0.44,
        "other/brand_new": 0.55,
        "n_spots": 10,
        "n_batches": 2,
    }
    row.update(metrics)
    if fingerprint is not None:
        row["embedding"] = fingerprint
    return pd.DataFrame([row])


def test_the_uncorrected_panel_is_diagnoses_rather_than_a_second_computation(tmp_path):
    """The `none` row's batch columns are read, not recomputed.

    `diagnose` already scores this exact embedding through the same
    `scib.benchmark` call at the same `diagnostics.seed`, and `vgtfm.results`
    checks the two against each other. Computing it twice is what made kBET
    disagree across the stages: it is the only metric of the panel whose path
    reaches `scipy.sparse.linalg.eigsh` without a fixed start vector, so unlike
    iLISI, cLISI and graph connectivity it does not reproduce across processes.
    """
    from vgtfm.diagnostics.integration import _stored_panel
    from vgtfm.provenance import array_fingerprint

    cfg = _integrate_config(tmp_path)
    X = np.arange(24, dtype=np.float32).reshape(6, 4)
    _panel_frame(array_fingerprint(X)).to_csv(
        cfg.sub("diagnostics") / "scib_panel.csv", index=False
    )

    cols = _stored_panel(cfg, X, 42)

    assert cols == {
        "batch/ilisi": 0.11,
        "batch/kbet": 0.22,
        "batch/graph_connectivity": 0.33,
        "bio/clisi": 0.44,
    }, "the panel's own values, and only its two namespaces"


def test_the_uncorrected_panel_refuses_a_panel_scored_on_another_embedding(tmp_path):
    """A reused panel is only safe if both stages hold the same matrix.

    Without the check the batch columns would describe one embedding and the probe
    columns beside them another, and every row would still look consistent.
    """
    from vgtfm.diagnostics.integration import _stored_panel
    from vgtfm.provenance import array_fingerprint

    cfg = _integrate_config(tmp_path)
    X = np.arange(24, dtype=np.float32).reshape(6, 4)
    other = np.zeros((6, 4), dtype=np.float32)
    _panel_frame(array_fingerprint(other)).to_csv(
        cfg.sub("diagnostics") / "scib_panel.csv", index=False
    )

    with pytest.raises(SystemExit, match="re-run without the other"):
        _stored_panel(cfg, X, 42)


def test_the_uncorrected_panel_refuses_a_panel_that_records_no_fingerprint(tmp_path):
    """An older `diagnose` wrote no `embedding` column. Reusing a panel that cannot
    be tied to a matrix is the failure this check exists to prevent, so it refuses
    rather than trusting the row."""
    from vgtfm.diagnostics.integration import _stored_panel

    cfg = _integrate_config(tmp_path)
    _panel_frame(None).to_csv(cfg.sub("diagnostics") / "scib_panel.csv", index=False)

    with pytest.raises(SystemExit, match="no embedding fingerprint"):
        _stored_panel(cfg, np.arange(24, dtype=np.float32).reshape(6, 4), 42)


def test_the_uncorrected_panel_refuses_when_diagnose_has_not_run(tmp_path):
    from vgtfm.diagnostics.integration import _stored_panel

    cfg = _integrate_config(tmp_path)
    with pytest.raises(SystemExit, match="run `python run.py diagnose`"):
        _stored_panel(cfg, np.zeros((6, 4), dtype=np.float32), 42)


def test_the_uncorrected_panel_refuses_a_seed_diagnose_did_not_score(tmp_path):
    """Naming a seed the panel does not carry must not pass silently: the row would
    be missing its batch columns while the probe columns beside it stayed."""
    from vgtfm.diagnostics.integration import _stored_panel
    from vgtfm.provenance import array_fingerprint

    cfg = _integrate_config(tmp_path)
    X = np.zeros((6, 4), dtype=np.float32)
    _panel_frame(array_fingerprint(X), seed=43).to_csv(
        cfg.sub("diagnostics") / "scib_panel.csv", index=False
    )

    with pytest.raises(SystemExit, match="no 'pca \\(seed 42\\)' row"):
        _stored_panel(cfg, X, 42)


def test_the_scib_panel_records_the_embedding_it_scored(tmp_path, monkeypatch):
    """`integrate` reuses this panel keyed by the fingerprint, so `diagnose` has to
    write one — the two files are otherwise unrelated."""
    import sys
    import types as _types

    import pandas as pd

    from vgtfm.provenance import array_fingerprint

    class _Bench:
        def __init__(self, *a, **k):
            pass

        def benchmark(self):
            pass

        def get_results(self, min_max_scale=False):
            return pd.DataFrame([{"iLISI": 0.2, "cLISI": 0.9}])

    module = _types.ModuleType("scib_metrics.benchmark")
    module.Benchmarker = _Bench
    module.BatchCorrection = lambda *a, **k: None
    module.BioConservation = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "scib_metrics.benchmark", module)

    table = diagnostics_table()
    table.meta["annotation"] = ["Tumor", "Stroma"] * (len(table.gene) // 2)
    cfg = _integrate_config(tmp_path)
    cfg.diagnostics.columns = ()
    Z = np.arange(len(table.gene) * 3, dtype=np.float32).reshape(len(table.gene), 3)

    panel = scib.run(cfg, table, {"pca (seed 42)": Z})

    assert panel.loc[0, "embedding"] == array_fingerprint(Z)


def test_the_uncorrected_panel_is_scored_here_when_diagnose_omits_it(tmp_path):
    """`diagnostics.run_scib=false` is the recorded way to skip the panel. There is
    then no `diagnose` row to reuse and none to disagree with, so the uncorrected
    row is scored in this stage as every row was before — refusing would make a
    supported opt-out unrunnable."""
    from vgtfm.config import Config
    from vgtfm.diagnostics.integration import _reuses_diagnoses_panel

    assert Config().diagnostics.run_scib is True, "the reuse is the default path"

    cfg = _integrate_config(tmp_path)
    assert _reuses_diagnoses_panel(cfg) is True

    cfg.diagnostics.run_scib = False
    assert _reuses_diagnoses_panel(cfg) is False
