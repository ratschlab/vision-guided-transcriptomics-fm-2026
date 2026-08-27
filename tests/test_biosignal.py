"""Per-gene ridge R^2, its streaming accumulator, and the GSEA plumbing.

The ridge is hand-rolled for speed, so it is checked against sklearn rather than
against itself; the accumulator is checked against scoring the pooled data in one
go, which is the thing it claims to be equivalent to.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vgtfm.biosignal.enrichment import GENE_SET_SCHEMAS, _normalise_output, check_coverage, load_net
from vgtfm.biosignal.ridge import (
    R2Accumulator,
    alpha_grid,
    detrend_on_baseline,
    fit_ridge_svd,
    grid_saturation,
    predict_fold,
    predict_fold_gcv,
    quartile_table,
)
from vgtfm.evaluate.probes import standardize


# ── the ridge itself ─────────────────────────────────────────────────


def test_ridge_matches_sklearn_without_an_intercept():
    """The closed form must be numerically identical to the reference solver."""
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 12))
    Y = rng.standard_normal((200, 5))
    for alpha in (0.1, 1.0, 25.0):
        W = fit_ridge_svd(X, Y, alpha)
        ref = Ridge(alpha=alpha, fit_intercept=False).fit(X, Y).coef_.T
        assert np.allclose(W, ref, atol=1e-5)


def test_ridge_solves_the_penalised_normal_equations():
    """``(X'X + aI) W = X'Y`` — the definition, independent of any library."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((80, 6))
    Y = rng.standard_normal((80, 3))
    alpha = 2.5
    W = fit_ridge_svd(X, Y, alpha).astype(np.float64)
    lhs = X.T @ X @ W + alpha * W
    assert np.allclose(lhs, X.T @ Y, atol=1e-6)


def test_ridge_shrinks_toward_zero_as_alpha_grows():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((100, 8))
    Y = X @ rng.standard_normal((8, 2))
    norms = [np.linalg.norm(fit_ridge_svd(X, Y, a)) for a in (0.01, 1.0, 100.0)]
    assert norms[0] > norms[1] > norms[2]


def test_ridge_is_stable_when_there_are_more_features_than_spots():
    """The wide case is the real one: 1152 features, sometimes few hundred spots."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((30, 200))
    Y = rng.standard_normal((30, 4))
    W = fit_ridge_svd(X, Y, 1.0)
    assert W.shape == (200, 4)
    assert np.isfinite(W).all()


def test_predict_fold_recovers_a_planted_linear_map():
    """Fit-and-apply, with both sides standardised on training rows as the stage does.

    There is no intercept term, so the targets must be centred on the training
    statistics too — which is exactly what the biosignal stage does before calling
    this, and what makes R^2 comparable across folds.
    """
    rng = np.random.default_rng(4)
    X = rng.standard_normal((500, 5))
    Y = X @ rng.standard_normal((5, 3))
    Xtr, Xte = standardize(X[:400], X[400:])
    Ytr, Yte = standardize(Y[:400], Y[400:])
    pred = predict_fold(Xtr, Ytr, Xte, alpha=1e-6)
    assert np.allclose(pred, Yte, atol=1e-3)


# ── the fitted penalty ───────────────────────────────────────────────


def test_alpha_grid_is_log_spaced_and_rejects_a_degenerate_spec():
    g = alpha_grid((1e-2, 1e2, 5))
    assert np.allclose(g, [1e-2, 1e-1, 1e0, 1e1, 1e2])
    for bad in [(0.0, 10.0, 5), (10.0, 1.0, 5), (1.0, 10.0, 1)]:
        with pytest.raises(ValueError, match="alpha grid"):
            alpha_grid(bad)


def test_a_one_value_grid_reproduces_the_fixed_penalty_path():
    """The search is a generalisation of the fixed fit, not a different estimator."""
    rng = np.random.default_rng(7)
    X, Y, Xte = (
        rng.standard_normal((200, 30)),
        rng.standard_normal((200, 12)),
        rng.standard_normal((60, 30)),
    )
    Y -= Y.mean(axis=0, keepdims=True)
    for alpha in (0.5, 3.0, 250.0):
        want = predict_fold(X, Y, Xte, alpha)
        got, chosen = predict_fold_gcv(X, Y, Xte, np.array([alpha]))
        assert np.allclose(chosen, alpha)
        assert np.allclose(got, want, atol=1e-5)


def test_gcv_picks_the_penalty_that_maximises_held_out_r2():
    """GCV is an estimate of held-out error; on planted data it should find it."""
    rng = np.random.default_rng(11)
    n, p, g = 300, 60, 40
    B = rng.standard_normal((p, g)) * 0.3
    X, Xte = rng.standard_normal((n, p)), rng.standard_normal((2000, p))
    Y = X @ B + rng.standard_normal((n, g))
    Yte = Xte @ B + rng.standard_normal((2000, g))
    Y -= Y.mean(axis=0, keepdims=True)

    grid = alpha_grid((1e-2, 1e5, 30))
    curve = [_r2_reference(Yte, predict_fold(X, Y, Xte, a)).mean() for a in grid]
    oracle = grid[int(np.argmax(curve))]
    got, chosen = predict_fold_gcv(X, Y, Xte, grid)

    assert np.median(chosen) == pytest.approx(oracle, rel=0.5)
    assert _r2_reference(Yte, got).mean() >= max(curve) - 0.01


def test_the_fitted_penalty_removes_the_width_term_a_fixed_one_leaves():
    """The reason this exists.

    Two designs over the same rows, one wide and one narrow, against targets that
    carry no signal at all. Their true R^2 is zero, so any gap between them is the
    optimism of least squares, ~p/n. A fixed alpha=1 reports that gap as if it were
    a difference between the representations; a fitted penalty does not.
    """
    rng = np.random.default_rng(3)
    n, nt = 600, 400
    wide, narrow = rng.standard_normal((n, 300)), None
    wide_te = rng.standard_normal((nt, 300))
    narrow, narrow_te = wide[:, :30].copy(), wide_te[:, :30].copy()
    Y, Yte = rng.standard_normal((n, 80)), rng.standard_normal((nt, 80))
    Y -= Y.mean(axis=0, keepdims=True)

    fixed = (
        _r2_reference(Yte, predict_fold(narrow, Y, narrow_te, 1.0)).mean()
        - _r2_reference(Yte, predict_fold(wide, Y, wide_te, 1.0)).mean()
    )
    grid = alpha_grid((1e-1, 1e7, 25))
    fitted = (
        _r2_reference(Yte, predict_fold_gcv(narrow, Y, narrow_te, grid)[0]).mean()
        - _r2_reference(Yte, predict_fold_gcv(wide, Y, wide_te, grid)[0]).mean()
    )

    assert fixed > 0.2, "the fixed-penalty width term should be large here"
    assert abs(fitted) < 0.02, f"fitted penalty still leaves a width term: {fitted}"


def test_a_signal_free_target_shrinks_all_the_way_to_the_training_mean():
    rng = np.random.default_rng(5)
    X, Xte = rng.standard_normal((300, 50)), rng.standard_normal((100, 50))
    Y = rng.standard_normal((300, 20))
    Y -= Y.mean(axis=0, keepdims=True)
    grid = alpha_grid((1e-1, 1e8, 30))
    P, chosen = predict_fold_gcv(X, Y, Xte, grid)
    assert np.median(chosen) > grid[len(grid) // 2]
    assert np.abs(P).max() < 0.5, "a no-signal fit should predict near the mean"


def test_grid_saturation_reports_mass_on_each_endpoint():
    grid = alpha_grid((1.0, 100.0, 3))
    lo, hi = grid_saturation(np.array([1.0, 1.0, 10.0, 100.0]), grid)
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(0.25)
    assert grid_saturation(np.array([]), grid) == (0.0, 0.0)


def test_standardize_uses_training_statistics_and_survives_constant_columns():
    """A gene-side column with no variance must not produce inf or nan."""
    train = np.array([[0.0, 5.0], [2.0, 5.0], [4.0, 5.0]])
    test = np.array([[6.0, 5.0]])
    tr, te = standardize(train, test)
    assert np.isfinite(tr).all() and np.isfinite(te).all()
    assert tr[:, 1].tolist() == [0.0, 0.0, 0.0]
    assert tr[:, 0].mean() == pytest.approx(0.0)


# ── streaming R^2 ────────────────────────────────────────────────────


def _r2_reference(y_true, y_pred):
    """Pooled per-gene R^2, computed the obvious way."""
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / ss_tot


def test_accumulator_matches_scoring_the_pooled_data_in_one_pass():
    """The whole point of streaming: identical answer, O(n_genes) memory."""
    rng = np.random.default_rng(0)
    n_genes = 7
    acc = R2Accumulator(n_genes, ("a", "b"))
    truths, preds_a, preds_b = [], [], []
    for n in (40, 90, 25):  # three folds of different sizes
        Y = rng.standard_normal((n, n_genes))
        Pa = Y + 0.3 * rng.standard_normal((n, n_genes))
        Pb = rng.standard_normal((n, n_genes))
        acc.update(Y, {"a": Pa, "b": Pb})
        truths.append(Y)
        preds_a.append(Pa)
        preds_b.append(Pb)

    Y = np.vstack(truths)
    got = acc.r2()
    assert acc.n == len(Y)
    assert np.allclose(got["a"], _r2_reference(Y, np.vstack(preds_a)), atol=1e-10)
    assert np.allclose(got["b"], _r2_reference(Y, np.vstack(preds_b)), atol=1e-10)
    # The informative predictor must win on every gene.
    assert (got["a"] > got["b"]).all()


def test_accumulator_scores_a_perfect_predictor_at_one():
    rng = np.random.default_rng(1)
    Y = rng.standard_normal((50, 4))
    acc = R2Accumulator(4, ("perfect",))
    acc.update(Y, {"perfect": Y.copy()})
    assert np.allclose(acc.r2()["perfect"], 1.0)


def test_accumulator_scores_the_pooled_mean_at_zero():
    """R^2 is measured against the pooled mean, so predicting it scores exactly 0."""
    rng = np.random.default_rng(2)
    Y = rng.standard_normal((300, 3))
    acc = R2Accumulator(3, ("mean",))
    acc.update(Y, {"mean": np.tile(Y.mean(axis=0), (300, 1))})
    assert np.allclose(acc.r2()["mean"], 0.0, atol=1e-10)


def test_accumulator_reports_nan_for_a_gene_with_no_variance():
    Y = np.column_stack([np.ones(20), np.arange(20.0)])
    acc = R2Accumulator(2, ("p",))
    acc.update(Y, {"p": Y.copy()})
    r2 = acc.r2()["p"]
    assert np.isnan(r2[0])  # constant gene: undefined
    assert r2[1] == pytest.approx(1.0)


def test_accumulator_rejects_an_unregistered_prediction():
    acc = R2Accumulator(2, ("frozen",))
    with pytest.raises(KeyError):
        acc.update(np.zeros((3, 2)), {"typo": np.zeros((3, 2))})


def test_empty_accumulator_is_all_nan():
    assert np.isnan(R2Accumulator(3, ("a",)).r2()["a"]).all()


# ── quartile view ────────────────────────────────────────────────────


def test_quartile_table_exposes_regression_toward_the_mean():
    """The pattern the mean delta-R^2 hides.

    A refinement that shrinks every prediction toward zero lifts genes the frozen
    embedding predicted badly and pushes down the ones it predicted well. That is
    monotone across quartiles and invisible in the overall average.
    """
    frozen = np.linspace(-1.0, 0.9, 400)
    shrunk = 0.5 * frozen  # everything pulled toward zero
    table = quartile_table(frozen, shrunk - frozen)

    assert list(table["quartile"]) == ["Q1", "Q2", "Q3", "Q4"]
    deltas = table["mean_delta_r2"].to_numpy()
    assert (np.diff(deltas) < 0).all()  # monotone decreasing
    assert deltas[0] > 0 > deltas[-1]
    assert table["pct_improved"].iloc[0] == pytest.approx(100.0)
    assert table["pct_improved"].iloc[-1] == pytest.approx(0.0)
    assert table["n_genes"].sum() == 400


def test_quartile_table_drops_nan_genes_and_degenerate_input():
    frozen = np.array([0.1, 0.2, np.nan, 0.4, 0.5, 0.6, 0.7, 0.8])
    table = quartile_table(frozen, np.zeros_like(frozen))
    assert table["n_genes"].sum() == 7
    assert quartile_table(np.array([]), np.array([])).empty
    assert quartile_table(np.full(10, 0.5), np.zeros(10)).empty  # no quantiles


def test_detrending_removes_a_monotone_dependence_on_the_baseline():
    """The shrinkage trend is what made every curated set land at the bottom.

    A delta that is a pure function of the frozen R^2 carries no gene-specific
    signal at all, so detrending must flatten it to ~0 rather than merely shrink it.
    """
    rng = np.random.default_rng(0)
    frozen = rng.uniform(-0.2, 0.8, size=4000)
    pure_trend = -0.4 * frozen

    # What binning cannot remove is the trend's own slope across a single bin,
    # here 0.4 * (range / 50); anything beyond that would be trend left standing.
    out = detrend_on_baseline(frozen, pure_trend, bins=50)
    assert np.abs(out).max() < 0.4 * (frozen.max() - frozen.min()) / 50
    assert np.abs(out).max() < 0.05 * np.abs(pure_trend - pure_trend.mean()).max()

    # A gene-specific offset on top of the trend survives it.
    spike = pure_trend.copy()
    spike[7] += 0.5
    out = detrend_on_baseline(frozen, spike, bins=50)
    assert out[7] == pytest.approx(0.5, abs=1e-2)
    assert np.corrcoef(frozen, out)[0, 1] == pytest.approx(0.0, abs=0.05)


def test_detrending_is_a_no_op_when_it_cannot_bin():
    """Fewer genes than bins, or bins switched off, leaves the ranking untouched."""
    stat = np.array([3.0, 1.0, 2.0])
    for bins in (0, 1, 50):
        assert detrend_on_baseline(np.arange(3.0), stat, bins=bins) == pytest.approx(stat)


def test_detrending_preserves_the_gene_order_within_a_baseline_bin():
    """It re-centres bins; it must not reorder genes that share one."""
    frozen = np.zeros(100)
    stat = np.arange(100.0)
    out = detrend_on_baseline(frozen, stat, bins=2)
    assert np.all(np.diff(out[:50]) > 0) and np.all(np.diff(out[50:]) > 0)


# ── enrichment plumbing ──────────────────────────────────────────────


def test_coverage_guard_fires_when_symbols_do_not_match():
    """The common failure: Ensembl ids where HGNC symbols were expected."""
    net = pd.DataFrame({"source": ["SET"] * 4, "target": ["TP53", "EGFR", "MYC", "KRAS"]})
    coverage, n_covered, n_universe = check_coverage(
        {"TP53", "EGFR", "MYC", "KRAS"}, net, "hallmark"
    )
    assert coverage == pytest.approx(1.0)
    assert (n_covered, n_universe) == (4, 4)

    with pytest.raises(RuntimeError, match="coverage"):
        check_coverage({"ENSG00000141510"}, net, "hallmark")


def test_coverage_floors_are_declared_for_every_collection():
    for name, schema in GENE_SET_SCHEMAS.items():
        assert 0.0 < schema["min_coverage"] < 1.0, name
        assert schema["filename"].endswith(".parquet")


@pytest.mark.parametrize(
    "frames,expected",
    [
        (2, ["norm", "padj"]),  # current decoupler
        (4, ["score", "norm", "pval", "padj"]),  # older releases
        (3, ["norm", "pval", "padj"]),
    ],
)
def test_gsea_output_is_normalised_across_decoupler_return_shapes(frames, expected):
    """``norm`` is always the normalised score, whichever tuple shape arrives."""
    sets = ["HALLMARK_EMT", "HALLMARK_HYPOXIA"]
    result = tuple(
        pd.DataFrame([[float(i + 1), float(i + 2)]], columns=sets) for i in range(frames)
    )
    out = _normalise_output(result, "delta_r2")

    assert list(out.columns) == ["sample", "source", *expected]
    assert set(out["source"]) == set(sets)
    assert (out["sample"] == "delta_r2").all()
    assert out["norm"].notna().all()


def test_gsea_output_accepts_a_long_frame_and_renames_its_columns():
    frame = pd.DataFrame({"source": ["A", "B"], "nes": [2.0, -3.0], "fdr": [0.2, 0.01]})
    out = _normalise_output(frame, "delta_r2")
    assert {"norm", "padj", "sample"} <= set(out.columns)
    # Sorted by adjusted p-value, so the significant set comes first.
    assert out.iloc[0]["source"] == "B"


def test_gsea_output_of_nothing_is_empty():
    assert _normalise_output((None, None), "delta_r2").empty


# ── the vendored collections ─────────────────────────────────────────

RESOURCES = Path(__file__).resolve().parents[1] / "vgtfm" / "biosignal" / "resources"


@pytest.mark.parametrize("gene_set", sorted(GENE_SET_SCHEMAS))
def test_the_shipped_gene_sets_load_into_decouplers_schema(gene_set):
    """They are vendored so compute nodes need no network; a parquet that no longer
    carries the columns the schema names would only fail mid-run otherwise."""
    net = load_net(RESOURCES, gene_set)

    assert list(net.columns[:2]) == ["source", "target"]
    assert net["source"].nunique() > 10
    assert net["target"].nunique() > 100
    assert net["target"].astype(str).str.isupper().mean() > 0.8, "HGNC symbols"


def test_an_unknown_collection_names_the_known_ones():
    with pytest.raises(ValueError) as e:
        load_net(RESOURCES, "reactome")
    for name in GENE_SET_SCHEMAS:
        assert name in str(e.value)


def test_a_collection_that_is_not_vendored_says_how_to_fetch_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="--fetch"):
        load_net(tmp_path, "hallmark")


def test_the_shipped_hallmark_clears_its_own_coverage_floor():
    """The floor exists to catch Ensembl ids ranked against symbol-keyed sets. It
    must not be set so high that a correct run trips it."""
    net = load_net(RESOURCES, "hallmark")
    universe = set(net["target"].astype(str))
    coverage, covered, total = check_coverage(universe, net, "hallmark")

    assert coverage == 1.0 and covered == total
