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

from vgtfm.biosignal import enrichment as enrichment_mod
from vgtfm.biosignal.enrichment import (
    GENE_SET_SCHEMAS,
    RESULT_COLUMNS,
    SET_MEAN_COLUMNS,
    benjamini_hochberg,
    check_coverage,
    collection_provenance,
    load_net,
    run_gsea,
    run_set_mean,
    set_members,
)
from vgtfm.biosignal.ridge import (
    R2Accumulator,
    alpha_grid,
    baseline_strata,
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

RESOURCES = Path(__file__).resolve().parents[1] / "vgtfm" / "biosignal" / "resources"


def _ranking(n: int = 4000, seed: int = 0) -> pd.Series:
    """A ranking over real Hallmark symbols, so the coverage floor is cleared."""
    symbols = sorted(load_net(RESOURCES, "hallmark")["target"].astype(str).unique())
    rng = np.random.default_rng(seed)
    genes = symbols[: min(n, len(symbols))]
    return pd.Series(rng.normal(size=len(genes)), index=genes)


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


def test_the_decoupler_kernel_still_returns_scores_and_unadjusted_p_values():
    """The one thing this module depends on decoupler's internals for.

    ``dc.mt.gsea`` overwrites the permutation p-value with its own BH adjustment, so
    the unadjusted column is only reachable from the kernel below it. A release that
    changes that contract has to fail here, not in a table with a mis-labelled p.
    """
    pytest.importorskip("decoupler")
    from decoupler.mt._gsea import _func_gsea
    from decoupler.pp.net import idxmat, prune

    stat = _ranking(seed=3)
    net = load_net(RESOURCES, "hallmark")
    pruned = prune(features=stat.index.to_numpy(), net=net, tmin=15, verbose=False)
    sources, cnct, starts, offsets = idxmat(features=stat.index.to_numpy(), net=pruned)
    norm, pval = _func_gsea(stat.to_numpy(float)[None, :], cnct, starts, offsets, times=50, seed=42)

    assert norm.shape == pval.shape == (1, len(sources))
    # Unadjusted: a permutation count over a finite null, so it reaches the floor.
    assert pval.min() >= 0.0 and pval.max() <= 1.0
    # `offsets` is the set size on the background, which is what the table prints.
    assert np.asarray(offsets).min() >= 15


# ── the vendored collections ─────────────────────────────────────────


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


# ── the statistics a reader has to be given ──────────────────────────
#
# The columns exist, the permutation floor is reported as a bound rather than as a
# zero, and the scores still agree with decoupler's own entry point.


def test_benjamini_hochberg_matches_the_textbook_definition():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
    n = p.size
    expected = np.minimum.accumulate((p * n / np.arange(1, n + 1))[::-1])[::-1]
    assert benjamini_hochberg(p) == pytest.approx(expected)


def test_benjamini_hochberg_is_monotone_and_bounded():
    rng = np.random.default_rng(0)
    p = rng.random(200)
    q = benjamini_hochberg(p)
    assert q.max() <= 1.0 and q.min() >= 0.0
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)


def test_benjamini_hochberg_excludes_nan_from_the_family():
    p = np.array([0.01, np.nan, 0.02])
    q = benjamini_hochberg(p)
    assert np.isnan(q[1])
    # Two hypotheses in the family, not three.
    assert q[0] == pytest.approx(0.02)


def test_run_gsea_reports_every_column_a_reader_needs():
    pytest.importorskip("decoupler")
    res = run_gsea(_ranking(), RESOURCES, "hallmark", times=200, seed=42)

    assert list(res.table.columns) == RESULT_COLUMNS
    # The set size is the count on the *background*: never above it, never below the
    # minimum that was asked for.
    assert res.table["set_size"].between(res.min_n, res.n_ranked).all()
    assert res.n_ranked == len(_ranking())

    meta = res.meta()
    for field in (
        "collection",
        "citation",
        "test",
        "background_n_genes",
        "min_set_size",
        "permutations",
        "pval_resolution",
        "multiple_testing",
    ):
        assert meta[field] not in (None, ""), field
    assert meta["background_n_genes"] == res.n_ranked
    assert meta["pval_resolution"] == pytest.approx(1 / 200)


def test_an_unresolvable_p_value_never_becomes_an_fdr_of_zero():
    """A permutation test cannot deliver q = 0, and a table must not print one."""
    pytest.importorskip("decoupler")
    res = run_gsea(_ranking(seed=1), RESOURCES, "hallmark", times=200, seed=42)
    assert (res.table["padj"] > 0).all()
    # BH was applied at the floor, so no adjusted value can sit below it.
    assert res.table["padj"].min() >= 1 / res.permutations - 1e-12
    # The unadjusted column keeps the raw count, zeros included: rounding those away
    # would hide the resolution.
    assert (res.table["pval"] >= 0).all()


def test_run_gsea_reproduces_decouplers_own_scores():
    """The extra columns come from calling decoupler one level down, not from a
    different test: diverging scores would mean GSEA has been reimplemented here and
    the caption's citation is no longer true."""
    dc = pytest.importorskip("decoupler")
    stat = _ranking(seed=2)
    res = run_gsea(stat, RESOURCES, "hallmark", label="d", times=200, seed=7)

    mat = stat.to_frame(name="d").T
    mat.index.name = "sample"
    nes, _ = dc.mt.gsea(
        data=mat, net=load_net(RESOURCES, "hallmark"), tmin=15, times=200, seed=7, verbose=False
    )
    ours = res.table.set_index("source")["norm"]
    assert ours.to_numpy() == pytest.approx(nes.iloc[0].reindex(ours.index).to_numpy())


@pytest.mark.parametrize("gene_set", sorted(GENE_SET_SCHEMAS))
def test_every_collection_can_name_and_identify_itself(gene_set):
    """A name is not a citation and a citation is not a version; the caption quotes
    all three."""
    prov = collection_provenance(RESOURCES, gene_set)
    assert prov["collection"] and prov["citation"]
    assert len(prov["sha256"]) == 64 and len(prov["md5"]) == 32
    assert prov["n_sets"] > 0 and prov["n_genes"] > 0


# ── baseline strata ──────────────────────────────────────────────────


def test_baseline_strata_are_equal_count_and_ordered():
    rng = np.random.default_rng(0)
    baseline = rng.normal(size=200)
    strata = baseline_strata(baseline, bins=4)
    counts = np.bincount(strata)
    assert list(counts) == [50, 50, 50, 50]
    # A higher bin id means a higher baseline: the bins are ranks, not values.
    means = [baseline[strata == k].mean() for k in range(4)]
    assert means == sorted(means)


def test_baseline_strata_refuse_to_bin_what_they_cannot():
    assert baseline_strata(np.arange(10.0), bins=1) is None
    assert baseline_strata(np.arange(3.0), bins=5) is None


def test_detrending_still_centres_within_the_strata_it_reports():
    rng = np.random.default_rng(1)
    baseline = rng.normal(size=120)
    stat = 2.0 * baseline + rng.normal(scale=0.01, size=120)
    strata = baseline_strata(baseline, bins=6)
    out = detrend_on_baseline(baseline, stat, bins=6)
    for k in range(6):
        assert abs(out[strata == k].mean()) < 1e-12


# ── set membership ───────────────────────────────────────────────────


def test_set_members_keeps_only_sets_that_clear_the_floor():
    net = pd.DataFrame(
        {
            "source": ["big"] * 5 + ["small"] * 2 + ["absent"] * 4,
            "target": list("ABCDE") + list("AB") + list("WXYZ"),
        }
    )
    sets = set_members(np.array(list("ABCDEF")), net, min_n=3)
    assert sorted(sets) == ["big"]
    assert list(sets["big"]) == [0, 1, 2, 3, 4]


def test_set_members_ignores_duplicate_pairs():
    net = pd.DataFrame({"source": ["s"] * 4, "target": ["A", "A", "B", "C"]})
    (only,) = set_members(np.array(list("ABC")), net, min_n=3).values()
    assert list(only) == [0, 1, 2]


@pytest.mark.parametrize("gene_set", sorted(GENE_SET_SCHEMAS))
def test_set_sizes_agree_with_the_kernel_the_gsea_table_uses(gene_set):
    """The two tables sit side by side, so their ``set_size`` columns must match.

    Membership is resolved here without decoupler; this is what keeps that
    independence from turning into a silent disagreement with the GSEA beside it.
    """
    prune = pytest.importorskip("decoupler.pp.net").prune
    idxmat = pytest.importorskip("decoupler.pp.net").idxmat

    resources = Path(__file__).resolve().parents[1] / "vgtfm" / "biosignal" / "resources"
    net = load_net(resources, gene_set)
    features = np.array(sorted(set(net["target"].astype(str)))[:1500])

    mine = set_members(features, net, min_n=15)
    sources, _cnct, _starts, offsets = idxmat(
        features=features, net=prune(features=features, net=net, tmin=15, verbose=False)
    )
    assert dict(zip([str(s) for s in sources], np.asarray(offsets, dtype=int))) == {
        k: len(v) for k, v in mine.items()
    }


# ── the per-set mean and its two nulls ───────────────────────────────


def _planted_ranking(n_genes=8000, seed=0):
    """A background of noise, one set displaced, one set not.

    The background is wide enough that the displaced set barely moves its own
    reference: 40 genes at $-0.5$ shift the mean of 8,000 by $-0.0025$, well inside
    what a typical set's interval covers. A narrow background would make *every* set
    differ from a mean it had itself dragged down, which is a property of the fixture
    rather than of the test.
    """
    rng = np.random.default_rng(seed)
    genes = np.array([f"G{i:04d}" for i in range(n_genes)])
    values = rng.normal(scale=0.02, size=n_genes)
    values[:40] -= 0.5  # the displaced set's members
    net = pd.DataFrame(
        {
            "source": ["moved"] * 40 + ["typical"] * 40,
            "target": list(genes[:40]) + list(genes[100:140]),
        }
    )
    return pd.Series(values, index=genes), net


def test_the_null_permutes_gene_labels_rather_than_resampling_sets():
    """One permutation of the whole background per replicate, read by every set.

    Two properties the appendix states and the p-values depend on: set sizes are held
    fixed, so the null's spread is the sampling variability of a set mean of that size;
    and the null centres on the background mean, which is what lets one null serve both
    the against-zero and the against-the-background reading.
    """
    rng = np.random.default_rng(0)
    values = rng.normal(size=500)
    members = [np.arange(20), np.arange(100, 400)]

    null = enrichment_mod._permutation_null(values, members, times=4000, seed=1)

    assert null.shape == (2, 4000)
    assert null.mean(axis=1) == pytest.approx(values.mean(), abs=0.01)
    # The spread is set by the set's size and by the size of the background it is
    # drawn from, with the finite-population correction that says this is a
    # permutation of a fixed list of values rather than a bootstrap over them.
    n_bg = len(values)
    for row, n in zip(null, (20, 300)):
        expected = values.std() / np.sqrt(n) * np.sqrt((n_bg - n) / (n_bg - 1))
        assert row.std() == pytest.approx(expected, rel=0.1)


def test_the_matched_null_permutes_only_within_a_stratum():
    """ "Permuting within equal-count bins of the frozen R^2" has to mean exactly that.

    Each stratum's multiset of values is preserved, so a set's null draws keep its own
    baseline-predictivity profile and only which gene of a given predictivity carries
    which change is randomised. The two-stratum case below is the smallest one where
    an unstratified permutation would visibly differ: every value in the low stratum is
    below every value in the high one, so a mixed draw shows up in the mean.
    """
    values = np.concatenate([np.full(200, -1.0), np.full(200, 1.0)])
    strata = np.concatenate([np.zeros(200, dtype=np.int64), np.ones(200, dtype=np.int64)])
    members = [np.arange(180)]  # entirely inside the low stratum

    matched = enrichment_mod._permutation_null(values, members, times=200, seed=1, strata=strata)
    plain = enrichment_mod._permutation_null(values, members, times=200, seed=1)

    # Within its own stratum the set can only ever draw -1, which is what it observes.
    assert np.all(matched == -1.0)
    # Without the strata the same set draws from both halves and lands near zero.
    assert plain.mean() == pytest.approx(0.0, abs=0.05)


def test_the_matched_null_reproduces_the_bins_the_stage_permutes_within():
    """The strata the test uses are the ones :func:`baseline_strata` defines.

    The appendix quotes one number for the binning -- 20 equal-count bins of the frozen
    R^2 -- and it has to describe both the detrended ranking and this null, or the two
    corrections are not the same correction applied in two places.
    """
    rng = np.random.default_rng(0)
    stat = pd.Series(rng.normal(size=400), index=[f"G{i:04d}" for i in range(400)])
    baseline = pd.Series(rng.uniform(size=400), index=stat.index)

    strata = enrichment_mod._strata_for(stat, baseline, 20)
    assert np.array_equal(strata, baseline_strata(baseline.to_numpy(), 20))
    assert np.bincount(strata).tolist() == [20] * 20
    # A baseline that does not cover the ranking leaves the matched columns NaN
    # rather than binning on a silently reindexed vector.
    assert enrichment_mod._strata_for(stat, baseline.iloc[:399], 20) is None
    assert enrichment_mod._strata_for(stat, None, 20) is None


def test_run_set_mean_writes_every_column_a_caption_needs(tmp_path, monkeypatch):
    stat, net = _planted_ranking()
    monkeypatch.setattr(enrichment_mod, "load_net", lambda *_a, **_k: net)
    monkeypatch.setattr(enrichment_mod, "check_coverage", lambda *_a, **_k: (1.0, 80, 80))

    res = run_set_mean(stat, tmp_path, "hallmark", times=2000, n_boot=500, min_n=15)
    assert list(res.table.columns) == SET_MEAN_COLUMNS
    assert res.n_sets == 2
    assert res.table["set_size"].tolist() == [40, 40]
    # Ranked by effect size, most negative first, which is the order the figure draws.
    assert res.table["source"].tolist() == ["moved", "typical"]
    meta = res.meta()
    assert meta["permutations"] == 2000
    assert meta["background_n_genes"] == len(stat)
    assert meta["pval_resolution"] == pytest.approx(1 / 2000)


def test_a_displaced_set_clears_both_tests_and_a_typical_one_clears_neither(tmp_path, monkeypatch):
    stat, net = _planted_ranking()
    monkeypatch.setattr(enrichment_mod, "load_net", lambda *_a, **_k: net)
    monkeypatch.setattr(enrichment_mod, "check_coverage", lambda *_a, **_k: (1.0, 80, 80))

    t = run_set_mean(stat, tmp_path, "hallmark", times=2000, n_boot=500).table.set_index("source")
    assert t.loc["moved", "padj_vs_background"] < 0.01
    # The untouched set sits on the background mean, which the whole ranking shares.
    assert t.loc["typical", "padj_vs_background"] > 0.05
    # The bars the figure draws, on the same fact: a set the data cannot separate from
    # the transcriptome-wide average is one whose interval covers the line drawn at it.
    assert t.loc["typical", "ci_lo"] <= t.loc["typical", "background_mean"]
    assert t.loc["typical", "background_mean"] <= t.loc["typical", "ci_hi"]
    assert t.loc["moved", "ci_hi"] < t.loc["moved", "background_mean"]


def test_shifting_the_whole_background_leaves_the_selectivity_test_alone(tmp_path, monkeypatch):
    """``p_vs_background`` is about a set, not about where the transcriptome sits.

    That is the property E.3 leans on: displacing every gene by the same amount moves
    the background with the sets, so a typical set stays typical.
    """
    stat, net = _planted_ranking()
    stat = stat - 0.5  # shift the whole background, sets included
    monkeypatch.setattr(enrichment_mod, "load_net", lambda *_a, **_k: net)
    monkeypatch.setattr(enrichment_mod, "check_coverage", lambda *_a, **_k: (1.0, 80, 80))

    t = run_set_mean(stat, tmp_path, "hallmark", times=2000, n_boot=500).table.set_index("source")
    assert t.loc["typical", "padj_vs_background"] > 0.05
    assert t.loc["moved", "padj_vs_background"] < 0.01


def test_a_set_displaced_only_by_its_composition_fails_the_matched_null(tmp_path, monkeypatch):
    """The confound the matched null exists for, planted deliberately.

    ``stat`` here is a deterministic function of the baseline and nothing else, and
    the set is built from the genes with the largest baseline. Against the whole
    background it looks strongly displaced; against genes of its own baseline it is
    exactly typical, which is the correct reading.
    """
    rng = np.random.default_rng(3)
    n = 600
    genes = np.array([f"G{i:04d}" for i in range(n)])
    baseline = rng.uniform(size=n)
    values = -baseline + rng.normal(scale=0.01, size=n)
    top = np.argsort(baseline)[-60:]
    net = pd.DataFrame({"source": ["well_predicted"] * 60, "target": list(genes[top])})
    stat = pd.Series(values, index=genes)

    monkeypatch.setattr(enrichment_mod, "load_net", lambda *_a, **_k: net)
    monkeypatch.setattr(enrichment_mod, "check_coverage", lambda *_a, **_k: (1.0, 60, 60))
    res = run_set_mean(
        stat,
        tmp_path,
        "hallmark",
        times=2000,
        n_boot=500,
        baseline=pd.Series(baseline, index=genes),
        baseline_bins=10,
    )
    row = res.table.iloc[0]
    assert row["padj_vs_background"] < 0.01
    assert row["padj_vs_matched"] > 0.05
    # The matched null centres where the set's composition puts it, not on the
    # background mean, which is the number that makes the column readable.
    assert row["expected_matched"] == pytest.approx(row["mean_delta"], abs=0.02)
    assert res.meta()["baseline_matched_null_bins"] == 10


def test_the_matched_columns_stay_nan_when_no_baseline_is_given(tmp_path, monkeypatch):
    stat, net = _planted_ranking()
    monkeypatch.setattr(enrichment_mod, "load_net", lambda *_a, **_k: net)
    monkeypatch.setattr(enrichment_mod, "check_coverage", lambda *_a, **_k: (1.0, 80, 80))

    t = run_set_mean(stat, tmp_path, "hallmark", times=500, n_boot=200).table
    assert t["expected_matched"].isna().all()
    assert t["p_vs_matched"].isna().all()
    assert t["padj_vs_matched"].isna().all()
    # Only the matched columns: the rest of the row still has to be usable.
    assert np.isfinite(t[["mean_delta", "ci_lo", "ci_hi", "p_vs_background"]]).all().all()


def test_the_provenance_names_no_bins_when_the_matched_null_could_not_run(tmp_path, monkeypatch):
    """A baseline that does not cover the ranking leaves the matched columns NaN.

    The bins have to be reported from the strata that were actually built, or the
    provenance names a null that never ran.
    """
    stat, net = _planted_ranking()
    monkeypatch.setattr(enrichment_mod, "load_net", lambda *_a, **_k: net)
    monkeypatch.setattr(enrichment_mod, "check_coverage", lambda *_a, **_k: (1.0, 80, 80))

    partial = pd.Series(1.0, index=stat.index[:-1])  # one gene short of the ranking
    res = run_set_mean(
        stat, tmp_path, "hallmark", times=200, n_boot=100, baseline=partial, baseline_bins=20
    )
    assert res.table["p_vs_matched"].isna().all()
    assert res.baseline_bins == 0
    assert res.meta()["baseline_matched_null_bins"] == 0

    covering = pd.Series(np.arange(len(stat), dtype=float), index=stat.index)
    ran = run_set_mean(
        stat, tmp_path, "hallmark", times=200, n_boot=100, baseline=covering, baseline_bins=20
    )
    assert ran.table["p_vs_matched"].notna().all()
    assert ran.meta()["baseline_matched_null_bins"] == 20


def test_a_set_with_no_member_in_the_background_is_never_scored():
    """``min_n`` is floored at 1. A set present only by name has nothing to average,
    and admitting it would divide by zero in the permutation null."""
    net = pd.DataFrame({"source": ["absent", "present"], "target": ["NOPE", "G0001"]})
    sets = enrichment_mod.set_members(np.array(["G0001", "G0002"]), net, min_n=0)
    assert set(sets) == {"present"}


def test_an_unresolvable_set_mean_p_value_never_becomes_an_fdr_of_zero(tmp_path, monkeypatch):
    stat, net = _planted_ranking()
    monkeypatch.setattr(enrichment_mod, "load_net", lambda *_a, **_k: net)
    monkeypatch.setattr(enrichment_mod, "check_coverage", lambda *_a, **_k: (1.0, 80, 80))

    t = run_set_mean(stat, tmp_path, "hallmark", times=500, n_boot=200).table
    assert (t["p_vs_background"] == 0).any()  # the planted set is unreachable at 500 draws
    assert (t["padj_vs_background"] > 0).all()


def test_the_bootstrap_interval_brackets_the_mean_it_is_drawn_for(tmp_path, monkeypatch):
    stat, net = _planted_ranking()
    monkeypatch.setattr(enrichment_mod, "load_net", lambda *_a, **_k: net)
    monkeypatch.setattr(enrichment_mod, "check_coverage", lambda *_a, **_k: (1.0, 80, 80))

    t = run_set_mean(stat, tmp_path, "hallmark", times=500, n_boot=2000, ci=0.95).table
    assert (t["ci_lo"] <= t["mean_delta"]).all()
    assert (t["mean_delta"] <= t["ci_hi"]).all()
    # A wider interval covers more, so it cannot be narrower than the 95% one.
    wide = run_set_mean(stat, tmp_path, "hallmark", times=500, n_boot=2000, ci=0.99).table
    assert ((wide["ci_hi"] - wide["ci_lo"]) >= (t["ci_hi"] - t["ci_lo"]) - 1e-12).all()


def test_the_set_mean_is_reproducible_at_a_fixed_seed(tmp_path, monkeypatch):
    stat, net = _planted_ranking()
    monkeypatch.setattr(enrichment_mod, "load_net", lambda *_a, **_k: net)
    monkeypatch.setattr(enrichment_mod, "check_coverage", lambda *_a, **_k: (1.0, 80, 80))

    kw = dict(times=500, n_boot=200, seed=7)
    a = run_set_mean(stat, tmp_path, "hallmark", **kw).table
    b = run_set_mean(stat, tmp_path, "hallmark", **kw).table
    pd.testing.assert_frame_equal(a, b)
