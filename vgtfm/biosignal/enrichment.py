"""Gene-set readings of the delta-R^2 ranking.

Genes are ranked by how much a refinement changed their predictivity, and the
pathways at either end summarise ~16,000 individual numbers. The ranking is read
twice, because neither reading is sufficient on its own:

:func:`run_gsea`      *position*: a strongly negative enrichment score means a set's
                      members concentrate at the bottom. Invariant to how far
                      anything moved, so it cannot say whether a set lost 0.005 of
                      R^2 or 0.2.
:func:`run_set_mean`  *effect size*, on the scale the paper quotes.

Two collections, chosen because they answer different questions:

``hallmark``  MSigDB Hallmark — broad, well-characterised biological programmes.
``progeny``   PROGENy — footprints of signalling pathways inferred from
              perturbation experiments, so it speaks to signalling activity rather
              than to membership in a curated set.

The parquets are vendored under ``resources/`` so this runs on compute nodes with
no network access; :func:`fetch_resources` refreshes them from a machine that has
one. A coverage floor guards the common failure where gene identifiers are
Ensembl rather than HGNC symbols and the overlap silently collapses.

A pathway name and a score cannot be checked on their own, so every row carries the
rest of it: the collection, the set's size on the background it was scored against,
an unadjusted permutation p-value and its Benjamini-Hochberg adjustment
(:attr:`EnrichmentResult.table`), with the per-ranking half in
:meth:`EnrichmentResult.meta`. decoupler's public entry point overwrites the
permutation p-value with its own adjustment in place and surfaces neither the pruned
set sizes nor the background, so :func:`run_gsea` calls the same scoring kernel one
level down and adjusts itself. The normalised scores are identical to
``dc.mt.gsea``'s at the same ranking, seed and permutation count; only the adjusted
p-value differs, and only where decoupler's is zero.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

GENE_SET_SCHEMAS = {
    "hallmark": {
        "filename": "msigdb_hallmark.parquet",
        "source_col": "geneset",
        "target_col": "genesymbol",
        # Visium panels typically profile 45-55% of Hallmark even when symbol
        # matching is working correctly.
        "min_coverage": 0.45,
        # What the paper has to cite for this collection, and where it came from.
        "collection": "MSigDB Hallmark (h.all), retrieved with decoupler dc.op.hallmark",
        "citation": "Liberzon et al., Cell Syst. 2015",
    },
    "progeny": {
        "filename": "progeny.parquet",
        "source_col": "source",
        "target_col": "target",
        "min_coverage": 0.60,
        "collection": "PROGENy, retrieved with decoupler dc.op.progeny",
        "citation": "Schubert et al., Nat. Commun. 2018",
    },
}

#: Columns every enrichment table carries, in the order they are written. ``norm``
#: is the normalised enrichment score, ``pval`` the unadjusted permutation p-value,
#: ``padj`` its Benjamini-Hochberg adjustment over the sets scored on one ranking,
#: and ``set_size`` the members present *in the background* — the number the test
#: used, smaller than the set's nominal size.
RESULT_COLUMNS = ["sample", "source", "set_size", "norm", "pval", "padj"]

#: Columns of a set-mean table, in the order they are written. ``mean_delta`` is the
#: mean of the ranked quantity over the set's members present in the background, and
#: ``background_mean`` the same average over the whole background, repeated on every
#: row so the file stands on its own. ``ci_lo``/``ci_hi`` bound ``mean_delta`` by a
#: bootstrap over member genes. The two ``p_*`` columns are the permutation tests of
#: :func:`run_set_mean`, each Benjamini-Hochberg adjusted within its own family.
SET_MEAN_COLUMNS = [
    "sample",
    "source",
    "set_size",
    "mean_delta",
    "ci_lo",
    "ci_hi",
    "background_mean",
    "expected_matched",
    "p_vs_background",
    "padj_vs_background",
    "p_vs_matched",
    "padj_vs_matched",
]


@dataclass
class EnrichmentResult:
    """One collection scored against one ranking, with what it takes to cite it."""

    gene_set: str
    table: pd.DataFrame
    coverage: float
    #: Distinct genes in the collection, and how many of them the ranking carries.
    n_universe: int
    n_covered: int
    #: Genes in the ranking — the background the enrichment score walks down.
    n_ranked: int = 0
    #: Sets that cleared ``min_n`` members on that background and were scored.
    n_sets: int = 0
    permutations: int = 0
    min_n: int = 0
    seed: int = 0

    def meta(self) -> dict:
        """The per-ranking half of the method, for a caption or a provenance file."""
        schema = GENE_SET_SCHEMAS.get(self.gene_set, {})
        return {
            "gene_set": self.gene_set,
            "collection": schema.get("collection", self.gene_set),
            "citation": schema.get("citation", ""),
            "test": "GSEA on a ranked gene list (decoupler `gsea`)",
            "n_sets_scored": int(self.n_sets),
            "n_genes_in_collection": int(self.n_universe),
            "n_genes_in_collection_covered": int(self.n_covered),
            "coverage": float(self.coverage),
            "background_n_genes": int(self.n_ranked),
            "min_set_size": int(self.min_n),
            "permutations": int(self.permutations),
            "pval_resolution": 1.0 / self.permutations if self.permutations else float("nan"),
            "seed": int(self.seed),
            "multiple_testing": (
                "Benjamini-Hochberg over the sets scored on this ranking, applied to "
                "p-values floored at the permutation resolution, so padj is an upper bound"
            ),
        }


def load_net(resources_dir: str | Path, gene_set: str) -> pd.DataFrame:
    """Read a vendored collection and normalise it to decoupler's schema."""
    if gene_set not in GENE_SET_SCHEMAS:
        raise ValueError(f"unknown gene set '{gene_set}'. Known: {sorted(GENE_SET_SCHEMAS)}")
    schema = GENE_SET_SCHEMAS[gene_set]
    path = Path(resources_dir) / schema["filename"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m vgtfm.biosignal.enrichment --fetch` "
            f"on a machine with network access."
        )
    net = pd.read_parquet(path).rename(
        columns={schema["source_col"]: "source", schema["target_col"]: "target"}
    )
    keep = ["source", "target"] + [c for c in net.columns if c not in {"source", "target"}]
    return net[keep]


def collection_provenance(resources_dir: str | Path, gene_set: str) -> dict:
    """Identify the vendored parquet a run scored against.

    MSigDB revises Hallmark between releases and the parquets carry no version of
    their own, so the file's digest stands in for one; unlike a modification time it
    is stable across machines. ``sha256`` is the one to trust; ``md5`` is short enough
    to print in a caption and is the digest the paper quotes. Neither is used for
    security, hence ``usedforsecurity=False`` so this still runs under FIPS.
    """
    schema = GENE_SET_SCHEMAS[gene_set]
    path = Path(resources_dir) / schema["filename"]
    net = load_net(resources_dir, gene_set)
    record = path.with_suffix(".json")
    fetched = json.loads(record.read_text()) if record.exists() else {}
    return {
        "gene_set": gene_set,
        "collection": schema["collection"],
        "citation": schema["citation"],
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "md5": hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest(),
        "n_sets": int(net["source"].nunique()),
        "n_genes": int(net["target"].nunique()),
        **({"fetched": fetched} if fetched else {}),
    }


def scoring_versions() -> dict:
    """Installed versions the enrichment provenance has to name to be reproducible.

    :func:`run_gsea` scores through a *private* decoupler kernel, so its numbers are a
    property of that release as much as of the ranking. Recorded beside every table so
    the version is read off an artefact rather than assumed from ``requirements.txt``.
    The import is unguarded: provenance is only written after :func:`run_gsea` has
    already scored something, so a failure here is a broken install rather than a
    missing optional extra.
    """
    import decoupler

    return {"decoupler": str(decoupler.__version__)}


def check_coverage(
    universe: set[str], net: pd.DataFrame, gene_set: str, min_coverage: float | None = None
) -> tuple[float, int, int]:
    net_genes = set(net["target"].astype(str).unique())
    covered = net_genes & set(universe)
    coverage = len(covered) / max(1, len(net_genes))
    floor = min_coverage if min_coverage is not None else GENE_SET_SCHEMAS[gene_set]["min_coverage"]
    if coverage < floor:
        raise RuntimeError(
            f"gene-set '{gene_set}' coverage {coverage:.3f} is below {floor}. "
            f"The measured gene names are probably not HGNC symbols, or the "
            f"expression filter is too aggressive."
        )
    return coverage, len(covered), len(net_genes)


def benjamini_hochberg(pvalues) -> np.ndarray:
    """BH-adjusted p-values, with the usual monotonicity enforced.

    Written out here because the family it runs over — the sets scored on *one*
    ranking — is part of what the paper reports, and belongs beside the caption that
    states it. NaNs are carried through untouched, so a set that could not be scored
    does not enter the family.
    """
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    n = int(ok.sum())
    if n == 0:
        return out
    vals = p[ok]
    order = np.argsort(vals, kind="stable")
    ranked = vals[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.clip(ranked, 0.0, 1.0)
    out[ok] = adj
    return out


def _score_ranking(
    statistic: pd.Series, net: pd.DataFrame, *, min_n: int, times: int, seed: int
) -> pd.DataFrame:
    """``source, set_size, norm, pval`` for one ranking.

    ``dc.mt.gsea`` runs exactly this kernel and then discards the unadjusted p-value
    and the pruned set sizes; both are free one level below it.

    These helpers are private to decoupler and ``requirements.txt`` pins the version
    they belong to. If a release moves them the ``ImportError`` travels, rather than
    falling back to the wrapper and writing an enrichment table with an empty p
    column — the defect this module exists to fix — as an apparently successful run.
    """
    from decoupler.mt._gsea import _func_gsea
    from decoupler.pp.net import idxmat, prune

    features = statistic.index.to_numpy()
    pruned = prune(features=features, net=net, tmin=min_n, verbose=False)
    sources, cnct, starts, offsets = idxmat(features=features, net=pruned, verbose=False)
    norm, pval = _func_gsea(
        statistic.to_numpy(dtype=float)[None, :], cnct, starts, offsets, times=times, seed=seed
    )
    return pd.DataFrame(
        {
            "source": [str(s) for s in sources],
            "set_size": np.asarray(offsets, dtype=int),
            "norm": np.asarray(norm[0], dtype=float),
            "pval": np.asarray(pval[0], dtype=float),
        }
    )


def run_gsea(
    statistic: pd.Series,
    resources_dir: str | Path,
    gene_set: str,
    *,
    label: str = "delta_r2",
    times: int = 10_000,
    min_n: int = 15,
    seed: int = 42,
    min_coverage: float | None = None,
) -> EnrichmentResult:
    """GSEA of one ranked per-gene statistic against a vendored collection.

    *times* sets the resolution of the permutation p-value: a set whose observed
    score no permutation reached comes back with ``pval = 0``, which means
    ``p < 1/times``, and is to be reported as that bound.

    ``padj`` is Benjamini-Hochberg over the sets scored on this ranking, taken after
    flooring each p at that resolution. It is therefore an upper bound on the
    adjusted p-value rather than decoupler's point value, and it is never zero: a
    permutation test cannot deliver an FDR of exactly zero, and a table that prints
    one invites the reader to believe it.
    """
    statistic = statistic.dropna()
    statistic.index = statistic.index.astype(str)
    statistic.index.name = "gene"
    statistic.name = label

    net = load_net(resources_dir, gene_set)
    coverage, n_covered, n_universe = check_coverage(
        set(statistic.index), net, gene_set, min_coverage
    )

    table = _score_ranking(statistic, net, min_n=min_n, times=times, seed=seed)
    if not table.empty:
        # Flooring before adjusting is monotone in each p, so `padj` stays a valid
        # upper bound on the true q instead of reporting an unreachable zero.
        table["padj"] = benjamini_hochberg(np.maximum(table["pval"], 1.0 / times))
        table.insert(0, "sample", label)
        table = table[RESULT_COLUMNS].sort_values(
            ["padj", "pval", "norm"], na_position="last", kind="stable"
        )
        table = table.reset_index(drop=True)

    return EnrichmentResult(
        gene_set=gene_set,
        table=table,
        coverage=coverage,
        n_universe=n_universe,
        n_covered=n_covered,
        n_ranked=int(len(statistic)),
        n_sets=int(len(table)),
        permutations=int(times),
        min_n=int(min_n),
        seed=int(seed),
    )


@dataclass
class SetMeanResult:
    """One collection's per-set mean of a ranked statistic, with both its tests."""

    gene_set: str
    table: pd.DataFrame
    #: The same statistic over the whole background: what the gene-label null centres
    #: on, and what ``p_vs_background`` compares against.
    background_mean: float
    background_ci: tuple[float, float]
    coverage: float
    n_universe: int
    n_covered: int
    n_ranked: int = 0
    n_sets: int = 0
    permutations: int = 0
    bootstrap: int = 0
    ci: float = 0.95
    min_n: int = 0
    seed: int = 0
    #: Bins the matched null actually permuted within, 0 when it did not run. Set
    #: from the strata that were built, not from the request: a baseline that does
    #: not cover the ranking leaves the matched columns NaN, and the provenance has
    #: to say so rather than name the bins that were asked for.
    baseline_bins: int = 0

    def meta(self) -> dict:
        """The per-ranking half of the method, for a caption or a provenance file."""
        schema = GENE_SET_SCHEMAS.get(self.gene_set, {})
        return {
            "gene_set": self.gene_set,
            "collection": schema.get("collection", self.gene_set),
            "citation": schema.get("citation", ""),
            "test": (
                "mean of the ranked statistic over a set's members, against a "
                "gene-label permutation null, referred to the background mean and to "
                "a null permuted within equal-count bins of the baseline"
            ),
            "statistic": "mean over member genes",
            "n_sets_scored": int(self.n_sets),
            "n_genes_in_collection": int(self.n_universe),
            "n_genes_in_collection_covered": int(self.n_covered),
            "coverage": float(self.coverage),
            "background_n_genes": int(self.n_ranked),
            "background_mean": float(self.background_mean),
            "background_ci": [float(self.background_ci[0]), float(self.background_ci[1])],
            "min_set_size": int(self.min_n),
            "permutations": int(self.permutations),
            "pval_resolution": 1.0 / self.permutations if self.permutations else float("nan"),
            "bootstrap_resamples": int(self.bootstrap),
            "ci_level": float(self.ci),
            "baseline_matched_null_bins": int(self.baseline_bins),
            "seed": int(self.seed),
            "multiple_testing": (
                "Benjamini-Hochberg within each family separately — the sets scored on "
                "this ranking, once per reference — applied to p-values floored at the "
                "permutation resolution, so padj is an upper bound"
            ),
            "families": ["p_vs_background", "p_vs_matched"],
        }


def set_members(features, net: pd.DataFrame, *, min_n: int) -> dict[str, np.ndarray]:
    """``source -> positions in *features*`` for the sets big enough to score.

    Resolved here rather than through decoupler so the set-mean test does not inherit
    a private-API dependency it has no other use for. The test suite checks these
    sizes against ``decoupler.pp.net.idxmat``'s, since a ``set_size`` disagreeing with
    the GSEA table beside it would be worse than either number alone.

    *min_n* is floored at 1: a set with no member in the background has nothing to
    average, and admitting one would divide by zero in :func:`_permutation_null`.
    """
    position = {str(g): i for i, g in enumerate(np.asarray(features).astype(str))}
    pairs = net[["source", "target"]].astype(str).drop_duplicates()
    floor = max(int(min_n), 1)
    out = {}
    for source, members in pairs.groupby("source", sort=True)["target"]:
        idx = np.array([position[g] for g in members if g in position], dtype=np.int64)
        if len(idx) >= floor:
            out[str(source)] = np.sort(idx)
    return out


def _permutation_null(
    values: np.ndarray,
    members: list[np.ndarray],
    times: int,
    seed: int,
    strata: np.ndarray | None = None,
):
    """``(n_sets, times)`` set means under permuted gene labels.

    One permutation of the whole background per replicate, read by every set at once:
    set sizes are held fixed and the statistic is re-attached to genes at random.
    Sampling each set independently would give the same marginal null but destroy the
    correlation between overlapping sets.

    With *strata* the permutation runs *within* each stratum, so a set's null draws
    keep its own mix of well- and badly-predicted genes and only which gene of a given
    predictivity carries which change is randomised. The unstratified null centres on
    the background mean by construction; the stratified one centres wherever the set's
    composition puts it, so its reference is returned rather than assumed.
    """
    rng = np.random.default_rng(seed)
    sizes = np.array([len(m) for m in members], dtype=np.int64)
    flat = np.concatenate(members) if members else np.zeros(0, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    null = np.empty((len(members), times), dtype=np.float64)
    n = len(values)
    groups = (
        None
        if strata is None
        else [np.flatnonzero(strata == k) for k in np.unique(np.asarray(strata))]
    )
    permuted = np.empty(n, dtype=np.float64)
    for b in range(times):
        if groups is None:
            permuted[:] = values[rng.permutation(n)]
        else:
            for g in groups:
                permuted[g] = values[g][rng.permutation(len(g))]
        null[:, b] = np.add.reduceat(permuted[flat], starts) / sizes
    return null


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int, ci: float, rng) -> tuple[float, float]:
    """Percentile interval for the mean of *values*, resampling the values themselves.

    Over member genes, not over spots or folds, so it describes how much of a set's
    displacement is carried by a few of its genes. It is a spread, not a test: it
    treats genes as independent and holds the background fixed, so whether it covers
    ``background_mean`` is not the question ``p_vs_background`` answers. It is also
    not an interval over donors — the per-gene R^2 it averages is already pooled over
    a level's folds.
    """
    if len(values) < 2 or n_boot < 2:
        m = float(np.mean(values)) if len(values) else float("nan")
        return m, m
    draws = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    lo = (1.0 - ci) / 2.0 * 100.0
    return float(np.percentile(draws, lo)), float(np.percentile(draws, 100.0 - lo))


def _strata_for(statistic: pd.Series, baseline: pd.Series | None, bins: int):
    """Equal-count bins of *baseline*, aligned to *statistic*'s genes.

    ``None`` when no baseline was given or it does not cover the ranking, which leaves
    the matched columns NaN rather than silently binning on nothing. The caller records
    the bins from what comes back, so provenance never names a null that did not run.
    """
    if baseline is None:
        return None
    from .ridge import baseline_strata

    aligned = pd.Series(baseline).astype(float)
    aligned.index = aligned.index.astype(str)
    aligned = aligned.reindex(statistic.index)
    if aligned.isna().any():
        return None
    return baseline_strata(aligned.to_numpy(), bins)


def run_set_mean(
    statistic: pd.Series,
    resources_dir: str | Path,
    gene_set: str,
    *,
    label: str = "delta_r2",
    times: int = 10_000,
    n_boot: int = 10_000,
    min_n: int = 15,
    ci: float = 0.95,
    seed: int = 42,
    min_coverage: float | None = None,
    baseline: pd.Series | None = None,
    baseline_bins: int = 20,
) -> SetMeanResult:
    """Per-set mean of a ranked statistic, against the background and a matched null.

    The effect size GSEA cannot give: a normalised enrichment score is invariant to
    how far the statistic moved, so it cannot say whether a pathway lost 0.005 of R^2
    or 0.2. Two tests, because they license different claims:

    ``p_vs_background``  the set's mean differs from the mean over the whole
                         background — the selectivity claim.
    ``p_vs_matched``     the set's mean differs from what genes of the *same baseline
                         predictivity* did, when *baseline* is given. Change in R^2 is
                         strongly anti-correlated with the frozen R^2 and curated sets
                         are built from well-predicted genes, so a set can clear
                         ``p_vs_background`` on composition alone — 41 of the 50
                         Hallmark sets do, and 3 survive this. ``expected_matched`` is
                         where that null centres: the displacement the set's
                         composition alone predicts.

    Both are two-sided against a gene-label permutation null
    (:func:`_permutation_null`), floored at ``1/times`` and Benjamini-Hochberg
    adjusted within their own family. ``ci_lo``/``ci_hi`` are a separate percentile
    bootstrap over member genes, drawn by the figure as a spread rather than as a
    test; see :func:`_bootstrap_mean_ci`.

    The bootstrap and the two nulls draw from independent substreams of *seed*, so
    neither null borrows the other's permutations.
    """
    statistic = statistic.dropna()
    statistic.index = statistic.index.astype(str)
    statistic.index.name = "gene"
    statistic.name = label

    net = load_net(resources_dir, gene_set)
    coverage, n_covered, n_universe = check_coverage(
        set(statistic.index), net, gene_set, min_coverage
    )

    values = statistic.to_numpy(dtype=float)
    background = float(values.mean()) if len(values) else float("nan")
    boot_seed, plain_seed, matched_seed = np.random.SeedSequence(seed).generate_state(3)
    rng = np.random.default_rng(int(boot_seed))
    background_ci = _bootstrap_mean_ci(values, n_boot, ci, rng)

    sets = set_members(statistic.index.to_numpy(), net, min_n=min_n)
    sources = list(sets)
    members = [sets[s] for s in sources]

    strata = _strata_for(statistic, baseline, baseline_bins)
    table = pd.DataFrame(columns=SET_MEAN_COLUMNS)
    if sources:
        observed = np.array([values[m].mean() for m in members])
        null = _permutation_null(values, members, times, int(plain_seed))
        # The null's spread is the sampling variability of a size-matched set mean;
        # it is centred on the background, which is the hypothesis being tested.
        p_background = (np.abs(null - background) >= np.abs(observed - background)[:, None]).mean(
            axis=1
        )

        expected, p_matched = np.full(len(sources), np.nan), np.full(len(sources), np.nan)
        if strata is not None:
            matched = _permutation_null(values, members, times, int(matched_seed), strata=strata)
            expected = matched.mean(axis=1)
            p_matched = (
                np.abs(matched - expected[:, None]) >= np.abs(observed - expected)[:, None]
            ).mean(axis=1)

        bounds = [_bootstrap_mean_ci(values[m], n_boot, ci, rng) for m in members]
        floor = 1.0 / times
        table = pd.DataFrame(
            {
                "sample": label,
                "source": sources,
                "set_size": [len(m) for m in members],
                "mean_delta": observed,
                "ci_lo": [b[0] for b in bounds],
                "ci_hi": [b[1] for b in bounds],
                "background_mean": background,
                "expected_matched": expected,
                "p_vs_background": p_background,
                "padj_vs_background": benjamini_hochberg(np.maximum(p_background, floor)),
                "p_vs_matched": p_matched,
                "padj_vs_matched": benjamini_hochberg(np.maximum(p_matched, floor)),
            }
        )[SET_MEAN_COLUMNS]
        table = table.sort_values("mean_delta", kind="stable").reset_index(drop=True)

    return SetMeanResult(
        gene_set=gene_set,
        table=table,
        background_mean=background,
        background_ci=background_ci,
        coverage=coverage,
        n_universe=n_universe,
        n_covered=n_covered,
        n_ranked=int(len(statistic)),
        n_sets=int(len(table)),
        permutations=int(times),
        bootstrap=int(n_boot),
        ci=float(ci),
        min_n=int(min_n),
        seed=int(seed),
        baseline_bins=int(baseline_bins) if strata is not None else 0,
    )


def fetch_resources(resources_dir: str | Path) -> None:
    """Download Hallmark and PROGENy to ``resources/``. Needs network access.

    A JSON record beside each parquet says when it was fetched and with which
    decoupler, so a later run can name the revision it scored against.
    """
    import decoupler as dc

    out = Path(resources_dir)
    out.mkdir(parents=True, exist_ok=True)
    for gene_set, getter in (("hallmark", dc.op.hallmark), ("progeny", dc.op.progeny)):
        schema = GENE_SET_SCHEMAS[gene_set]
        net = getter(organism="human")
        path = out / schema["filename"]
        net.to_parquet(path, index=False)
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "collection": schema["collection"],
                    "citation": schema["citation"],
                    "fetched_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "decoupler": dc.__version__,
                    "n_sets": int(net[schema["source_col"]].nunique()),
                    "n_genes": int(net[schema["target_col"]].nunique()),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"wrote {path} and {path.with_suffix('.json')}")


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true", help="refresh the vendored parquets")
    ap.add_argument("--resources-dir", default=str(Path(__file__).parent / "resources"))
    args = ap.parse_args()
    if args.fetch:
        fetch_resources(args.resources_dir)
    else:
        for gs in GENE_SET_SCHEMAS:
            print(json.dumps(collection_provenance(args.resources_dir, gs), indent=2))
