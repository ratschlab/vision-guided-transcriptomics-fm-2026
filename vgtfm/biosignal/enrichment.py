"""Gene-set enrichment on the delta-R^2 ranking.

Genes are ranked by how much a refinement changed their predictivity, and the
pathways at either end summarise ~16,000 individual numbers. A strongly negative
normalised enrichment score means a pathway's members are concentrated at the
*bottom* of the ranking — the refinement degraded them systematically.

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
        """The per-ranking half of the method, for a caption or a sidecar file."""
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
    is stable across machines.
    """
    schema = GENE_SET_SCHEMAS[gene_set]
    path = Path(resources_dir) / schema["filename"]
    net = load_net(resources_dir, gene_set)
    sidecar = path.with_suffix(".json")
    fetched = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    return {
        "gene_set": gene_set,
        "collection": schema["collection"],
        "citation": schema["citation"],
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "n_sets": int(net["source"].nunique()),
        "n_genes": int(net["target"].nunique()),
        **({"fetched": fetched} if fetched else {}),
    }


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


def fetch_resources(resources_dir: str | Path) -> None:
    """Download Hallmark and PROGENy to ``resources/``. Needs network access.

    A JSON sidecar records when, and with which decoupler, so a later run can say
    which revision of a collection it scored against.
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
