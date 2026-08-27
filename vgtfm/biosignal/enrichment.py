"""Gene-set enrichment on the delta-R^2 ranking.

Genes are ranked by how much a refinement changed their predictivity, and the
pathways at either end summarise ~11,000 individual numbers. A strongly negative
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
"""

from __future__ import annotations

from dataclasses import dataclass
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
    },
    "progeny": {
        "filename": "progeny.parquet",
        "source_col": "source",
        "target_col": "target",
        "min_coverage": 0.60,
    },
}


@dataclass
class EnrichmentResult:
    gene_set: str
    table: pd.DataFrame
    coverage: float
    n_universe: int
    n_covered: int


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


def _normalise_output(result, label: str) -> pd.DataFrame:
    """Flatten decoupler's GSEA return value into one row per pathway.

    The shape has changed across decoupler 2.x releases. Current versions return
    ``(nes, padj)``; older ones returned ``(es, nes, pval, padj)``; some return a
    single long frame. All three are accepted, and the column named ``norm`` is
    always the normalised enrichment score — the one worth reading, because the
    raw score scales with set size.
    """
    if isinstance(result, tuple):
        frames = [f for f in result if f is not None]
        if not frames:
            return pd.DataFrame()
        if len(frames) == 2:
            names = ["norm", "padj"]
        elif len(frames) >= 4:
            names = ["score", "norm", "pval", "padj"]
        else:
            names = ["norm", "pval", "padj"][: len(frames)]
        df = pd.DataFrame({"source": list(frames[0].columns)})
        for name, frame in zip(names, frames):
            df[name] = np.asarray(frame.iloc[0], dtype=float)
        df.insert(0, "sample", label)
    else:
        df = pd.DataFrame(result)
        df = df.rename(columns={"fdr": "padj", "nes": "norm", "es": "score"})
        if "sample" not in df.columns:
            df.insert(0, "sample", label)
    sort_key = next((c for c in ("padj", "pval", "norm") if c in df.columns), df.columns[-1])
    return df.sort_values(sort_key).reset_index(drop=True)


def run_gsea(
    statistic: pd.Series,
    resources_dir: str | Path,
    gene_set: str,
    *,
    label: str = "delta_r2",
    times: int = 1000,
    min_n: int = 15,
    seed: int = 42,
    min_coverage: float | None = None,
) -> EnrichmentResult:
    """GSEA of one ranked per-gene statistic against a vendored collection."""
    import decoupler as dc

    statistic = statistic.dropna()
    statistic.index = statistic.index.astype(str)
    statistic.index.name = "gene"

    net = load_net(resources_dir, gene_set)
    coverage, n_covered, n_universe = check_coverage(
        set(statistic.index), net, gene_set, min_coverage
    )

    mat = statistic.to_frame(name=label).T
    mat.index.name = "sample"
    result = dc.mt.gsea(data=mat, net=net, tmin=min_n, times=times, seed=seed, verbose=False)
    return EnrichmentResult(
        gene_set=gene_set,
        table=_normalise_output(result, label),
        coverage=coverage,
        n_universe=n_universe,
        n_covered=n_covered,
    )


def fetch_resources(resources_dir: str | Path) -> None:
    """Download Hallmark and PROGENy to ``resources/``. Needs network access."""
    import decoupler as dc

    out = Path(resources_dir)
    out.mkdir(parents=True, exist_ok=True)
    hallmark = dc.op.hallmark(organism="human")
    hallmark.to_parquet(out / GENE_SET_SCHEMAS["hallmark"]["filename"], index=False)
    progeny = dc.op.progeny(organism="human")
    progeny.to_parquet(out / GENE_SET_SCHEMAS["progeny"]["filename"], index=False)
    print(f"wrote {out}/msigdb_hallmark.parquet and progeny.parquet")


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
            net = load_net(args.resources_dir, gs)
            print(f"{gs}: {net.source.nunique()} sets, {net.target.nunique()} genes")
