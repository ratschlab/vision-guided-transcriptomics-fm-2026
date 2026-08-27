"""Pooling a cohort's slides into one AnnData, and splitting the result back out.

Geneformer ranks a spot's expressed genes against a fixed vocabulary, so embedding
one slide alone gives exactly the answer it gives inside a cohort, and it takes the
per-slide route. scGPT and CancerFoundation select highly variable genes before
tokenising, so for them the unit the selection is made over is a real choice with no
free lunch:

* **Per slide** each slide gets the gene set that best describes *it*, and no small
  cohort is outvoted by a large one. But each slide's embedding then summarises a
  different set of genes, so a difference between two slides mixes biology with the
  selection.
* **Pooled** every spot is summarised over the same genes, at the cost of a selection
  the cohort's composition drives: whichever genes carry variance in the cohorts with
  the most spots, or in the most cohorts, depending on the flavour (see below).
  Programs specific to an under-represented cohort can lose their genes entirely.

Note what per-slide selection does *not* cost, because it is easy to assume it does:
these encoders consume ``(gene token, binned value)`` pairs and return their ``<cls>``
token, so the output space is the same 512 or 256 dimensions however the input set was
chosen. Different gene sets are not different coordinate systems — a point the
upstream ``merged/cancerfoundation_embed_mixed.py`` is built on, and the reason a
per-group selection is a coherent thing to do at all.

``global`` is the default because it is what produced the artefacts the paper reports:
``config.paths.merged_datasets`` names ``none_midnight_scgpt_merged_...`` and
``..._cancerfoundation_merged_...``, the output of the upstream
``merged/precompute_*.py``, not of the per-slide ``precompute_models/*.py``. That is
provenance, not a verdict. All three strategies distort; they distort differently, and
which distortion is acceptable depends on what is being claimed from the embedding, so
the choice is a flag and is recorded next to its output.

The batch key is not a neutral fix for the second bullet, and the two models do not
even fail the same way, because scanpy's two HVG flavours combine batches by
different rules:

* scGPT's ``seurat_v3`` sorts on the **median rank** across the batches where a gene
  made the top-n, with the batch count only as a tiebreak. A gene that is rank 1 in
  one small cohort and absent elsewhere beats a gene that is rank 50 in all ten.
* CancerFoundation's ``cell_ranger`` sorts on the **batch count** first, dispersion
  second. A gene highly variable in one cohort out of ten loses to anything shared by
  five, so cohort-specific programs are suppressed by construction.

So the two substrates carry opposite gene-selection biases, which is worth knowing
when their downstream numbers are put in the same table.

The mechanics below are shared by whichever unit is chosen: one concatenated AnnData,
one pass, then the rows split back into the same per-slide parquets the per-slide
route writes. Nothing downstream can tell which route a parquet came by, which is why
the route has to be recorded rather than inferred.

Slides are joined on version-stripped Ensembl ids rather than on symbols, because
symbols are not unique and ``var_names_make_unique`` breaks the tie by position — two
slides whose genes are ordered differently would disagree about which one is
``TBCE-1``. ``join="inner"`` keeps the genes every slide carries: a gene absent from
one slide is a structural zero there and a measurement everywhere else, which is
exactly the kind of difference an HVG selection would then read as biology.

Visium barcodes repeat across slides — every slide has an ``AAACAAGTATCTCCCA-1`` — so
the pooled index is ``<dataset_id>__<sample_id>|<barcode>`` and the original barcode
stays in ``obs['spot_id']``. Splitting reads those columns back rather than parsing
the index apart, so a barcode that happens to contain the separator cannot be
mis-attributed to another slide.

Pooling holds the whole cohort's counts in memory at once (sparse: tens of GB for the
full ten cohorts, a few for one). The alternative — ``anndata.experimental.concat_on_disk``
— buys that back at the cost of a second copy on disk; if a site needs it, this is the
one function to change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd


class Slide(NamedTuple):
    """One slide of raw counts, and where the embeddings it yields belong."""

    dataset_id: str
    sample_id: str
    h5ad: str
    gene_id_column: str = "gene_ids"

    @property
    def source(self) -> str:
        """The key that survives pooling and names the slide on the way out."""
        return f"{self.dataset_id}__{self.sample_id}"


# ── the manifest ─────────────────────────────────────────────────────


def write_manifest(path: str | Path, model: str, slides: list[Slide]) -> Path:
    """Record which slides one pooled run covers.

    The manifest is what makes the pooled environments self-contained: they never
    read ``configs/datasets.json``, a site profile or a :class:`~vgtfm.config.Config`,
    only this file. It also pins the *set* a gene selection was made over, which is
    part of what the embeddings mean and is otherwise unrecorded.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "slides": [s._asdict() for s in slides]}
    path.write_text(json.dumps(payload, indent=2))
    return path


def read_manifest(path: str | Path) -> list[Slide]:
    payload = json.loads(Path(path).read_text())
    slides = payload["slides"] if isinstance(payload, dict) else payload
    missing = [s["h5ad"] for s in slides if not Path(s["h5ad"]).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(slides)} manifest entries have no .h5ad, "
            f"starting with {missing[0]}"
        )
    return [Slide(**s) for s in slides]


# ── pooling ──────────────────────────────────────────────────────────


def _standardise_var(adata, gene_id_column: str) -> None:
    """Index a slide's genes by version-stripped Ensembl id, keeping the symbol.

    The cohorts follow the 10x convention — ``var_names`` are HGNC symbols and
    ``var['gene_ids']`` the Ensembl ids — but a cohort that has only one of the two
    still has to pool, so the missing side falls back to the side that exists.
    """
    if gene_id_column in adata.var.columns:
        ensembl = [str(g).split(".")[0] for g in adata.var[gene_id_column]]
        symbols = [str(s) for s in adata.var_names]
    else:
        ensembl = [str(g).split(".")[0] for g in adata.var_names]
        symbols = list(ensembl)
    adata.var = pd.DataFrame(
        {"ensembl_id": ensembl, "gene_symbol": symbols}, index=pd.Index(ensembl)
    )
    adata.var_names_make_unique()


def _read_slide(slide: Slide):
    """One slide as raw counts, stripped of everything pooling does not need."""
    import anndata as ad
    import scipy.sparse as sp

    a = ad.read_h5ad(slide.h5ad)
    # X is raw counts in every cohort used here; the layer is the unambiguous
    # source when a cohort ships both.
    X = a.layers["counts"] if "counts" in a.layers else a.X
    X = sp.csr_matrix(X)
    obs = pd.DataFrame(
        {
            "spot_id": a.obs_names.astype(str),
            "dataset_id": slide.dataset_id,
            "sample_id": slide.sample_id,
            "sample_source": slide.source,
        },
        index=pd.Index([f"{slide.source}|{b}" for b in a.obs_names.astype(str)]),
    )
    part = ad.AnnData(X=X, obs=obs, var=a.var.copy())
    _standardise_var(part, slide.gene_id_column)
    return part


def pool(slides: list[Slide], *, verbose: bool = True):
    """Concatenate every slide into one AnnData of raw counts on shared genes."""
    import anndata as ad

    if not slides:
        raise ValueError("no slides to pool")
    parts = []
    for slide in slides:
        part = _read_slide(slide)
        if verbose:
            print(f"    {slide.source}: {part.n_obs:,} spots x {part.n_vars:,} genes")
        parts.append(part)

    # merge="first" carries gene_symbol through; the joined genes are shared, so
    # every part agrees on it.
    pooled = parts[0] if len(parts) == 1 else ad.concat(parts, join="inner", merge="first")
    if verbose:
        print(
            f"  pooled {len(slides)} slide(s): {pooled.n_obs:,} spots x "
            f"{pooled.n_vars:,} shared genes"
        )
    if pooled.n_vars == 0:
        raise ValueError(
            "the slides share no genes after joining on Ensembl id — check that "
            "gene_id_column names the right .var column for every cohort"
        )
    return pooled


def filter_genes(adata, min_cells: int = 100, *, verbose: bool = True):
    """Drop genes expressed in fewer than *min_cells* spots across the cohort.

    A gene seen in a handful of spots carries no dispersion an HVG selection can
    read, but it can still win a rank on a near-empty variance estimate.
    """
    import scipy.sparse as sp

    X = adata.X
    counts = (
        np.asarray(X.getnnz(axis=0)).ravel()
        if sp.issparse(X)
        else np.asarray((X != 0).sum(axis=0)).ravel()
    )
    keep = counts >= min_cells
    if verbose:
        print(f"  gene filter (min_cells={min_cells}): {int(keep.sum()):,}/{adata.n_vars:,} kept")
    return adata if keep.all() else adata[:, keep].copy()


def drop_batch_zero_genes(adata, batch_key: str = "dataset_id", *, verbose: bool = True):
    """Drop genes that are all-zero in *any* batch.

    Batch-aware HVG selection fits a dispersion per batch and then ranks the
    per-batch results; a gene with no counts at all in one batch has no dispersion
    there, and scanpy raises rather than skipping it. Removing those genes up front
    is what lets ``batch_key`` be used at all.
    """
    import scipy.sparse as sp

    keep = np.ones(adata.n_vars, dtype=bool)
    for batch in pd.unique(adata.obs[batch_key]):
        X = adata[adata.obs[batch_key] == batch].X
        nonzero = (
            np.asarray(X.getnnz(axis=0)).ravel() > 0
            if sp.issparse(X)
            else np.asarray((X != 0).sum(axis=0)).ravel() > 0
        )
        keep &= nonzero
    if verbose:
        print(f"  batch zero-gene filter ({batch_key}): {int(keep.sum()):,}/{adata.n_vars:,} kept")
    return adata if keep.all() else adata[:, keep].copy()


def symbols_as_var_names(adata, *, verbose: bool = True):
    """Re-index the pooled genes by HGNC symbol, which is what both vocabularies use.

    Order matters here, and not obviously. Symbols are not unique — several Ensembl
    ids map to one — so ``var_names_make_unique`` appends ``-1`` to the loser, and a
    gene named ``TBCE-1`` matches no vocabulary and is silently dropped by the model.
    *Which* copy loses depends on how many genes are in the frame when the tie is
    broken, so doing this before the QC filters and before HVG selection (as
    ``merged/precompute_*.py`` did) and doing it after give different gene sets.

    This runs where the upstream scripts ran it: immediately after pooling, on the
    full joined gene set.
    """
    if "gene_symbol" not in adata.var.columns:
        return adata
    symbols = pd.Index(adata.var["gene_symbol"].astype(str))
    # Counted before the rename, not pattern-matched after: plenty of real symbols
    # already end in -<digits>, and a diagnostic that cannot tell those from a
    # collision is worse than none.
    collided = int(symbols.duplicated(keep="first").sum())
    adata.var_names = symbols
    adata.var_names_make_unique()
    if verbose:
        print(
            f"  gene symbols restored over {adata.n_vars:,} genes: {collided:,} "
            f"collided and were suffixed (a suffixed symbol matches no vocabulary)"
        )
    return adata


def prepare_cohort(
    slides: list[Slide],
    *,
    min_cells: int = 100,
    batch_key: str = "dataset_id",
    var_names: str = "symbol",
    verbose: bool = True,
):
    """Pool, name the genes, then QC-filter — the order ``merged/precompute_*.py`` used.

    Both filters run *before* any model-specific preprocessing, so every model sees
    the same gene universe and a difference between two substrates cannot be a
    difference in which genes were on the table.
    """
    adata = pool(slides, verbose=verbose)
    if var_names == "symbol":
        adata = symbols_as_var_names(adata, verbose=verbose)
    adata = filter_genes(adata, min_cells=min_cells, verbose=verbose)
    adata = drop_batch_zero_genes(adata, batch_key=batch_key, verbose=verbose)
    return adata


# ── splitting back out ───────────────────────────────────────────────


def split_to_parquets(obs: pd.DataFrame, features: np.ndarray, out_dir: str | Path) -> list[Path]:
    """Write one ``<dataset_id>/<sample_id>.parquet`` per slide, from pooled rows.

    *obs* must be the pooled ``adata.obs`` reindexed to the rows *features* actually
    covers — the embedders return the row labels they produced for exactly this
    reason, so a model that drops a spot drops it here too instead of shifting every
    later row into the wrong slide.
    """
    from .gene_fm import write_parquet

    if len(obs) != len(features):
        raise ValueError(f"{len(obs)} rows of obs vs {len(features)} embeddings")
    out_dir = Path(out_dir)
    written: list[Path] = []
    for (dataset_id, sample_id), rows in obs.groupby(["dataset_id", "sample_id"], sort=True):
        idx = obs.index.get_indexer(rows.index)
        target = out_dir / str(dataset_id) / f"{sample_id}.parquet"
        write_parquet(target, rows["spot_id"].to_numpy(), features[idx])
        written.append(target)
    return written


# ── choosing the genes ───────────────────────────────────────────────

#: Where gene selection is made, and whether groups share the result.
HVG_STRATEGIES = ("global", "mixed", "per_slide")


def _hvg(adata, *, n_top: int, flavor: str, layer: str | None, batch_key: str | None):
    """One scanpy HVG call, returning the selected genes in ``var`` order.

    Both flavours estimate a mean-variance relation across genes, and both give up
    when too many genes share a near-zero mean — in different words, neither of
    which names the cause::

        seurat_v3    ValueError: b'reciprocal condition number  2.5833e-16'
        cell_ranger  ValueError: Bin edges must be unique: Index([-inf, 0.00666...

    The first is the loess design matrix going singular, the second is twenty
    quantiles of the gene means landing on the same edge. The cause is the same one:
    too many genes at a near-zero mean for the number of spots *in this call*.

    Which number that is depends on the strategy, and it is not the cohort size.
    ``global`` fits over every spot; ``mixed`` fits within each group; ``per_slide``
    within one slide. Since :func:`filter_genes` applies its floor to the pooled
    cohort, the same cohort needs a very different floor depending on which is asked
    for. Measured on four USZ slides, all 17,845 genes, taking the floor at which
    *every* slide fits:

        ======  ===========  ==========  ============  ==============
        pooled  strategy      seurat_v3   cell_ranger   genes at that
        ======  ===========  ==========  ============  ==============
        600     global        >= 10       >= 20         13,676
        600     mixed         >= 30       >= 30         10,362
        600     per_slide     >= 100      >= 150           923
        1,600   per_slide     >= 40       >= 200         5,648
        ======  ===========  ==========  ============  ==============

    Note what that is not: a ratio that generalises. ``cell_ranger`` fits one slide
    at 12,604 genes and refuses another at 1,904, because what breaks it is how many
    genes share a mean, which is a property of the slide. ``per_slide`` is the
    expensive case — at 600 pooled spots the floor it needs leaves fewer genes than
    the token budget asks for — and that is a real cost of the strategy on a small
    cohort, not an artefact of checking it.

    None of this binds on the production cohort — ~328k spots at a floor of 100 —
    but it binds immediately on any subset, which is what anyone checking this code
    runs.

    Raising the floor here to make the call succeed would be the wrong repair: it
    would change which genes the embeddings summarise, silently, and differently for
    the two substrates. So this reports the cause and stops.

    One HVG call is out of reach of this: under ``global`` CancerFoundation delegates
    selection to upstream's ``embed()``, which calls scanpy itself, and that one still
    raises the bare message.
    """
    import scanpy as sc

    try:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=min(n_top, adata.n_vars),
            flavor=flavor,
            layer=layer,
            batch_key=batch_key,
        )
    except ValueError as e:
        raise ValueError(
            f"{flavor} could not fit a mean-variance relation over "
            f"{adata.n_obs:,} spots x {adata.n_vars:,} genes"
            f"{'' if batch_key is None else f' (batch_key={batch_key!r})'}: {e}. "
            f"Too many genes sit at a near-zero mean for this many spots. Note the "
            f"spot count: it is the unit this selection is fitted over, not the "
            f"cohort, so a per-group strategy needs a higher --min-cells than the "
            f"same cohort does under `global` — about 20 * cohort / unit, and "
            f"--min-cells is applied to the cohort. Raise it, or embed more spots. "
            f"Changing the floor changes which genes the embeddings summarise, so "
            f"it belongs in the recorded run parameters rather than in a retry."
        ) from e
    return adata.var_names[adata.var["highly_variable"]].tolist()


def shared_genes(
    adata,
    *,
    group_key: str,
    n_shared: int,
    n_candidates: int,
    flavor: str,
    layer: str | None = None,
    min_group_size: int = 50,
    verbose: bool = True,
) -> list[str]:
    """Genes that rank highly in *many* groups, ranked by how many and then how high.

    Ported from ``merged/cancerfoundation_embed_mixed.py``. Selection is run inside
    each group and the per-group lists are then combined, so a cohort of 40k spots
    and one of 2k each contribute one list; group size enters only through how well
    that group's own dispersion is estimated.
    """
    from collections import Counter

    groups = pd.unique(adata.obs[group_key])
    in_groups: Counter = Counter()
    rank_sum: dict[str, float] = {}
    rank_n: dict[str, int] = {}

    for group in groups:
        sub = adata[adata.obs[group_key] == group]
        if sub.n_obs < min_group_size:
            if verbose:
                print(f"    {group}: {sub.n_obs} spots < {min_group_size}, not voting on shared")
            continue
        for rank, gene in enumerate(
            _hvg(sub.copy(), n_top=n_candidates, flavor=flavor, layer=layer, batch_key=None)
        ):
            in_groups[gene] += 1
            rank_sum[gene] = rank_sum.get(gene, 0.0) + rank
            rank_n[gene] = rank_n.get(gene, 0) + 1

    ranked = sorted(in_groups, key=lambda g: (-in_groups[g], rank_sum[g] / max(rank_n[g], 1)))
    chosen = ranked[:n_shared]
    if verbose and ranked:
        print(
            f"    shared: {len(chosen):,} genes; the top one is highly variable in "
            f"{in_groups[ranked[0]]}/{len(groups)} groups"
        )
    return chosen


def select_genes(
    adata,
    *,
    strategy: str = "global",
    n_genes: int,
    flavor: str,
    layer: str | None = None,
    batch_key: str | None = None,
    group_key: str = "dataset_id",
    shared_ratio: float = 0.70,
    n_candidate_factor: int = 2,
    min_group_size: int = 50,
    verbose: bool = True,
) -> dict[str, list[str]]:
    """Which genes each group of spots is described by. ``{group -> genes}``.

    The three strategies differ only in the unit the selection is made over and in
    how much of the budget is spent on genes every group shares:

    ``global``      one set for the whole cohort, from a single batch-aware call.
                    Every spot is described by the same genes; a program carried by
                    one under-represented cohort can lose its genes outright, and
                    which way that goes depends on the flavour (see the module
                    docstring — ``seurat_v3`` and ``cell_ranger`` lean opposite ways).
    ``mixed``       ``shared_ratio`` of the budget on genes that rank highly across
                    groups, the rest on each group's own. Ported from
                    ``merged/cancerfoundation_embed_mixed.py``.
    ``per_slide``   each slide chooses alone, sharing nothing.

    ``mixed`` and ``per_slide`` give different groups different genes, which for these
    models is a smaller thing than it sounds: the encoder consumes ``(gene token,
    binned value)`` pairs and returns its ``<cls>`` token, so the output space is the
    same however the input set was chosen. What differs is *which* genes a group's
    embedding summarises, so a difference between two groups mixes biology with the
    selection. That is the cost being traded, not a change of basis.
    """
    if strategy not in HVG_STRATEGIES:
        raise ValueError(f"unknown hvg strategy {strategy!r}; known: {HVG_STRATEGIES}")

    if strategy == "global":
        genes = _hvg(adata, n_top=n_genes, flavor=flavor, layer=layer, batch_key=batch_key)
        if verbose:
            print(f"  genes ({strategy}): {len(genes):,} for the whole cohort")
        return {"all": genes}

    key = "sample_source" if strategy == "per_slide" else group_key
    ratio = 0.0 if strategy == "per_slide" else shared_ratio
    n_shared = int(n_genes * ratio)
    n_candidates = max(n_shared * n_candidate_factor, n_genes)
    if verbose:
        print(
            f"  genes ({strategy}, over {key}): {n_shared:,} shared + "
            f"{n_genes - n_shared:,} specific"
        )

    common = (
        shared_genes(
            adata,
            group_key=key,
            n_shared=n_shared,
            n_candidates=n_candidates,
            flavor=flavor,
            layer=layer,
            min_group_size=min_group_size,
            verbose=verbose,
        )
        if n_shared
        else []
    )
    common_set = set(common)

    out: dict[str, list[str]] = {}
    for group in pd.unique(adata.obs[key]):
        sub = adata[adata.obs[key] == group]
        present = [g for g in common if g in set(sub.var_names)]
        if sub.n_obs < min_group_size and common:
            # Too few spots to estimate a dispersion on; the shared set is the
            # honest fallback, and saying so beats a silently noisier selection.
            if verbose:
                print(f"    {group}: {sub.n_obs} spots, shared genes only")
            out[str(group)] = present
            continue
        specific = [
            g
            for g in _hvg(
                sub.copy(), n_top=n_candidates, flavor=flavor, layer=layer, batch_key=None
            )
            if g not in common_set
        ]
        own = specific[: n_genes - len(present)]
        out[str(group)] = present + own
        if verbose:
            print(f"    {group}: {len(present):,} shared + {len(own):,} own")
    return out


def group_masks(adata, strategy: str, group_key: str) -> dict[str, np.ndarray]:
    """Row masks matching the keys :func:`select_genes` returned."""
    if strategy == "global":
        return {"all": np.ones(adata.n_obs, dtype=bool)}
    key = "sample_source" if strategy == "per_slide" else group_key
    values = adata.obs[key].astype(str).to_numpy()
    return {str(g): values == str(g) for g in pd.unique(values)}


def write_provenance(out_dir: str | Path, payload: dict) -> Path:
    """Record which basis these parquets were built in, beside the parquets.

    The parquets themselves are just ``spot_id`` and numbers, and two runs under
    different strategies are indistinguishable once written — which is exactly why
    the strategy, the group key, the seed and the gene sets have to be on disk rather
    than in whoever's shell history.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "provenance.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"  provenance -> {target}")
    return target


def report_missing(slides: list[Slide], written: list[Path]) -> list[str]:
    """Name the manifest entries no parquet was written for.

    A slide that survives pooling and then produces nothing is not visible in the
    merge — it simply contributes no spots — so it is named here instead.
    """
    got = {(p.parent.name, p.stem) for p in written}
    return [s.source for s in slides if (s.dataset_id, s.sample_id) not in got]
