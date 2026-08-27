"""scIB integration metrics, reported as two panels that are never combined.

``scib-metrics`` produces two families of scores: batch-removal metrics (iLISI,
kBET, batch ASW, PCR comparison, graph connectivity) and bio-conservation metrics
(NMI, ARI, label ASW, cLISI, isolated labels). The library also offers a weighted
"total"; this module does not compute one.

The batch axis is trivially gameable: an embedding that replaces every spot with
its slide's mean scores well on some batch metrics while destroying the biology,
and one that discards all structure scores well on others. A composite lets a
method trade away the thing being measured for the thing being controlled and
reports the trade as an improvement. The two panels side by side make it visible.

Both panels are computed on annotated spots only, with ``sample_id`` as the batch
key and the pathology annotation as the label key.
"""

from __future__ import annotations

import os

import numpy as np

from ..labels import labeled_mask
from ..degraded import refuse
from ..evaluate.probes import stratified_subsample
from ..provenance import array_fingerprint

#: Metric names in each panel, as emitted by scib-metrics (lower-cased, underscored).
#: Older releases emit ``silhouette_batch`` where current ones emit ``bras``; both
#: are listed so the panel is complete across versions.
BATCH_METRICS = (
    "ilisi",
    "kbet",
    "graph_connectivity",
    "pcr_comparison",
    "bras",
    "silhouette_batch",
)
BIO_METRICS = ("isolated_labels", "kmeans_nmi", "kmeans_ari", "silhouette_label", "clisi")
#: Never reported. A weighted total lets a method trade biology for batch mixing
#: and report the trade as an improvement; see this module's docstring.
COMPOSITE_METRICS = ("batch_correction", "bio_conservation", "total")


def _configure_jax() -> None:
    """Keep JAX on CPU in a low-memory configuration.

    scib-metrics runs its neighbour computations through JAX; on a machine with a
    small GPU it will otherwise try to allocate the whole device and fail in the
    middle of a long benchmark.
    """
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")


#: Neighbourhood size scib-metrics scores each graph metric at, read off
#: ``Benchmarker._neighbor_values`` and its ``MetricAnnDataAPI`` dispatch table. A
#: supplied graph is only comparable to one the Benchmarker built for itself if it is
#: truncated to the same k, and BBKNN returns far more neighbours than any of these
#: (``neighbors_within_batch`` x one per slide).
GRAPH_METRIC_K = {"ilisi": 90, "clisi": 90, "kbet": 50, "graph_connectivity": 15}


def scoring_rows(
    batch_labels, class_labels, *, max_spots: int = 10_000, seed: int = 42
) -> np.ndarray:
    """The spots :func:`benchmark` will score, as indices into the full table.

    A corrected *matrix* can be subsampled after the fact; a corrected *graph* cannot
    — drop 97% of the spots and the survivors' neighbours mostly no longer exist. So
    a graph-only method has to be fitted on the scored spots instead, which means
    knowing which ones those are before it runs. Same mask and same draw as
    :func:`benchmark`, so the two agree row for row.
    """
    mask = labeled_mask(np.asarray(class_labels))
    rows = np.flatnonzero(mask)
    cls = np.asarray(class_labels).astype(str)[mask]
    (kept,) = stratified_subsample(
        np.arange(len(rows)), labels=cls, max_samples=max_spots, random_seed=seed
    )
    return rows[kept]


def benchmark_graph(graph, batch_labels, class_labels) -> dict:
    """Run the graph half of the scIB panel on an already-corrected neighbour graph.

    ``Benchmarker`` takes embeddings and nothing else: it builds its own kNN graph
    from each one in ``prepare()``, and exposes no way to supply one. But three of the
    five batch metrics (iLISI, kBET, graph connectivity) and one of the five bio
    metrics (cLISI) read nothing except that graph, so a method that corrects the
    graph directly can still be scored on them by calling the functions underneath.

    The metrics that need coordinates — BRAS, PCR comparison, isolated labels, k-means
    NMI/ARI, label silhouette — are absent rather than zero. That is the honest
    reading for a method that produces no coordinates, and it is the point the
    comparison is making about BBKNN.

    *graph* is a :class:`..diagnostics.integration.NeighborGraph`; only ``.indices``
    and ``.distances`` are used, so any object carrying those two arrays will do.
    """
    _configure_jax()
    from scib_metrics import clisi_knn, graph_connectivity, ilisi_knn, kbet_per_label
    from scib_metrics.nearest_neighbors import NeighborsResults

    batch = np.asarray(batch_labels).astype(str)
    cls = np.asarray(class_labels).astype(str)
    n, width = graph.indices.shape
    if len(batch) != n or len(cls) != n:
        refuse(
            "the graph panel",
            f"the graph has {n} rows but {len(batch)} batch and {len(cls)} class "
            f"labels were supplied",
            hint="a graph-only method must be fitted on scib.scoring_rows(...), "
            "and scored against the labels of those same rows",
        )
    need = max(GRAPH_METRIC_K.values())
    if width < need:
        refuse(
            "the graph panel",
            f"the graph carries {width} neighbours per spot and the panel scores "
            f"iLISI/cLISI at k={need}",
            hint="raise the correction's per-batch neighbour count, or drop the "
            "graph-only method from diagnostics.integration_methods",
        )

    def at(k: int) -> "NeighborsResults":
        return NeighborsResults(indices=graph.indices[:, :k], distances=graph.distances[:, :k])

    out = {
        "ilisi": float(ilisi_knn(at(GRAPH_METRIC_K["ilisi"]), batch)),
        "kbet": float(kbet_per_label(at(GRAPH_METRIC_K["kbet"]), batch, cls)),
        "graph_connectivity": float(
            graph_connectivity(at(GRAPH_METRIC_K["graph_connectivity"]), cls)
        ),
        "clisi": float(clisi_knn(at(GRAPH_METRIC_K["clisi"]), cls)),
    }
    out["n_spots"] = int(n)
    out["n_batches"] = int(len(np.unique(batch)))
    return out


def benchmark(
    embedding: np.ndarray,
    batch_labels,
    class_labels,
    *,
    max_spots: int = 10_000,
    seed: int = 42,
    n_jobs: int = 4,
) -> dict:
    """Run the scIB panel on one embedding; returns flat ``{metric: value}``."""
    _configure_jax()
    import anndata as ad
    import pandas as pd
    from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation

    # Through the same helper a graph-only method is fitted on, so the two cannot
    # drift apart and score different spots.
    rows = scoring_rows(batch_labels, class_labels, max_spots=max_spots, seed=seed)
    X = np.asarray(embedding, dtype=np.float32)[rows]
    batch = np.asarray(batch_labels).astype(str)[rows]
    cls = np.asarray(class_labels).astype(str)[rows]
    if len(X) == 0:
        return {"error": "no annotated spots"}

    adata = ad.AnnData(X=X)
    adata.obsm["X_emb"] = X
    adata.obs["batch"] = pd.Categorical(batch)
    adata.obs["cell_type"] = pd.Categorical(cls)

    single_batch = len(np.unique(batch)) < 2
    if single_batch:
        print("      only one batch present — bio-conservation metrics only")

    bm = Benchmarker(
        adata,
        batch_key="batch",
        label_key="cell_type",
        embedding_obsm_keys=["X_emb"],
        bio_conservation_metrics=BioConservation(),
        batch_correction_metrics=None if single_batch else BatchCorrection(),
        n_jobs=n_jobs,
        progress_bar=False,
    )
    bm.benchmark()

    # min_max_scale=False keeps absolute values; the scaled variant is only
    # meaningful relative to whatever other embeddings happened to be in the run.
    row = bm.get_results(min_max_scale=False).iloc[0]
    out = {}
    for col, val in row.items():
        key = str(col).lower().replace(" ", "_").replace("-", "_")
        if key in COMPOSITE_METRICS:
            continue  # never reported; see module docstring
        out[key] = float(val) if val == val and not isinstance(val, str) else None
    unknown = set(out) - set(BATCH_METRICS) - set(BIO_METRICS)
    if unknown:
        print(
            f"      note: unclassified scib metrics {sorted(unknown)} "
            f"(scib-metrics version mismatch) — reported under 'other/'"
        )
    out["n_spots"] = int(len(X))
    out["n_batches"] = int(len(np.unique(batch)))
    return out


def split_panels(metrics: dict) -> tuple[dict, dict, dict]:
    """Separate a flat result dict into batch, bio and unclassified panels.

    Composite scores are dropped here as well as in :func:`benchmark`, so no
    weighted total reaches a panel by any route.
    """
    skip = {"n_spots", "n_batches", "error", *COMPOSITE_METRICS}
    batch = {k: v for k, v in metrics.items() if k in BATCH_METRICS}
    bio = {k: v for k, v in metrics.items() if k in BIO_METRICS}
    other = {k: v for k, v in metrics.items() if k not in batch and k not in bio and k not in skip}
    return batch, bio, other


def run(cfg, table, embeddings: dict[str, np.ndarray] | None = None):
    """Score the frozen features and, if provided, each learned embedding."""
    import pandas as pd

    d = cfg.diagnostics
    rows = []
    targets: dict[str, np.ndarray] = {c: table.features(c) for c in d.columns}
    if embeddings:
        targets.update(embeddings)

    for name, X in targets.items():
        print(f"    scIB panel: {name}")
        try:
            m = benchmark(
                X, table.sample_id, table.annotation, max_spots=d.scib_max_spots, seed=d.seed
            )
        except ImportError:
            # The caller turns a missing scib-metrics into an actionable message,
            # including the config field that omits the panel on purpose.
            raise
        except Exception as e:
            refuse(f"the scIB panel for '{name}'", f"{type(e).__name__}: {e}")
        batch, bio, other = split_panels(m)
        rows.append(
            {
                "substrate": cfg.data.substrate,
                "representation": name,
                **{f"batch/{k}": v for k, v in batch.items()},
                **{f"bio/{k}": v for k, v in bio.items()},
                **{f"other/{k}": v for k, v in other.items()},
                # The matrix this row scored. `integrate` reuses the `pca` row
                # rather than recomputing the panel on the same embedding, and a
                # reused score is only safe if the two stages are demonstrably
                # holding the same matrix -- see
                # :func:`..diagnostics.integration._stored_panel`.
                "embedding": array_fingerprint(X),
                "n_spots": m.get("n_spots"),
                "n_batches": m.get("n_batches"),
            }
        )
        for label, panel in (("batch", batch), ("bio", bio)):
            if panel:
                print(
                    f"      {label:6s}"
                    + "  ".join(f"{k}={v:.3f}" for k, v in panel.items() if v is not None)
                )
    return pd.DataFrame(rows)
