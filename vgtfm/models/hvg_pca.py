"""The baseline that skips the foundation model entirely.

Every other gene-side representation here starts from a foundation model's output,
so ``pca`` measures what the autoencoder adds on top of Geneformer and nothing
measures what Geneformer adds on top of the counts. This model closes that gap: the
standard scanpy workflow — CP10k, ``log1p``, highly-variable-gene selection,
per-gene scaling, PCA — on the raw Visium counts the foundation models were given.

Three details are deliberate. Two concern leakage:

* **HVG ranking uses the fit split only.** Which genes *exist* is a property of the
  assay, so the vocabulary spans every slide that will be embedded — a gene absent
  from a slide cannot be a feature there. Which genes are *variable* is learned
  from data, so it is learned from the spots that fit the PCA. Ranking on the pooled
  cohort would let the held-out donors choose the features used to classify them.
* **The scaler is fitted on the fit split** and applied unchanged at embed time,
  for the same reason.

The third concerns memory. This model reads 120 per-slide ``.h5ad`` files, and
holding them dense at once is tens of GB. Every pass over the cohort is chunked by
slide with the cache evicted between chunks, keeping the resident set at one slide
plus the output matrix.

The result is a function of counts alone, so it satisfies the same deployment
contract as the other gene models.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from ..evaluate.probes import stratified_subsample
from .base import Inputs, Model


def seurat_v3_dispersion(counts: np.ndarray) -> np.ndarray:
    """Per-gene normalised variance, as in scanpy's ``flavor="seurat_v3"``.

    Ranks genes by the variance of their *variance-stabilised* counts: fit
    ``log10(var) ~ log10(mean)`` across genes, standardise each gene by the
    expected standard deviation that fit predicts, clip at ``sqrt(n)`` so a handful
    of extreme spots cannot carry a gene, and take the variance of the result.

    Implemented here rather than pulled from scanpy so a model build does not need
    scanpy and an AnnData round-trip. Genes with zero variance score 0.0.
    """
    n = counts.shape[0]
    mean = counts.mean(axis=0)
    var = counts.var(axis=0)

    out = np.zeros_like(var, dtype=np.float64)
    ok = (var > 0) & (mean > 0)
    if ok.sum() < 3:
        return var.astype(np.float64)

    # Expected dispersion as a function of abundance, in log space.
    coef = np.polyfit(np.log10(mean[ok]), np.log10(var[ok]), deg=2)
    expected_sd = np.sqrt(10.0 ** np.polyval(coef, np.log10(mean[ok])))

    clip = np.sqrt(n)
    z = (counts[:, ok] - mean[ok]) / np.maximum(expected_sd, 1e-12)
    np.clip(z, -clip, clip, out=z)
    # sum(z^2)/(n-1), not var(z): scanpy standardises about the *unclipped* mean
    # and never re-centres, so a gene whose clipping is one-sided keeps that
    # asymmetry in its score.
    out[ok] = (z**2).sum(axis=0) / (n - 1)
    return out


class HVGPCA(Model):
    """CP10k -> log1p -> top-N HVG -> scale -> PCA, starting from raw counts."""

    name = "hvg_pca"

    def __init__(self, cfg, seed: int = 42):
        super().__init__(cfg, seed)
        self.hvg = cfg.models.hvg
        self.n_components = int(cfg.models.pca_components)
        self.genes: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.pca: PCA | None = None
        self._index = None

    # -- expression access ------------------------------------------
    def _expression_index(self):
        from ..biosignal.expression import ExpressionIndex

        if self._index is None:
            self._index = ExpressionIndex(self.cfg)
        return self._index

    def _matrix(
        self,
        inputs: Inputs,
        genes: np.ndarray,
        rows: np.ndarray | None = None,
        *,
        normalize: bool = True,
    ):
        """``(Y, found)`` over *rows* of *inputs*, restricted to *genes*.

        Chunked by slide with the slide cache dropped between chunks; see this
        module's docstring. Rows are grouped so each slide is opened exactly once.
        """
        index = self._expression_index()
        index.genes = np.asarray(genes)

        idx = np.arange(inputs.n) if rows is None else np.asarray(rows)
        Y = np.zeros((len(idx), len(index.genes)), dtype=np.float32)
        found = np.zeros(len(idx), dtype=bool)

        sample = np.asarray(inputs.sample_id).astype(str)[idx]
        order = np.argsort(sample, kind="stable")
        chunk = max(1, int(self.hvg.slides_per_chunk))
        slides = list(dict.fromkeys(sample[order].tolist()))
        for start in range(0, len(slides), chunk):
            batch = set(slides[start : start + chunk])
            take = order[np.isin(sample[order], list(batch))]
            src = idx[take]
            y, ok = index.matrix(
                inputs.dataset_id[src],
                inputs.sample_id[src],
                inputs.spot_id[src],
                normalize=normalize,
            )
            Y[take] = y
            found[take] = ok
            index.drop_cache()
        return Y, found

    # -- Model ------------------------------------------------------
    def fit(self, inputs: Inputs) -> dict:
        from ..data import tables

        index = self._expression_index()

        # Vocabulary over *every* slide the embedding will cover, not just the fit
        # split: a gene missing from a held-out cohort's panel cannot be a feature
        # there. Metadata only — this does not materialise the feature matrices.
        cohort = tables.load_meta(self.cfg)
        shared = index.build_vocabulary(
            cohort["dataset_id"].to_numpy(),
            cohort["sample_id"].to_numpy(),
            cohort["spot_id"].to_numpy(),
            min_expressed=int(self.hvg.min_expressed),
            max_spots=int(self.hvg.vocabulary_max_spots),
            seed=self.seed,
            stream=True,
        )
        index.drop_cache()

        # Rank on a slide-balanced subsample of the fit spots. Gene variance is
        # estimated fine from tens of thousands of spots, and the full fit split at
        # the full vocabulary would be a ~12 GB dense matrix.
        rank_rows = np.arange(inputs.n)
        (rank_rows,) = stratified_subsample(
            rank_rows,
            labels=np.asarray(inputs.sample_id).astype(str),
            max_samples=int(self.hvg.rank_max_spots),
            random_seed=self.seed,
        )
        # Raw counts, not log-CP10k: seurat_v3 standardises each gene against a
        # mean-variance trend fitted across genes, and library-size normalisation
        # changes that trend, so ranking on normalised values silently selects a
        # different gene set than the recipe names.
        Y_rank, ok_rank = self._matrix(inputs, shared, rank_rows, normalize=False)
        if ok_rank.sum() < 2:
            raise SystemExit(
                f"hvg_pca resolved {int(ok_rank.sum())} of {len(rank_rows)} sampled "
                f"fit spots in the raw .h5ad files.\nCheck paths.raw_h5ad in "
                f"environments.yaml — every cohort in the fit split needs its "
                f"per-slide .h5ad reachable."
            )

        n_top = min(int(self.hvg.n_top_genes), len(shared))
        dispersion = seurat_v3_dispersion(Y_rank[ok_rank])
        keep = np.sort(np.argsort(dispersion)[::-1][:n_top])
        self.genes = shared[keep]
        del Y_rank

        # Second pass over *all* fit rows, now only 2000 genes wide, so the scaler
        # and the PCA see the same spots the other baselines are fitted on.
        Y, found = self._matrix(inputs, self.genes)
        n_found = int(found.sum())
        if n_found < 2:
            raise SystemExit("hvg_pca resolved no usable fit spots; see above")
        Y = Y[found]

        self.mean_ = Y.mean(axis=0)
        if self.hvg.scale:
            self.scale_ = Y.std(axis=0)
            self.scale_[self.scale_ < 1e-8] = 1.0
        else:
            self.scale_ = np.ones_like(self.mean_)
        Y = (Y - self.mean_) / self.scale_

        n = min(self.n_components, Y.shape[0], Y.shape[1])
        if n < self.n_components:
            print(f"  [{self.name}] capping components {self.n_components} -> {n} (data {Y.shape})")
        self.pca = PCA(n_components=n, random_state=self.seed)
        self.pca.fit(Y)

        evr = float(np.sum(self.pca.explained_variance_ratio_))
        self.history = {
            "n_components": int(n),
            "n_fit_spots": n_found,
            "n_fit_spots_unresolved": int(inputs.n - n_found),
            "input_dim": int(len(self.genes)),
            "n_shared_genes": int(len(shared)),
            "n_rank_spots": int(ok_rank.sum()),
            "explained_variance_ratio": evr,
            "scaled": bool(self.hvg.scale),
        }
        print(
            f"  [{self.name}] {len(self.genes):,} HVG of {len(shared):,} shared "
            f"genes, {n_found:,} fit spots -> {n} PCs, {evr:.4f} of variance"
        )
        return self.history

    def embed(self, inputs: Inputs) -> np.ndarray:
        if self.pca is None or self.genes is None:
            raise RuntimeError(f"{self.name}.embed called before fit")

        Y, found = self._matrix(inputs, self.genes)
        Z = np.zeros((inputs.n, int(self.pca.n_components_)), dtype=np.float32)
        if found.any():
            Z[found] = self.pca.transform((Y[found] - self.mean_) / self.scale_).astype(np.float32)
        missing = int((~found).sum())
        if missing:
            # Zero rows rather than dropped rows: every stage downstream indexes
            # the embedding by the spot table's row order, so the array has to keep
            # its shape. The count is reported because a silent block of identical
            # rows would read as a collapsed representation rather than absent data.
            print(
                f"  [{self.name}] {missing:,} of {inputs.n:,} spots had no raw "
                f".h5ad row and are embedded as zeros"
            )
        return Z

    def state(self) -> dict:
        return {**super().state(), "n_genes": int(len(self.genes)) if self.genes is not None else 0}
