"""Effective rank of the frozen embeddings, with the controls that make it mean
something.

Roy & Vetterli's effective rank is ``exp(H(p))`` over the normalised spectrum of
squared singular values — the *exponential*, which puts it on the scale of a
dimension count rather than of an entropy.

A bare effective rank is uninterpretable, so every column is scored against:

``gaussian``          i.i.d. normal noise at the same (n, d). Saturates near
                      ``min(n, d)`` and so marks the top of the scale.
``column_shuffled``   the real matrix with each column independently permuted.
                      Preserves every marginal and destroys every correlation, so
                      the gap to the observed value is exactly the part owed to
                      correlated structure rather than per-dimension variance.
``pca<k>``            a k-component PCA of the real matrix, i.e. the ``pca``
                      baseline representation.

Compare fractions, never raw ranks: 78.7 of 3072 is a *smaller* share than 29.3 of
1152.
"""

from __future__ import annotations

import numpy as np

from ..models.nn import effective_rank
from . import split_rows


def bootstrap_effective_rank(
    X: np.ndarray, *, n_samples: int = 8000, n_iters: int = 5, seed: int = 42
) -> dict:
    """Effective rank over ``n_iters`` random subsamples of ``n_samples`` rows.

    Subsampling keeps the SVD affordable on hundreds of thousands of spots, and the
    spread across resamples shows the estimate is not an artefact of which spots
    were drawn.
    """
    n, d = X.shape
    rng = np.random.default_rng(seed)
    take = min(n_samples, n)
    ranks = []
    for _ in range(n_iters):
        idx = rng.choice(n, size=take, replace=False)
        ranks.append(effective_rank(X[idx]))
    ranks_arr = np.asarray(ranks, dtype=float)
    return {
        "n_total": int(n),
        "dim": int(d),
        "n_samples": int(take),
        "n_iters": int(n_iters),
        "eff_rank": float(ranks_arr.mean()),
        "eff_rank_std": float(ranks_arr.std(ddof=0)),
        "eff_rank_min": float(ranks_arr.min()),
        "eff_rank_max": float(ranks_arr.max()),
        "pct_of_dim": float(100.0 * ranks_arr.mean() / d),
    }


def mean_pairwise_cosine(X: np.ndarray, *, n_samples: int = 4000, seed: int = 42) -> float:
    """Average cosine similarity between distinct rows.

    A complementary view of the same collapse: when every spot's embedding points in
    nearly the same direction (Geneformer sits near 0.97), the variation lives in a
    thin shell and kNN distances are dominated by a handful of directions.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    A = np.asarray(X[idx], dtype=np.float64)
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    A = A / norms
    G = A @ A.T
    n = len(A)
    off = (G.sum() - np.trace(G)) / (n * (n - 1))
    return float(off)


def control_matrices(
    X: np.ndarray, *, seed: int = 42, pca_components: int = 50
) -> dict[str, np.ndarray]:
    """Reference matrices that bracket the effective-rank scale."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    out: dict[str, np.ndarray] = {}

    out["gaussian"] = rng.standard_normal((n, d)).astype(np.float32)

    shuffled = np.array(X, dtype=np.float32, copy=True)
    for j in range(d):
        rng.shuffle(shuffled[:, j])
    out["column_shuffled"] = shuffled

    k = min(pca_components, n, d)
    from sklearn.decomposition import PCA

    out[f"pca{k}"] = PCA(n_components=k, random_state=seed).fit_transform(X).astype(np.float32)
    return out


def run(cfg, table) -> "pd.DataFrame":  # noqa: F821 - lazy import
    """Score every configured feature column, plus controls, on one split."""
    import pandas as pd

    d = cfg.diagnostics
    rows = split_rows(cfg, table)

    recs = []
    for column in d.columns:
        X = table.features(column)[rows]
        stats = bootstrap_effective_rank(X, n_samples=d.n_samples, n_iters=d.n_iters, seed=d.seed)
        recs.append(
            {
                "substrate": cfg.data.substrate,
                "column": column,
                "variant": "observed",
                "mean_pairwise_cosine": mean_pairwise_cosine(X, seed=d.seed),
                **stats,
            }
        )
        print(
            f"    {column:16s} observed        dim={stats['dim']:5d} "
            f"eff_rank={stats['eff_rank']:7.2f} ({stats['pct_of_dim']:.1f}% of dim)"
        )

        if not d.include_controls:
            continue
        # Controls are built on a subsample: an (n x d) Gaussian at full cohort
        # size would be tens of GB for no extra information.
        sub = X[
            np.random.default_rng(d.seed).choice(
                len(X), size=min(d.n_samples, len(X)), replace=False
            )
        ]
        for variant, M in control_matrices(sub, seed=d.seed).items():
            s = bootstrap_effective_rank(
                M, n_samples=d.n_samples, n_iters=max(1, d.n_iters // 2), seed=d.seed
            )
            recs.append(
                {
                    "substrate": cfg.data.substrate,
                    "column": column,
                    "variant": variant,
                    "mean_pairwise_cosine": mean_pairwise_cosine(M, seed=d.seed),
                    **s,
                }
            )
            print(
                f"    {column:16s} {variant:15s} dim={s['dim']:5d} "
                f"eff_rank={s['eff_rank']:7.2f} ({s['pct_of_dim']:.1f}% of dim)"
            )

    return pd.DataFrame(recs)
