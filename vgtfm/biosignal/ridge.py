"""Per-gene ridge regression from an embedding to expression.

For each gene: how much of its variation across held-out spots can be read
linearly out of the embedding? Comparing the frozen embedding's per-gene R^2
against the refined one's shows which genes a refinement preserves and which it
erases.

Two implementation choices, both about scale rather than statistics:

* One SVD of the standardised training design serves all genes, since the ridge
  solution shares its left factors across targets. Numerically identical to
  looping ``sklearn.Ridge`` per gene, and orders of magnitude faster.
* Folds are pooled with streaming accumulators. A cohort-wide (spots x genes)
  float32 matrix is gigabytes and there would be three; the per-gene sums R^2
  needs are vectors of length ``n_genes``.

Any frozen-vs-refined drop is partly dimensionality: a 1152-wide embedding has more
directions to fit with than a 128-wide one. Scoring ``PCA_k(frozen)`` at the refined
width separates capacity from information.
"""

from __future__ import annotations

import numpy as np


def fit_ridge_svd(X_train: np.ndarray, Y_train: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge weights ``W = V diag(s / (s^2 + alpha)) U^T Y``.

    Assumes both matrices are already centred/standardised, so no intercept.
    """
    U, s, Vt = np.linalg.svd(np.asarray(X_train, dtype=np.float64), full_matrices=False)
    d = s / (s * s + alpha)
    return (Vt.T @ (d[:, None] * (U.T @ Y_train))).astype(np.float32)


def predict_fold(X_train, Y_train, X_test, alpha: float) -> np.ndarray:
    """Fit on the training rows, predict the held-out rows."""
    return X_test @ fit_ridge_svd(X_train, Y_train, alpha)


def alpha_grid(spec: tuple[float, float, int]) -> np.ndarray:
    """Log-spaced penalties from a ``(lo, hi, num)`` triple."""
    lo, hi, num = spec
    if not (lo > 0 and hi > lo and int(num) >= 2):
        raise ValueError(f"alpha grid must be 0 < lo < hi with num >= 2, got {spec}")
    return np.logspace(np.log10(float(lo)), np.log10(float(hi)), int(num))


def predict_fold_gcv(X_train, Y_train, X_test, alphas: np.ndarray):
    """Ridge with the penalty chosen per gene by GCV, then the held-out prediction.

    A fixed penalty makes this comparison a comparison of *widths*. The held-out
    R^2 of an unregularised least-squares fit sits near ``-p/n`` when the target
    carries no signal, so at ``alpha=1`` a 1152-wide design with ~4.3k training
    rows pays ~0.25 that a 128-wide one does not — larger than any plausible
    difference in what the two embeddings know. Fitting the penalty per fold, per
    representation and per gene removes that term, and what survives is
    information.

    Generalised cross-validation, not a held-out split: the one SVD the genes
    already share makes the whole grid a pair of matrix-vector products, so no
    refit is needed. With ``X = U S V^T`` and ``a_g = U^T y_g``,

        RSS_g(alpha) = ||y_g||^2 - ||a_g||^2 + sum_i (1 - d_i)^2 a_gi^2
        GCV_g(alpha) = RSS_g(alpha) / (1 - (1/n) sum_i d_i)^2,  d_i = s_i^2/(s_i^2+alpha)

    Selection reads training rows only. The evaluation fold is a different slide,
    so choosing here cannot borrow from what the R^2 is scored on.

    Returns ``(predictions, chosen_alpha_per_gene)``. A penalty that piles up on
    either end of the grid is the grid's fault, not the data's, so the caller is
    expected to check :func:`grid_saturation` on what comes back.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    U, s, Vt = np.linalg.svd(X_train, full_matrices=False)
    n = X_train.shape[0]
    s2 = s * s

    A = U.T @ np.asarray(Y_train, dtype=np.float64)  # (k, n_genes)
    A2 = A * A
    ss_y = (np.asarray(Y_train, dtype=np.float64) ** 2).sum(axis=0)
    ss_proj = A2.sum(axis=0)

    best = np.full(A.shape[1], np.inf)
    pick = np.zeros(A.shape[1], dtype=np.int64)
    for j, alpha in enumerate(alphas):
        d = s2 / (s2 + alpha)
        # `w @ A2` is the per-gene shrinkage residual; one gemv, not a refit.
        rss = ss_y - ss_proj + ((1.0 - d) ** 2) @ A2
        denom = 1.0 - d.sum() / n
        gcv = rss / (denom * denom) if denom > 0 else np.full_like(rss, np.inf)
        better = gcv < best
        best = np.where(better, gcv, best)
        pick = np.where(better, j, pick)

    # Predict through the k-dimensional right factors rather than forming the
    # p x n_genes weight matrix: same arithmetic, one fewer p-sized temporary, and
    # it is what lets each gene carry its own penalty.
    XV = np.asarray(X_test, dtype=np.float64) @ Vt.T  # (n_test, k)
    P = np.empty((XV.shape[0], A.shape[1]), dtype=np.float32)
    for j, alpha in enumerate(alphas):
        cols = np.flatnonzero(pick == j)
        if cols.size == 0:
            continue
        P[:, cols] = (XV @ ((s / (s2 + alpha))[:, None] * A[:, cols])).astype(np.float32)
    return P, alphas[pick]


class R2Accumulator:
    """Streaming per-gene R^2 over folds that share one gene vocabulary.

    Accumulates, per gene, the residual sum of squares and enough moments of the
    truth to recover the pooled total sum of squares:
    ``SS_tot = sum(y^2) - sum(y)^2 / n``. Memory is O(n_genes) regardless of how
    many spots are pooled.
    """

    def __init__(self, n_genes: int, names: tuple[str, ...]):
        self.n_genes = int(n_genes)
        self.names = tuple(names)
        self.n = 0
        self.sum_y = np.zeros(n_genes, dtype=np.float64)
        self.sum_y2 = np.zeros(n_genes, dtype=np.float64)
        self.ss_res = {name: np.zeros(n_genes, dtype=np.float64) for name in names}

    def update(self, Y_true: np.ndarray, predictions: dict[str, np.ndarray]) -> None:
        Y = np.asarray(Y_true, dtype=np.float64)
        self.n += Y.shape[0]
        self.sum_y += Y.sum(axis=0)
        self.sum_y2 += (Y**2).sum(axis=0)
        for name, P in predictions.items():
            if name not in self.ss_res:
                raise KeyError(f"unregistered prediction '{name}'")
            self.ss_res[name] += ((Y - np.asarray(P, dtype=np.float64)) ** 2).sum(axis=0)

    def r2(self) -> dict[str, np.ndarray]:
        if self.n == 0:
            return {name: np.full(self.n_genes, np.nan) for name in self.names}
        ss_tot = self.sum_y2 - (self.sum_y**2) / self.n
        out = {}
        for name, ss_res in self.ss_res.items():
            with np.errstate(invalid="ignore", divide="ignore"):
                r2 = 1.0 - ss_res / ss_tot
            r2[ss_tot <= 0] = np.nan
            out[name] = r2
        return out


def quartile_table(r2_frozen: np.ndarray, delta: np.ndarray) -> "pd.DataFrame":  # noqa: F821
    """Mean delta-R^2 within quartiles of the frozen R^2.

    A refinement that shrinks every prediction toward zero produces large positive
    deltas on genes the frozen embedding predicted badly (negative R^2 pulled up
    toward 0) and negative deltas on the genes it predicted well. That pattern is
    monotone across quartiles and invisible in the overall mean, which averages the
    two effects against each other.
    """
    import pandas as pd

    df = pd.DataFrame({"r2_frozen": r2_frozen, "delta_r2": delta}).dropna()
    if df.empty:
        return pd.DataFrame()
    try:
        df["quartile"] = pd.qcut(
            df["r2_frozen"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
        )
    except ValueError:
        return pd.DataFrame()
    return (
        df.groupby("quartile", observed=True)
        .agg(
            n_genes=("delta_r2", "size"),
            frozen_r2_min=("r2_frozen", "min"),
            frozen_r2_max=("r2_frozen", "max"),
            mean_delta_r2=("delta_r2", "mean"),
            pct_improved=("delta_r2", lambda s: 100.0 * float((s > 0).mean())),
        )
        .reset_index()
    )


def detrend_on_baseline(baseline: np.ndarray, stat: np.ndarray, bins: int = 50) -> np.ndarray:
    """Centre *stat* within equal-count bins of *baseline*.

    The same shrinkage :func:`quartile_table` tabulates is what makes the raw
    delta-R^2 unusable as a *ranking*: it is monotone in the frozen R^2 (Spearman
    around -0.6 on these cohorts), so ordering genes by it largely orders them by
    how well they were predicted to begin with. Curated gene sets are enriched for
    well-expressed, well-predicted genes, so every set lands at the bottom and the
    enrichment reports that gradient rather than a biological programme.

    Binning on the baseline's ranks rather than fitting a curve keeps this
    agnostic about the shape of the trend, which is neither linear nor monotone at
    the extremes. What survives is the part of the change not explained by the
    gene's starting predictivity — the part a pathway claim can rest on.
    """
    stat = np.asarray(stat, dtype=np.float64)
    strata = baseline_strata(baseline, bins)
    if strata is None:
        return stat.copy()
    out = stat.copy()
    for k in range(strata.max() + 1):
        idx = np.flatnonzero(strata == k)
        if len(idx):
            out[idx] -= stat[idx].mean()
    return out


def baseline_strata(baseline: np.ndarray, bins: int = 50) -> np.ndarray | None:
    """Equal-count bin id per gene, ranked on *baseline*; ``None`` if it cannot bin.

    The grouping :func:`detrend_on_baseline` centres within, exposed separately
    because the baseline-matched permutation null needs the same strata: shuffling
    gene labels *within* a bin of the frozen R^2 holds a set's predictivity profile
    fixed and randomises only which gene of comparable predictivity carries which
    change.
    """
    baseline = np.asarray(baseline, dtype=np.float64)
    if bins < 2 or len(baseline) < bins:
        return None
    out = np.empty(len(baseline), dtype=np.int64)
    for k, idx in enumerate(np.array_split(np.argsort(baseline, kind="stable"), bins)):
        out[idx] = k
    return out


def grid_saturation(chosen: np.ndarray, alphas: np.ndarray) -> tuple[float, float]:
    """Fraction of genes that selected the lowest and the highest penalty offered.

    An interior optimum means the grid bracketed it. Mass on an endpoint means it
    did not, and the fitted penalty is then a statement about the grid: at the low
    end the fit is unregularised least squares, at the high end it is the training
    mean. Both are legitimate answers for some genes — a gene with no signal really
    does want infinite shrinkage — so this reports rather than refuses.
    """
    if chosen.size == 0:
        return 0.0, 0.0
    return (float(np.mean(chosen <= alphas[0])), float(np.mean(chosen >= alphas[-1])))


def scope_name(tissues) -> str:
    """Directory name for one biosignal scope: ``skin``, ``skin-lung``, ``all``.

    The stage writes under ``biosignal/<scope>/`` so the organ runs coexist and an
    artefact says which cohort produced it. Shared with the figures stage, which
    must resolve the same path from the same config.
    """
    return "-".join(sorted(str(t) for t in tissues)) if tissues else "all"
