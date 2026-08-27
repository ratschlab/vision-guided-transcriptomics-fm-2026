"""Cross-modal canonical correlation, against controls for slide identity.

A raw canonical correlation between the gene and morphology embeddings is close to
uninterpretable, because two spots from the same slide are coupled through stain,
scanner and tissue composition before any biology is involved. Three references
separate that from spot-level signal:

* **within-slide permutation** — permute morphology rows inside each slide. Slide
  coupling survives; spot correspondence does not. This is the null that matters.
* **global permutation** — permute across all spots, destroying slide coupling
  too. A floor that everything clears, and so uninformative on its own.
* **leave-one-slide-out** — fit on all but one slide, apply without refitting.
  Held-out slides are split by whether a same-patient slide stayed in the training
  set, which makes patient leakage measurable.
"""

from __future__ import annotations

import numpy as np

from ..data.tables import load_registry
from . import split_rows


#: Smallest fold a held-out CCA is fitted and scored on. Below these the loadings
#: are fitted on too few spots for the held-out correlation to mean anything, and a
#: slide that cannot be scored is skipped rather than scored badly.
MIN_TRAIN_SPOTS, MIN_TEST_SPOTS = 50, 20


def _inv_sqrt(S: np.ndarray, ridge: float) -> np.ndarray:
    """Symmetric inverse square root with a ridge relative to the top eigenvalue."""
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, 0.0)
    reg = ridge * (w.max() if w.size else 1.0)
    inv = 1.0 / np.sqrt(w + reg)
    return (V * inv) @ V.T


def cca(X: np.ndarray, Y: np.ndarray, ridge: float = 1e-4, n_components: int | None = None) -> dict:
    """Canonical correlations and loadings for two centred views.

    Covariance-based with a whitened SVD, which is numerically stabler than solving
    the generalised eigenproblem when both views are wide relative to the number of
    spots.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    n = len(X)

    Sxx = X.T @ X / n
    Syy = Y.T @ Y / n
    Sxy = X.T @ Y / n

    Wx = _inv_sqrt(Sxx, ridge)
    Wy = _inv_sqrt(Syy, ridge)
    M = Wx @ Sxy @ Wy
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    k = len(s) if n_components is None else min(n_components, len(s))
    return {
        "rho": np.clip(s[:k], 0.0, 1.0),
        "Ax": Wx @ U[:, :k],  # X -> canonical variates
        "Ay": Wy @ Vt[:k].T,  # Y -> canonical variates
    }


def _corr_columns(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson correlation of matching columns of two matrices."""
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    na = np.linalg.norm(A, axis=0)
    nb = np.linalg.norm(B, axis=0)
    denom = na * nb
    out = np.zeros(A.shape[1])
    ok = denom > 0
    out[ok] = np.abs((A * B).sum(axis=0)[ok] / denom[ok])
    return out


def _reduce(X: np.ndarray, dim: int, seed: int) -> np.ndarray:
    """PCA pre-reduction, capped so at least five spots back every dimension.

    Without the cap, CCA on a wide matrix with few spots finds spurious directions
    and reports correlations near 1.0 that mean nothing.
    """
    from sklearn.decomposition import PCA

    k = min(dim, X.shape[1], max(2, X.shape[0] // 5))
    return PCA(n_components=k, random_state=seed).fit_transform(X).astype(np.float64)


def permute_within(groups: np.ndarray, rng) -> np.ndarray:
    """Row permutation that keeps every row inside its own group."""
    perm = np.arange(len(groups))
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        perm[idx] = shuffled
    return perm


def spectrum_with_nulls(
    Xr: np.ndarray,
    Yr: np.ndarray,
    slides: np.ndarray,
    *,
    ridge: float = 1e-4,
    n_permutations: int = 50,
    n_report: int = 10,
    seed: int = 42,
) -> dict:
    """Observed canonical spectrum plus global and within-slide permutation nulls."""
    rng = np.random.default_rng(seed)
    obs = cca(Xr, Yr, ridge=ridge)["rho"]
    k = min(n_report, len(obs))

    glob = np.empty((n_permutations, k))
    within = np.empty((n_permutations, k))
    for i in range(n_permutations):
        glob[i] = cca(Xr, Yr[rng.permutation(len(Yr))], ridge=ridge)["rho"][:k]
        within[i] = cca(Xr, Yr[permute_within(slides, rng)], ridge=ridge)["rho"][:k]

    obs_k = obs[:k]
    obs_var = float((obs_k**2).sum())
    within_mean = within.mean(axis=0)
    within_q95 = np.percentile(within, 95, axis=0)
    return {
        "rho_observed": obs_k.tolist(),
        "rho_null_global_mean": glob.mean(axis=0).tolist(),
        "rho_null_global_q95": np.percentile(glob, 95, axis=0).tolist(),
        "rho_null_within_mean": within_mean.tolist(),
        "rho_null_within_q95": within_q95.tolist(),
        "rho1_observed": float(obs_k[0]),
        "rho1_null_within_q95": float(within_q95[0]),
        # Share of the top-k shared variance that the slide-mean null cannot
        # reproduce. Small values mean the coupling is slide identity.
        "spot_level_fraction_mean_null": float(
            max(0.0, obs_var - float((within_mean**2).sum())) / obs_var
        )
        if obs_var > 0
        else float("nan"),
        "spot_level_fraction_q95_null": float(
            max(0.0, obs_var - float((within_q95**2).sum())) / obs_var
        )
        if obs_var > 0
        else float("nan"),
        "n_components_reported": int(k),
        "n_permutations": int(n_permutations),
    }


def leave_one_slide_out(
    Xr: np.ndarray,
    Yr: np.ndarray,
    slides: np.ndarray,
    donors: np.ndarray,
    *,
    ridge: float = 1e-4,
    n_report: int = 10,
) -> dict:
    """Fit CCA on all but one slide; report the held-out correlation per slide.

    Each held-out slide is tagged ``paired`` when another slide from the same donor
    stayed in the training set, and ``unpaired`` otherwise. The gap between those
    two groups is the size of the patient-leakage effect.
    """
    uniq = np.unique(slides)
    rows = []
    for held in uniq:
        test = slides == held
        train = ~test
        if train.sum() < MIN_TRAIN_SPOTS or test.sum() < MIN_TEST_SPOTS:
            continue
        fit = cca(Xr[train], Yr[train], ridge=ridge, n_components=n_report)
        # Apply the training loadings to the held-out slide (no refit).
        xt = Xr[test] - Xr[train].mean(axis=0, keepdims=True)
        yt = Yr[test] - Yr[train].mean(axis=0, keepdims=True)
        rho_test = _corr_columns(xt @ fit["Ax"], yt @ fit["Ay"])
        donor = donors[test][0]
        paired = bool(np.any((donors == donor) & train))
        rows.append(
            {
                "slide": str(held),
                "donor": str(donor),
                "paired": paired,
                "rho1_train": float(fit["rho"][0]),
                "rho1_test": float(rho_test[0]),
                "n_test": int(test.sum()),
            }
        )

    if not rows:
        return {"per_slide": [], "rho1_test_mean": float("nan")}

    test_rho = np.array([r["rho1_test"] for r in rows])
    paired = np.array([r["paired"] for r in rows])
    out = {
        "per_slide": rows,
        "n_slides": len(rows),
        "rho1_train_mean": float(np.mean([r["rho1_train"] for r in rows])),
        "rho1_test_mean": float(test_rho.mean()),
        "rho1_test_paired_mean": float(test_rho[paired].mean()) if paired.any() else float("nan"),
        "rho1_test_unpaired_mean": float(test_rho[~paired].mean())
        if (~paired).any()
        else float("nan"),
        "n_paired": int(paired.sum()),
        "n_unpaired": int((~paired).sum()),
    }
    out["patient_leakage_gap"] = out["rho1_test_paired_mean"] - out["rho1_test_unpaired_mean"]
    # A NaN gap has two very different causes, and the reader cannot tell them
    # apart from the number: either no held-out slide had a same-patient sibling in
    # training, or every one did. Say which, rather than leaving a bare NaN to be
    # read as a failed computation.
    if not paired.any():
        out["patient_leakage_gap_note"] = (
            f"not measurable: none of the {len(rows)} held-out slides has another "
            f"slide from the same patient in this split"
        )
    elif paired.all():
        out["patient_leakage_gap_note"] = (
            f"not measurable: all {len(rows)} held-out slides have a same-patient "
            f"sibling in this split, so there is no unpaired group to compare to"
        )
    return out


def run(cfg, table) -> dict:
    """Full ceiling diagnostic on a subsample of the configured split."""
    d = cfg.diagnostics
    rows = split_rows(cfg, table)
    rng = np.random.default_rng(d.seed)
    if len(rows) > d.cca_max_spots:
        rows = np.sort(rng.choice(rows, size=d.cca_max_spots, replace=False))

    slides = table.sample_id[rows].astype(str)
    # The registry's patient key, not SpotTable.donor: the whole point of the
    # paired/unpaired split is to find slides that share a patient, and the fold key
    # cannot see the ones that do — De Zuani's `P10-*` slides are one patient and
    # live in the train split this diagnostic runs on.
    donors = load_registry(cfg.paths.datasets_json).patient_keys(
        slides, table.col("dataset_id")[rows]
    )

    print(
        f"    CCA on {len(rows):,} spots from {len(np.unique(slides))} slides "
        f"(PCA-{d.cca_pca_dim} per view, {d.cca_n_permutations} permutations)"
    )
    Xr = _reduce(table.gene[rows], d.cca_pca_dim, d.seed)
    Yr = _reduce(table.patch[rows], d.cca_pca_dim, d.seed)

    spec = spectrum_with_nulls(
        Xr, Yr, slides, ridge=d.cca_ridge, n_permutations=d.cca_n_permutations, seed=d.seed
    )
    loso = leave_one_slide_out(Xr, Yr, slides, donors, ridge=d.cca_ridge)

    print(
        f"    rho1 in-sample {spec['rho1_observed']:.3f} vs within-slide null q95 "
        f"{spec['rho1_null_within_q95']:.3f}"
    )
    print(
        f"    top-{spec['n_components_reported']} shared variance not explained by "
        f"slide identity: {spec['spot_level_fraction_mean_null']:.3f}"
    )
    if loso.get("n_slides"):
        print(
            f"    held-out rho1 {loso['rho1_test_mean']:.3f} "
            f"(paired {loso['rho1_test_paired_mean']:.3f} on {loso['n_paired']} "
            f"slide(s) / unpaired {loso['rho1_test_unpaired_mean']:.3f} on "
            f"{loso['n_unpaired']})"
        )
        if loso.get("patient_leakage_gap_note"):
            print(f"    patient leakage gap {loso['patient_leakage_gap_note']}")

    return {
        "substrate": cfg.data.substrate,
        "spectrum": spec,
        "loso": loso,
        "n_spots": int(len(rows)),
        "n_slides": int(len(np.unique(slides))),
    }
