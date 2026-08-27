"""Gene <-> morphology correspondence ablation.

Each transform maps the morphology features of shape ``(n_spots, d)`` to a new
array of the same shape, leaving the gene side untouched. Only the autoencoder's
*target* changes, so the marginal distribution of the target is preserved and the
only thing destroyed is which patch belongs to which spot.

``none``           identity — matched pairs, the condition of interest.
``within-sample``  permute rows inside each slide. Slide-level structure (stain,
                   scanner, tissue composition) survives; within-slide spot
                   correspondence does not. The milder control.
``global``         permute rows across all spots. Every gene is paired with an
                   unrelated patch from an unrelated slide. The strongest break
                   that still uses real morphology features.
``gaussian``       replace patches with i.i.d. noise matched to the per-dimension
                   mean and standard deviation of the real ones. No structure of
                   any kind remains, so this is the floor on what the encoder can
                   extract from genes alone.

If matched and shuffled score the same, the encoder was only reducing the gene side.
"""

from __future__ import annotations

import numpy as np

PATCH_TRANSFORMS = ("none", "within-sample", "global", "gaussian")


def apply_patch_transform(
    strategy: str, patch: np.ndarray, sample_id: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(transformed_patch, row_permutation)``.

    The permutation is the identity for ``none`` and ``gaussian``, and is returned
    in both cases so callers can record a fixed-point fraction and check that the
    intended condition ran.
    """
    n = len(patch)
    identity = np.arange(n, dtype=np.int64)

    if strategy == "none":
        return patch, identity
    if strategy == "global":
        perm = rng.permutation(n).astype(np.int64)
        return patch[perm], perm
    if strategy == "within-sample":
        perm = identity.copy()
        for sid in np.unique(sample_id):
            rows = np.where(sample_id == sid)[0]
            shuffled = rows.copy()
            rng.shuffle(shuffled)
            perm[rows] = shuffled
        return patch[perm], perm
    if strategy == "gaussian":
        mean = patch.mean(axis=0, keepdims=True).astype(np.float32)
        std = patch.std(axis=0, keepdims=True).astype(np.float32) + 1e-8
        noise = rng.standard_normal(patch.shape).astype(np.float32)
        return noise * std + mean, identity

    raise ValueError(
        f"Unknown patch transform {strategy!r}. Choices: {', '.join(PATCH_TRANSFORMS)}"
    )
