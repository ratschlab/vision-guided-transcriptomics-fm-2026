"""Batch-corrected gene inputs: guidance fitted on a corrected source.

A model named ``<method>_<inner>`` is the ``<inner>`` model fitted and embedded on
gene features corrected by ``<method>``::

    harmony_pca     PCA of Harmony-corrected gene features   (corrected, unguided)
    harmony_ae      the gene->morphology AE on the same      (corrected, guided)
    combat_pca, combat_ae, harmony_cdann, ...

``integrate`` corrects the ``pca`` *embedding* and probes it, which answers whether
batch correction replaces guidance. These arms correct the features every model
reads, which answers whether it enables guidance. ``eval`` therefore scores
``harmony_ae`` against ``harmony_pca`` (:func:`reference_for`), so the delta is
guidance within the corrected condition rather than guidance and correction at
once. ``harmony_pca`` is not ``integrate``'s Harmony row: that row is
``PCA(fit_split) -> Harmony(all spots)``, this one is ``Harmony(all spots) ->
PCA(fit_split)``, and only this one shares its input with ``harmony_ae``.

The correction is transductive, and that favours this arm. Harmony and ComBat align
batches by looking at every spot at once and have no out-of-sample transform, so the
correction sees the held-out donors' slides; there is no way to fit either on
``train.fit_split`` alone, which is also why ``integrate`` corrects across the whole
cohort. If guidance does not help here, it does not help under a setup more
favourable than any deployable one.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
from sklearn.decomposition import PCA

from ..degraded import refuse
from ..diagnostics import integration
from ..provenance import array_fingerprint

#: Corrections that can be applied to the gene features. Both are the ones
#: ``integrate`` uses, so "Harmony" cannot come to mean two things in two tables.
CORRECTIONS = ("harmony", "combat")

#: Corrections whose output depends on the run seed, and which are therefore cached
#: per seed. ComBat is a closed-form location/scale adjustment and is cached once
#: for every seed; Harmony converges to a seed-dependent solution, so the spread
#: across seeds covers the correction's variability as well as the model's.
STOCHASTIC = ("harmony",)

#: Inner models that must not be wrapped, and why. ``hvg_pca`` reads raw counts
#: through ``dataset_id``/``spot_id`` and never touches ``inputs.gene``, so its
#: corrected row would silently duplicate the uncorrected one; the oracles read
#: H&E, which this does not correct.
REFUSED_INNER = {
    "hvg_pca": "it starts from raw counts and never reads the gene features",
    "pca_oracle": "it reads H&E, which this corrects nothing of",
    "pca_oracle_matched": "it reads H&E, which this corrects nothing of",
}


def parse(name: str) -> tuple[str, str] | None:
    """``"harmony_ae"`` -> ``("harmony", "ae")``; ``None`` for an ordinary model.

    Split on the first underscore only, and only when the head is a correction, so
    ``hvg_pca``, ``pca_oracle`` and ``pca_oracle_matched`` parse as themselves.
    """
    head, _, rest = name.partition("_")
    if not rest or head not in CORRECTIONS:
        return None
    return head, rest


def validate(names) -> None:
    """Refuse a corrected name whose inner model cannot read a corrected input.

    Called before any fitting: an unusable model list is a config error and should
    cost a second, not the hour the correction itself takes.
    """
    for name in names:
        spec = parse(name)
        if spec is None:
            continue
        method, inner = spec
        if inner in REFUSED_INNER:
            refuse(
                f"the model '{name}'",
                f"'{inner}' cannot be fitted on corrected gene features: {REFUSED_INNER[inner]}",
                hint=f"drop '{name}' from models.names; score '{inner}' uncorrected "
                f"instead, and read the correction's effect from '{method}_pca'",
            )
        if parse(inner) is not None:
            refuse(
                f"the model '{name}'",
                "corrections cannot be stacked",
                hint=f"use one of {', '.join(f'{m}_<model>' for m in CORRECTIONS)}",
            )


def reference_for(name: str) -> str | None:
    """The model a corrected arm's paired delta should be measured against.

    ``None`` for a model with no corrected reference of its own — including a
    corrected baseline, which is its own reference — which the caller reads as "use
    the run's default reference".
    """
    spec = parse(name)
    if spec is None:
        return None
    ref = f"{spec[0]}_pca"
    return None if name == ref else ref


def _cache_path(cfg, method: str, seed: int):
    stem = f"{method}__seed-{seed}" if method in STOCHASTIC else method
    return cfg.sub("train", "corrected") / f"{stem}.npy"


def _reduce(cfg, X: np.ndarray, fit_rows: np.ndarray, seed: int) -> np.ndarray:
    """Project to ``models.correction_input_dim`` PCs before correcting.

    Fitted on *fit_rows* rather than on every spot, for the reason
    :func:`vgtfm.diagnostics.integration._reference_embedding` gives: a projection
    fitted on the whole cohort has seen the held-out donors.
    """
    d = int(cfg.models.correction_input_dim)
    n = min(d, len(fit_rows), X.shape[1])
    if n < d:
        print(f"  [correct] capping correction_input_dim {d} -> {n} (data {X.shape})")
    pca = PCA(n_components=n, random_state=seed)
    pca.fit(X[fit_rows])
    evr = float(np.sum(pca.explained_variance_ratio_))
    print(
        f"  [correct] reduced {X.shape[1]} -> {n} dims before correcting "
        f"({evr:.4f} of variance, fitted on {len(fit_rows):,} spots)"
    )
    return pca.transform(X).astype(np.float32)


def _apply(method: str, X: np.ndarray, batch: np.ndarray, seed: int) -> np.ndarray:
    if method == "harmony":
        return integration.correct_harmony(X, batch, seed=seed)
    if method == "combat":
        return integration.correct_combat(X, batch)
    refuse(
        f"a '{method}' correction",
        "no such method",
        hint=f"models.names accepts {', '.join(f'{m}_<model>' for m in CORRECTIONS)}",
    )


def corrected_gene(cfg, table, method: str, seed: int, fit_rows: np.ndarray) -> np.ndarray:
    """The gene feature matrix corrected by *method*, computed once and cached.

    The cache is keyed by the method (and the seed, where the method is stochastic)
    and guarded by a fingerprint of the gene features it was built from, so a run
    that changes substrate or data filters recomputes rather than silently reusing
    another cohort's correction.
    """
    X = np.asarray(table.gene, dtype=np.float32)
    if int(cfg.models.correction_input_dim) > 0:
        X = _reduce(cfg, X, fit_rows, seed)
    want = array_fingerprint(X)

    path = _cache_path(cfg, method, seed)
    side = path.with_suffix(".json")
    if path.exists() and side.exists():
        if json.loads(side.read_text()).get("source_fingerprint") == want:
            Z = np.load(path)
            print(f"  [correct] {method}: reusing {path} {Z.shape}")
            return Z
        print(f"  [correct] {method}: {path} was built from other features — recomputing")

    # The batch variable is the slide, the same one `integrate` and the scIB panel
    # use: the claim under test is that the embeddings are structured by slide
    # identity, so the correction has to be given the variable that claim is about.
    batch = np.asarray(table.sample_id).astype(str)
    print(
        f"  [correct] {method} on {X.shape[0]:,}x{X.shape[1]} gene features, "
        f"{len(set(batch))} slides as batches (seed {seed})"
    )
    Z = np.ascontiguousarray(_apply(method, X, batch, seed), dtype=np.float32)
    if Z.shape != X.shape:
        refuse(
            f"the {method}-corrected gene features",
            f"the correction returned {Z.shape} for a {X.shape} input",
            hint="the models downstream address spots by row and assume the width is unchanged",
        )
    np.save(path, Z)
    side.write_text(
        json.dumps(
            {
                "method": method,
                "seed": seed if method in STOCHASTIC else None,
                "batch_key": "sample_id",
                "n_batches": len(set(batch)),
                "correction_input_dim": int(cfg.models.correction_input_dim),
                "source_fingerprint": want,
                "corrected_fingerprint": array_fingerprint(Z),
            },
            indent=2,
        )
    )
    print(f"  [correct] {method} -> {path}")
    return Z


def corrected_inputs(cfg, table, inputs, method: str, seed: int, fit_rows: np.ndarray):
    """*inputs* with the gene block replaced by its corrected form.

    Every other block is passed through untouched — in particular ``patch``, the
    supervision target, which is never corrected: the question is whether a
    corrected source can be guided towards the same morphology, not whether two
    corrected views agree.
    """
    return replace(inputs, gene=corrected_gene(cfg, table, method, seed, fit_rows))


def resolve(cfg, table, inputs, name: str, seed: int, fit_rows: np.ndarray):
    """``(inputs the inner model reads, inner model name, correction applied)``.

    The one call ``train`` needs: an ordinary name passes straight through, so
    ``build`` only ever sees registry names and the inner model of a corrected arm
    is bit-for-bit the one the uncorrected arm reports.
    """
    spec = parse(name)
    if spec is None:
        return inputs, name, "none"
    method, inner = spec
    return corrected_inputs(cfg, table, inputs, method, seed, fit_rows), inner, method
