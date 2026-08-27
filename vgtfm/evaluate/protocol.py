"""Evaluation protocols — who trains the probe, and who is scored.

Four protocols in decreasing strictness, listed in :data:`PROTOCOLS`. All operate
on annotated spots only (:mod:`vgtfm.labels`) and return per-spot predictions, so
every downstream statistic is computed from one shared prediction vector. What each
one leaks is documented on the individual functions below; reporting them side by
side is what makes the cost of each leak visible.
"""

from __future__ import annotations

import numpy as np

from ..labels import labeled_mask
from .probes import knn_predict, majority_predict, standardize, stratified_subsample

PROTOCOLS = ("heldout_donor", "loso_donor", "pooled_loso", "pooled_loso_on_fold")


def _fit_predict(Z_train, y_train, Z_test, *, cfg, model_is_majority: bool):
    if model_is_majority:
        return majority_predict(y_train, len(Z_test))
    if cfg.eval.max_train_samples:
        Z_train, y_train = stratified_subsample(
            Z_train,
            y_train,
            labels=y_train,
            max_samples=cfg.eval.max_train_samples,
            random_seed=cfg.eval.subsample_seed,
        )
    if cfg.eval.standardize:
        Z_train, Z_test = standardize(Z_train, Z_test)
    return knn_predict(Z_train, y_train, Z_test, k=cfg.eval.knn_k)


def _annotated(Z, meta):
    """Restrict to annotated spots; returns (Z, meta, row_index_into_original)."""
    mask = labeled_mask(meta["annotation"].to_numpy())
    rows = np.where(mask)[0]
    return Z[rows], meta.iloc[rows].reset_index(drop=True), rows


def heldout_donor(Z, meta, fold, *, cfg, majority: bool = False) -> dict:
    """Fit on the fold's training slides, score its held-out slides. The default.

    Neither the representation nor the probe has seen the evaluated patient, so this
    is the strict reading of "zero-shot". Leaks nothing.

    Returns ``{}`` when either side has no annotated spots, which happens for
    cohorts whose slides carry no pathology annotation.
    """
    Za, m, _ = _annotated(Z, meta)
    sid = m["sample_id"].to_numpy().astype(str)
    train_mask = np.isin(sid, np.asarray(fold.train_sample_ids, dtype=str))
    test_mask = np.isin(sid, np.asarray(fold.eval_sample_ids, dtype=str))
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        return {}

    y = m["annotation"].to_numpy().astype(str)
    y_pred = _fit_predict(
        Za[train_mask], y[train_mask], Za[test_mask], cfg=cfg, model_is_majority=majority
    )
    return {
        "y_true": y[test_mask],
        "y_pred": np.asarray(y_pred).astype(str),
        "donor": m["donor"].to_numpy().astype(str)[test_mask],
        "sample_id": sid[test_mask],
        "tissue": m["tissue"].to_numpy().astype(str)[test_mask],
        "n_train": int(train_mask.sum()),
    }


def _leave_one_group_out(
    Z, meta, group_col: str, *, cfg, majority: bool, restrict_samples=None
) -> dict:
    """Within each tissue, hold out one value of *group_col* at a time.

    ``restrict_samples`` limits the probe to a subset of slides, which is what
    reproduces the legacy fold protocol: there the probe ran over the union of a
    fold's training and evaluation slides, so the fold determined *which* slides
    were in play without separating them into train and test.
    """
    Za, m, _ = _annotated(Z, meta)
    if restrict_samples is not None:
        keep = np.isin(
            m["sample_id"].to_numpy().astype(str), np.asarray(restrict_samples, dtype=str)
        )
        if keep.sum() == 0:
            return {}
        Za, m = Za[keep], m.loc[keep].reset_index(drop=True)
    y = m["annotation"].to_numpy().astype(str)
    tissue = m["tissue"].to_numpy().astype(str)
    group = m[group_col].to_numpy().astype(str)
    sid = m["sample_id"].to_numpy().astype(str)
    donor = m["donor"].to_numpy().astype(str)

    y_true_parts, y_pred_parts, keep_parts = [], [], []
    for t in np.unique(tissue):
        t_mask = tissue == t
        groups = np.unique(group[t_mask])
        if len(groups) < 2:
            print(
                f"    {group_col} LOO: tissue '{t}' has {len(groups)} group(s) "
                f"— skipped (need >= 2)"
            )
            continue
        for held in groups:
            test_mask = t_mask & (group == held)
            train_mask = t_mask & ~test_mask
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue
            y_pred = _fit_predict(
                Za[train_mask], y[train_mask], Za[test_mask], cfg=cfg, model_is_majority=majority
            )
            y_true_parts.append(y[test_mask])
            y_pred_parts.append(np.asarray(y_pred).astype(str))
            keep_parts.append(np.where(test_mask)[0])

    if not y_true_parts:
        return {}
    keep = np.concatenate(keep_parts)
    return {
        "y_true": np.concatenate(y_true_parts),
        "y_pred": np.concatenate(y_pred_parts),
        "donor": donor[keep],
        "sample_id": sid[keep],
        "tissue": tissue[keep],
        "n_train": int(len(Za)),
    }


def loso_donor(Z, meta, *, cfg, majority: bool = False) -> dict:
    """Leave one *donor* out within tissue, over every annotated slide. No folds.

    The right version of :func:`pooled_loso` for TuPro, whose 18 slides come from
    7 patients.
    """
    return _leave_one_group_out(Z, meta, "donor", cfg=cfg, majority=majority)


def pooled_loso(Z, meta, *, cfg, majority: bool = False, restrict_samples=None) -> dict:
    """Leave one *slide* out within tissue, over every annotated slide.

    Weaker than :func:`loso_donor`: the held-out slide's technical replicate — same
    patient, same tissue block — stays in the probe's training set.
    """
    return _leave_one_group_out(
        Z, meta, "sample_id", cfg=cfg, majority=majority, restrict_samples=restrict_samples
    )


def pooled_loso_on_fold(Z, meta, fold, *, cfg, majority: bool = False) -> dict:
    """The legacy protocol, reproduced exactly.

    The fold selects which slides participate; the probe then runs
    leave-one-slide-out within tissue across that whole set, with no train/eval
    separation. The held-out donor's own slides therefore contribute training rows
    on every other slide's turn, and the encoder has already seen the fold's
    training slides. Kept so earlier numbers stay checkable against the corrected
    protocol.
    """
    samples = (*fold.train_sample_ids, *fold.eval_sample_ids)
    return pooled_loso(Z, meta, cfg=cfg, majority=majority, restrict_samples=samples)


def is_fold_protocol(name: str) -> bool:
    """Whether the protocol consumes a :class:`~vgtfm.data.folds.FoldSpec`."""
    return name in ("heldout_donor", "pooled_loso_on_fold")


def run(name: str, Z, meta, *, cfg, fold=None, majority: bool = False) -> dict:
    if name == "heldout_donor":
        if fold is None:
            raise ValueError("heldout_donor requires a fold")
        return heldout_donor(Z, meta, fold, cfg=cfg, majority=majority)
    if name == "pooled_loso_on_fold":
        if fold is None:
            raise ValueError("pooled_loso_on_fold requires a fold")
        return pooled_loso_on_fold(Z, meta, fold, cfg=cfg, majority=majority)
    if name == "loso_donor":
        return loso_donor(Z, meta, cfg=cfg, majority=majority)
    if name == "pooled_loso":
        return pooled_loso(Z, meta, cfg=cfg, majority=majority)
    raise SystemExit(f"unknown eval protocol '{name}'. Known: {PROTOCOLS}")
