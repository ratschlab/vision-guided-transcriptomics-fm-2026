"""Turning prediction vectors into result rows.

Every stage that scores an embedding — ``eval``, ``ablate`` and ``integrate`` —
derives all of its statistics from one prediction vector per (condition, protocol,
level). The rules for how that vector becomes rows live here rather than in any of
them, because all three must agree: a macro average over a different denominator, or
a paired delta against a differently-ordered reference, produces numbers that cannot
be compared.
"""

from __future__ import annotations

import numpy as np

from . import bootstrap as boot
from .probes import classification_metrics


#: Separator between an organ and a class name in the ``global`` scope's labels.
#: No cohort's annotation vocabulary contains a pipe, so the split is unambiguous.
TISSUE_SEP = " | "


def scopes(pred: dict):
    """Yield ``(scope_name, row_mask)`` for the global pool and each tissue."""
    yield "global", np.ones(len(pred["y_true"]), dtype=bool)
    tissue = pred["tissue"]
    for t in sorted(set(tissue.tolist())):
        yield t, tissue == t


def _qualify(labels, tissue) -> np.ndarray:
    """``Tumor`` in skin and ``Tumor`` in lung as two separately averaged cells."""
    labels = np.asarray(labels).astype(str)
    return np.array([f"{t}{TISSUE_SEP}{c}" for t, c in zip(tissue, labels)], dtype=object)


def scope_view(pred: dict, scope: str, mask, classes, vocab: dict):
    """What the scoring functions need for one scope: ``(qualify, vocab, scored)``.

    ``qualify`` maps any label array over the masked rows into the scope's own label
    space; ``vocab`` is the vocabulary its confusion matrix spans; ``scored`` is the
    subset the macro average runs over.

    A tissue scope is the identity. The **global** scope qualifies every label with
    its organ, so its macro average runs over ``(tissue, class)`` cells rather than
    over bare class names. That is the only denominator that stays well posed once
    more than one cohort is evaluated: the cohorts were annotated independently and
    the probe is fitted within tissue, so a lung ``TLS`` and a skin ``Stroma`` are
    different questions, while ``Tumor`` — the one name both vocabularies use — is
    two different questions wearing one label. Averaging them into a single cell
    weights it by whichever organ contributed more spots and silently scores each
    fold against classes that cannot occur in it.

    With a single tissue in play this is exactly the old bare-class macro: the same
    per-class F1 values under longer names.
    """
    if scope != "global":
        qualify = lambda y: np.asarray(y).astype(str)[mask]  # noqa: E731
        full = list(classes)
    else:
        tissue = np.asarray(pred["tissue"]).astype(str)[mask]
        qualify = lambda y: _qualify(np.asarray(y)[mask], tissue)  # noqa: E731
        full = [f"{t}{TISSUE_SEP}{c}" for t in sorted(set(tissue.tolist())) for c in classes]
    # A fold spans one tissue, so the pooled vocabulary names cells it cannot
    # contain. Intersecting here is what keeps `n_classes_scored` honest and stops a
    # skin fold being averaged against lung cells it has no way to predict.
    known = set(full)
    scored = [c for c in _scored_classes(vocab, scope, qualify(pred["y_true"])) if c in known]
    return qualify, full, (scored or sorted(set(qualify(pred["y_true"]).tolist())))


def score_vocabulary(pred: dict) -> dict[str, list[str]]:
    """Classes the macro average runs over, per scope.

    Derived from the *pooled* ground truth of a (protocol, level), which depends
    only on which spots the protocol evaluates and never on the model. Every model
    and seed is therefore averaged over the same denominator, while a fold is not
    penalised for the classes of a cohort it does not contain.

    The ``global`` entry is in :func:`scope_view`'s ``(tissue, class)`` label space.
    """
    out = {}
    for scope, mask in scopes(pred):
        if not mask.sum():
            continue
        y = np.asarray(pred["y_true"]).astype(str)[mask]
        if scope == "global":
            y = _qualify(y, np.asarray(pred["tissue"]).astype(str)[mask])
        out[scope] = sorted(set(y.tolist()))
    return out


def _scored_classes(vocab: dict, scope: str, y_true: np.ndarray) -> list[str]:
    return vocab.get(scope) or sorted(set(np.asarray(y_true).tolist()))


def metric_rows(pred: dict, classes, base: dict, vocab: dict[str, list[str]]):
    """Expand one prediction vector into ``(macro_rows, per_class_rows)``."""
    macro_rows, class_rows = [], []
    for scope, mask in scopes(pred):
        if mask.sum() == 0:
            continue
        qualify, full, score = scope_view(pred, scope, mask, classes, vocab)
        y_true, y_pred = qualify(pred["y_true"]), qualify(pred["y_pred"])
        # The confusion matrix always spans the full cohort vocabulary, so a
        # prediction outside `score` still counts as an error for its true class.
        m = classification_metrics(y_true, y_pred, full, score_classes=score)
        row = {
            **base,
            "scope": scope,
            "n_classes_scored": len(score),
            "n_donors": int(len(set(pred["donor"][mask].tolist()))),
            "n_slides": int(len(set(pred["sample_id"][mask].tolist()))),
        }
        macro_rows.append({**row, **{k: v for k, v in m.items() if k != "per_class"}})
        keep = set(score)
        for cls, cm in m["per_class"].items():
            if cls in keep:
                class_rows.append({**row, "class": cls, **cm})
    return macro_rows, class_rows


def pool(preds: list[dict]) -> dict:
    """Concatenate per-fold predictions into one vector per level."""
    preds = [p for p in preds if p]
    if not preds:
        return {}
    keys = ("y_true", "y_pred", "donor", "sample_id", "tissue")
    out = {k: np.concatenate([p[k] for p in preds]) for k in keys}
    out["n_train"] = int(np.mean([p["n_train"] for p in preds]))
    return out


def bootstrap_rows(pred: dict, classes, base: dict, cfg, vocab, *, return_draws: bool = False):
    """Donor-level CIs for the macro and per-class F1 of one prediction vector.

    With *return_draws*, also returns ``{(scope, class): replicate draws}`` so the
    caller can pool them across seeds; see :func:`pooled_columns`.
    """
    rows: list[dict] = []
    draws: dict[tuple[str, str], object] = {}
    for scope, mask in scopes(pred):
        if mask.sum() == 0:
            continue
        qualify, full, score = scope_view(pred, scope, mask, classes, vocab)
        res = boot.donor_bootstrap(
            qualify(pred["y_true"]),
            qualify(pred["y_pred"]),
            pred["donor"][mask],
            classes=full,
            score_classes=score,
            n_boot=cfg.eval.bootstrap_n,
            seed=cfg.eval.bootstrap_seed,
            ci=cfg.eval.bootstrap_ci,
            return_draws=return_draws,
        )
        ci = res["ci"].get("f1_score", (np.nan, np.nan))
        for cls, d in res.get("draws", {}).items():
            draws[(scope, "macro" if cls == "macro" else cls)] = d
        rows.append(
            {
                **base,
                "scope": scope,
                "class": "macro",
                "value": res["point"]["f1_score"],
                "ci_lo": ci[0],
                "ci_hi": ci[1],
                "n_donors": res["n_donors"],
                "n_boot": res["n_boot"],
            }
        )
        keep = set(score)
        for cls, cci in res["per_class_ci"].items():
            if cls not in keep:
                continue
            rows.append(
                {
                    **base,
                    "scope": scope,
                    "class": cls,
                    "value": res["point"]["per_class_f1"][cls],
                    "ci_lo": cci[0],
                    "ci_hi": cci[1],
                    "n_donors": res["n_donors"],
                    "n_boot": res["n_boot"],
                }
            )
    return (rows, draws) if return_draws else rows


def paired_delta_rows(
    pred: dict, y_pred_a, y_pred_b, base: dict, classes, cfg, vocab, *, return_draws: bool = False
):
    """Rows for ``metric(a) - metric(b)`` on a shared donor resample, per scope.

    *pred* supplies the ground truth, donors and tissues; ``y_pred_a`` and
    ``y_pred_b`` are two predictions of those same rows. Order matters and is the
    caller's to decide: ``eval`` passes (model, PCA) so a positive delta means the
    model beat the baseline, ``ablate`` passes (matched, shuffled) so a positive
    delta means the correspondence helped.
    """
    rows: list[dict] = []
    draws: dict[tuple[str, str], object] = {}
    for scope, mask in scopes(pred):
        if mask.sum() == 0:
            continue
        qualify, full, score = scope_view(pred, scope, mask, classes, vocab)
        res = boot.paired_delta_ci(
            qualify(pred["y_true"]),
            qualify(y_pred_a),
            qualify(y_pred_b),
            pred["donor"][mask],
            classes=full,
            score_classes=score,
            n_boot=cfg.eval.bootstrap_n,
            seed=cfg.eval.bootstrap_seed,
            ci=cfg.eval.bootstrap_ci,
            return_draws=return_draws,
        )
        ci = res["ci"].get("delta_f1_score", (np.nan, np.nan))
        for cls, d in res.get("draws", {}).items():
            draws[(scope, cls)] = d
        # n_donors travels with the interval it was built from. Without it a
        # two-donor bracket (cross_region, USZ kidney) is indistinguishable on the
        # page from a seven-donor one, though the first has three possible
        # resamples and the second has 1,716.
        rows.append(
            {
                **base,
                "scope": scope,
                "class": "macro",
                "delta": res["delta_f1_score"],
                "ci_lo": ci[0],
                "ci_hi": ci[1],
                "p_two_sided": res.get("p_two_sided", np.nan),
                "n_donors": res["n_donors"],
            }
        )
        keep = set(score)
        for cls, d in res["per_class_delta"].items():
            if cls not in keep:
                continue
            cci = res["per_class_ci"].get(cls, (np.nan, np.nan))
            rows.append(
                {
                    **base,
                    "scope": scope,
                    "class": cls,
                    "delta": d,
                    "ci_lo": cci[0],
                    "ci_hi": cci[1],
                    "p_two_sided": res["per_class_p"].get(cls, np.nan),
                    "n_donors": res["n_donors"],
                }
            )
    return (rows, draws) if return_draws else rows


#: Row fields that identify a group of runs differing only in the model seed.
POOL_KEY = ("protocol", "level", "scope", "class")


def pooled_columns(draws_by_group: dict, cfg, *, with_p: bool) -> dict:
    """Pooled interval per group, keyed as the caller keyed *draws_by_group*.

    Each value is a column dict ready to merge onto every per-seed row of that
    group. The columns are deliberately constant within a group, so a downstream
    ``groupby(...).mean()`` over seeds returns them unchanged and the table code
    needs no special case.

    Pooling the replicate draws rather than averaging the per-seed bounds is what
    makes the published interval cover training variability as well as donor
    sampling; :func:`vgtfm.evaluate.bootstrap.pool_replicates` documents the
    trade-off.
    """
    out = {}
    for key, draws in draws_by_group.items():
        lo, hi, p, n = boot.pool_replicates(draws, ci=cfg.eval.bootstrap_ci)
        cols = {"ci_lo_pooled": lo, "ci_hi_pooled": hi, "n_seeds_pooled": n}
        if with_p:
            cols["p_two_sided_pooled"] = p
        out[key] = cols
    return out


def attach_pooled(
    rows: list[dict], draws_by_group: dict, cfg, *, group_fields, with_p: bool
) -> None:
    """Merge :func:`pooled_columns` onto *rows* in place, matching on *group_fields*."""
    if not draws_by_group:
        return
    cols = pooled_columns(draws_by_group, cfg, with_p=with_p)
    for row in rows:
        got = cols.get(tuple(row[f] for f in group_fields))
        if got:
            row.update(got)


#: Scope name for the organ-balanced headline statistic. Not a row mask like the
#: scopes in :func:`scopes` — it is an average *over* them, so it exists only on a
#: pooled prediction and never on a single fold, which spans one organ.
TISSUE_BALANCED = "tissue_balanced"


def tissue_balanced_rows(
    pred: dict, classes, base: dict, cfg, vocab: dict[str, list[str]], *, with_boot: bool = True
):
    """``(macro_row, boot_row, draws)`` for the organ-balanced macro-F1.

    Returns ``(None, None, None)`` when fewer than two organs are in play, where the
    statistic is just that organ's macro and the global scope already reports it.
    """
    organs = [s for s, _ in scopes(pred) if s != "global"]
    if len(organs) < 2:
        return None, None, None
    by_tissue = {t: vocab[t] for t in organs if vocab.get(t)}
    if len(by_tissue) < 2:
        return None, None, None

    res = boot.tissue_balanced_bootstrap(
        pred["y_true"],
        pred["y_pred"],
        pred["donor"],
        pred["tissue"],
        classes=classes,
        score_classes_by_tissue=by_tissue,
        n_boot=cfg.eval.bootstrap_n if with_boot else 0,
        seed=cfg.folds.seed,
        ci=cfg.eval.bootstrap_ci,
        return_draws=True,
    )
    if not res:
        return None, None, None

    row = {
        **base,
        "scope": TISSUE_BALANCED,
        "n_classes_scored": sum(len(v) for v in by_tissue.values()),
        "n_organs": res["n_organs"],
        "n_donors": res["n_donors"],
        "n_slides": int(len(set(pred["sample_id"].tolist()))),
        "f1_score": res["point"],
        "n_test": int(len(pred["y_true"])),
    }
    if "ci" not in res:
        return row, None, None
    lo, hi = res["ci"]
    # "value", not "point": this row is concatenated into the same bootstrap.csv
    # that :func:`bootstrap_rows` writes, and a second name for the point estimate
    # lands as a NaN in the column every consumer reads.
    boot_row = {
        **base,
        "scope": TISSUE_BALANCED,
        "class": "macro",
        "value": res["point"],
        "ci_lo": lo,
        "ci_hi": hi,
        "n_donors": res["n_donors"],
        "n_organs": res["n_organs"],
        "n_boot": res["n_boot"],
    }
    return row, boot_row, res.get("draws")
