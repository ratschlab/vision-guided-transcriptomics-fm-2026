"""Donor-level bootstrap confidence intervals.

Resampling *spots* treats the ~1,500 spots of a slide as 1,500 independent
observations. They are not: spots within a slide share a patient, a tissue block,
a stain batch and a scanner, so a spot-level interval understates the real
uncertainty by roughly an order of magnitude (on TuPro stroma, a donor-level
standard deviation of 0.098 against a spot-level 0.005).

The unit of replication here is the **donor**. With 7 melanoma patients and 6 USZ
patients, absolute per-class F1 is barely resolvable. Paired deltas on the same
donor resample are far more informative, because the donor effect cancels, so
:func:`paired_delta_ci` is what comparisons between conditions should use.

Labels are integer-encoded once and each replicate is a single ``bincount`` over
the confusion matrix, so 2000 replicates over 100k predictions take seconds.
"""

from __future__ import annotations

import numpy as np

from .probes import LabelCodec, confusion_counts, metrics_from_confusion

_MACRO_KEYS = ("accuracy", "precision", "recall", "f1_score")


def _percentile_ci(values: np.ndarray, ci: float) -> tuple[float, float]:
    lo = (1.0 - ci) / 2.0 * 100.0
    return (float(np.percentile(values, lo)), float(np.percentile(values, 100.0 - lo)))


def _two_sided_p(draws: np.ndarray) -> float:
    """Two-sided bootstrap p-value: how often the resampled delta changes sign.

    Both tails are counted with a non-strict inequality, so a replicate whose delta
    is exactly zero falls in *both* — deliberately conservative, since a delta that
    never resolves a sign should not read as evidence. The clamp is what that
    convention costs: when every replicate is exactly zero (both arms score the
    same, which happens on classes no model ever predicts) each tail is 1.0 and the
    doubled minimum would be 2.0, i.e. not a probability.

    A returned 0.0 means "no replicate crossed zero", i.e. ``p < 1 / n_boot``.
    """
    return float(min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean())))


def pool_replicates(draws, *, ci: float = 0.95) -> tuple[float, float, float, int]:
    """Interval over the *union* of several runs' replicate draws.

    Returns ``(lo, hi, p_two_sided, n_runs)``.

    Every seed is bootstrapped over the same donor resamples (``bootstrap_seed`` is
    fixed independently of the model seed), so replicate *b* means the same patients
    in every run. Concatenating the draws therefore mixes two sources into one
    distribution: which patients were recruited, and which random initialisation the
    model got. The percentile interval of that mixture covers both.

    The alternative — averaging the per-seed interval bounds — describes a *single*
    run while being printed beside a point estimate that is the mean over runs, and
    it hides training variability exactly where it is largest. On this cohort the
    seed is a median 0.5% of the variance, so the two agree almost everywhere; they
    part company on the handful of rows where an autoencoder's initialisation moves
    the score more than the patients do, and there the wider answer is the honest
    one.

    This is a mixture, not a variance decomposition: with three seeds it cannot
    populate the tails of the training-noise distribution, so treat it as
    conservative rather than exact.
    """
    kept = [np.asarray(d, dtype=float) for d in draws if d is not None and len(np.asarray(d))]
    if not kept:
        return float("nan"), float("nan"), float("nan"), 0
    stacked = np.concatenate(kept)
    lo, hi = _percentile_ci(stacked, ci)
    return lo, hi, _two_sided_p(stacked), len(kept)


def _donor_rows(donors: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    uniq = np.unique(donors)
    return uniq, [np.where(donors == d)[0] for d in uniq]


def _scored(cm: np.ndarray, score_idx):
    m = metrics_from_confusion(cm, score_idx)
    _p, _r, f1, _s = m.pop("_per_class")
    return m, f1


def _prepare(y_true, y_pred, donors, classes, score_classes):
    codec = LabelCodec(classes)
    t = codec.encode(y_true)
    p = codec.encode(y_pred)
    score_idx = None if score_classes is None else codec.subset_index(score_classes)
    donors = np.asarray(donors).astype(str)
    return codec, t, p, score_idx, donors


def donor_bootstrap(
    y_true,
    y_pred,
    donors,
    *,
    classes=None,
    score_classes=None,
    n_boot: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
    return_draws: bool = False,
) -> dict:
    """Bootstrap macro and per-class F1 by resampling donors with replacement.

    Point estimates come from the observed data, not the bootstrap mean, which is
    biased for a bounded statistic like F1.
    """
    if classes is None:
        classes = np.unique(
            np.concatenate([np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str)])
        )
    codec, t, p, score_idx, donors = _prepare(y_true, y_pred, donors, classes, score_classes)
    K = len(codec)

    point_m, point_f1 = _scored(confusion_counts(t, p, K), score_idx)
    out = {
        "point": {
            **point_m,
            "per_class_f1": {str(c): float(point_f1[i]) for i, c in enumerate(codec.classes)},
        },
        "n_boot": int(n_boot),
    }

    uniq, rows_by_donor = _donor_rows(donors)
    out["n_donors"] = int(len(uniq))
    if len(uniq) < 2 or n_boot <= 0:
        out["ci"] = {}
        out["per_class_ci"] = {}
        if return_draws:
            out["draws"] = {}
        return out

    rng = np.random.default_rng(seed)
    macro = {k: np.empty(n_boot) for k in _MACRO_KEYS}
    per_class = np.empty((n_boot, K))
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        rows = np.concatenate([rows_by_donor[i] for i in pick])
        m, f1 = _scored(confusion_counts(t[rows], p[rows], K), score_idx)
        for k in _MACRO_KEYS:
            macro[k][b] = m[k]
        per_class[b] = f1

    out["ci"] = {k: _percentile_ci(v, ci) for k, v in macro.items()}
    out["per_class_ci"] = {
        str(c): _percentile_ci(per_class[:, i], ci) for i, c in enumerate(codec.classes)
    }
    if return_draws:
        # Kept so the caller can pool them across seeds; see pool_replicates.
        out["draws"] = {
            "macro": macro["f1_score"].copy(),
            **{str(c): per_class[:, i].copy() for i, c in enumerate(codec.classes)},
        }
    return out


def paired_delta_ci(
    y_true,
    pred_a,
    pred_b,
    donors,
    *,
    classes=None,
    score_classes=None,
    n_boot: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
    return_draws: bool = False,
) -> dict:
    """CI for ``metric(a) - metric(b)`` under a shared donor resample.

    Both conditions are scored on the *same* resampled donors in each replicate, so
    between-donor variance cancels. At this cohort size an interval on either
    condition alone is several times wider than the difference between them.
    """
    y_true = np.asarray(y_true)
    if len(pred_a) != len(y_true) or len(pred_b) != len(y_true):
        raise ValueError("paired_delta_ci requires predictions aligned to the same rows")
    if classes is None:
        classes = np.unique(
            np.concatenate(
                [
                    np.asarray(y_true).astype(str),
                    np.asarray(pred_a).astype(str),
                    np.asarray(pred_b).astype(str),
                ]
            )
        )
    codec, t, pa, score_idx, donors = _prepare(y_true, pred_a, donors, classes, score_classes)
    pb = codec.encode(pred_b)
    K = len(codec)

    ma, fa = _scored(confusion_counts(t, pa, K), score_idx)
    mb, fb = _scored(confusion_counts(t, pb, K), score_idx)
    out = {
        "delta_f1_score": ma["f1_score"] - mb["f1_score"],
        "delta_accuracy": ma["accuracy"] - mb["accuracy"],
        "per_class_delta": {str(c): float(fa[i] - fb[i]) for i, c in enumerate(codec.classes)},
    }

    uniq, rows_by_donor = _donor_rows(donors)
    out["n_donors"] = int(len(uniq))
    if len(uniq) < 2 or n_boot <= 0:
        out["ci"] = {}
        out["per_class_ci"] = {}
        out["per_class_p"] = {}
        if return_draws:
            out["draws"] = {}
        return out

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    pc = np.empty((n_boot, K))
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        rows = np.concatenate([rows_by_donor[i] for i in pick])
        m_a, f_a = _scored(confusion_counts(t[rows], pa[rows], K), score_idx)
        m_b, f_b = _scored(confusion_counts(t[rows], pb[rows], K), score_idx)
        deltas[b] = m_a["f1_score"] - m_b["f1_score"]
        pc[b] = f_a - f_b

    out["ci"] = {"delta_f1_score": _percentile_ci(deltas, ci)}
    out["p_two_sided"] = _two_sided_p(deltas)
    out["per_class_ci"] = {
        str(c): _percentile_ci(pc[:, i], ci) for i, c in enumerate(codec.classes)
    }
    out["per_class_p"] = {str(c): _two_sided_p(pc[:, i]) for i, c in enumerate(codec.classes)}
    if return_draws:
        out["draws"] = {
            "macro": deltas.copy(),
            **{str(c): pc[:, i].copy() for i, c in enumerate(codec.classes)},
        }
    return out


def tissue_balanced_bootstrap(
    y_true,
    y_pred,
    donors,
    tissues,
    *,
    classes,
    score_classes_by_tissue: dict[str, list[str]],
    n_boot: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
    return_draws: bool = False,
) -> dict:
    """Two-level interval for the macro-F1 averaged over organs with equal weight.

    The headline statistic weights each organ equally, so the cohort is a two-level
    design — donors nested in organs — and an interval has to resample both levels:
    organs with replacement, then donors with replacement inside each organ drawn.
    Resampling only donors would hold the organ means fixed and report the spread of
    a statistic that is mostly *between* organs as if it were within them.

    Each organ is scored over its own vocabulary, which is what makes the average
    well posed: the pooled ``(tissue, class)`` denominator counts a cell only kidney
    can contain against lung, whereas this asks each organ its own question and then
    gives the three answers equal weight.

    With three organs the outer level can only draw ten distinct multisets, so the
    interval is honest but granular — report the organ count beside it.
    """
    tissues = np.asarray(tissues).astype(str)
    donors = np.asarray(donors).astype(str)
    codec = LabelCodec(classes)
    t_all, p_all = codec.encode(y_true), codec.encode(y_pred)
    K = len(codec)

    organs = sorted(set(tissues.tolist()) & set(score_classes_by_tissue))
    if len(organs) < 2:
        return {}

    # A confusion matrix is additive over disjoint row sets, and F1 is a function of
    # it alone. So each donor's matrix is computed once here and a replicate is a
    # sum of the ones it drew — which turns the inner loop from rescoring ~90k rows
    # into adding a handful of KxK arrays.
    cms, idx_by_organ = {}, {}
    for org in organs:
        m = tissues == org
        base = np.where(m)[0]
        _, rows = _donor_rows(donors[m])
        cms[org] = np.stack([confusion_counts(t_all[base[r]], p_all[base[r]], K) for r in rows])
        idx_by_organ[org] = codec.subset_index(score_classes_by_tissue[org])

    def f1_of(org, cm) -> float:
        return float(_scored(cm, idx_by_organ[org])[0]["f1_score"])

    point = float(np.mean([f1_of(o, cms[o].sum(axis=0)) for o in organs]))
    out = {
        "point": point,
        "n_organs": len(organs),
        "n_donors": int(len(np.unique(donors))),
        "n_boot": int(n_boot),
    }
    if n_boot <= 0:
        return out

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        vals = []
        for oi in rng.integers(0, len(organs), len(organs)):  # organs, with repl.
            org = organs[oi]
            n_d = len(cms[org])
            pick = rng.integers(0, n_d, n_d)  # donors, with repl.
            vals.append(f1_of(org, cms[org][pick].sum(axis=0)))
        draws[b] = float(np.mean(vals))
    out["ci"] = _percentile_ci(draws, ci)
    if return_draws:
        out["draws"] = draws
    return out
