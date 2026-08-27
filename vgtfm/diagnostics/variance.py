"""Between- versus within-group variance decomposition.

The simplest statement of the batch problem. For spots :math:`x_i` in groups
:math:`b`, the total sum of squares splits exactly:

.. math::
    \\sum_i \\|x_i - \\bar x\\|^2
      = \\sum_b n_b \\|\\bar x_b - \\bar x\\|^2
      + \\sum_b \\sum_{i \\in b} \\|x_i - \\bar x_b\\|^2

The between fraction is a multivariate :math:`\\eta^2` — the share of variance a
one-hot group indicator alone would explain. Scored against slide, patient, tissue
and cohort, because "batch effect" means different things at those levels.

``patient`` comes from the registry's ``donor_pattern``, not from
:attr:`~vgtfm.data.tables.SpotTable.donor`. The latter parses TuPro ids and falls
back to the slide id, so on the train split — where the multi-slide patients
actually live — it makes every slide its own patient and the between-patient
fraction identical to the between-slide one by construction.

It is invariant to rotation but *not* to per-dimension rescaling, so it must be
computed on raw features, never on standardised or whitened ones.
"""

from __future__ import annotations

import numpy as np

from . import split_rows


def decompose(X: np.ndarray, groups: np.ndarray) -> dict:
    """Exact between/within decomposition of the total sum of squares."""
    X = np.asarray(X, dtype=np.float64)
    groups = np.asarray(groups)
    n = len(X)
    if n < 2:
        return {"n": int(n), "n_groups": 0, "between_fraction": float("nan")}

    grand = X.mean(axis=0)
    total = float(((X - grand) ** 2).sum())
    between = 0.0
    uniq = np.unique(groups)
    for g in uniq:
        m = groups == g
        n_g = int(m.sum())
        if n_g == 0:
            continue
        between += n_g * float(((X[m].mean(axis=0) - grand) ** 2).sum())

    within = total - between
    frac = between / total if total > 0 else float("nan")
    return {
        "n": int(n),
        "n_groups": int(len(uniq)),
        "total_ss": total,
        "between_ss": between,
        "within_ss": within,
        "between_fraction": float(frac),
        "within_fraction": float(1.0 - frac) if total > 0 else float("nan"),
    }


def decompose_subsampled(
    X: np.ndarray, groups: np.ndarray, *, max_spots: int = 50_000, seed: int = 42
) -> dict:
    """Decomposition on a random subsample, for cohorts too large to hold twice."""
    n = len(X)
    if n <= max_spots:
        return decompose(X, groups)
    idx = np.random.default_rng(seed).choice(n, size=max_spots, replace=False)
    out = decompose(X[idx], groups[idx])
    out["subsampled_from"] = int(n)
    return out


def _grouping_values(cfg, table, rows, name: str) -> np.ndarray:
    """Group labels for one grouping variable over *rows*.

    Everything but ``patient`` is a column of the spot table. ``patient`` is derived
    here from the registry rather than cached, so the fold and bootstrap keys are
    left exactly as they are; see :meth:`~vgtfm.data.tables.Registry.patient_keys`.
    """
    if name != "patient":
        return table.col(name)[rows]
    from ..data.tables import load_registry

    registry = load_registry(cfg.paths.datasets_json)
    return registry.patient_keys(table.sample_id[rows], table.col("dataset_id")[rows])


def run(cfg, table, *, group_cols=("sample_id", "patient", "tissue", "dataset_id")):
    """Decompose every configured feature column against every grouping variable."""
    import pandas as pd

    d = cfg.diagnostics
    rows = split_rows(cfg, table)

    recs = []
    for column in d.columns:
        X = table.features(column)[rows]
        for gc in group_cols:
            groups = _grouping_values(cfg, table, rows, gc)
            if len(np.unique(groups)) < 2:
                continue
            res = decompose_subsampled(X, groups, seed=d.seed)
            recs.append({"substrate": cfg.data.substrate, "column": column, "grouping": gc, **res})
            print(
                f"    {column:16s} by {gc:11s} "
                f"between={res['between_fraction']:.3f} "
                f"within={res['within_fraction']:.3f} "
                f"({res['n_groups']} groups, n={res['n']:,})"
            )
    # Name the columns even when there are no rows: every consumer selects on
    # `grouping`, and a column-less frame turns "this split has one slide" into an
    # AttributeError several frames away from the cause.
    return pd.DataFrame(
        recs,
        columns=[
            "substrate",
            "column",
            "grouping",
            "n",
            "n_groups",
            "total_ss",
            "between_ss",
            "within_ss",
            "between_fraction",
            "within_fraction",
            "subsampled_from",
        ],
    )
