"""``data`` stage — materialise the feature cache and describe the cohort.

Cheap, deterministic, and the only stage that touches the cached HuggingFace
datasets. It writes:

``data/cohort.csv``     one row per slide: cohort, tissue, donor, split, spot and
                        annotation counts. The dataset appendix is built from this,
                        so its counts are what the pipeline actually loaded.
``data/classes.csv``    per-class spot counts, overall and per tissue.
``data/folds.json``     every hierarchical fold, fully enumerated.
``data/summary.json``   dimensions, totals, and the feature-cache location.
"""

from __future__ import annotations

import json

import pandas as pd

from ..labels import labeled_mask
from . import folds as folds_mod
from . import tables


def _cohort_table(table) -> pd.DataFrame:
    meta = table.meta.copy()
    meta["labeled"] = labeled_mask(meta["annotation"].to_numpy())
    rows = []
    for sid, g in meta.groupby("sample_id", sort=True):
        classes = sorted(set(g.loc[g["labeled"], "annotation"].astype(str)))
        rows.append(
            {
                "sample_id": sid,
                "dataset_id": g["dataset_id"].iloc[0],
                "tissue": g["tissue"].iloc[0],
                "donor": g["donor"].iloc[0],
                "region": int(g["region"].iloc[0]),
                "replicate": int(g["replicate"].iloc[0]),
                "split": g["split"].iloc[0],
                "n_spots": int(len(g)),
                "n_annotated": int(g["labeled"].sum()),
                "n_classes": len(classes),
                "classes": "|".join(classes),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset_id", "sample_id"])


def _class_table(table) -> pd.DataFrame:
    """Per-class spot counts, per tissue and pooled.

    ``annotation`` is the canonical class; ``raw_labels`` lists the cohort spellings
    that fed it, pipe-separated, so a merge is visible on the ``ALL`` rows —
    ``Tumor`` there reads ``TUM|Tumor``.
    """
    meta = table.meta.copy()
    # A class demoted by the support floor is unlabelled everywhere else, but the
    # dataset appendix still has to name it: an organ whose rare class was dropped
    # reads as an organ that never had one otherwise. `scored` says which is which.
    floored = (
        meta["annotation_floored"].astype(str) != ""
        if "annotation_floored" in meta.columns
        else pd.Series(False, index=meta.index)
    )
    if floored.any():
        meta.loc[floored, "annotation"] = meta.loc[floored, "annotation_floored"]
    mask = labeled_mask(meta["annotation"].to_numpy()) | floored
    lab = meta.loc[mask]
    cols = ["tissue", "annotation", "raw_labels", "n_spots", "n_slides", "scored"]
    if lab.empty:
        return pd.DataFrame(columns=cols)
    # `annotation_raw` is absent only for a frame built before cache schema 2.
    raw_col = "annotation_raw" if "annotation_raw" in lab.columns else "annotation"
    agg = dict(
        n_spots=("spot_id", "size"),
        n_slides=("sample_id", "nunique"),
        raw_labels=(raw_col, lambda s: "|".join(sorted(set(s.astype(str))))),
        scored=("_scored", "all"),
    )
    lab = lab.assign(_scored=~floored.loc[lab.index])
    out = (
        lab.groupby(["tissue", "annotation"])
        .agg(**agg)
        .reset_index()
        .sort_values(["tissue", "n_spots"], ascending=[True, False])
    )
    total = (
        lab.groupby("annotation")
        .agg(**agg)
        .reset_index()
        .assign(tissue="ALL")
        .sort_values("n_spots", ascending=False)
    )
    return pd.concat([out[cols], total[cols]], ignore_index=True)


def run(cfg) -> None:
    out = cfg.sub("data")

    table = tables.load(cfg)

    cohort = _cohort_table(table)
    cohort.to_csv(out / "cohort.csv", index=False)

    classes = _class_table(table)
    classes.to_csv(out / "classes.csv", index=False)

    fold_map = folds_mod.generate_for(cfg, table)
    (out / "folds.json").write_text(
        json.dumps({lvl: [f.as_dict() for f in specs] for lvl, specs in fold_map.items()}, indent=2)
    )

    mask = table.labeled
    summary = {
        "substrate": table.substrate,
        "source": str(cfg.data_path()),
        "feature_cache": str(cfg.cache_dir()),
        "n_spots": table.n,
        "n_annotated_spots": int(mask.sum()),
        "n_slides": int(table.meta.sample_id.nunique()),
        "n_annotated_slides": int(cohort.loc[cohort.n_annotated > 0, "sample_id"].nunique()),
        "n_donors": int(table.meta.donor.nunique()),
        "gene_dim": table.gene_dim,
        "patch_dim": table.patch_dim,
        "spots_per_split": table.meta.split.value_counts().to_dict(),
        "slides_per_split": table.meta.groupby("split").sample_id.nunique().to_dict(),
        "spots_per_tissue": table.meta.tissue.value_counts().to_dict(),
        "folds": {lvl: len(specs) for lvl, specs in fold_map.items()},
        # What the pooled macro average is actually taken over, and what the floor
        # removed from it — the two numbers needed to read any macro-F1 here.
        "class_floor": {
            "min_class_spots": int(cfg.data.min_class_spots),
            "min_class_slides": int(cfg.data.min_class_slides),
            "n_classes_scored": int(
                (classes.tissue != "ALL").sum()
                - (~classes.loc[classes.tissue != "ALL", "scored"]).sum()
            ),
            "dropped": [
                {k: r[k] for k in ("tissue", "annotation", "n_spots", "n_slides")}
                for r in classes.loc[(classes.tissue != "ALL") & ~classes.scored].to_dict("records")
            ],
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(
        f"\n  {table.n:,} spots | {summary['n_slides']} slides "
        f"({summary['n_annotated_slides']} annotated) | "
        f"{summary['n_donors']} donors | {int(mask.sum()):,} annotated spots"
    )
    print(f"  gene {table.gene_dim}d, patch {table.patch_dim}d")
    print("  folds: " + ", ".join(f"{k}={v}" for k, v in summary["folds"].items()))
    print(f"  wrote {out}/cohort.csv, classes.csv, folds.json, summary.json")

    # Free the (potentially several GB) feature matrices before the next stage.
    del table
    import gc

    gc.collect()
