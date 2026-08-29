"""``train`` stage — fit each representation once per seed and embed every spot.

The representation is fitted on the cohort split named by ``cfg.train.fit_split``
(by default the unannotated MOSAIC and De Zuani slides). The annotated TuPro and
USZ slides live in the ``test`` split and are never seen while fitting, so the
embedding that the evaluation stage scores is genuinely out-of-cohort.

A model named ``<correction>_<model>`` (``harmony_ae``, ``combat_pca``, ...) is that
model fitted on batch-corrected gene features instead of the raw ones; see
:mod:`vgtfm.models.corrected`.

One embedding matrix per (model, seed) is written for *all* spots, and the
evaluation stage only ever slices it. Nothing is refitted per fold unless
``cfg.train.refit_per_fold`` is set, in which case the evaluation stage does that
refitting itself — keeping this stage's outputs meaningful in both modes.
"""

from __future__ import annotations

import json

import numpy as np

from .. import perf
from ..data import tables
from . import corrected
from .base import Inputs, build
from .nn import effective_rank


def embedding_path(cfg, model: str, seed: int):
    return cfg.sub("embeddings", model) / f"seed-{seed}.npy"


def load_embedding(cfg, model: str, seed: int) -> np.ndarray:
    """One stored embedding, exactly as this stage wrote it.

    Every stage that reads an embedding reads it through here, and none of them may
    cast on the way in. ``eval`` fingerprints the matrix it scores and ``integrate``
    checks that fingerprint against the matrix it corrects
    (:func:`vgtfm.provenance.array_fingerprint`, which hashes the dtype along with
    the bytes). Two call sites free to differ by an ``astype`` would make a float32
    view of a float64 file a *different* fingerprint, and ``integrate`` would then
    refuse with "`train` and `eval` are out of step" on a pair of files that are
    perfectly in step -- a failure no rerun can clear. Loading in one place is what
    makes "the matrix `eval` scored" name a single array.

    The caller checks that the file exists: a missing embedding means something
    different to each of them (a stage that has not run, a seed that was never
    trained, a model dropped from the config) and each says so in its own terms.
    """
    return np.load(embedding_path(cfg, model, seed))


def fit_rows(cfg, table) -> np.ndarray:
    """Rows of the cohort split the representation is fitted on."""
    rows = np.where(table.col("split") == cfg.train.fit_split)[0]
    if len(rows) == 0:
        raise SystemExit(
            f"train.fit_split='{cfg.train.fit_split}' selects no spots. "
            f"Available splits: {sorted(set(table.col('split')))}"
        )
    return rows


def fit_and_embed(cfg, inputs: Inputs, rows: np.ndarray, name: str, seed: int):
    """Fit one model on *rows* and embed every spot in *inputs*."""
    model = build(name, cfg, seed)
    history = model.fit(inputs.select(rows))
    Z = model.embed(inputs)
    perf.free_cuda()
    return model, Z, history


def run(cfg) -> None:
    out = cfg.sub("train")
    # Ahead of the table load and of every fit, which is what costs the hour.
    corrected.validate(cfg.models.names)
    table = tables.load(cfg)
    inputs = Inputs.from_table(table)
    rows = fit_rows(cfg, table)

    print(
        f"  fitting on split '{cfg.train.fit_split}': {len(rows):,} spots, "
        f"{table.meta.iloc[rows].sample_id.nunique()} slides"
    )
    if cfg.train.refit_per_fold:
        print(
            "  note: train.refit_per_fold is on — the eval stage will additionally "
            "refit each model on every fold's training slides (legacy protocol)"
        )

    records = []
    for seed in cfg.seeds:
        for name in cfg.models.names:
            print(f"\n  -- {name} (seed {seed}) --")
            model_inputs, inner, correction = corrected.resolve(
                cfg, table, inputs, name, seed, rows
            )
            model, Z, history = fit_and_embed(cfg, model_inputs, rows, inner, seed)
            # Fixed subsample seed, independent of the model seed, so the effective
            # ranks of different models and seeds are measured on the same spots.
            rank = effective_rank(
                Z[np.random.default_rng(0).choice(len(Z), size=min(8000, len(Z)), replace=False)]
            )
            rec = {
                "model": name,
                "correction": correction,
                "inner_model": inner,
                "seed": seed,
                "dim": int(Z.shape[1]),
                "effective_rank": rank,
                **{k: v for k, v in history.items() if not isinstance(v, list)},
            }
            records.append(rec)
            print(f"  [{name}] embedding {Z.shape} eff_rank={rank:.1f}/{Z.shape[1]}")

            if cfg.train.save_embeddings:
                path = embedding_path(cfg, name, seed)
                np.save(path, Z)
                print(f"  [{name}] -> {path}")
            # Keyed by the outer name: `model.state()` only knows the inner model.
            (cfg.sub("train", "history") / f"{name}__seed-{seed}.json").write_text(
                json.dumps(
                    {"model": name, "correction": correction, **model.state()},
                    indent=2,
                    default=str,
                )
            )
            del model, Z, model_inputs
            perf.free_cuda()

    import pandas as pd

    df = pd.DataFrame(records)
    df.to_csv(out / "train_summary.csv", index=False)
    _print_width_warning(df)

    print(f"\n  wrote {out}/train_summary.csv")
    print(df.to_string(index=False))


def _print_width_warning(df) -> None:
    """Warn when the embeddings being compared do not share a width.

    kNN distances depend on dimensionality, so models of different widths are not
    directly comparable on the annotation probe. `pca_oracle_matched` is excluded
    because being narrower is its entire purpose, and a warning that fires on every
    run stops being read.
    """
    comparable = df[df["model"] != "pca_oracle_matched"] if not df.empty else df
    widths = sorted(set(comparable["dim"])) if not comparable.empty else []
    if len(widths) > 1:
        print(
            f"\n  WARNING: embeddings have different widths {widths}. The kNN probe "
            f"is sensitive to dimensionality — set every model's latent_dim (and "
            f"models.pca_components) to the same value for a fair comparison."
        )
