"""``eval`` stage — score every embedding with every protocol.

For each (model, seed, protocol) the stage produces one prediction per annotated
spot, then derives everything else from that single vector: macro metrics, a
per-tissue breakdown, per-class F1, and donor-level bootstrap intervals. Sharing
one prediction vector is what keeps the paired comparisons in
:mod:`vgtfm.evaluate.bootstrap` aligned row for row.

Outputs (under ``eval/``):

``results.csv``     macro metrics, one row per model/seed/protocol/level/fold/scope
``per_class.csv``   per-class precision/recall/F1/support at the same granularity
``bootstrap.csv``   donor-level CIs on fold-pooled predictions
``deltas.csv``      paired model-vs-reference deltas on a shared donor resample
``predictions/``    the raw prediction vectors, for re-analysis without re-running
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..data import folds as folds_mod
from ..data import tables
from ..degraded import refuse
from ..labels import cohort_classes, labeled_mask
from .. import provenance
from ..models import corrected
from ..models.train import embedding_path, fit_and_embed, load_embedding
from . import protocol as proto
from .reporting import (
    POOL_KEY,
    TISSUE_BALANCED,
    attach_pooled,
    bootstrap_rows,
    metric_rows,
    paired_delta_rows,
    pool,
    score_vocabulary,
    tissue_balanced_rows,
)

#: Model name used for the chance-level baseline row.
MAJORITY = "majority"
#: Model paired deltas are computed against, unless the arm has a closer
#: reference of its own (:func:`_reference_model`).
REFERENCE = "pca"


def _load_embedding(cfg, model: str, seed: int, n_spots: int) -> np.ndarray:
    if model == MAJORITY:
        return np.zeros((n_spots, 1), dtype=np.float32)
    path = embedding_path(cfg, model, seed)
    if not path.exists():
        raise SystemExit(
            f"missing embedding {path}. Run `python run.py train` first "
            f"(or drop '{model}' from models.names)."
        )
    # Through the shared loader, uncast: `integrate` checks the fingerprint of what
    # it loads against the one stamped here, and a cast on either side would make
    # two readings of one file two different matrices. See `train.load_embedding`.
    Z = load_embedding(cfg, model, seed)
    if len(Z) != n_spots:
        raise SystemExit(
            f"{path} has {len(Z)} rows but the loaded cohort has {n_spots}. "
            f"The data filters changed since training — rerun the train stage."
        )
    return Z


def _predict_all(cfg, table, Z, model: str, fold_map) -> dict[tuple[str, str], dict]:
    """Return ``{(protocol, level): pooled_prediction}`` for one model+seed."""
    meta = table.meta
    is_majority = model == MAJORITY
    out: dict[tuple[str, str], dict] = {}

    for name in cfg.eval.protocols:
        before = len(out)
        if not proto.is_fold_protocol(name):
            pred = proto.run(name, Z, meta, cfg=cfg, majority=is_majority)
            if pred:
                out[(name, "all")] = pred
        else:
            for level, specs in fold_map.items():
                per_fold = []
                for spec in specs:
                    pred = proto.run(name, Z, meta, cfg=cfg, fold=spec, majority=is_majority)
                    if pred:
                        pred["fold"] = spec.name
                        per_fold.append(pred)
                pooled = pool(per_fold)
                if pooled:
                    pooled["per_fold"] = per_fold
                    out[(name, level)] = pooled
                elif specs:
                    print(
                        f"    {name}/{level}: none of the {len(specs)} fold(s) "
                        f"produced a prediction"
                    )
        # A protocol that scores nothing at all is a configuration that does not fit
        # this cohort, not a result. Left alone it vanishes from results.csv and the
        # figures stage draws the remaining protocols as if they were what was asked
        # for.
        if len(out) == before:
            refuse(
                f"predictions from protocol '{name}' for model '{model}'",
                "it scored no spot at any fold level",
                hint="check eval.protocols and folds.levels against the slides in this cohort",
            )
    return out


def _score_prediction(pred: dict, classes, base: dict, cfg, vocab: dict):
    """Every row one pooled prediction yields: macro, per-class, bootstrap, draws.

    One place rather than four call sites, because the four have to agree on the
    denominator. ``draws`` is keyed by ``(scope, class)`` so the caller can group the
    replicate draws across seeds without knowing how a scope is named.
    """
    macro, per_class = metric_rows(pred, classes, base, vocab)
    boot: list[dict] = []
    draws: dict[tuple[str, str], list] = {}

    # The organ-balanced headline: an average *over* the tissue scopes, so it belongs
    # to the pooled prediction only, never to a single fold.
    tb_row, tb_boot, tb_draws = tissue_balanced_rows(
        pred, classes, base, cfg, vocab, with_boot=cfg.eval.bootstrap_n > 0
    )
    if tb_row:
        macro.append(tb_row)
    if tb_boot:
        boot.append(tb_boot)
        draws[(TISSUE_BALANCED, "macro")] = tb_draws

    for fold_pred in pred.get("per_fold", []):
        fold_base = {**base, "fold": fold_pred["fold"]}
        fold_macro, fold_class = metric_rows(fold_pred, classes, fold_base, vocab)
        macro.extend(fold_macro)
        per_class.extend(fold_class)

    if cfg.eval.bootstrap_n > 0:
        rows, bdraws = bootstrap_rows(pred, classes, base, cfg, vocab, return_draws=True)
        boot.extend(rows)
        draws.update(bdraws)
    return macro, per_class, boot, draws


def run(cfg) -> None:
    out = cfg.sub("eval")
    table = tables.load(cfg)
    meta = table.meta
    fold_map = folds_mod.generate_for(cfg, table)

    classes = cohort_classes(table)
    n_lab = int(labeled_mask(meta["annotation"].to_numpy()).sum())
    print(f"  {n_lab:,} annotated spots, {len(classes)} classes: {classes}")

    models = [MAJORITY, *cfg.models.names]
    macro_rows: list[dict] = []
    class_rows: list[dict] = []
    boot_rows: list[dict] = []
    delta_rows: list[dict] = []

    # Cache predictions so paired deltas can be computed against the reference,
    # and the scoring vocabulary so every model shares one denominator.
    preds_by_key: dict[tuple, dict] = {}
    vocab_by_key: dict[tuple[str, str], dict[str, list[str]]] = {}
    # Replicate draws per group of seeds, so the published interval can be the
    # percentile of their union rather than the mean of their separate bounds.
    boot_draws: dict[tuple, list] = {}

    fingerprints: dict[tuple, str] = {}

    inputs = None
    if cfg.train.refit_per_fold:
        from ..models.base import Inputs

        inputs = Inputs.from_table(table)

    for seed in cfg.seeds:
        for model in models:
            print(f"\n  -- {model} (seed {seed}) --")
            Z = _load_embedding(cfg, model, seed, table.n)
            # Tie the predictions about to be cached to the matrix that produced
            # them: `integrate` reads them back beside the embedding it loads
            # itself, and only this says the two are the same representation.
            fingerprints[(seed, model)] = provenance.array_fingerprint(Z)
            per_protocol = _predict_all(cfg, table, Z, model, fold_map)

            # The majority baseline has nothing to refit; every other model does,
            # including the oracle.
            if cfg.train.refit_per_fold and model != MAJORITY:
                _refuse_refit_of_corrected(model)
                per_protocol.update(_refit_predictions(cfg, table, inputs, model, seed, fold_map))

            for (protocol, level), pred in per_protocol.items():
                preds_by_key[(seed, model, protocol, level)] = pred
                vocab = vocab_by_key.setdefault((protocol, level), score_vocabulary(pred))
                base = {
                    "substrate": cfg.data.substrate,
                    "model": model,
                    "seed": seed,
                    "protocol": protocol,
                    "level": level,
                    "fold": "pooled",
                }
                macro, per_class, boot, draws = _score_prediction(pred, classes, base, cfg, vocab)
                macro_rows.extend(macro)
                class_rows.extend(per_class)
                boot_rows.extend(boot)
                for key, d in draws.items():
                    boot_draws.setdefault((model, protocol, level, *key), []).append(d)

            del Z

    attach_pooled(boot_rows, boot_draws, cfg, with_p=False, group_fields=("model", *POOL_KEY))

    results = pd.DataFrame(macro_rows)
    per_class = pd.DataFrame(class_rows)
    results.to_csv(out / "results.csv", index=False)
    per_class.to_csv(out / "per_class.csv", index=False)
    if boot_rows:
        pd.DataFrame(boot_rows).to_csv(out / "bootstrap.csv", index=False)

    delta_rows = _delta_rows(preds_by_key, classes, cfg, vocab_by_key)
    if delta_rows:
        pd.DataFrame(delta_rows).to_csv(out / "deltas.csv", index=False)

    _save_predictions(cfg, preds_by_key, fingerprints)
    _print_headline(results, cfg)
    print(
        f"\n  wrote {out}/results.csv, per_class.csv"
        f"{', bootstrap.csv' if boot_rows else ''}"
        f"{', deltas.csv' if delta_rows else ''}"
    )


# ── helpers ──────────────────────────────────────────────────────────


def _refuse_refit_of_corrected(model: str) -> None:
    """``train.refit_per_fold`` cannot be combined with a batch-corrected arm.

    A refit here would fit on the fold's own training slides while their features
    carry a correction fitted across the whole cohort — Harmony and ComBat have no
    per-fold form — so the protocol the row claims would not be the one that ran.
    """
    if corrected.parse(model) is None:
        return
    refuse(
        f"a per-fold refit of '{model}'",
        "its input was corrected across the whole cohort and the correction has no per-fold form",
        hint=f"set train.refit_per_fold=false, or drop '{model}' from models.names",
    )


def _refit_predictions(cfg, table, inputs, model, seed, fold_map) -> dict:
    """Legacy mode: refit the representation on each fold's training slides.

    The model is fitted on the unannotated cohorts and then refitted on the fold's
    own training slides, using those slides *only* — a union with the 192k
    unannotated spots would let the larger cohort dominate the fit and would not be
    the same protocol.

    It costs one fit per fold, and the evaluated embedding has seen the fold's
    training donors, which is why the default is to score a single embedding fitted
    once on the unannotated cohorts.
    """
    out: dict = {}
    for level, specs in fold_map.items():
        per_fold: dict[str, list] = {}
        for spec in specs:
            rows = table.rows_for_samples(spec.train_sample_ids)
            if len(rows) == 0:
                continue
            print(f"    refit {model} on {level}/{spec.name}: {len(rows):,} spots")
            _m, Z_fold, _h = fit_and_embed(cfg, inputs, rows, model, seed)
            for protocol, fn in (
                ("heldout_donor_refit", proto.heldout_donor),
                ("pooled_loso_refit", proto.pooled_loso_on_fold),
            ):
                pred = fn(Z_fold, table.meta, spec, cfg=cfg)
                if pred:
                    pred["fold"] = spec.name
                    per_fold.setdefault(protocol, []).append(pred)
            del Z_fold
        for protocol, preds in per_fold.items():
            pooled = pool(preds)
            if pooled:
                pooled["per_fold"] = preds
                out[(protocol, level)] = pooled
    return out


def _reference_model(model: str, cfg) -> str:
    """Which model *model*'s paired delta is measured against.

    ``pca`` for everything the paper reports, and the matching corrected baseline
    for a batch-corrected arm, so that delta is what guidance adds *given* a
    corrected source rather than the sum of guidance and correction. Falls back to
    ``pca`` when that baseline is not in the run: the delta then answers the
    two-change question, and the ``reference`` column says so.
    """
    ref = corrected.reference_for(model)
    return ref if ref and ref in cfg.models.names else REFERENCE


def _delta_rows(preds_by_key: dict, classes, cfg, vocab_by_key) -> list[dict]:
    """Paired model-minus-reference deltas wherever both scored exactly the same rows.

    ``deltas.csv`` carries the reference per row (:func:`_reference_model`), so two
    arms measured against different references are never read as one column.
    """
    rows: list[dict] = []
    draws: dict[tuple, list] = {}
    for (seed, model, protocol, level), pred in preds_by_key.items():
        if model == MAJORITY:
            continue
        reference = _reference_model(model, cfg)
        if model == reference:
            continue
        ref = preds_by_key.get((seed, reference, protocol, level))
        if not ref or len(ref["y_true"]) != len(pred["y_true"]):
            continue
        if not np.array_equal(ref["y_true"], pred["y_true"]):
            continue  # different row order; not comparable
        base = {
            "substrate": cfg.data.substrate,
            "model": model,
            "reference": reference,
            "seed": seed,
            "protocol": protocol,
            "level": level,
        }
        drows, ddraws = paired_delta_rows(
            pred,
            pred["y_pred"],
            ref["y_pred"],
            base,
            classes,
            cfg,
            vocab_by_key.get((protocol, level), {}),
            return_draws=True,
        )
        rows.extend(drows)
        for (scope, cls), d in ddraws.items():
            draws.setdefault((model, protocol, level, scope, cls), []).append(d)
    attach_pooled(rows, draws, cfg, with_p=True, group_fields=("model", *POOL_KEY))
    return rows


def prediction_path(cfg, model: str, seed, protocol: str, level: str):
    """Where one pooled prediction vector is cached.

    Named here rather than formatted at the call site because ``integrate`` reads
    these files back: two places formatting the same stem independently is how the
    reader and the writer drift apart.
    """
    return cfg.sub("eval", "predictions") / f"{model}__seed-{seed}__{protocol}__{level}.npz"


def _save_predictions(cfg, preds_by_key: dict, fingerprints: dict) -> None:
    d = cfg.sub("eval", "predictions")
    index = []
    for (seed, model, protocol, level), pred in preds_by_key.items():
        path = prediction_path(cfg, model, seed, protocol, level)
        stem = path.stem
        # `refit_per_fold` refits the representation inside every fold, so these
        # predictions are not a function of the stored embedding at all. Recording
        # it lets a reader refuse rather than compare against a matrix that never
        # produced them.
        fp = "" if cfg.train.refit_per_fold else fingerprints.get((seed, model), "")
        np.savez_compressed(
            path,
            y_true=pred["y_true"],
            y_pred=pred["y_pred"],
            donor=pred["donor"],
            sample_id=pred["sample_id"],
            tissue=pred["tissue"],
            embedding_fingerprint=np.array(fp),
            refit_per_fold=np.array(bool(cfg.train.refit_per_fold)),
        )
        index.append(
            {
                "file": f"{stem}.npz",
                "model": model,
                "seed": seed,
                "protocol": protocol,
                "level": level,
                "n": int(len(pred["y_true"])),
                "embedding_fingerprint": fp,
                "refit_per_fold": bool(cfg.train.refit_per_fold),
            }
        )
    (d / "index.json").write_text(json.dumps(index, indent=2))


def _print_headline(results: pd.DataFrame, cfg) -> None:
    if results.empty:
        print("  no results produced")
        return
    view = results[(results.scope == "global") & (results.fold == "pooled")]
    if view.empty:
        return
    piv = (
        view.groupby(["protocol", "level", "model"])["f1_score"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "macro_f1", "std": "sd_over_seeds", "count": "n_seeds"})
    )
    print("\n  macro-F1 (annotated spots, pooled over folds):")
    print(piv.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
