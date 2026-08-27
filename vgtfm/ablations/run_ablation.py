"""``ablate`` stage — does the autoencoder use the gene/morphology correspondence?

Train the same autoencoder four times, changing only which patch each spot is
paired with (see :mod:`vgtfm.ablations.patch_shuffle`), and score all four with the
same probes. The seed for the permutation is kept separate from the seed for
network initialisation, so matched and shuffled runs start from identical weights
and the only difference between them is the correspondence.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import perf
from ..data import folds as folds_mod
from ..data import tables
from ..evaluate import protocol as proto
from ..evaluate.reporting import (
    POOL_KEY,
    attach_pooled,
    bootstrap_rows,
    metric_rows,
    paired_delta_rows,
    pool,
    score_vocabulary,
)
from ..labels import cohort_classes
from ..models.base import Inputs
from ..models.train import fit_and_embed, fit_rows
from .patch_shuffle import PATCH_TRANSFORMS

#: The condition the others are compared against.
REFERENCE_CONDITION = "none"


def run(cfg) -> None:
    out = cfg.sub("ablation")
    table = tables.load(cfg)
    inputs = Inputs.from_table(table)
    rows = fit_rows(cfg, table)
    fold_map = folds_mod.generate_for(cfg, table)

    classes = cohort_classes(table)

    conditions = [c for c in cfg.ablation.conditions if c in PATCH_TRANSFORMS]
    unknown = set(cfg.ablation.conditions) - set(PATCH_TRANSFORMS)
    if unknown:
        print(f"  ignoring unknown condition(s) {sorted(unknown)}; known: {PATCH_TRANSFORMS}")

    model_name = cfg.ablation.model
    print(
        f"  {len(conditions)} condition(s) x {len(cfg.seeds)} seed(s) "
        f"on '{model_name}', fitted on {len(rows):,} spots"
    )

    macro_rows: list[dict] = []
    class_rows: list[dict] = []
    boot_rows: list[dict] = []
    preds_by_key: dict[tuple, dict] = {}
    vocab_by_key: dict[tuple[str, str], dict] = {}
    histories: list[dict] = []
    # See reporting.pooled_columns: the published interval pools the seeds'
    # replicate draws rather than averaging their separate bounds.
    boot_draws: dict[tuple, list] = {}

    # The stage mutates the shared model config to select each condition; restore it
    # afterwards even on failure, so a later stage in the same process cannot be
    # silently trained against shuffled morphology.
    original = (cfg.models.ae.patch_transform, cfg.models.ae.shuffle_seed)
    try:
        _train_conditions(
            cfg,
            conditions,
            model_name,
            inputs,
            rows,
            table,
            fold_map,
            classes,
            macro_rows,
            class_rows,
            boot_rows,
            preds_by_key,
            vocab_by_key,
            histories,
            boot_draws,
        )
    finally:
        cfg.models.ae.patch_transform, cfg.models.ae.shuffle_seed = original

    attach_pooled(boot_rows, boot_draws, cfg, with_p=False, group_fields=("condition", *POOL_KEY))

    results = pd.DataFrame(macro_rows)
    per_class = pd.DataFrame(class_rows)
    results.to_csv(out / "results.csv", index=False)
    per_class.to_csv(out / "per_class.csv", index=False)
    pd.DataFrame(histories).to_csv(out / "training_history.csv", index=False)
    if boot_rows:
        pd.DataFrame(boot_rows).to_csv(out / "bootstrap.csv", index=False)

    delta_rows = _paired_deltas(preds_by_key, vocab_by_key, classes, cfg)
    if delta_rows:
        pd.DataFrame(delta_rows).to_csv(out / "paired_deltas.csv", index=False)

    (out / "summary.json").write_text(
        json.dumps(
            {
                "substrate": cfg.data.substrate,
                "model": model_name,
                "conditions": conditions,
                "seeds": list(cfg.seeds),
            },
            indent=2,
        )
    )
    _print_summary(results, per_class)
    print(
        f"\n  wrote {out}/results.csv, per_class.csv"
        f"{', bootstrap.csv' if boot_rows else ''}, paired_deltas.csv"
    )


def _train_conditions(
    cfg,
    conditions,
    model_name,
    inputs,
    rows,
    table,
    fold_map,
    classes,
    macro_rows,
    class_rows,
    boot_rows,
    preds_by_key,
    vocab_by_key,
    histories,
    boot_draws,
) -> None:
    """Train and score one autoencoder per (condition, seed)."""
    for seed in cfg.seeds:
        for condition in conditions:
            print(f"\n  -- {condition} (seed {seed}) --")
            # The permutation seed follows the run seed, so the spread across seeds
            # covers both initialisation and permutation.
            cfg.models.ae.patch_transform = condition
            cfg.models.ae.shuffle_seed = seed
            model, Z, history = fit_and_embed(cfg, inputs, rows, model_name, seed)
            histories.append(
                {
                    "condition": condition,
                    "seed": seed,
                    **{k: v for k, v in history.items() if not isinstance(v, list)},
                }
            )
            del model

            for protocol in cfg.eval.protocols:
                if proto.is_fold_protocol(protocol):
                    for level, specs in fold_map.items():
                        per_fold = []
                        for spec in specs:
                            p = proto.run(protocol, Z, table.meta, cfg=cfg, fold=spec)
                            if p:
                                p["fold"] = spec.name
                                per_fold.append(p)
                        pooled = pool(per_fold)
                        if pooled:
                            _record(
                                pooled,
                                protocol,
                                level,
                                condition,
                                seed,
                                cfg,
                                classes,
                                vocab_by_key,
                                macro_rows,
                                class_rows,
                                boot_rows,
                                preds_by_key,
                                boot_draws,
                            )
                else:
                    p = proto.run(protocol, Z, table.meta, cfg=cfg)
                    if p:
                        _record(
                            p,
                            protocol,
                            "all",
                            condition,
                            seed,
                            cfg,
                            classes,
                            vocab_by_key,
                            macro_rows,
                            class_rows,
                            boot_rows,
                            preds_by_key,
                            boot_draws,
                        )
            del Z
            perf.free_cuda()


def _record(
    pred,
    protocol,
    level,
    condition,
    seed,
    cfg,
    classes,
    vocab_by_key,
    macro_rows,
    class_rows,
    boot_rows,
    preds_by_key,
    boot_draws,
) -> None:
    vocab = vocab_by_key.setdefault((protocol, level), score_vocabulary(pred))
    base = {
        "substrate": cfg.data.substrate,
        "model": cfg.ablation.model,
        "condition": condition,
        "seed": seed,
        "protocol": protocol,
        "level": level,
        "fold": "pooled",
    }
    m_rows, c_rows = metric_rows(pred, classes, base, vocab)
    macro_rows.extend(m_rows)
    class_rows.extend(c_rows)
    # Absolute per-class intervals. Read them next to `paired_deltas.csv`: these
    # overlap heavily between conditions at 7 and 6 patients, while the paired
    # difference on a shared donor resample still resolves.
    if cfg.eval.bootstrap_n > 0:
        brows, bdraws = bootstrap_rows(pred, classes, base, cfg, vocab, return_draws=True)
        boot_rows.extend(brows)
        for (scope, cls), d in bdraws.items():
            boot_draws.setdefault((condition, protocol, level, scope, cls), []).append(d)
    preds_by_key[(seed, condition, protocol, level)] = pred


def _paired_deltas(preds_by_key, vocab_by_key, classes, cfg) -> list[dict]:
    """Matched-minus-shuffled deltas, resampling the same donors for both.

    The reference is passed first, so a positive delta means the real
    gene/morphology correspondence helped.
    """
    rows: list[dict] = []
    draws: dict[tuple, list] = {}
    for (seed, condition, protocol, level), pred in preds_by_key.items():
        if condition == REFERENCE_CONDITION:
            continue
        ref = preds_by_key.get((seed, REFERENCE_CONDITION, protocol, level))
        if not ref or not np.array_equal(ref["y_true"], pred["y_true"]):
            continue
        base = {
            "substrate": cfg.data.substrate,
            "condition": condition,
            "reference": REFERENCE_CONDITION,
            "seed": seed,
            "protocol": protocol,
            "level": level,
        }
        drows, ddraws = paired_delta_rows(
            pred,
            ref["y_pred"],
            pred["y_pred"],
            base,
            classes,
            cfg,
            vocab_by_key.get((protocol, level), {}),
            return_draws=True,
        )
        rows.extend(drows)
        for (scope, cls), d in ddraws.items():
            draws.setdefault((condition, protocol, level, scope, cls), []).append(d)
    attach_pooled(rows, draws, cfg, with_p=True, group_fields=("condition", *POOL_KEY))
    return rows


def _print_summary(results: pd.DataFrame, per_class: pd.DataFrame) -> None:
    if results.empty:
        print("  no ablation results produced")
        return
    view = results[results.scope != "global"]
    if view.empty:
        view = results
    piv = view.pivot_table(
        index=["protocol", "level", "scope"], columns="condition", values="f1_score", aggfunc="mean"
    )
    print("\n  macro-F1 by condition:")
    print(piv.to_string(float_format=lambda x: f"{x:.4f}"))
