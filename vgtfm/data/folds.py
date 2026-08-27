"""Hierarchical held-out splits over the annotated cohorts.

Four notions of "held out", by how much of the evaluated slide stays in training:

``cross_replicate``  hold out one technical replicate of a (donor, region) pair —
                     same patient, same tissue block, different capture.
``cross_region``     hold out a whole region of a donor — same patient, different
                     tissue block.
``cross_slide``      hold out one annotated slide, training on the tissue's others.
                     On TuPro that keeps the patient, via the held-out slide's own
                     replicate; on a cohort with one slide per patient it does not.
``cross_donor``      hold out a patient entirely — the zero-shot setting, and the
                     only level that keeps nothing back on every cohort.

The first two need a replicate and a region to hold out, which only TuPro's
``DONOR-REGION-REPLICATE`` slide ids supply; USZ records ``region = replicate = -1``
because there is no second capture of a block and no second block per patient. That
is a property of the cohort, not of the id format, so those two levels stay
TuPro-only.

``cross_donor`` and ``cross_slide`` need only a patient key and an organ, both of
which every cohort carries in ``cohort.csv``. USZ's kidney and lung slides are one
patient each, so leave-one-slide-out and leave-one-donor-out coincide there and
neither leaks a replicate. Both levels are therefore built from the table's own
``donor`` / ``tissue`` columns rather than by parsing slide ids.

Every level groups **within tissue**, because the probe is fitted within tissue: the
cohorts were annotated independently and do not share a class vocabulary, so a fold
trained on another organ would be predicting classes that cannot occur.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .tables import TuProSampleID, parse_tupro_sample_id

LEVELS = ("cross_replicate", "cross_region", "cross_donor", "cross_slide")

#: Levels that need a replicate or a region to hold out, and so can only be built
#: from ids that encode one. See the module docstring: this is a limit of what the
#: cohort captured, not of what the ids can be parsed into.
STRUCTURED_LEVELS = ("cross_replicate", "cross_region")


@dataclass(frozen=True)
class FoldSpec:
    """One held-out split: which slides train the representation, which are scored."""

    name: str
    level: str
    train_sample_ids: tuple[str, ...]
    eval_sample_ids: tuple[str, ...]

    @property
    def eval_donors(self) -> tuple[str, ...]:
        donors = []
        for s in self.eval_sample_ids:
            p = parse_tupro_sample_id(s)
            donors.append(p.donor if p else s)
        return tuple(dict.fromkeys(donors))

    def rows(self, table) -> tuple[np.ndarray, np.ndarray]:
        """Resolve the fold to ``(train_rows, eval_rows)`` in a :class:`SpotTable`."""
        return (
            table.rows_for_samples(self.train_sample_ids),
            table.rows_for_samples(self.eval_sample_ids),
        )

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "level": self.level,
            "train_sample_ids": list(self.train_sample_ids),
            "eval_sample_ids": list(self.eval_sample_ids),
        }


# ── generators ───────────────────────────────────────────────────────


def _cross_replicate(samples: list[TuProSampleID]) -> list[FoldSpec]:
    groups: dict[tuple[str, int], list[TuProSampleID]] = defaultdict(list)
    for s in samples:
        groups[(s.donor, s.region)].append(s)

    folds = []
    for (donor, region), members in sorted(groups.items()):
        if len(members) < 2:
            continue
        for held_out in members:
            train = tuple(m.raw for m in members if m.raw != held_out.raw)
            folds.append(
                FoldSpec(
                    name=f"cross_rep__{donor}-{region}__eval_{held_out.replicate}",
                    level="cross_replicate",
                    train_sample_ids=train,
                    eval_sample_ids=(held_out.raw,),
                )
            )
    return folds


def _cross_region(samples: list[TuProSampleID]) -> list[FoldSpec]:
    by_donor: dict[str, dict[int, list[TuProSampleID]]] = defaultdict(lambda: defaultdict(list))
    for s in samples:
        by_donor[s.donor][s.region].append(s)

    folds = []
    for donor in sorted(by_donor):
        regions = by_donor[donor]
        if len(regions) < 2:
            continue
        for held_out_region in sorted(regions):
            train = tuple(
                s.raw
                for r, members in sorted(regions.items())
                if r != held_out_region
                for s in members
            )
            eval_ids = tuple(s.raw for s in regions[held_out_region])
            folds.append(
                FoldSpec(
                    name=f"cross_reg__{donor}__eval_region_{held_out_region}",
                    level="cross_region",
                    train_sample_ids=train,
                    eval_sample_ids=eval_ids,
                )
            )
    return folds


def _group_by_tissue_then(sample_ids, key_of: dict, tissue_of: dict):
    """``{tissue: {key: [slide, ...]}}``, deterministic in the input order."""
    out: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for sid in sorted(set(map(str, sample_ids))):
        out[tissue_of.get(sid, "unknown")][key_of.get(sid, sid)].append(sid)
    return out


def _cross_donor(
    sample_ids, donor_of: dict, tissue_of: dict, max_splits: int, seed: int
) -> list[FoldSpec]:
    """Hold out one patient at a time, training on the same tissue's other patients.

    Needs a patient key and an organ, nothing else, so every cohort enters — not
    only the one whose slide ids happen to encode the structure. On USZ each slide
    is a different patient, which makes this level identical to
    :func:`_cross_slide` there and leaves no technical replicate behind to leak.

    ``max_splits`` caps the folds *per tissue*. A cohort-wide cap would let the
    largest cohort crowd out a smaller one's folds entirely, which is the opposite
    of what a cap is for.
    """
    folds = []
    for tissue, by_donor in sorted(_group_by_tissue_then(sample_ids, donor_of, tissue_of).items()):
        donors = sorted(by_donor)
        if len(donors) < 2:
            continue
        if max_splits and len(donors) > max_splits:
            donors = sorted(random.Random(seed).sample(donors, max_splits))
        for held in donors:
            folds.append(
                FoldSpec(
                    name=f"cross_donor__{tissue}__eval_{held}",
                    level="cross_donor",
                    train_sample_ids=tuple(
                        s for d in sorted(by_donor) if d != held for s in by_donor[d]
                    ),
                    eval_sample_ids=tuple(by_donor[held]),
                )
            )
    return folds


def _cross_slide(sample_ids, tissue_of: dict[str, str]) -> list[FoldSpec]:
    """Leave one annotated slide out, training on same-tissue slides.

    Cohort-agnostic: it needs no id structure, only a tissue grouping, so it is the
    level that covers USZ and any future single-slide-per-patient cohort.
    """
    folds = []
    for tissue, by_slide in sorted(_group_by_tissue_then(sample_ids, {}, tissue_of).items()):
        slides = sorted(by_slide)
        if len(slides) < 2:
            continue
        for held_out in slides:
            folds.append(
                FoldSpec(
                    name=f"cross_slide__{tissue}__eval_{held_out}",
                    level="cross_slide",
                    train_sample_ids=tuple(s for s in slides if s != held_out),
                    eval_sample_ids=(held_out,),
                )
            )
    return folds


def generate(
    sample_ids,
    *,
    levels=LEVELS,
    max_cross_donor_splits: int = 10,
    seed: int = 42,
    tissue_of: dict[str, str] | None = None,
    donor_of: dict[str, str] | None = None,
    verbose: bool = True,
) -> dict[str, list[FoldSpec]]:
    """Build every requested fold level from a list of slide ids.

    ``donor_of`` and ``tissue_of`` map slide id to patient key and organ; entries
    given by the caller win over the defaults below, per slide.

    The default patient key is :func:`parse_tupro_sample_id`'s donor where the id
    encodes one and the slide id otherwise — the same rule
    :attr:`~vgtfm.data.tables.SpotTable.donor` applies, and the only safe one: a
    bare per-slide fallback would quietly turn TuPro's 18 slides into 18 "patients"
    and hand ``cross_donor`` a leave-one-slide-out that leaks the replicate. Tissue
    defaults to a single unknown organ, which groups everything together.
    """
    ids = sorted(set(map(str, sample_ids)))
    tissue_of = dict(tissue_of or {})

    parsed, skipped = [], []
    default_donor = {}
    for sid in ids:
        p = parse_tupro_sample_id(sid)
        (parsed if p else skipped).append(p or sid)
        default_donor[sid] = p.donor if p else sid
    donor_of = {**default_donor, **dict(donor_of or {})}

    wanted_structured = [lv for lv in levels if lv in STRUCTURED_LEVELS]
    if verbose:
        if skipped and wanted_structured:
            print(
                f"  folds: {len(skipped)} slide(s) carry no region/replicate and "
                f"enter every level except {', '.join(wanted_structured)}"
            )
        donors = {donor_of.get(sid, sid) for sid in ids}
        tissues = {tissue_of.get(sid, "unknown") for sid in ids}
        regions = {(s.donor, s.region) for s in parsed}
        print(
            f"  folds: {len(ids)} slides, {len(donors)} donors, "
            f"{len(regions)} regions, {len(tissues)} tissue(s)"
        )

    out: dict[str, list[FoldSpec]] = {}
    for lvl in levels:
        if lvl == "cross_replicate":
            out[lvl] = _cross_replicate(parsed)
        elif lvl == "cross_region":
            out[lvl] = _cross_region(parsed)
        elif lvl == "cross_donor":
            out[lvl] = _cross_donor(ids, donor_of, tissue_of, max_cross_donor_splits, seed)
        elif lvl == "cross_slide":
            out[lvl] = _cross_slide(ids, tissue_of)
        else:
            raise SystemExit(f"unknown fold level '{lvl}' (known: {LEVELS})")
        if verbose:
            print(f"    {lvl}: {len(out[lvl])} fold(s)")
            # A tissue with two patients gives folds whose probe sees one slide.
            # That is a real fold and not a leak, but it is an anecdote, and it
            # belongs in the log rather than being discovered from an odd number.
            # Only worth saying where the level means to train on several patients:
            # holding out one of two replicates leaves exactly one slide by
            # definition, and saying so on all 18 of them is noise.
            thin = (
                [f.name for f in out[lvl] if len(f.train_sample_ids) < 2]
                if lvl not in STRUCTURED_LEVELS
                else []
            )
            if thin:
                print(
                    f"      {len(thin)} of them fit the probe on a single slide, "
                    f"so read them as anecdotes: {', '.join(thin)}"
                )
    return out


def generate_for(cfg, table) -> dict[str, list[FoldSpec]]:
    """Folds for the slides actually present in *table*, per ``cfg.folds``."""
    from ..labels import labeled_mask

    # A slide with no annotation cannot be evaluated, so it must not define a fold.
    meta = table.meta
    annotated = meta.loc[labeled_mask(meta["annotation"].to_numpy())]
    if cfg.folds.tissues:
        keep = annotated["tissue"].astype(str).isin(set(cfg.folds.tissues))
        if not keep.any():
            raise SystemExit(
                f"folds.tissues={list(cfg.folds.tissues)} matches no annotated "
                f"organ (present: {sorted(set(annotated['tissue'].astype(str)))})"
            )
        annotated = annotated.loc[keep]
    ids = annotated["sample_id"].astype(str)
    tissue_of = dict(zip(ids, annotated["tissue"].astype(str)))
    # `donor` is the patient key the cohort itself records — the TuPro donor where
    # the id encodes one, the slide id where each slide is its own patient. Falling
    # back to the slide id keeps a table built before the column existed working,
    # and is the same answer for every single-slide-per-patient cohort.
    donor_of = dict(zip(ids, annotated["donor"].astype(str))) if "donor" in annotated else {}
    return generate(
        ids.unique(),
        levels=tuple(cfg.folds.levels),
        max_cross_donor_splits=cfg.folds.max_cross_donor_splits,
        seed=cfg.folds.seed,
        tissue_of=tissue_of,
        donor_of=donor_of,
    )
