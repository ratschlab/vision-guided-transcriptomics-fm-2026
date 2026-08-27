"""Hierarchical folds: what each level holds out, and what it must not.

A fold whose "held-out" donor also appears in the training set produces a number
that looks like generalisation and is not. Every level asserts its own invariant.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from vgtfm.config import load_config
from vgtfm.data import folds as folds_mod
from vgtfm.data.folds import LEVELS, FoldSpec
from vgtfm.data.tables import SpotTable, parse_tupro_sample_id


@pytest.fixture
def tupro_ids():
    """3 donors x 2 regions x 2 replicates — the TuPro structure in miniature."""
    return [f"{d}-{r}-{k}" for d in ("A", "B", "C") for r in (1, 2) for k in (1, 2)]


# ── leakage invariants ───────────────────────────────────────────────


def test_no_level_ever_leaks_an_evaluated_slide(tupro_ids):
    for level, specs in folds_mod.generate(tupro_ids, verbose=False).items():
        assert specs, f"{level} produced no folds"
        for spec in specs:
            overlap = set(spec.train_sample_ids) & set(spec.eval_sample_ids)
            assert not overlap, f"{spec.name} leaks {overlap}"
            assert spec.train_sample_ids and spec.eval_sample_ids


def test_cross_donor_holds_out_the_whole_patient(tupro_ids):
    """The zero-shot level: no slide of the evaluated donor may be trained on."""
    specs = folds_mod.generate(tupro_ids, verbose=False)["cross_donor"]
    assert len(specs) == 3  # one per donor
    for spec in specs:
        train = {parse_tupro_sample_id(s).donor for s in spec.train_sample_ids}
        held = {parse_tupro_sample_id(s).donor for s in spec.eval_sample_ids}
        assert not (train & held)
        assert len(held) == 1
        assert len(spec.eval_sample_ids) == 4  # all 4 of that donor


def test_cross_region_holds_out_a_block_but_keeps_the_patient(tupro_ids):
    for spec in folds_mod.generate(tupro_ids, verbose=False)["cross_region"]:
        train = {parse_tupro_sample_id(s) for s in spec.train_sample_ids}
        held = {parse_tupro_sample_id(s) for s in spec.eval_sample_ids}
        assert {p.donor for p in train} == {p.donor for p in held}
        assert not {p.region for p in train} & {p.region for p in held}


def test_cross_replicate_holds_out_a_capture_but_keeps_the_block(tupro_ids):
    for spec in folds_mod.generate(tupro_ids, verbose=False)["cross_replicate"]:
        train = {parse_tupro_sample_id(s) for s in spec.train_sample_ids}
        held = {parse_tupro_sample_id(s) for s in spec.eval_sample_ids}
        assert {(p.donor, p.region) for p in train} == {(p.donor, p.region) for p in held}


def test_the_levels_are_ordered_from_easiest_to_hardest(tupro_ids):
    """Each level holds out strictly more than the one before it."""
    out = folds_mod.generate(tupro_ids, verbose=False)
    held = {lvl: max(len(s.eval_sample_ids) for s in specs) for lvl, specs in out.items() if specs}
    assert held["cross_replicate"] < held["cross_region"] < held["cross_donor"]


# ── cohorts without replicate structure ──────────────────────────────


def test_cross_slide_covers_single_slide_per_patient_cohorts():
    ids = ["KC1", "KC3", "LC1", "LC2"]
    tissue = {"KC1": "kidney", "KC3": "kidney", "LC1": "lung", "LC2": "lung"}
    specs = folds_mod.generate(ids, levels=("cross_slide",), tissue_of=tissue, verbose=False)[
        "cross_slide"
    ]
    assert len(specs) == 4
    for spec in specs:
        held = spec.eval_sample_ids[0]
        assert held not in spec.train_sample_ids
        assert {tissue[s] for s in spec.train_sample_ids} == {tissue[held]}


def test_a_tissue_with_one_slide_yields_no_cross_slide_fold():
    specs = folds_mod.generate(
        ["KC1", "LC1", "LC2"],
        levels=("cross_slide",),
        tissue_of={"KC1": "kidney", "LC1": "lung", "LC2": "lung"},
        verbose=False,
    )["cross_slide"]
    assert {s.eval_sample_ids[0] for s in specs} == {"LC1", "LC2"}


def test_non_tupro_ids_enter_cross_donor_but_not_the_structured_levels():
    """USZ has a patient per slide, so cross_donor applies; region/replicate do not."""
    out = folds_mod.generate(["KC1", "KC3"], tissue_of={"KC1": "k", "KC3": "k"}, verbose=False)
    assert out["cross_region"] == [] and out["cross_replicate"] == []
    assert len(out["cross_donor"]) == 2
    assert len(out["cross_slide"]) == 2
    # One slide per patient makes the two levels the same split.
    assert [(s.train_sample_ids, s.eval_sample_ids) for s in out["cross_donor"]] == [
        (s.train_sample_ids, s.eval_sample_ids) for s in out["cross_slide"]
    ]


def test_mixed_cohorts_split_across_the_right_levels():
    ids = ["A-1-1", "A-1-2", "B-1-1", "KC1", "KC3"]
    out = folds_mod.generate(
        ids,
        tissue_of={
            "KC1": "kidney",
            "KC3": "kidney",
            "A-1-1": "mel",
            "A-1-2": "mel",
            "B-1-1": "mel",
        },
        verbose=False,
    )
    # Every cohort enters cross_donor: 2 TuPro patients + 2 USZ patients.
    assert len(out["cross_donor"]) == 4
    # ... but the structured levels still need a region and a replicate to hold out.
    structured = {
        s
        for lvl in ("cross_replicate", "cross_region")
        for spec in out[lvl]
        for s in (*spec.train_sample_ids, *spec.eval_sample_ids)
    }
    assert not structured & {"KC1", "KC3"}
    assert len(out["cross_slide"]) == 5  # all of them, per tissue


def test_cross_donor_never_trains_on_another_organ():
    """The probe is fitted within tissue; a fold that crossed organs would predict
    classes that cannot occur in the evaluated slide."""
    tissue = {
        "A-1-1": "mel",
        "A-1-2": "mel",
        "B-1-1": "mel",
        "KC1": "kidney",
        "KC3": "kidney",
        "LC1": "lung",
        "LC2": "lung",
    }
    for spec in folds_mod.generate(sorted(tissue), tissue_of=tissue, verbose=False)["cross_donor"]:
        organs = {tissue[s] for s in (*spec.train_sample_ids, *spec.eval_sample_ids)}
        assert len(organs) == 1, f"{spec.name} mixes {organs}"


def test_cross_donor_holds_out_the_patient_not_the_slide_for_usz():
    """Each USZ slide is a different patient, so nothing of it may remain behind."""
    tissue = {s: "lung" for s in ("LC1", "LC2", "LC3", "LC5")}
    for spec in folds_mod.generate(sorted(tissue), tissue_of=tissue, verbose=False)["cross_donor"]:
        assert len(spec.eval_sample_ids) == 1
        assert not set(spec.train_sample_ids) & set(spec.eval_sample_ids)
        assert len(spec.train_sample_ids) == 3


def test_the_default_patient_key_does_not_split_a_tupro_donor(tupro_ids):
    """Regression guard: a bare per-slide fallback would turn 3 patients into 12 and
    make cross_donor a leave-one-slide-out that keeps the replicate."""
    specs = folds_mod.generate(tupro_ids, verbose=False)["cross_donor"]
    assert len(specs) == 3
    assert all(len(s.eval_sample_ids) == 4 for s in specs)


def test_an_explicit_donor_map_overrides_the_parsed_default():
    """The cohort's own key wins: De Zuani's P10-* slides are one patient."""
    ids = ["P10-B1", "P10-B2", "P10-T1", "Q4-B1"]
    donor = {"P10-B1": "P10", "P10-B2": "P10", "P10-T1": "P10", "Q4-B1": "Q4"}
    specs = folds_mod.generate(
        ids, donor_of=donor, tissue_of={s: "lung" for s in ids}, verbose=False
    )["cross_donor"]
    assert len(specs) == 2
    held = {s.name: set(s.eval_sample_ids) for s in specs}
    assert held["cross_donor__lung__eval_P10"] == {"P10-B1", "P10-B2", "P10-T1"}


def test_the_cross_donor_cap_is_applied_per_tissue():
    """A cohort-wide cap would let the largest cohort crowd out a smaller one."""
    tissue = {**{f"D{i}-1-1": "mel" for i in range(8)}, "KC1": "kidney", "KC3": "kidney"}
    specs = folds_mod.generate(
        sorted(tissue), tissue_of=tissue, max_cross_donor_splits=3, seed=42, verbose=False
    )["cross_donor"]
    per_tissue = Counter(s.name.split("__")[1] for s in specs)
    assert per_tissue == {"mel": 3, "kidney": 2}


def test_a_single_donor_cannot_produce_a_cross_donor_fold():
    assert folds_mod.generate(["A-1-1", "A-1-2"], verbose=False)["cross_donor"] == []


# ── configuration ────────────────────────────────────────────────────


def test_the_cross_donor_fold_count_can_be_capped():
    ids = [f"D{i}-1-1" for i in range(10)]
    uncapped = folds_mod.generate(ids, verbose=False)["cross_donor"]
    capped = folds_mod.generate(ids, max_cross_donor_splits=4, seed=42, verbose=False)[
        "cross_donor"
    ]
    assert len(uncapped) == 10 and len(capped) == 4
    # Capping is a deterministic subsample, not a truncation of the sorted list.
    assert (
        capped
        == folds_mod.generate(ids, max_cross_donor_splits=4, seed=42, verbose=False)["cross_donor"]
    )


def test_an_unknown_level_names_the_known_ones():
    with pytest.raises(SystemExit, match="unknown fold level"):
        folds_mod.generate(["A-1-1", "B-1-1"], levels=("cross_batch",), verbose=False)


def test_only_requested_levels_are_generated(tupro_ids):
    out = folds_mod.generate(tupro_ids, levels=("cross_donor",), verbose=False)
    assert set(out) == {"cross_donor"}


def test_fold_generation_is_deterministic(tupro_ids):
    a = folds_mod.generate(tupro_ids, verbose=False)
    b = folds_mod.generate(list(reversed(tupro_ids)), verbose=False)
    assert a == b  # input order must not matter


def test_all_declared_levels_are_generatable(tupro_ids):
    out = folds_mod.generate(
        tupro_ids, levels=LEVELS, tissue_of={s: "mel" for s in tupro_ids}, verbose=False
    )
    assert set(out) == set(LEVELS)
    assert all(specs for specs in out.values())


# ── FoldSpec ─────────────────────────────────────────────────────────


def test_eval_donors_deduplicates_and_falls_back_to_the_slide_id():
    tupro = FoldSpec("f", "cross_donor", ("B-1-1",), ("A-1-1", "A-1-2", "A-2-1"))
    assert tupro.eval_donors == ("A",)
    usz = FoldSpec("f", "cross_slide", ("KC3",), ("KC1",))
    assert usz.eval_donors == ("KC1",)


def test_a_fold_resolves_to_row_indices_in_a_table():
    meta = pd.DataFrame(
        {"sample_id": ["A-1-1"] * 3 + ["B-1-1"] * 2, "donor": ["A"] * 3 + ["B"] * 2}
    )
    table = SpotTable(
        gene=np.zeros((5, 2), dtype=np.float32),
        patch=np.zeros((5, 2), dtype=np.float32),
        meta=meta,
        substrate="x",
    )
    train, evaluate = FoldSpec("f", "cross_donor", ("A-1-1",), ("B-1-1",)).rows(table)
    assert train.tolist() == [0, 1, 2]
    assert evaluate.tolist() == [3, 4]
    assert not set(train) & set(evaluate)


def test_a_fold_serialises_for_the_run_record():
    spec = FoldSpec("f", "cross_donor", ("A-1-1",), ("B-1-1",))
    assert spec.as_dict() == {
        "name": "f",
        "level": "cross_donor",
        "train_sample_ids": ["A-1-1"],
        "eval_sample_ids": ["B-1-1"],
    }


# ── generate_for ─────────────────────────────────────────────────────


def _mixed_table():
    ids = ["A-1-1", "A-1-2", "B-1-1", "B-1-2", "KC1", "KC3"]
    meta = pd.DataFrame(
        {
            "sample_id": np.repeat(ids, 2),
            "donor": np.repeat(["A", "A", "B", "B", "KC1", "KC3"], 2),
            "tissue": np.repeat(["skin"] * 4 + ["kidney"] * 2, 2),
            "annotation": ["TUM", "Stroma"] * 6,
        }
    )
    return SpotTable(
        gene=np.zeros((12, 2), dtype=np.float32),
        patch=np.zeros((12, 2), dtype=np.float32),
        meta=meta,
        substrate="x",
    )


def test_generate_for_uses_the_tables_own_donor_column():
    """USZ enters cross_donor; its patient key is the slide id and that is correct."""
    out = folds_mod.generate_for(load_config(None, {"folds.levels": "cross_donor"}), _mixed_table())
    assert {s.name for s in out["cross_donor"]} == {
        "cross_donor__skin__eval_A",
        "cross_donor__skin__eval_B",
        "cross_donor__kidney__eval_KC1",
        "cross_donor__kidney__eval_KC3",
    }


def test_folds_can_be_pinned_to_one_organ():
    """What `regression_check` needs: reproduce a melanoma-only published protocol
    without the cohorts that were not in it."""
    cfg = load_config(None, {"folds.levels": "cross_donor", "folds.tissues": "skin"})
    out = folds_mod.generate_for(cfg, _mixed_table())
    assert {s.name for s in out["cross_donor"]} == {
        "cross_donor__skin__eval_A",
        "cross_donor__skin__eval_B",
    }


def test_pinning_to_an_absent_organ_is_refused_not_silently_empty():
    cfg = load_config(None, {"folds.levels": "cross_donor", "folds.tissues": "pancreas"})
    with pytest.raises(SystemExit, match="matches no annotated organ"):
        folds_mod.generate_for(cfg, _mixed_table())


def test_folds_are_built_only_from_annotated_slides():
    """An unannotated slide cannot be evaluated, so it must not define a fold."""
    meta = pd.DataFrame(
        {
            "sample_id": np.repeat(["A-1-1", "A-1-2", "B-1-1", "MW-B-001a-vis"], 4),
            "donor": np.repeat(["A", "A", "B", "MW-B-001a-vis"], 4),
            "tissue": np.repeat(["mel"], 16),
            "annotation": ["TUM", "STR"] * 6 + ["UNASSIGNED"] * 4,
        }
    )
    table = SpotTable(
        gene=np.zeros((16, 2), dtype=np.float32),
        patch=np.zeros((16, 2), dtype=np.float32),
        meta=meta,
        substrate="x",
    )
    out = folds_mod.generate_for(load_config(None), table)

    seen = {
        s
        for specs in out.values()
        for spec in specs
        for s in (*spec.train_sample_ids, *spec.eval_sample_ids)
    }
    assert "MW-B-001a-vis" not in seen
    assert seen == {"A-1-1", "A-1-2", "B-1-1"}
