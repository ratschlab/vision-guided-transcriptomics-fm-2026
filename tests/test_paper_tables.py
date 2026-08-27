"""``scripts/paper_tables.py`` — the last join, and the gate in front of it.

The tables in the manuscript are the one output no stage rebuilds, and they drifted
a whole batch behind their own source once already. Two things are pinned here: the
consistency gate refuses in both of the ways it can (a cell computed twice with two
values, and an alias that compared nothing at all), and the per-organ table reads
the organ columns off ``integration.csv`` without mistaking an interval suffix for a
fold level.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from conftest import ROOT


def _module():
    """Import the script by path: ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location(
        "paper_tables", ROOT / "scripts" / "paper_tables.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pt = _module()


def _run(
    root: Path, name: str, *, eval_f1: float, integrate_f1: float, panel_seed: int = 42
) -> Path:
    """One run directory carrying the same cell in three stages."""
    d = root / name
    (d / "eval").mkdir(parents=True)
    (d / "integration").mkdir(parents=True)
    (d / "diagnostics").mkdir(parents=True)
    (d / "config.resolved.json").write_text(
        json.dumps({"run_name": name, "data": {"substrate": "sub"}, "diagnostics": {"seed": 42}})
    )
    pd.DataFrame(
        [
            {
                "substrate": "sub",
                "model": "pca",
                "seed": 42,
                "protocol": "heldout_donor",
                "level": "cross_donor",
                "fold": "pooled",
                "scope": "kidney",
                "f1_score": eval_f1,
                "n_donors": 2,
            }
        ]
    ).to_csv(d / "eval" / "results.csv", index=False)
    pd.DataFrame(
        [
            {
                "method": "none",
                "batch/kbet": 0.15,
                "f1/kidney/cross_donor": integrate_f1,
                "f1/kidney/cross_donor_lo": 0.3,
                "f1/kidney/cross_donor_hi": 0.5,
                "f1/kidney/cross_donor_n_donors": 2,
            }
        ]
    ).to_csv(d / "integration" / "integration.csv", index=False)
    pd.DataFrame(
        [{"substrate": "sub", "representation": f"pca (seed {panel_seed})", "batch/kbet": 0.15}]
    ).to_csv(d / "diagnostics" / "scib_panel.csv", index=False)
    return d


def test_the_gate_refuses_to_write_a_table_the_stages_disagree_on(tmp_path):
    """The failure it exists for: `integrate`'s uncorrected row and Table 1's PCA row
    describing two different representations, in one manuscript, with nothing able to
    compare them."""
    _run(tmp_path, "a", eval_f1=0.4496, integrate_f1=0.3738)

    with pytest.raises(SystemExit, match="computed by two stages"):
        pt._check_consistency(tmp_path, ["a"])


def test_the_gate_refuses_a_check_that_compared_nothing(tmp_path):
    """A check that quietly stopped checking prints the same clean line as one that
    checked everything, and only one of them says the manuscript is safe."""
    _run(tmp_path, "a", eval_f1=0.45, integrate_f1=0.45, panel_seed=99)

    with pytest.raises(SystemExit, match="compared nothing"):
        pt._check_consistency(tmp_path, ["a"])


def test_the_gate_passes_a_run_whose_stages_agree(tmp_path, capsys):
    """The guards above must not turn every run into a refusal — and the line they
    print has to carry the number of comparisons, not just the verdict."""
    _run(tmp_path, "a", eval_f1=0.45, integrate_f1=0.45)
    pt._check_consistency(tmp_path, ["a"])

    out = capsys.readouterr().out
    assert "computed more than once and all in agreement" in out
    assert "0 computed more than once" not in out


def test_the_organ_columns_are_read_off_the_table_not_hardcoded():
    """`f1/<scope>/<level>` folds the scope and the level into a column name, with
    the interval hung off it as `_lo`/`_hi`/`_n_donors`. Matching on the shape alone
    reads `f1/kidney/cross_donor_lo` as a fold level named `cross_donor_lo`.
    """
    wide = pd.DataFrame(
        [
            {
                "method": "none",
                "f1/global/cross_donor": 0.35,
                "f1/global/cross_donor_lo": 0.2,
                "f1/global/cross_donor_hi": 0.4,
                "f1/global/cross_donor_n_donors": 13,
                "f1/kidney/cross_donor": 0.45,
                "f1/kidney/cross_donor_lo": 0.3,
                "f1/kidney/cross_donor_hi": 0.5,
                "f1/kidney/cross_donor_n_donors": 2,
                "f1/skin/cross_donor": 0.52,
                "f1/skin/cross_donor_lo": 0.4,
                "f1/skin/cross_donor_hi": 0.6,
                "f1/skin/cross_donor_n_donors": 7,
                "f1/skin/cross_replicate": 0.61,
                "f1/skin/cross_replicate_lo": 0.5,
                "f1/skin/cross_replicate_hi": 0.7,
                "f1/skin/cross_replicate_n_donors": 7,
            }
        ]
    ).set_index("method")

    # Ordered by donor count ascending, and `global` dropped: this table exists
    # because the pooled macro hides the organs.
    assert pt._organ_scopes(wide, "cross_donor") == [("kidney", 2), ("skin", 7)]
    assert pt._organ_scopes(wide, "cross_replicate") == [("skin", 7)]
    assert pt._organ_scopes(wide, "cross_region") == []


# ── the seeds a table averages over ──────────────────────────────────


def _wide(seeds, f1s, *, pooled=True) -> pd.DataFrame:
    """`integration.csv` as `integrate` writes it: one row per (method, seed)."""
    rows = []
    for s, f1 in zip(seeds, f1s):
        r = {
            "method": "none",
            "seed": s,
            "dim": 32,
            "note": None,
            "batch/ilisi": 0.04 + 0.001 * s,
            "f1/kidney/cross_donor": f1,
            "f1/kidney/cross_donor_lo": f1 - 0.10,
            "f1/kidney/cross_donor_hi": f1 + 0.10,
            "f1/kidney/cross_donor_n_donors": 2,
        }
        if pooled:
            # Constant within a method, as `_attach_pooled_probe` writes it.
            r |= {
                "f1/kidney/cross_donor_lo_pooled": 0.20,
                "f1/kidney/cross_donor_hi_pooled": 0.70,
                "f1/kidney/cross_donor_n_seeds": len(seeds),
            }
        rows.append(r)
    return pd.DataFrame(rows)


def test_a_method_is_one_row_however_many_fits_it_was_scored_on():
    """`integrate` scores every seed `eval` scores; the manuscript reports methods."""
    from vgtfm.figures.tables import integration_over_seeds

    out = integration_over_seeds(_wide((42, 43, 44), (0.40, 0.44, 0.42)))

    assert len(out) == 1
    row = out.set_index("method").loc["none"]
    assert row["f1/kidney/cross_donor"] == pytest.approx(0.42)
    assert row["n_seeds"] == 3
    # The published bracket is the percentile of the seeds' replicate draws taken
    # together, not the mean of their separate bounds: it has to cover training
    # variability as well as donor sampling.
    assert row["f1/kidney/cross_donor_lo"] == pytest.approx(0.20)
    assert row["f1/kidney/cross_donor_hi"] == pytest.approx(0.70)


def test_a_run_without_pooled_columns_falls_back_to_its_own_bounds():
    """A run made before the stage pooled over seeds still builds its table, at the
    cost of an interval that describes the mean of three per-seed brackets."""
    from vgtfm.figures.tables import integration_over_seeds

    out = integration_over_seeds(_wide((42, 43, 44), (0.40, 0.44, 0.42), pooled=False)).set_index(
        "method"
    )
    assert out.loc["none", "f1/kidney/cross_donor_lo"] == pytest.approx(0.32)


def test_a_table_written_before_the_seed_was_recorded_passes_through():
    from vgtfm.figures.tables import integration_over_seeds

    wide = _wide((42,), (0.40,)).drop(columns=["seed"])
    assert integration_over_seeds(wide).equals(wide)


def test_the_deltas_prefer_the_seed_pooled_interval_and_p_value():
    """A per-seed p-value printed beside a pooled one is not the same test."""
    d = pd.DataFrame(
        [
            {
                "method": "harmony",
                "seed": s,
                "level": "cross_donor",
                "scope": "kidney",
                "delta_f1": v,
                "ci_lo": v - 0.05,
                "ci_hi": v + 0.05,
                "p_two_sided": 0.04,
                "n_donors": 2,
                "ci_lo_pooled": -0.12,
                "ci_hi_pooled": 0.14,
                "p_two_sided_pooled": 0.31,
            }
            for s, v in ((42, 0.01), (43, 0.03), (44, 0.02))
        ]
        + [
            {
                "method": "harmony",
                "seed": 42,
                "level": "cross_region",
                "scope": "kidney",
                "delta_f1": 9.0,
                "ci_lo": 9.0,
                "ci_hi": 9.0,
                "p_two_sided": 0.0,
                "n_donors": 2,
                "ci_lo_pooled": 9.0,
                "ci_hi_pooled": 9.0,
                "p_two_sided_pooled": 0.0,
            }
        ]
    )

    out = pt._deltas_over_seeds(d, "cross_donor")

    assert len(out) == 1  # the other level is not this table's
    row = out.iloc[0]
    assert row.delta_f1 == pytest.approx(0.02)
    assert (row.ci_lo, row.ci_hi) == pytest.approx((-0.12, 0.14))
    assert row.p_two_sided == pytest.approx(0.31)
    assert row.n_seeds == 3
