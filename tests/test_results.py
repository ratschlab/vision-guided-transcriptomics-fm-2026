"""The shared results store: one frame per run, and the check that two stages
computing one quantity have not computed it differently.

The failure these tests pin actually happened. ``integrate`` fitted its own PCA over
every spot in the cohort while ``train`` fitted the ``pca`` baseline on
``train.fit_split`` alone, so the integration table's uncorrected row and Table 1's
PCA row described different representations and disagreed by up to 0.076 macro-F1 --
with nothing in the repository able to compare them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from vgtfm import results


def _run_dir(
    tmp_path: Path,
    *,
    eval_f1: float,
    integrate_f1: float | None,
    run_name: str = "r",
    seed: int = 42,
) -> Path:
    """A run directory carrying one scored cell per stage, both the same cell."""
    d = tmp_path / run_name
    (d / "eval").mkdir(parents=True)
    (d / "config.resolved.json").write_text(
        json.dumps(
            {"run_name": run_name, "data": {"substrate": "sub"}, "diagnostics": {"seed": seed}}
        )
    )
    pd.DataFrame(
        [
            {
                "substrate": "sub",
                "model": "pca",
                "seed": seed,
                "protocol": "heldout_donor",
                "level": "cross_donor",
                "fold": "pooled",
                "scope": "kidney",
                "f1_score": eval_f1,
                "n_donors": 2,
            }
        ]
    ).to_csv(d / "eval" / "results.csv", index=False)
    if integrate_f1 is not None:
        (d / "integration").mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "method": "none",
                    "batch/ilisi": 0.04,
                    "f1/kidney/cross_donor": integrate_f1,
                    "f1/kidney/cross_donor_lo": 0.3,
                    "f1/kidney/cross_donor_hi": 0.5,
                    "f1/kidney/cross_donor_n_donors": 2,
                }
            ]
        ).to_csv(d / "integration" / "integration.csv", index=False)
    return d


def test_the_wide_integration_table_is_unfolded_into_the_shared_schema(tmp_path):
    """Its column names carry the scope and the fold level, which is what made it
    incomparable with `eval`'s long format in the first place."""
    df = results.collect_run(_run_dir(tmp_path, eval_f1=0.45, integrate_f1=0.37))
    probe = df[(df.stage == "integrate") & (df.metric == "f1_score")]

    assert len(probe) == 1
    row = probe.iloc[0]
    assert (row.scope, row.level, row["class"]) == ("kidney", "cross_donor", "macro")
    assert row.protocol == "heldout_donor" and row.value == pytest.approx(0.37)
    # The scIB columns are cohort-wide, not a tissue's: the panel scores one
    # stratified sample of the whole annotated cohort.
    mixing = df[(df.stage == "integrate") & (df.metric == "ilisi")]
    assert list(mixing.scope) == [results.COHORT_SCOPE]


def test_two_stages_disagreeing_on_one_cell_are_reported(tmp_path):
    df = results.collect_run(_run_dir(tmp_path, eval_f1=0.4496, integrate_f1=0.3738))
    conf = results.conflicts(df)

    assert len(conf) == 1
    row = conf.iloc[0]
    assert (row.scope, row.metric) == ("kidney", "f1_score")
    assert row.spread == pytest.approx(0.0758, abs=1e-4)
    # The report has to name the files, or it says only that something is wrong.
    assert "eval/results.csv" in row["values"]
    assert "integration/integration.csv" in row["values"]


def test_two_stages_agreeing_on_one_cell_are_not_reported(tmp_path):
    """`integrate` reusing `eval`'s prediction vector is the fixed state."""
    df = results.collect_run(_run_dir(tmp_path, eval_f1=0.4496, integrate_f1=0.4496))
    assert results.conflicts(df).empty


def test_a_disagreement_too_small_to_print_is_not_a_conflict(tmp_path):
    """The manuscript prints these to three decimals, and two float paths may differ
    in their last bits without anyone being wrong."""
    df = results.collect_run(_run_dir(tmp_path, eval_f1=0.4496, integrate_f1=0.44961))
    assert results.conflicts(df).empty


def test_a_stage_with_no_counterpart_cannot_conflict(tmp_path):
    """Only the pairs `ALIASES` names are the same quantity. Everything else shares
    no cell, and a check that flagged them would cry wolf on every run."""
    df = results.collect_run(_run_dir(tmp_path, eval_f1=0.45, integrate_f1=None))
    assert results.conflicts(df).empty
    assert not df[df.stage == "eval"].empty


def test_collect_spans_runs_and_keeps_them_apart(tmp_path):
    """A cell is identified within its run: two substrates scoring the same tissue
    are not two answers to one question."""
    _run_dir(tmp_path, eval_f1=0.45, integrate_f1=0.45, run_name="a")
    _run_dir(tmp_path, eval_f1=0.61, integrate_f1=0.61, run_name="b")
    df = results.collect(tmp_path, ["a", "b"])

    assert set(df.run_name) == {"a", "b"}
    assert results.conflicts(df).empty


def test_the_results_stage_refuses_when_two_stages_disagree(tmp_path):
    from vgtfm.config import Config

    _run_dir(tmp_path, eval_f1=0.4496, integrate_f1=0.3738, run_name="r")
    cfg = Config()
    cfg.run_name = "r"
    cfg.paths.artifact_root = str(tmp_path)

    with pytest.raises(SystemExit, match="computed twice with different values"):
        results.run(cfg)
    # The frame is still written: the point is to show the reader the conflict, not
    # to withhold the evidence for it.
    assert (tmp_path / "r" / "results" / "conflicts.csv").exists()
    assert (tmp_path / "r" / "results" / "scores.csv").exists()


def _panel_dirs(
    tmp_path: Path,
    *,
    diagnose_kbet: float,
    integrate_kbet: float,
    seed: int = 42,
    run_name: str = "r",
) -> Path:
    """A run whose `diagnose` and `integrate` stages both score the scIB panel on
    the `pca` embedding — one computation, written down in two files."""
    d = tmp_path / run_name
    (d / "diagnostics").mkdir(parents=True)
    (d / "integration").mkdir(parents=True)
    (d / "config.resolved.json").write_text(
        json.dumps(
            {"run_name": run_name, "data": {"substrate": "sub"}, "diagnostics": {"seed": seed}}
        )
    )
    pd.DataFrame(
        [
            {
                "substrate": "sub",
                "representation": f"pca (seed {seed})",
                "batch/kbet": diagnose_kbet,
                "bio/clisi": 0.98,
            }
        ]
    ).to_csv(d / "diagnostics" / "scib_panel.csv", index=False)
    pd.DataFrame([{"method": "none", "batch/kbet": integrate_kbet, "bio/clisi": 0.98}]).to_csv(
        d / "integration" / "integration.csv", index=False
    )
    return d


def test_the_scib_panel_is_checked_across_the_two_stages_that_score_it(tmp_path):
    """`diagnose` and `integrate` call the same `scib.benchmark` with the same seed
    on the same embedding, and that call is deterministic — so a difference here is
    a difference in the matrix, which is exactly what went wrong.
    """
    agreeing = results.collect_run(
        _panel_dirs(tmp_path / "ok", diagnose_kbet=0.1517, integrate_kbet=0.1517)
    )
    assert results.conflicts(agreeing).empty

    differing = results.collect_run(
        _panel_dirs(tmp_path / "bad", diagnose_kbet=0.1517, integrate_kbet=0.1533)
    )
    conf = results.conflicts(differing)
    assert list(conf.metric) == ["kbet"]
    assert "diagnostics/scib_panel.csv" in conf.iloc[0]["values"]


def test_the_seed_is_lifted_out_of_the_representation_name(tmp_path):
    """`diagnose` writes `pca (seed 42)`. Left in the name, that row can never be
    matched against the same model scored by another stage."""
    df = results.collect_run(_panel_dirs(tmp_path, diagnose_kbet=0.15, integrate_kbet=0.15))
    row = df[(df.stage == "diagnose") & (df.metric == "kbet")].iloc[0]

    assert row.model == "pca" and int(row.seed) == 42
    # A panel row is not a probe result: stamping a protocol on it is what stopped
    # these two stages matching in the first place. It is unset in the raw frame;
    # `conflicts` is what normalises an absent field to the empty string.
    assert pd.isna(row.protocol) or row.protocol == ""
    assert row.scope == results.COHORT_SCOPE


def test_a_panel_scored_at_another_seed_is_not_a_disagreement(tmp_path):
    """Two seeds are two fits, not two answers to one question."""
    d = _panel_dirs(tmp_path, diagnose_kbet=0.15, integrate_kbet=0.28, seed=42)
    panel = pd.read_csv(d / "diagnostics" / "scib_panel.csv")
    panel["representation"] = "pca (seed 43)"
    panel.to_csv(d / "diagnostics" / "scib_panel.csv", index=False)

    assert results.conflicts(results.collect_run(d)).empty


# ── the check has to be checking something ───────────────────────────


def test_one_stage_computing_a_number_twice_is_also_checked(tmp_path):
    """`eval` writes the macro-F1 of one prediction vector into `results.csv` through
    `classification_metrics` and into `bootstrap.csv` as `donor_bootstrap`'s point
    estimate. Two code paths, one quantity — grouping by stage would put them out of
    scope, and on the three paper runs that is 10,461 of the 10,515 comparisons.
    """
    d = _run_dir(tmp_path, eval_f1=0.4496, integrate_f1=None)
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
                "class": "macro",
                "value": 0.4496,
                "ci_lo": 0.3,
                "ci_hi": 0.5,
                "n_donors": 2,
                "n_boot": 2000,
            }
        ]
    ).to_csv(d / "eval" / "bootstrap.csv", index=False)
    assert results.conflicts(results.collect_run(d)).empty

    boot = pd.read_csv(d / "eval" / "bootstrap.csv")
    boot["value"] = 0.4711
    boot.to_csv(d / "eval" / "bootstrap.csv", index=False)
    conf = results.conflicts(results.collect_run(d))

    assert len(conf) == 1
    assert "eval/results.csv" in conf.iloc[0]["values"]
    assert "eval/bootstrap.csv" in conf.iloc[0]["values"]


def test_the_number_of_cells_actually_compared_is_reported(tmp_path):
    """ "No stage disagrees" has two readings — everything agreed, or nothing was
    compared — and they print identically. This is the count that separates them."""
    df = results.collect_run(_run_dir(tmp_path, eval_f1=0.45, integrate_f1=0.45))
    checks = results.cross_checks(df)

    assert int(checks["n_cells"].sum()) == 1
    row = checks.iloc[0]
    assert row.metric == "f1_score"
    # Naming the files is what makes the count auditable rather than reassuring.
    assert "eval/results.csv" in row.sources
    assert "integration/integration.csv" in row.sources


def test_an_alias_that_compared_nothing_is_a_failure_not_a_pass(tmp_path):
    """The way this check dies quietly: one stage renames a seed, a protocol or a
    level, every alias-matched cell lands in a different bucket, and `conflicts`
    returns empty because it compared nothing at all."""
    d = _panel_dirs(tmp_path, diagnose_kbet=0.1517, integrate_kbet=0.1533)
    assert results.unexercised(results.collect_run(d)) == []

    panel = pd.read_csv(d / "diagnostics" / "scib_panel.csv")
    panel["representation"] = "pca (seed 99)"
    panel.to_csv(d / "diagnostics" / "scib_panel.csv", index=False)
    df = results.collect_run(d)

    assert results.conflicts(df).empty  # the disagreement has gone quiet
    idle = results.unexercised(df)
    assert len(idle) == 1, idle
    assert "diagnose/pca and integrate/none" in idle[0]
    assert "never on the same cell" in idle[0]


def test_an_alias_with_no_metric_in_common_is_idle_on_purpose(tmp_path):
    """`diagnose`'s panel and `eval`'s probe are both the `pca` baseline and share no
    metric, so their alias lying unused in a run without `integrate` is correct. A
    vacuity check that fired here would cry wolf on every core run."""
    d = _run_dir(tmp_path, eval_f1=0.45, integrate_f1=None)
    (d / "diagnostics").mkdir(parents=True)
    pd.DataFrame(
        [{"substrate": "sub", "representation": "pca (seed 42)", "batch/kbet": 0.15}]
    ).to_csv(d / "diagnostics" / "scib_panel.csv", index=False)

    assert results.unexercised(results.collect_run(d)) == []


def test_the_results_stage_refuses_when_an_alias_compared_nothing(tmp_path):
    from vgtfm.config import Config

    d = _panel_dirs(tmp_path, diagnose_kbet=0.15, integrate_kbet=0.15, run_name="r")
    panel = pd.read_csv(d / "diagnostics" / "scib_panel.csv")
    panel["representation"] = "pca (seed 99)"
    panel.to_csv(d / "diagnostics" / "scib_panel.csv", index=False)
    cfg = Config()
    cfg.run_name = "r"
    cfg.paths.artifact_root = str(tmp_path)

    with pytest.raises(SystemExit, match="alias.*compared nothing"):
        results.run(cfg)


def test_a_stage_scoring_three_fits_is_checked_against_eval_fit_by_fit(tmp_path):
    """`integrate` scores every seed `eval` scores, so the comparison is per fit.

    Collapsing the seeds before comparing would let a stage that got seed 43 wrong
    hide behind a mean that still matches; keying on the seed is what makes each
    fit's uncorrected row answerable for itself.
    """
    d = tmp_path / "r"
    (d / "eval").mkdir(parents=True)
    (d / "integration").mkdir(parents=True)
    (d / "config.resolved.json").write_text(
        json.dumps({"run_name": "r", "data": {"substrate": "sub"}, "diagnostics": {"seed": 42}})
    )
    pd.DataFrame(
        [
            {
                "substrate": "sub",
                "model": "pca",
                "seed": s,
                "protocol": "heldout_donor",
                "level": "cross_donor",
                "fold": "pooled",
                "scope": "kidney",
                "f1_score": f1,
                "n_donors": 2,
            }
            for s, f1 in ((42, 0.40), (43, 0.44))
        ]
    ).to_csv(d / "eval" / "results.csv", index=False)
    pd.DataFrame(
        [
            {
                "method": "none",
                "seed": s,
                "f1/kidney/cross_donor": f1,
                "f1/kidney/cross_donor_lo": 0.3,
                "f1/kidney/cross_donor_hi": 0.5,
            }
            for s, f1 in ((42, 0.40), (43, 0.61))
        ]
    ).to_csv(d / "integration" / "integration.csv", index=False)

    conf = results.conflicts(results.collect_run(d))
    assert len(conf) == 1
    assert conf.iloc[0].seed == "43.0" or float(conf.iloc[0].seed) == 43
    assert conf.iloc[0].spread == pytest.approx(0.17, abs=1e-6)


def test_the_integration_seed_comes_from_the_row_not_the_config(tmp_path):
    """The row is what says which fit it describes. Reading it from the resolved
    config stamps one seed on all three, and every cell then keys against the wrong
    counterpart in `eval` — or against none at all."""
    d = tmp_path / "r"
    (d / "integration").mkdir(parents=True)
    (d / "config.resolved.json").write_text(
        json.dumps({"run_name": "r", "data": {"substrate": "sub"}, "diagnostics": {"seed": 42}})
    )
    pd.DataFrame([{"method": "none", "seed": s, "batch/kbet": 0.15} for s in (42, 43, 44)]).to_csv(
        d / "integration" / "integration.csv", index=False
    )

    df = results.collect_run(d)
    assert sorted(df[df.metric == "kbet"].seed.tolist()) == [42, 43, 44]
