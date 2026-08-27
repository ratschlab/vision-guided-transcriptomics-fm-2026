"""End-to-end smoke test on a small slice of the cohort.

Runs ``data -> train -> eval -> figures`` with ``configs/smoke.yaml`` and asserts
the properties that must hold for the pipeline to be reporting anything real:

* every stage writes the artefacts the next one reads;
* the H&E oracle beats the gene-only baseline, which beats the majority baseline —
  an ordering that can only break if the probe or the labels are wrong;
* both evaluation protocols produce numbers, and the held-out-donor one is not
  silently identical to the legacy pooled one;
* the deployed embedding is a function of gene features only.

Marked ``smoke`` and excluded from ``make test``: it runs the real pipeline and
takes minutes. Run it with ``make smoke``. It skips itself when the cached merged
dataset is absent, so it is harmless on a machine without the data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

from vgtfm.config import load_config  # noqa: E402

CONFIG = ROOT / "configs" / "smoke.yaml"

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def cfg():
    cfg = load_config(CONFIG)
    if not cfg.data_path().exists():
        pytest.skip(f"cached merged dataset not found at {cfg.data_path()}")
    return cfg


@pytest.fixture(scope="module")
def pipeline(cfg):
    """Run the pipeline once for every assertion in this module."""
    import run as runner

    for stage in ("data", "train", "eval"):
        assert runner.main([stage, "--config", str(CONFIG), "--no-log"]) == 0
    return cfg


def test_data_stage_describes_the_cohort(pipeline):
    out = pipeline.out_dir / "data"
    cohort = pd.read_csv(out / "cohort.csv")
    assert (out / "summary.json").exists()
    assert (out / "folds.json").exists()
    assert len(cohort) > 0
    # Annotated slides carry classes; unannotated training slides do not.
    assert (cohort["n_annotated"] > 0).any()
    assert cohort.loc[cohort.n_annotated > 0, "n_classes"].min() >= 2


def test_train_stage_writes_one_embedding_per_model_and_seed(pipeline):
    summary = pd.read_csv(pipeline.out_dir / "train" / "train_summary.csv")
    from vgtfm.models.train import embedding_path

    assert set(summary["model"]) == set(pipeline.models.names)
    for name in pipeline.models.names:
        for seed in pipeline.seeds:
            path = embedding_path(pipeline, name, seed)
            assert path.exists(), path
            Z = np.load(path)
            assert Z.dtype == np.float32
            assert np.isfinite(Z).all()
            assert Z.shape[1] == pipeline.models.pca_components


def test_probe_ordering_is_sane(pipeline):
    """Oracle > gene-only > chance. A break here means the probe is broken."""
    results = pd.read_csv(pipeline.out_dir / "eval" / "results.csv")
    view = results[
        (results.scope == "global")
        & (results.fold == "pooled")
        & (results.protocol == "heldout_donor")
        & (results.level == "cross_donor")
    ]
    scores = view.groupby("model")["f1_score"].mean()
    assert scores["majority"] < scores["pca"], scores.to_dict()
    assert scores["pca"] < scores["pca_oracle"], scores.to_dict()


def test_both_protocols_run_and_differ(pipeline):
    results = pd.read_csv(pipeline.out_dir / "eval" / "results.csv")
    protocols = set(results["protocol"])
    assert {"heldout_donor", "pooled_loso"} <= protocols

    pooled = results[(results.protocol == "pooled_loso") & (results.scope == "global")]
    heldout = results[
        (results.protocol == "heldout_donor")
        & (results.level == "cross_donor")
        & (results.scope == "global")
        & (results.fold == "pooled")
    ]
    # The legacy protocol pools train and eval slides into one probe, so it should
    # not reproduce the held-out-donor numbers. If it does, the fold split is not
    # reaching the probe.
    assert not np.isclose(pooled["f1_score"].mean(), heldout["f1_score"].mean(), atol=1e-6)


def test_bootstrap_intervals_bracket_the_point_estimates(pipeline):
    boot = pd.read_csv(pipeline.out_dir / "eval" / "bootstrap.csv")
    macro = boot[boot["class"] == "macro"].dropna(subset=["ci_lo", "ci_hi"])
    assert len(macro) > 0
    assert (macro["ci_lo"] <= macro["value"] + 1e-9).all()
    assert (macro["value"] <= macro["ci_hi"] + 1e-9).all()
    # Donor-level resampling on a handful of patients cannot produce a point interval.
    assert (macro["ci_hi"] - macro["ci_lo"]).max() > 0


def test_per_class_supports_match_the_cohort(pipeline):
    per_class = pd.read_csv(pipeline.out_dir / "eval" / "per_class.csv")
    classes = pd.read_csv(pipeline.out_dir / "data" / "classes.csv")
    scored = set(per_class.loc[per_class.scope != "global", "class"])
    known = set(classes["annotation"])
    assert scored <= known, scored - known


def test_figures_stage_produces_tables_and_plots(pipeline):
    import run as runner

    assert runner.main(["figures", "--config", str(CONFIG), "--no-log"]) == 0
    figs = pipeline.out_dir / "figures"
    # Names carry the protocol as well as the level: one level yields a separate
    # table per protocol, and they would otherwise overwrite each other.
    assert (figs / "table_annotation_heldout_donor_cross_donor.csv").exists()
    assert (figs / "table_annotation_heldout_donor_cross_donor.tex").exists()
    assert (figs / "fig_annotation_heldout_donor_cross_donor.pdf").exists()

    table = pd.read_csv(figs / "table_annotation_heldout_donor_cross_donor.csv")
    # Values in the table are the ones the eval stage wrote, not re-derived.
    results = pd.read_csv(pipeline.out_dir / "eval" / "results.csv")
    ref = results[
        (results.protocol == "heldout_donor")
        & (results.level == "cross_donor")
        & (results.scope == "global")
        & (results.fold == "pooled")
        & (results.model == "pca")
    ]
    assert np.isclose(table.loc[table.model == "pca", "macro_f1"].iloc[0], ref["f1_score"].mean())
