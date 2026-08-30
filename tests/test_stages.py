"""The stages end to end, on a synthetic cohort held in memory.

The unit tests elsewhere pin the statistics; these pin the wiring between stages —
that `train` writes the embedding `eval` looks for, that every artefact the `figures`
stage reads is actually produced, and that the columns each stage writes are the ones
the next stage selects on. A rename on either side of that boundary is silent
otherwise: the consumer filters on a column that no longer exists, gets an empty
frame, and reports it as a stage that has not run.

The real pipeline is covered by `test_smoke.py`, which needs the cached cohort. This
file needs nothing but a temporary directory.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from vgtfm.config import load_config
from vgtfm.data.tables import SpotTable

TISSUE = "mel"
#: 3 donors x 1 region x 2 replicates, annotated; plus two unannotated fit slides.
EVAL_SLIDES = [f"{d}-1-{r}" for d in ("A", "B", "C") for r in (1, 2)]
FIT_SLIDES = ["MW-B-001a-vis", "MW-G-031a-vis"]
CLASSES = ["Stroma", "Tumor"]


def synthetic_table(n_per_slide=30, gene_dim=8, patch_dim=6, seed=0) -> SpotTable:
    """A cohort whose annotation is linearly readable out of the gene features.

    The probe has to score above chance for the pooled tables to have any rows, so
    the classes are separated along the first gene dimension and the morphology
    carries the same signal in its own basis.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for slide in FIT_SLIDES + EVAL_SLIDES:
        annotated = slide in EVAL_SLIDES
        for i in range(n_per_slide):
            cls = CLASSES[i % len(CLASSES)] if annotated else "UNASSIGNED"
            rows.append(
                {
                    "sample_id": slide,
                    "spot_id": f"{slide}-bc{i}",
                    "array_row": i,
                    "array_col": i,
                    "dataset_id": "10x_TuPro" if annotated else "MOSAIC",
                    "annotation": cls,
                    "annotation_raw": cls,
                    "split": "test" if annotated else "train",
                    "tissue": TISSUE if annotated else "skin",
                    "donor": slide.split("-")[0],
                    "region": 1,
                    "replicate": int(slide[-1]) if annotated else -1,
                }
            )
    meta = pd.DataFrame(rows)

    signal = (meta["annotation"] == "Tumor").to_numpy().astype(np.float32)
    gene = rng.standard_normal((len(meta), gene_dim)).astype(np.float32) * 0.2
    gene[:, 0] += 4.0 * signal
    patch = rng.standard_normal((len(meta), patch_dim)).astype(np.float32) * 0.2
    patch[:, 1] += 4.0 * signal
    return SpotTable(gene=gene, patch=patch, meta=meta, substrate="geneformer")


@pytest.fixture
def cohort(monkeypatch):
    """Serve *table* to every stage in place of the cached merged dataset."""
    from vgtfm.data import tables as tables_mod

    table = synthetic_table()
    monkeypatch.setattr(tables_mod, "load", lambda cfg, substrate=None: table)
    monkeypatch.setattr(tables_mod, "load_meta", lambda cfg, substrate=None: table.meta)
    return table


@pytest.fixture
def cfg(tmp_path):
    return load_config(
        None,
        {
            "paths.artifact_root": str(tmp_path),
            "run_name": "t",
            "seeds": "42",
            "models.names": "pca,pca_oracle",
            "models.pca_components": "4",
            "folds.levels": "cross_replicate,cross_donor",
            "eval.protocols": "heldout_donor,pooled_loso",
            "eval.knn_k": "3",
            "eval.bootstrap_n": "25",
            "diagnostics.n_samples": "60",
            "diagnostics.n_iters": "2",
            "diagnostics.cca_max_spots": "150",
            "diagnostics.cca_pca_dim": "4",
            "diagnostics.cca_n_permutations": "3",
            "diagnostics.run_scib": "false",
            # This cohort's annotated organ. The stage defaults to skin because that is
            # the only organ TuPro's replicate and region ids cover; the fixture's organ
            # is `mel`, so name it rather than disabling the scope — every biosignal
            # test then runs through the same filtering the real runs do.
            "biosignal.tissues": TISSUE,
            "perf.device": "cpu",
        },
    )


def frame(cfg, *parts) -> pd.DataFrame:
    return pd.read_csv(cfg.out_dir.joinpath(*parts))


def bio_dir(cfg):
    """Where the biosignal stage writes for this config's organ scope."""
    from vgtfm.biosignal.ridge import scope_name

    return cfg.out_dir / "biosignal" / scope_name(cfg.biosignal.tissues)


# ── data ─────────────────────────────────────────────────────────────


def test_the_data_stage_describes_the_cohort_it_actually_loaded(cfg, cohort):
    from vgtfm.data import build

    build.run(cfg)

    cohort_csv = frame(cfg, "data", "cohort.csv")
    assert len(cohort_csv) == len(FIT_SLIDES) + len(EVAL_SLIDES)
    assert cohort_csv["n_spots"].sum() == cohort.n
    annotated = cohort_csv[cohort_csv.n_annotated > 0]
    assert sorted(annotated["sample_id"]) == sorted(EVAL_SLIDES)
    assert set(annotated["classes"]) == {"|".join(sorted(CLASSES))}


def test_the_data_stage_enumerates_every_requested_fold_level(cfg, cohort):
    from vgtfm.data import build

    build.run(cfg)
    folds = json.loads((cfg.out_dir / "data" / "folds.json").read_text())

    assert set(folds) == {"cross_replicate", "cross_donor"}
    assert len(folds["cross_donor"]) == 3, "one fold per donor"
    for spec in folds["cross_donor"]:
        assert not set(spec["train_sample_ids"]) & set(spec["eval_sample_ids"])

    summary = json.loads((cfg.out_dir / "data" / "summary.json").read_text())
    assert summary["n_slides"] == len(FIT_SLIDES) + len(EVAL_SLIDES)
    assert summary["n_annotated_spots"] == int(cohort.labeled.sum())


def test_only_annotated_classes_reach_the_class_table(cfg, cohort):
    from vgtfm.data import build

    build.run(cfg)
    classes = frame(cfg, "data", "classes.csv")

    assert set(classes["annotation"]) == set(CLASSES)
    assert set(classes["tissue"]) == {TISSUE, "ALL"}


# ── train ────────────────────────────────────────────────────────────


def test_the_fit_split_selects_the_unannotated_slides(cfg, cohort):
    from vgtfm.models.train import fit_rows

    rows = fit_rows(cfg, cohort)
    assert set(cohort.sample_id[rows]) == set(FIT_SLIDES)


def test_an_empty_fit_split_names_the_splits_that_do_exist(cfg, cohort):
    from vgtfm.models.train import fit_rows

    cfg.train.fit_split = "validation"
    with pytest.raises(SystemExit) as e:
        fit_rows(cfg, cohort)
    assert "train" in str(e.value) and "test" in str(e.value)


def test_the_train_stage_writes_one_embedding_per_model_and_seed(cfg, cohort):
    from vgtfm.models import train

    cfg.seeds = (42, 43)
    train.run(cfg)

    for name in cfg.models.names:
        for seed in cfg.seeds:
            Z = np.load(train.embedding_path(cfg, name, seed))
            assert Z.shape == (cohort.n, cfg.models.pca_components)
            assert Z.dtype == np.float32

    summary = frame(cfg, "train", "train_summary.csv")
    assert len(summary) == len(cfg.models.names) * len(cfg.seeds)
    assert set(summary["model"]) == set(cfg.models.names)
    assert (summary["effective_rank"] > 0).all()


def test_every_model_records_a_history_for_the_run_manifest(cfg, cohort):
    from vgtfm.models import train

    train.run(cfg)
    for name in cfg.models.names:
        state = json.loads(
            (cfg.out_dir / "train" / "history" / f"{name}__seed-42.json").read_text()
        )
        assert state["name"] == name and state["seed"] == 42
        assert state["n_fit_spots"] == len(FIT_SLIDES) * 30


def test_the_narrower_oracle_control_does_not_trip_the_width_warning(cfg, cohort, capsys):
    """Being narrower is the whole point of `pca_oracle_matched`, so a warning
    would fire on every default run and stop being read."""
    from vgtfm.models import train

    cfg.models.names = ("pca", "pca_oracle_matched")
    cfg.models.pca_oracle_matched_components = 2
    train.run(cfg)

    summary = frame(cfg, "train", "train_summary.csv")
    assert set(summary["dim"]) == {4, 2}, "the widths really do differ"
    assert "WARNING" not in capsys.readouterr().out


def test_two_models_of_different_widths_are_warned_about(cfg, cohort, capsys):
    """kNN distances scale with dimensionality, so a width mismatch would make the
    wider model look better for a reason that has nothing to do with the biology."""
    from vgtfm.models import train

    cfg.models.names = ("pca", "pca_oracle")
    train.run(cfg)
    capsys.readouterr()

    summary = pd.read_csv(cfg.out_dir / "train" / "train_summary.csv")
    summary.loc[summary.model == "pca_oracle", "dim"] = 2
    train._print_width_warning(summary)

    out = capsys.readouterr().out
    assert "WARNING" in out and "different widths" in out


# ── eval ─────────────────────────────────────────────────────────────


@pytest.fixture
def trained(cfg, cohort):
    from vgtfm.models import train

    train.run(cfg)
    return cfg


def test_the_eval_stage_scores_every_model_protocol_and_level(trained):
    from vgtfm.evaluate import run_eval

    run_eval.run(trained)
    results = frame(trained, "eval", "results.csv")

    assert set(results["model"]) == {"majority", "pca", "pca_oracle"}
    assert set(results["protocol"]) == {"heldout_donor", "pooled_loso"}
    assert set(results["level"]) == {"cross_replicate", "cross_donor", "all"}
    assert set(results["scope"]) == {"global", TISSUE}
    assert results["f1_score"].between(0, 1).all()


def test_a_separable_cohort_puts_the_probe_above_the_majority_baseline(trained):
    from vgtfm.evaluate import run_eval

    run_eval.run(trained)
    results = frame(trained, "eval", "results.csv")
    pooled = results[
        (results.scope == "global")
        & (results.fold == "pooled")
        & (results.protocol == "heldout_donor")
        & (results.level == "cross_donor")
    ]
    by_model = pooled.set_index("model")["f1_score"]

    assert by_model["pca"] > by_model["majority"]
    assert by_model["pca_oracle"] > by_model["majority"]


def test_per_fold_rows_accompany_the_pooled_ones(trained):
    from vgtfm.evaluate import run_eval

    run_eval.run(trained)
    results = frame(trained, "eval", "results.csv")
    folds = results[
        (results.protocol == "heldout_donor")
        & (results.level == "cross_donor")
        & (results.fold != "pooled")
    ]

    # Fold names carry the organ: a patient key is only unique within its cohort.
    assert set(folds["fold"]) == {f"cross_donor__mel__eval_{d}" for d in "ABC"}


def test_bootstrap_intervals_bracket_their_point_estimate(trained):
    from vgtfm.evaluate import run_eval

    run_eval.run(trained)
    boot = frame(trained, "eval", "bootstrap.csv")

    assert not boot.empty
    macro = boot[boot["class"] == "macro"]
    assert (macro["ci_lo"] <= macro["value"] + 1e-9).all()
    assert (macro["value"] <= macro["ci_hi"] + 1e-9).all()
    assert (macro["n_donors"] >= 1).all()


def test_paired_deltas_are_written_against_the_reference_model(trained):
    from vgtfm.evaluate import run_eval

    run_eval.run(trained)
    deltas = frame(trained, "eval", "deltas.csv")

    assert set(deltas["reference"]) == {run_eval.REFERENCE}
    assert run_eval.REFERENCE not in set(deltas["model"]), "never against itself"
    assert deltas["ci_lo"].le(deltas["ci_hi"]).all()


def test_the_published_interval_pools_the_seeds_rather_than_averaging_them(cfg, cohort):
    """`ci_*_pooled` is the percentile of every seed's draws taken together.

    Averaging the per-seed bounds describes one run while sitting beside a point
    estimate that is the mean over runs, and it hides training variability exactly
    where it is largest. The pooled bound must lie within the range of the per-seed
    bounds — it is a mixture quantile — and must be identical on every seed's row,
    so the tables can keep aggregating with a mean.
    """
    from vgtfm.evaluate import run_eval
    from vgtfm.models import train

    cfg.seeds = (42, 43)
    train.run(cfg)
    run_eval.run(cfg)

    for stem, cols in (
        ("bootstrap.csv", ("ci_lo", "ci_hi")),
        ("deltas.csv", ("ci_lo", "ci_hi", "p_two_sided")),
    ):
        f = frame(cfg, "eval", stem)
        group = ["model", "protocol", "level", "scope", "class"]
        assert (f["n_seeds_pooled"] == 2).all(), stem

        for col in cols:
            assert f"{col}_pooled" in f.columns, f"{stem} is missing {col}_pooled"
            # Constant within the group, so a groupby(...).mean() is a no-op on it.
            spread = f.groupby(group)[f"{col}_pooled"].nunique(dropna=False)
            assert (spread == 1).all(), f"{stem}: {col}_pooled varies within a group"

        # A mixture quantile sits between the components it mixes.
        per_seed = f.groupby(group)["ci_lo"].agg(["min", "max"])
        pooled = f.groupby(group)["ci_lo_pooled"].first()
        assert (pooled >= per_seed["min"] - 1e-9).all(), stem
        assert (pooled <= per_seed["max"] + 1e-9).all(), stem
        assert f["ci_lo_pooled"].le(f["ci_hi_pooled"]).all(), stem


def test_the_tables_prefer_the_pooled_interval_when_the_run_wrote_one(cfg, cohort):
    """Table 1's bracket must come from `ci_*_pooled`, not the per-seed mean."""
    from vgtfm.evaluate import run_eval
    from vgtfm.figures import tables as tbl
    from vgtfm.models import train

    cfg.seeds = (42, 43)
    train.run(cfg)
    run_eval.run(cfg)

    results = frame(cfg, "eval", "results.csv")
    boot = frame(cfg, "eval", "bootstrap.csv")
    out = cfg.sub("figures")
    t = tbl.annotation_table(results, boot, out, level="cross_donor")

    assert not t.empty and t["ci_is_pooled"].all()
    macro = boot[
        (boot["class"] == "macro")
        & (boot.protocol == "heldout_donor")
        & (boot.level == "cross_donor")
        & (boot.scope == "global")
    ]
    expected = macro.groupby("model")["ci_lo_pooled"].first()
    got = t.set_index("model")["donor_ci_lo"]
    for model, value in expected.items():
        assert got[model] == pytest.approx(value)


def test_predictions_are_saved_with_an_index_that_matches_them(trained):
    from vgtfm.evaluate import run_eval

    run_eval.run(trained)
    d = trained.out_dir / "eval" / "predictions"
    index = json.loads((d / "index.json").read_text())

    assert index
    for entry in index:
        npz = np.load(d / entry["file"], allow_pickle=True)
        assert len(npz["y_true"]) == entry["n"] == len(npz["y_pred"])
        assert set(npz.files) == {
            "y_true",
            "y_pred",
            "donor",
            "sample_id",
            "tissue",
            "embedding_fingerprint",
            "refit_per_fold",
        }
        # The fingerprint is what lets `integrate` reuse this vector instead of
        # recomputing it: without a way to check that the embedding it loaded is the
        # one that produced these predictions, reuse would be a guess.
        assert str(npz["embedding_fingerprint"]) == entry["embedding_fingerprint"]
        assert str(npz["embedding_fingerprint"])


def test_a_cached_prediction_records_the_embedding_that_produced_it(trained):
    """Not merely present — the fingerprint has to be the embedding's own."""
    from vgtfm.evaluate import run_eval
    from vgtfm.models.train import embedding_path
    from vgtfm.provenance import array_fingerprint

    run_eval.run(trained)
    d = trained.out_dir / "eval" / "predictions"
    entries = [e for e in json.loads((d / "index.json").read_text()) if e["model"] == "pca"]

    assert entries
    want = array_fingerprint(np.load(embedding_path(trained, "pca", entries[0]["seed"])))
    assert {e["embedding_fingerprint"] for e in entries} == {want}


def test_an_embedding_written_for_a_different_cohort_is_never_scored(trained):
    from vgtfm.evaluate import run_eval
    from vgtfm.models.train import embedding_path

    path = embedding_path(trained, "pca", 42)
    np.save(path, np.load(path)[:-5])
    with pytest.raises(SystemExit, match="rows"):
        run_eval.run(trained)


def test_a_missing_embedding_points_at_the_stage_that_writes_it(trained):
    from vgtfm.evaluate import run_eval
    from vgtfm.models.train import embedding_path

    embedding_path(trained, "pca", 42).unlink()
    with pytest.raises(SystemExit, match="run.py train"):
        run_eval.run(trained)


# ── diagnose ─────────────────────────────────────────────────────────


def test_the_diagnose_stage_writes_every_artefact_the_figures_stage_reads(cfg, cohort):
    from vgtfm.diagnostics import report

    cfg.diagnostics.split = "train"
    report.run(cfg)
    out = cfg.out_dir / "diagnostics"

    er = pd.read_csv(out / "effective_rank.csv")
    assert set(er["column"]) == set(cfg.diagnostics.columns)
    assert "observed" in set(er["variant"]) and len(set(er["variant"])) > 1

    var = pd.read_csv(out / "variance_decomposition.csv")
    assert "sample_id" in set(var["grouping"])
    assert var["between_fraction"].between(0, 1).all()

    ceiling = json.loads((out / "cca_ceiling.json").read_text())
    assert 0.0 <= ceiling["spectrum"]["rho1_observed"] <= 1.0
    assert len(ceiling["spectrum"]["rho_null_within_mean"]) >= 1

    summary = json.loads((out / "summary.json").read_text())
    assert set(summary["effective_rank"]) == set(cfg.diagnostics.columns)


def test_a_diagnostics_split_that_selects_nothing_stops_before_any_statistic(cfg, cohort):
    from vgtfm.diagnostics import report

    cfg.diagnostics.split = "validation"
    with pytest.raises(SystemExit, match="validation"):
        report.run(cfg)


# ── figures ──────────────────────────────────────────────────────────


def test_the_figures_stage_builds_what_the_earlier_stages_left(cfg, cohort):
    from vgtfm.data import build as data_build
    from vgtfm.diagnostics import report
    from vgtfm.evaluate import run_eval
    from vgtfm.figures import build as fig_build
    from vgtfm.models import train

    cfg.diagnostics.umap_max_spots = 60
    data_build.run(cfg)
    train.run(cfg)
    run_eval.run(cfg)
    report.run(cfg)
    fig_build.run(cfg)

    out = cfg.out_dir / "figures"
    assert (out / "table_datasets.csv").exists()
    assert (out / "table_effective_rank.tex").exists()
    assert (out / "fig_umap.pdf").exists()
    assert (out / "table_annotation_heldout_donor_cross_donor.csv").exists()

    table = pd.read_csv(out / "table_annotation_heldout_donor_cross_donor.csv")
    assert set(table["model"]) >= {"pca", "pca_oracle", "majority"}
    assert table["macro_f1"].notna().all()


def test_the_figures_stage_reports_stages_that_have_not_run_as_pending(cfg, cohort, capsys):
    """`ablate` and `biosignal` are opt-in, so the `all` job draws without them."""
    from vgtfm.data import build as data_build
    from vgtfm.diagnostics import report
    from vgtfm.evaluate import run_eval
    from vgtfm.figures import build as fig_build
    from vgtfm.models import train

    cfg.diagnostics.umap_max_spots = 60
    data_build.run(cfg)
    train.run(cfg)
    run_eval.run(cfg)
    report.run(cfg)
    capsys.readouterr()
    fig_build.run(cfg)

    out = capsys.readouterr().out
    assert "pending" in out
    assert "ablate" in out and "biosignal" in out
    assert not (cfg.out_dir / "figures" / "table_patch_shuffle.csv").exists()


def _setmean_csv(path, *, contrast: str = "guided_vs_frozen", n: int = 6):
    """A set-mean table shaped like the one the biosignal stage writes."""
    import numpy as np

    delta = np.linspace(-0.05, 0.01, n)
    pd.DataFrame(
        {
            "contrast": contrast,
            "sample": "cross_donor_guided_vs_frozen",
            "source": [f"HALLMARK_SET_{i}" for i in range(n)],
            "set_size": 30,
            "mean_delta": delta,
            # The first two intervals clear the background mean, the rest straddle it.
            "ci_lo": delta - np.where(np.arange(n) < 2, 0.002, 0.05),
            "ci_hi": delta + np.where(np.arange(n) < 2, 0.002, 0.05),
            "background_mean": -0.01,
            "expected_matched": delta + 0.001,
            "p_vs_background": 0.002,
            "padj_vs_background": [0.001] * 2 + [0.5] * (n - 2),
            "p_vs_matched": 0.4,
            "padj_vs_matched": [0.001] + [0.4] * (n - 1),
        }
    ).to_csv(path, index=False)


def test_the_dot_plot_caption_counts_come_off_the_table_it_draws(cfg, tmp_path):
    """The caption states a set count and two FDR counts, and the figure and that
    sentence have to be the same numbers or one of them is wrong."""
    from vgtfm.figures import build as fig_build

    _setmean_csv(tmp_path / "setmean_hallmark_cross_donor.csv")
    sub = pd.read_csv(tmp_path / "setmean_hallmark_cross_donor.csv")
    counts = fig_build._setmean_counts(sub, cfg.biosignal.fdr)

    assert counts["n_sets"] == 6
    assert counts["background_mean"] == pytest.approx(-0.01)
    # Four of the six planted sets sit below the background mean of -0.01.
    assert counts["n_below"] == 4
    assert counts["n_sig_vs_background"] == 2
    # The column the argument rests on: one set survives the baseline-matched null.
    assert counts["n_sig_vs_matched"] == 1


def test_the_dot_plot_is_drawn_from_the_stage_own_tables(cfg, tmp_path, capsys):
    """One PDF per collection and level the biosignal stage scored, for the
    published contrast, with the same counts printed as the caption quotes."""
    pytest.importorskip("matplotlib")
    from vgtfm.figures import build as fig_build

    bio = bio_dir(cfg)
    bio.mkdir(parents=True, exist_ok=True)
    cfg.biosignal.gene_sets = ("hallmark",)
    for level in ("cross_replicate", "cross_donor"):
        _setmean_csv(bio / f"setmean_hallmark_{level}.csv")

    notes: list = []
    found = fig_build._setmean_sources(bio, cfg, notes)
    assert [(g, level) for g, level, _ in found] == [
        ("hallmark", "cross_donor"),
        ("hallmark", "cross_replicate"),
    ]
    assert not notes

    out = tmp_path / "figures"
    counts = fig_build._fig_setmean(
        found[0][2], out, "fig_setmean_hallmark_cross_donor", fdr=cfg.biosignal.fdr
    )
    assert (out / "fig_setmean_hallmark_cross_donor.pdf").exists()
    assert f"{counts['n_sig_vs_matched']}" in capsys.readouterr().out


def test_a_run_whose_biosignal_wrote_no_set_mean_reports_the_dot_plot_as_pending(cfg, capsys):
    """A scope with a per-gene R^2 but no set-mean table has not finished the stage,
    which the figures stage reports rather than treating as an empty result."""
    from vgtfm.figures import build as fig_build

    bio = bio_dir(cfg)
    bio.mkdir(parents=True, exist_ok=True)
    (bio / "per_gene_r2.csv").write_text("gene\n")
    cfg.biosignal.gene_sets = ("hallmark",)

    notes: list = []
    assert fig_build._setmean_sources(bio, cfg, notes) == []
    assert len(notes) == 1
    assert "setmean_hallmark" in notes[0]


def test_the_per_gene_scatter_draws_the_levels_the_config_names(cfg):
    """The stage scores every level a cohort supports; the figure draws the two the
    manuscript reads. A level the scope never scored is dropped, not demanded: only
    TuPro carries replicate and region ids."""
    from vgtfm.figures import build as fig_build

    per_gene = pd.DataFrame({"level": ["cross_donor"] * 2 + ["cross_replicate"] * 2})
    assert fig_build._r2_levels(per_gene, ("cross_replicate", "cross_donor")) == [
        "cross_replicate",
        "cross_donor",
    ]
    # cross_region is configured but never scored here, so it is simply absent.
    assert fig_build._r2_levels(per_gene, ("cross_region", "cross_donor")) == ["cross_donor"]
    # Empty keeps every level, in the order the stage wrote them.
    assert fig_build._r2_levels(per_gene, ()) == ["cross_donor", "cross_replicate"]


def test_the_dot_plot_refuses_a_table_without_the_contrast_it_draws(cfg):
    """A table carrying only the other contrasts is a settings mismatch, not a
    stage that has yet to run, so it stops the figures pass."""
    from vgtfm.degraded import Incomplete
    from vgtfm.figures import build as fig_build

    bio = bio_dir(cfg)
    bio.mkdir(parents=True, exist_ok=True)
    cfg.biosignal.gene_sets = ("hallmark",)
    _setmean_csv(bio / "setmean_hallmark_cross_donor.csv", contrast="guided_vs_capacity")

    with pytest.raises(Incomplete, match="guided_vs_frozen"):
        fig_build._setmean_sources(bio, cfg, [])


# ── ablate ───────────────────────────────────────────────────────────


def test_the_ablate_stage_trains_and_scores_every_condition(cfg, cohort):
    pytest.importorskip("torch")
    from vgtfm.ablations import run_ablation

    cfg.ablation.conditions = ("none", "global")
    cfg.models.ae.latent_dim = 4
    cfg.models.ae.num_epochs = 2
    cfg.models.ae.patience = 2
    cfg.models.ae.batch_size = 16
    cfg.eval.protocols = ("pooled_loso",)
    cfg.eval.bootstrap_n = 20
    run_ablation.run(cfg)

    results = frame(cfg, "ablation", "results.csv")
    assert set(results["condition"]) == {"none", "global"}
    per_class = frame(cfg, "ablation", "per_class.csv")
    assert "condition" in per_class.columns
    # A tissue scope names its classes bare; the global scope qualifies them by
    # organ, because that is the space its macro average is taken in.
    by_tissue = per_class[per_class.scope != "global"]
    assert set(by_tissue["class"]) <= set(CLASSES)
    assert set(per_class[per_class.scope == "global"]["class"]) == {
        f"mel | {c}" for c in set(by_tissue["class"])
    }

    boot = frame(cfg, "ablation", "bootstrap.csv")
    assert set(boot["condition"]) == {"none", "global"}

    deltas = frame(cfg, "ablation", "paired_deltas.csv")
    assert set(deltas["condition"]) == {"global"}, "the reference is not its own delta"
    assert set(deltas["reference"]) == {run_ablation.REFERENCE_CONDITION}


def test_the_ablate_stage_restores_the_shared_model_config(cfg, cohort):
    """It mutates `models.ae` to select each condition. A later stage in the same
    process must not inherit a shuffled patch target."""
    pytest.importorskip("torch")
    from vgtfm.ablations import run_ablation

    cfg.ablation.conditions = ("global",)
    cfg.models.ae.num_epochs = 1
    cfg.models.ae.latent_dim = 4
    cfg.models.ae.batch_size = 16
    cfg.eval.protocols = ("pooled_loso",)
    cfg.eval.bootstrap_n = 0
    run_ablation.run(cfg)

    assert cfg.models.ae.patch_transform == "none"
    assert cfg.models.ae.shuffle_seed == 42


def test_an_unknown_ablation_condition_is_named_and_skipped(cfg, cohort, capsys):
    pytest.importorskip("torch")
    from vgtfm.ablations import run_ablation

    cfg.ablation.conditions = ("none", "shuffle-everything")
    cfg.models.ae.num_epochs = 1
    cfg.models.ae.latent_dim = 4
    cfg.models.ae.batch_size = 16
    cfg.eval.protocols = ("pooled_loso",)
    cfg.eval.bootstrap_n = 0
    run_ablation.run(cfg)

    assert "shuffle-everything" in capsys.readouterr().out
    assert set(frame(cfg, "ablation", "results.csv")["condition"]) == {"none"}


# ── biosignal ────────────────────────────────────────────────────────


@pytest.fixture
def raw_counts(cfg, cohort, tmp_path):
    """Per-slide `.h5ad` files covering the annotated spots, keyed by barcode."""
    ad = pytest.importorskip("anndata")

    root = tmp_path / "raw" / "tupro"
    root.mkdir(parents=True)
    rng = np.random.default_rng(1)
    genes = [f"G{i}" for i in range(24)]
    meta = cohort.meta
    for slide in EVAL_SLIDES:
        rows = meta.index[meta.sample_id == slide]
        barcodes = meta.loc[rows, "spot_id"].astype(str).tolist()
        signal = (meta.loc[rows, "annotation"] == "Tumor").to_numpy().astype(float)
        counts = rng.poisson(20, size=(len(rows), len(genes))).astype(np.float32)
        counts[:, :6] += 40.0 * signal[:, None]  # genes the embedding can predict
        a = ad.AnnData(
            X=counts,
            obs=pd.DataFrame(index=pd.Index(barcodes, dtype=str)),
            var=pd.DataFrame(index=pd.Index(genes, dtype=str)),
        )
        a.write_h5ad(root / f"{slide}.h5ad")

    cfg.paths.raw_data_root = str(tmp_path / "raw")
    cfg.paths.raw_h5ad["10x_TuPro"] = "tupro"
    return cfg


def test_the_biosignal_stage_scores_every_gene_against_the_frozen_embedding(raw_counts, cohort):
    from vgtfm.biosignal import run_biosignal
    from vgtfm.models import train

    cfg = raw_counts
    cfg.models.names = ("pca",)
    cfg.folds.levels = ("cross_donor",)
    cfg.biosignal.gene_sets = ()
    cfg.biosignal.min_expressed = 1
    train.run(cfg)
    run_biosignal.run(cfg)

    per_gene = pd.read_csv(bio_dir(cfg) / "per_gene_r2.csv")
    assert set(per_gene["level"]) == {"cross_donor"}
    assert len(per_gene) == 24
    for col in ("r2_frozen", "r2_refined", "r2_pca_control", "delta_r2", "delta_r2_pca_control"):
        assert col in per_gene.columns
    assert np.allclose(
        per_gene["delta_r2"], per_gene["r2_refined"] - per_gene["r2_frozen"], equal_nan=True
    )


def test_the_quartile_table_splits_genes_by_their_frozen_predictability(raw_counts):
    from vgtfm.biosignal import run_biosignal
    from vgtfm.models import train

    cfg = raw_counts
    cfg.models.names = ("pca",)
    cfg.folds.levels = ("cross_donor",)
    cfg.biosignal.gene_sets = ()
    cfg.biosignal.min_expressed = 1
    train.run(cfg)
    run_biosignal.run(cfg)

    quart = pd.read_csv(bio_dir(cfg) / "quartiles_cross_donor.csv")
    assert list(quart["quartile"]) == ["Q1", "Q2", "Q3", "Q4"]
    assert quart["n_genes"].sum() == 24
    assert quart["frozen_r2_min"].is_monotonic_increasing

    summary = json.loads((bio_dir(cfg) / "summary.json").read_text())
    assert summary["levels"]["cross_donor"]["n_genes"] == 24
    assert 0 <= summary["levels"]["cross_donor"]["pct_genes_improved"] <= 100


def test_unreachable_raw_counts_stop_the_stage_and_name_the_path(raw_counts):
    """Silently covering fewer folds would report a per-gene R^2 computed on a
    different cohort than the one the caption claims."""
    from vgtfm.biosignal import run_biosignal
    from vgtfm.models import train

    cfg = raw_counts
    cfg.models.names = ("pca",)
    cfg.folds.levels = ("cross_donor",)
    cfg.biosignal.gene_sets = ()
    train.run(cfg)
    cfg.paths.raw_h5ad["10x_TuPro"] = "nowhere"

    with pytest.raises(SystemExit, match="no shared genes"):
        run_biosignal.run(cfg)


def test_the_biosignal_scope_keeps_only_its_own_organ(raw_counts, cohort):
    """Scoping is what makes the three levels one cohort.

    Only TuPro carries replicate and region ids, so `cross_replicate` and
    `cross_region` are skin whatever else is loaded. Scoring every organ into one
    pooled R^2 left `cross_donor` describing a different cohort from the levels
    above it, and pooled fold residuals against a grand mean no fold ever saw.
    """
    from vgtfm.biosignal import run_biosignal
    from vgtfm.models import train

    cfg = raw_counts
    cfg.models.names = ("pca",)
    cfg.folds.levels = ("cross_donor",)
    cfg.biosignal.gene_sets = ()
    cfg.biosignal.min_expressed = 1
    train.run(cfg)

    cfg.biosignal.tissues = ("nosuchorgan",)
    with pytest.raises(SystemExit, match="no fold in any level"):
        run_biosignal.run(cfg)

    # And the scope decides where the artefact lands, so the organ runs coexist.
    cfg.biosignal.tissues = (TISSUE,)
    run_biosignal.run(cfg)
    assert (cfg.out_dir / "biosignal" / TISSUE / "per_gene_r2.csv").exists()
    assert not (cfg.out_dir / "biosignal" / "per_gene_r2.csv").exists()


def test_the_biosignal_stage_needs_a_trained_embedding(raw_counts):
    from vgtfm.biosignal import run_biosignal

    cfg = raw_counts
    cfg.models.names = ("pca",)
    with pytest.raises(SystemExit, match="run.py train"):
        run_biosignal.run(cfg)


# ── integration ──────────────────────────────────────────────────────


def test_a_correction_returned_the_other_way_round_is_transposed_back():
    """These libraries disagree on orientation and have changed it between
    releases; a transposed matrix embeds the wrong thing and raises nothing."""
    from vgtfm.diagnostics.integration import _as_cells_by_features

    Z = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert _as_cells_by_features(Z, 3, "m") is not None
    assert np.array_equal(_as_cells_by_features(Z, 3, "m"), Z)
    assert np.array_equal(_as_cells_by_features(Z.T, 3, "m"), Z)


def test_a_correction_matching_neither_axis_names_the_method():
    from vgtfm.diagnostics.integration import _as_cells_by_features

    with pytest.raises(ValueError, match="harmony"):
        _as_cells_by_features(np.zeros((5, 7), dtype=np.float32), 3, "harmony")


# ── the whole graph, once ────────────────────────────────────────────


def _write_integration(cfg):
    """One `integrate` run's two artefacts, in the layout the stage writes them."""
    out = cfg.sub("integration")
    pd.DataFrame(
        [
            {
                "method": "none",
                "dim": 8,
                "batch/ilisi": 0.04,
                "batch/kbet": 0.19,
                "bio/clisi": 0.98,
                "f1/global/cross_donor": 0.44,
                "f1/global/cross_donor_lo": 0.36,
                "f1/global/cross_donor_hi": 0.52,
                "f1/global/cross_donor_n_donors": 3,
                f"f1/{TISSUE}/cross_donor": 0.46,
                f"f1/{TISSUE}/cross_donor_lo": 0.38,
                f"f1/{TISSUE}/cross_donor_hi": 0.54,
                f"f1/{TISSUE}/cross_donor_n_donors": 3,
            },
            {
                "method": "harmony",
                "dim": 8,
                "batch/ilisi": 0.09,
                "batch/kbet": 0.20,
                "bio/clisi": 0.97,
                "f1/global/cross_donor": 0.42,
                "f1/global/cross_donor_lo": 0.33,
                "f1/global/cross_donor_hi": 0.50,
                "f1/global/cross_donor_n_donors": 3,
                f"f1/{TISSUE}/cross_donor": 0.41,
                f"f1/{TISSUE}/cross_donor_lo": 0.32,
                f"f1/{TISSUE}/cross_donor_hi": 0.49,
                f"f1/{TISSUE}/cross_donor_n_donors": 3,
            },
        ]
    ).to_csv(out / "integration.csv", index=False)
    pd.DataFrame(
        [
            {
                "method": "harmony",
                "level": "cross_donor",
                "scope": "global",
                "delta_f1": -0.026,
                "ci_lo": -0.048,
                "ci_hi": -0.004,
                "p_two_sided": 0.02,
                "n_donors": 3,
                "n_boot": 20,
            },
            {
                "method": "harmony",
                "level": "cross_donor",
                "scope": TISSUE,
                "delta_f1": -0.031,
                "ci_lo": -0.055,
                "ci_hi": -0.007,
                "p_two_sided": 0.03,
                "n_donors": 3,
                "n_boot": 20,
            },
        ]
    ).to_csv(out / "integration_deltas.csv", index=False)


def test_the_second_figures_pass_draws_the_opt_in_artefacts(raw_counts, cohort):
    """`ablate` and `biosignal` land after the `all` job, so the trailing figures
    pass is the one that draws Table 3 and the per-gene R^2 scatter."""
    pytest.importorskip("torch")
    from vgtfm.ablations import run_ablation
    from vgtfm.biosignal import run_biosignal
    from vgtfm.data import build as data_build
    from vgtfm.evaluate import run_eval
    from vgtfm.figures import build as fig_build
    from vgtfm.models import train

    cfg = raw_counts
    cfg.models.names = ("pca", "pca_oracle")
    cfg.folds.levels = ("cross_donor",)
    cfg.eval.protocols = ("heldout_donor", "pooled_loso")
    cfg.eval.bootstrap_n = 20
    cfg.diagnostics.umap_max_spots = 60
    cfg.ablation.conditions = ("none", "global")
    cfg.models.ae.latent_dim = 4
    cfg.models.ae.num_epochs = 2
    cfg.models.ae.patience = 2
    cfg.models.ae.batch_size = 16
    cfg.biosignal.gene_sets = ()
    cfg.biosignal.min_expressed = 1

    data_build.run(cfg)
    train.run(cfg)
    run_eval.run(cfg)
    run_ablation.run(cfg)
    run_biosignal.run(cfg)
    # `integrate` needs harmonypy, bbknn, scanpy and scvi, which this test does not
    # ask for. Its two artefacts are written directly instead: what is under test
    # here is that the figures stage finds them, not that the corrections run.
    _write_integration(cfg)
    fig_build.run(cfg)

    out = cfg.out_dir / "figures"
    for stem in (
        "table_patch_shuffle",
        "fig_patch_shuffle",
        "fig_patch_shuffle_deltas",
        "fig_per_gene_r2",
        "table_annotation_heldout_donor_cross_donor",
        "table_deltas_heldout_donor_cross_donor",
        "table_integration",
        f"table_integration_{TISSUE}",
    ):
        assert (out / f"{stem}.csv").exists() or (out / f"{stem}.pdf").exists(), stem

    shuffle = pd.read_csv(out / "table_patch_shuffle.csv")
    assert set(shuffle["condition"]) == {"none", "global"}
    assert set(shuffle["tissue"]) == {TISSUE}
    assert {"ci_lo", "ci_hi"} <= set(shuffle.columns), "the caption promises a CI"

    integration = pd.read_csv(out / "table_integration.csv")
    assert list(integration["method"]) == ["none", "harmony"]
    # The delta and its interval came from the second file, keyed on the level the
    # table reports — the wiring this test exists to guard.
    harmony = integration.set_index("method").loc["harmony"]
    assert harmony["delta_f1"] == pytest.approx(-0.026)
    assert harmony["delta_lo"] == pytest.approx(-0.048)

    # The stage scores every tissue scope, and each gets its own table beside the
    # pooled one, reading that scope's columns and no other's.
    per_organ = pd.read_csv(out / f"table_integration_{TISSUE}.csv")
    organ_harmony = per_organ.set_index("method").loc["harmony"]
    assert organ_harmony["f1"] == pytest.approx(0.41)
    assert organ_harmony["delta_f1"] == pytest.approx(-0.031)


def _fake_enrichment(enrichment, seen: dict | None = None):
    """A stand-in for :func:`run_gsea` that records how the stage called it.

    Its signature is complete on purpose: a fake swallowing the permutation count
    and minimum set size with ``**kwargs`` would let them stop being wired up from
    the config without a test noticing.
    """

    def fake(stat, resources_dir, gene_set, *, label, seed, times, min_n):
        if seen is not None:
            seen["genes"] = list(stat.index)
            seen["gene_set"] = gene_set
            seen["times"] = times
            seen["min_n"] = min_n
            seen.setdefault("labels", []).append(label)
        return enrichment.EnrichmentResult(
            gene_set=gene_set,
            table=pd.DataFrame(
                {
                    "sample": [label],
                    "source": ["HALLMARK_HYPOXIA"],
                    "set_size": [42],
                    "norm": [-2.1],
                    "pval": [0.0],
                    "padj": [0.01],
                }
            ),
            coverage=0.9,
            n_universe=100,
            n_covered=90,
            n_ranked=len(stat),
            n_sets=1,
            permutations=times,
            min_n=min_n,
            seed=seed,
        )

    return fake


def _fake_set_mean(enrichment, seen: dict | None = None):
    """A stand-in for :func:`run_set_mean`, complete in the same way as above.

    The synthetic cohort's genes are ``G0..G239``, which overlap Hallmark in nothing,
    so the real function correctly refuses on the coverage floor. What these stage
    tests check is the plumbing around it — that the stage forms one ranking per
    contrast, hands over the baseline, and writes the table and its provenance.
    """

    def fake(
        stat,
        resources_dir,
        gene_set,
        *,
        label,
        seed,
        times,
        n_boot,
        min_n,
        ci,
        baseline,
        baseline_bins,
    ):
        if seen is not None:
            seen["times"] = times
            seen["n_boot"] = n_boot
            seen["min_n"] = min_n
            seen["ci"] = ci
            seen["baseline_bins"] = baseline_bins
            seen["baseline_is_frozen_r2"] = baseline is not None and len(baseline) == len(stat)
            seen.setdefault("labels", []).append(label)
        return enrichment.SetMeanResult(
            gene_set=gene_set,
            table=pd.DataFrame(
                {
                    "sample": [label],
                    "source": ["HALLMARK_HYPOXIA"],
                    "set_size": [42],
                    "mean_delta": [-0.03],
                    "ci_lo": [-0.04],
                    "ci_hi": [-0.02],
                    "background_mean": [-0.01],
                    "expected_matched": [-0.02],
                    "p_vs_background": [0.002],
                    "padj_vs_background": [0.01],
                    "p_vs_matched": [0.4],
                    "padj_vs_matched": [0.4],
                }
            ),
            background_mean=-0.01,
            background_ci=(-0.012, -0.008),
            coverage=0.9,
            n_universe=100,
            n_covered=90,
            n_ranked=len(stat),
            n_sets=1,
            permutations=times,
            bootstrap=n_boot,
            ci=ci,
            min_n=min_n,
            seed=seed,
            baseline_bins=baseline_bins if baseline is not None else 0,
        )

    return fake


def test_gsea_runs_against_the_vendored_gene_sets(raw_counts, cohort, monkeypatch):
    """The ranking's identifiers have to be HGNC symbols; Ensembl ids would overlap
    the collections in nothing and the coverage floor is what catches that."""
    pytest.importorskip("decoupler")
    from vgtfm.biosignal import enrichment, run_biosignal
    from vgtfm.models import train

    cfg = raw_counts
    cfg.models.names = ("pca",)
    cfg.folds.levels = ("cross_donor",)
    cfg.biosignal.gene_sets = ("hallmark",)
    cfg.biosignal.min_expressed = 1

    cfg.biosignal.gsea_permutations = 250
    cfg.biosignal.gsea_min_set_size = 7

    seen: dict = {}
    monkeypatch.setattr(enrichment, "run_gsea", _fake_enrichment(enrichment, seen))
    monkeypatch.setattr(enrichment, "run_set_mean", _fake_set_mean(enrichment))
    train.run(cfg)
    run_biosignal.run(cfg)

    assert seen["gene_set"] == "hallmark"
    assert seen["genes"], "the ranking must not be empty"
    written = pd.read_csv(bio_dir(cfg) / "gsea_hallmark_cross_donor.csv")
    assert written["source"].unique().tolist() == ["HALLMARK_HYPOXIA"]

    # One ranking per contrast, so a pathway can be read against the control that
    # carries no morphology. Detrending is a sensitivity knob and stays off here.
    assert set(written["contrast"]) == set(run_biosignal.GSEA_CONTRASTS)
    assert set(written["detrended"]) == {False}
    assert len(written) == len(run_biosignal.GSEA_CONTRASTS)
    # The label decoupler is handed identifies the ranking, not just the level.
    assert "cross_donor_guided_vs_capacity" in seen["labels"]
    # The resolution and the minimum set size come from the config, not from
    # whatever default the enrichment module carries.
    assert (seen["times"], seen["min_n"]) == (250, 7)

    # The provenance file carries the other half of a result: what was tested, against
    # which collection, over what background, and how multiplicity was corrected.
    meta = json.loads((bio_dir(cfg) / "gsea_hallmark_cross_donor_meta.json").read_text())
    assert meta["collection"]["sha256"] and meta["collection"]["citation"]
    # The scores come out of a private decoupler kernel, so which release produced
    # them is part of the result and is read off the artefact, not off requirements.txt.
    assert meta["versions"]["decoupler"]
    assert {r["contrast"] for r in meta["rankings"]} == set(run_biosignal.GSEA_CONTRASTS)
    ranking = meta["rankings"][0]
    assert ranking["permutations"] == 250
    assert ranking["pval_resolution"] == pytest.approx(1 / 250)
    assert ranking["background_n_genes"] == len(seen["genes"])
    assert "Benjamini-Hochberg" in ranking["multiple_testing"]


def test_the_set_mean_tables_land_beside_the_gsea_ones(raw_counts, cohort, monkeypatch):
    """The second reading of the same ranking, and the config that shaped it.

    The two tables sit in one directory and are quoted in one appendix, so what is
    pinned here is that they cover the same contrasts and that the set-mean side gets
    the knobs it needs — the baseline the matched null permutes within above all,
    since without it the stage would silently write a NaN column.
    """
    pytest.importorskip("decoupler")
    from vgtfm.biosignal import enrichment, run_biosignal
    from vgtfm.models import train

    cfg = raw_counts
    cfg.models.names = ("pca",)
    cfg.folds.levels = ("cross_donor",)
    cfg.biosignal.gene_sets = ("hallmark",)
    cfg.biosignal.min_expressed = 1
    cfg.biosignal.gsea_permutations = 250
    cfg.biosignal.gsea_min_set_size = 7
    cfg.biosignal.setmean_bootstrap = 128
    cfg.biosignal.setmean_baseline_bins = 5
    # Detrending is a knob on the GSEA ranking; the set-mean test corrects the null
    # instead, so it must not pick up a second, doubly-corrected ranking from it.
    cfg.biosignal.gsea_detrend_bins = 4

    seen: dict = {}
    monkeypatch.setattr(enrichment, "run_gsea", _fake_enrichment(enrichment))
    monkeypatch.setattr(enrichment, "run_set_mean", _fake_set_mean(enrichment, seen))
    train.run(cfg)
    run_biosignal.run(cfg)

    written = pd.read_csv(bio_dir(cfg) / "setmean_hallmark_cross_donor.csv")
    assert set(written["contrast"]) == set(run_biosignal.GSEA_CONTRASTS)
    assert list(written.columns) == ["contrast", *enrichment.SET_MEAN_COLUMNS]
    assert not any("detrended" in label for label in seen["labels"])

    assert (seen["times"], seen["min_n"]) == (250, 7)
    assert (seen["n_boot"], seen["ci"], seen["baseline_bins"]) == (128, 0.95, 5)
    assert seen["baseline_is_frozen_r2"]

    meta = json.loads((bio_dir(cfg) / "setmean_hallmark_cross_donor_meta.json").read_text())
    assert meta["collection"]["sha256"] and meta["collection"]["citation"]
    assert meta["versions"]["decoupler"]
    assert {r["contrast"] for r in meta["rankings"]} == set(run_biosignal.GSEA_CONTRASTS)
    ranking = meta["rankings"][0]
    assert ranking["permutations"] == 250
    assert ranking["baseline_matched_null_bins"] == 5
    assert "Benjamini-Hochberg" in ranking["multiple_testing"]


def test_the_matched_null_is_switched_off_by_setting_its_bins_below_two(
    raw_counts, cohort, monkeypatch
):
    pytest.importorskip("decoupler")
    from vgtfm.biosignal import enrichment, run_biosignal
    from vgtfm.models import train

    cfg = raw_counts
    cfg.models.names = ("pca",)
    cfg.folds.levels = ("cross_donor",)
    cfg.biosignal.gene_sets = ("hallmark",)
    cfg.biosignal.min_expressed = 1
    cfg.biosignal.setmean_baseline_bins = 0

    seen: dict = {}
    monkeypatch.setattr(enrichment, "run_gsea", _fake_enrichment(enrichment))
    monkeypatch.setattr(enrichment, "run_set_mean", _fake_set_mean(enrichment, seen))
    train.run(cfg)
    run_biosignal.run(cfg)

    assert not seen["baseline_is_frozen_r2"]


def test_gsea_detrend_bins_add_a_second_ranking_per_contrast(raw_counts, cohort, monkeypatch):
    """The sensitivity analysis is opt-in, and adds to the raw ranking rather than
    replacing it — the pair is what says whether a set survives the correction."""
    pytest.importorskip("decoupler")
    from vgtfm.biosignal import enrichment, run_biosignal
    from vgtfm.models import train

    cfg = raw_counts
    cfg.models.names = ("pca",)
    cfg.folds.levels = ("cross_donor",)
    cfg.biosignal.gene_sets = ("hallmark",)
    cfg.biosignal.min_expressed = 1
    cfg.biosignal.gsea_detrend_bins = 4

    monkeypatch.setattr(enrichment, "run_gsea", _fake_enrichment(enrichment))
    monkeypatch.setattr(enrichment, "run_set_mean", _fake_set_mean(enrichment))
    train.run(cfg)
    run_biosignal.run(cfg)

    written = pd.read_csv(bio_dir(cfg) / "gsea_hallmark_cross_donor.csv")
    assert set(written["detrended"]) == {True, False}
    assert len(written) == 2 * len(run_biosignal.GSEA_CONTRASTS)


def test_gsea_without_the_capacity_control_scores_only_what_it_can(raw_counts, cohort, monkeypatch):
    """Turning the control off is allowed, and costs the contrasts that need it.

    Those contrasts are what separate a width change from lost information, so the
    stage says so rather than quietly reporting the confounded ranking alone.
    """
    pytest.importorskip("decoupler")
    from vgtfm.biosignal import enrichment, run_biosignal
    from vgtfm.models import train

    cfg = raw_counts
    cfg.models.names = ("pca",)
    cfg.folds.levels = ("cross_donor",)
    cfg.biosignal.gene_sets = ("hallmark",)
    cfg.biosignal.min_expressed = 1
    cfg.biosignal.include_pca_control = False

    monkeypatch.setattr(enrichment, "run_gsea", _fake_enrichment(enrichment))
    monkeypatch.setattr(enrichment, "run_set_mean", _fake_set_mean(enrichment))
    train.run(cfg)
    run_biosignal.run(cfg)

    written = pd.read_csv(bio_dir(cfg) / "gsea_hallmark_cross_donor.csv")
    assert set(written["contrast"]) == {"guided_vs_frozen"}


def test_gsea_refuses_a_contrast_it_cannot_form(raw_counts, cohort, monkeypatch):
    from vgtfm.biosignal import run_biosignal
    from vgtfm.degraded import Incomplete
    from vgtfm.models import train

    cfg = raw_counts
    cfg.models.names = ("pca",)
    cfg.folds.levels = ("cross_donor",)
    cfg.biosignal.gene_sets = ("hallmark",)
    cfg.biosignal.min_expressed = 1
    cfg.biosignal.gsea_contrasts = ("guided_vs_noise",)

    train.run(cfg)
    with pytest.raises(Incomplete, match="guided_vs_noise"):
        run_biosignal.run(cfg)


def test_gsea_that_returns_nothing_stops_the_stage(raw_counts, cohort, monkeypatch):
    from vgtfm.biosignal import enrichment, run_biosignal
    from vgtfm.degraded import Incomplete
    from vgtfm.models import train

    cfg = raw_counts
    cfg.models.names = ("pca",)
    cfg.folds.levels = ("cross_donor",)
    cfg.biosignal.gene_sets = ("hallmark",)
    cfg.biosignal.min_expressed = 1

    monkeypatch.setattr(
        enrichment,
        "run_gsea",
        lambda *a, **k: enrichment.EnrichmentResult("hallmark", pd.DataFrame(), 0.9, 1, 1),
    )
    train.run(cfg)
    with pytest.raises(Incomplete, match="HGNC"):
        run_biosignal.run(cfg)
