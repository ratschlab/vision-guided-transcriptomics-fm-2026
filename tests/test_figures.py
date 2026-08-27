"""The manuscript tables, built from a run's own outputs.

A wrong aggregation here — averaging over folds where seeds were meant, or dropping
a model from the ordering — produces a table that looks entirely reasonable.
"""

from __future__ import annotations

import pandas as pd
import pytest

from vgtfm.evaluate.reporting import TISSUE_BALANCED
from vgtfm.figures.tables import (
    MODEL_LABELS,
    TISSUE_LABELS,
    tex_escape,
    annotation_table,
    dataset_table,
    delta_table,
    effective_rank_table,
    fold_hierarchy_table,
    integration_table,
    per_organ_table,
    shuffle_table,
    tissue_label,
)


def results_frame():
    """Two models x two seeds, pooled rows plus three folds each."""
    rows = []
    for model, pooled, folds in (
        ("pca", 0.42, [0.30, 0.45, 0.48]),
        ("ae", 0.35, [0.25, 0.36, 0.40]),
    ):
        for seed in (42, 43):
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "protocol": "heldout_donor",
                    "level": "cross_donor",
                    "scope": "global",
                    "fold": "pooled",
                    "f1_score": pooled + 0.01 * (seed == 43),
                    "n_donors": 7,
                }
            )
            for i, f in enumerate(folds):
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "protocol": "heldout_donor",
                        "level": "cross_donor",
                        "scope": "global",
                        "fold": f"fold{i}",
                        "f1_score": f,
                        "n_donors": 1,
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_frame():
    return pd.DataFrame(
        [
            {
                "model": m,
                "protocol": "heldout_donor",
                "level": "cross_donor",
                "scope": "global",
                "class": "macro",
                "ci_lo": lo,
                "ci_hi": hi,
            }
            for m, lo, hi in (("pca", 0.36, 0.47), ("ae", 0.29, 0.41))
        ]
    )


# ── Table 1 ──────────────────────────────────────────────────────────


def test_annotation_table_averages_seeds_and_ranges_over_folds(tmp_path):
    df = annotation_table(results_frame(), bootstrap_frame(), tmp_path)

    pca = df.set_index("model").loc["pca"]
    assert pca["macro_f1"] == pytest.approx(0.425)  # mean of the two pooled rows
    assert pca["n_seeds"] == 2
    assert pca["fold_min"] == pytest.approx(0.30)  # from the per-fold rows only
    assert pca["fold_max"] == pytest.approx(0.48)
    assert pca["n_folds"] == 3
    assert (pca["donor_ci_lo"], pca["donor_ci_hi"]) == (0.36, 0.47)


def test_the_figures_stage_writes_one_table_per_organ_when_a_level_spans_several(tmp_path):
    """Once USZ enters cross_donor the global row is a macro over (tissue, class)
    cells. That is the right pooled number and the wrong one to read a single organ
    off, so each organ gets its own table."""
    from vgtfm.figures import build as fig_build

    rows = []
    for scope in ("global", "skin", "lung", "kidney"):
        for model in ("pca", "pca_oracle"):
            rows.append(
                {
                    "model": model,
                    "seed": 42,
                    "protocol": "heldout_donor",
                    "level": "cross_donor",
                    "scope": scope,
                    "fold": "pooled",
                    "f1_score": 0.4,
                    "n_donors": 7,
                }
            )
    made = fig_build._annotation_tables(pd.DataFrame(rows), None, tmp_path)

    assert "table_annotation_heldout_donor_cross_donor" in made
    for organ in ("skin", "lung", "kidney"):
        stem = f"table_annotation_heldout_donor_cross_donor_{organ}"
        assert stem in made and (tmp_path / f"{stem}.csv").exists()


def test_a_single_organ_level_gets_no_redundant_per_organ_table(tmp_path):
    """With one organ the global table already *is* the organ's table."""
    from vgtfm.figures import build as fig_build

    rows = [
        {
            "model": "pca",
            "seed": 42,
            "protocol": "heldout_donor",
            "level": "cross_donor",
            "scope": scope,
            "fold": "pooled",
            "f1_score": 0.4,
            "n_donors": 7,
        }
        for scope in ("global", "skin")
    ]
    made = fig_build._annotation_tables(pd.DataFrame(rows), None, tmp_path)
    assert made == [
        "table_annotation_heldout_donor_cross_donor",
        "fig_annotation_heldout_donor_cross_donor",
    ]


def test_the_pooled_score_and_the_fold_range_are_distinguishable(tmp_path):
    """A pooled macro average can sit outside its own fold range — the majority
    baseline at cross_replicate really does. `fold_mean` is what the range brackets,
    so the two statistics can be told apart rather than reading as an error."""
    rows = [
        {
            "model": "majority",
            "seed": 42,
            "protocol": "heldout_donor",
            "level": "cross_replicate",
            "scope": "global",
            "fold": "pooled",
            "f1_score": 0.2525,
            "n_donors": 7,
        }
    ]
    rows += [
        {
            "model": "majority",
            "seed": 42,
            "protocol": "heldout_donor",
            "level": "cross_replicate",
            "scope": "global",
            "fold": f"f{i}",
            "f1_score": v,
            "n_donors": 1,
        }
        for i, v in enumerate([0.13, 0.17, 0.19])
    ]
    df = annotation_table(pd.DataFrame(rows), None, tmp_path, level="cross_replicate")
    r = df.set_index("model").loc["majority"]
    assert r["macro_f1"] == pytest.approx(0.2525)  # pooled over all folds
    assert r["fold_mean"] == pytest.approx(0.49 / 3)  # mean of the per-fold ones
    assert (r["fold_min"], r["fold_max"]) == (0.13, 0.19)
    assert r["macro_f1"] > r["fold_max"], "the case this column exists to explain"


def test_annotation_table_orders_models_as_the_paper_lists_them(tmp_path):
    df = annotation_table(results_frame(), None, tmp_path)
    assert list(df["model"]) == ["pca", "ae"]
    assert df.loc[0, "label"] == MODEL_LABELS["pca"]


def test_annotation_table_writes_both_a_csv_and_a_latex_fragment(tmp_path):
    annotation_table(results_frame(), bootstrap_frame(), tmp_path)
    csv = tmp_path / "table_annotation_cross_donor.csv"
    tex = tmp_path / "table_annotation_cross_donor.tex"
    assert csv.exists() and tex.exists()

    body = tex.read_text()
    assert body.startswith(r"\begin{tabular}") and body.rstrip().endswith(r"\end{tabular}")
    assert body.count(r"\\") == 3  # header + one per model
    assert "0.425" in body and "[0.300, 0.480]" in body
    # The CSV and the LaTeX are generated from the same frame, never separately.
    assert len(pd.read_csv(csv)) == 2


def test_annotation_table_survives_a_missing_bootstrap(tmp_path):
    df = annotation_table(results_frame(), pd.DataFrame(), tmp_path)
    assert "donor_ci_lo" not in df.columns
    assert "--" in (tmp_path / "table_annotation_cross_donor.tex").read_text()


def test_annotation_table_of_an_absent_protocol_is_empty(tmp_path):
    assert annotation_table(results_frame(), None, tmp_path, protocol="loso_donor").empty
    assert not (tmp_path / "table_annotation_cross_donor.tex").exists()


def test_annotation_table_honours_the_requested_stem(tmp_path):
    annotation_table(results_frame(), None, tmp_path, stem="table_custom")
    assert (tmp_path / "table_custom.csv").exists()


# ── Table 2 ──────────────────────────────────────────────────────────


def effective_rank_frame():
    return pd.DataFrame(
        [
            {
                "column": "gene_features",
                "variant": "observed",
                "dim": 1152,
                "eff_rank": 29.3,
                "eff_rank_std": 0.4,
                "pct_of_dim": 2.5,
                "mean_pairwise_cosine": 0.979,
            },
            {
                "column": "patch_features",
                "variant": "observed",
                "dim": 3072,
                "eff_rank": 78.7,
                "eff_rank_std": 1.1,
                "pct_of_dim": 2.6,
                "mean_pairwise_cosine": 0.62,
            },
            {
                "column": "gene_features",
                "variant": "column_shuffled",
                "dim": 1152,
                "eff_rank": 900.0,
                "eff_rank_std": 2.0,
                "pct_of_dim": 78.1,
                "mean_pairwise_cosine": 0.01,
            },
        ]
    )


def test_effective_rank_table_reports_controls_in_their_own_block(tmp_path):
    df = effective_rank_table(effective_rank_frame(), tmp_path)
    assert list(df["representation"]) == ["gene", "patch", "gene"]

    body = (tmp_path / "table_effective_rank.tex").read_text()
    assert r"\emph{controls}" in body
    assert body.index("gene &") < body.index("controls")  # observed rows first
    # The percentage column is the one to argue from, and it is present per row.
    assert "2.5" in body and "2.6" in body and "78.1" in body


def test_effective_rank_table_escapes_identifiers_for_latex(tmp_path):
    effective_rank_table(effective_rank_frame(), tmp_path)
    body = (tmp_path / "table_effective_rank.tex").read_text()
    assert "column\\_shuffled" in body
    assert "column_shuffled" not in body.replace("column\\_shuffled", "")


def test_effective_rank_table_of_nothing_is_empty(tmp_path):
    assert effective_rank_table(pd.DataFrame(), tmp_path).empty


# ── Table 3 ──────────────────────────────────────────────────────────


def per_class_frame():
    rows = []
    for condition, f1 in (
        ("none", 0.51),
        ("within-sample", 0.50),
        ("global", 0.49),
        ("gaussian", 0.48),
    ):
        for seed in (42, 43):
            for scope, cls in (("melanoma", "TUM"), ("kidney", "TLS")):
                rows.append(
                    {
                        "protocol": "pooled_loso",
                        "level": "all",
                        "fold": "pooled",
                        "scope": scope,
                        "class": cls,
                        "condition": condition,
                        "seed": seed,
                        "f1_score": f1,
                        "support": 1200,
                    }
                )
    # A global-scope row that the table must ignore in favour of per-tissue rows.
    rows.append(
        {
            "protocol": "pooled_loso",
            "level": "all",
            "fold": "pooled",
            "scope": "global",
            "class": "TUM",
            "condition": "none",
            "seed": 42,
            "f1_score": 0.9,
            "support": 5000,
        }
    )
    return pd.DataFrame(rows)


def test_shuffle_table_is_per_tissue_and_per_condition(tmp_path):
    df = shuffle_table(per_class_frame(), tmp_path)
    assert set(df["tissue"]) == {"melanoma", "kidney"}  # global row dropped
    assert set(df["condition"]) == {"none", "within-sample", "global", "gaussian"}
    assert (df["n_seeds"] == 2).all()
    assert df[df.condition == "none"]["f1"].iloc[0] == pytest.approx(0.51)

    body = (tmp_path / "table_patch_shuffle.tex").read_text()
    for header in ("matched", "within-slide shuffle", "global shuffle", "Gaussian"):
        assert header in body


def test_shuffle_table_needs_a_condition_column(tmp_path):
    """Called on ordinary eval output it must decline, not invent a column."""
    plain = per_class_frame().drop(columns=["condition"])
    assert shuffle_table(plain, tmp_path).empty
    assert shuffle_table(pd.DataFrame(), tmp_path).empty


# ── paired deltas ────────────────────────────────────────────────────


def delta_frame():
    return pd.DataFrame(
        [
            {
                "model": "ae",
                "scope": "global",
                "protocol": "heldout_donor",
                "level": "cross_donor",
                "class": "macro",
                "delta": -0.07,
                "ci_lo": -0.12,
                "ci_hi": -0.02,
                "p_two_sided": 0.01,
            },
            {
                "model": "ae",
                "scope": "global",
                "protocol": "heldout_donor",
                "level": "cross_donor",
                "class": "TUM",
                "delta": -0.05,
                "ci_lo": -0.10,
                "ci_hi": 0.01,
                "p_two_sided": 0.2,
            },
        ]
    )


def test_delta_table_reports_macro_rows_with_signed_intervals(tmp_path):
    df = delta_table(delta_frame(), tmp_path)
    assert len(df) == 1  # per-class rows excluded
    assert df.loc[0, "delta"] == pytest.approx(-0.07)

    body = (tmp_path / "table_deltas_cross_donor.tex").read_text()
    assert "-0.070" in body and "[-0.120, -0.020]" in body


def test_delta_table_of_an_absent_slice_is_empty(tmp_path):
    assert delta_table(delta_frame(), tmp_path, level="cross_region").empty
    assert delta_table(pd.DataFrame(), tmp_path).empty


# ── LaTeX escaping ───────────────────────────────────────────────────


def test_tex_escaping_is_narrow_on_purpose():
    """Only the characters our identifiers actually contain; hand-written LaTeX passes."""
    assert tex_escape("pct_of_dim") == r"pct\_of\_dim"
    assert tex_escape("50%") == r"50\%"
    assert tex_escape("a#b") == r"a\#b"
    assert tex_escape(r"AE (gene$\rightarrow$H\&E)") == r"AE (gene$\rightarrow$H\&E)"
    assert tex_escape(3.5) == "3.5"


def test_every_model_label_is_latex_safe():
    """These strings are written into the paper verbatim."""
    for name, label in MODEL_LABELS.items():
        bare = label.replace(r"\&", "").replace(r"\_", "")
        assert "&" not in bare, name
        assert "_" not in bare, name
        assert label.count("$") % 2 == 0, name  # balanced math mode


# ── dataset appendix ─────────────────────────────────────────────────


def cohort_frame():
    """Three cohorts: a fitted one with replicate slides, and two held-out ones."""
    rows = []
    for sid, split in (
        ("P10-B1", "train"),
        ("P10-T1", "train"),
        ("P10-T2", "train"),
        ("P15-B1", "train"),
        ("P15-T1", "validation"),
    ):
        rows.append(
            {
                "sample_id": sid,
                "dataset_id": "LUAD",
                "tissue": "lung",
                "split": split,
                "n_spots": 1000,
            }
        )
    for sid in ("KC1", "KC3"):
        rows.append(
            {
                "sample_id": sid,
                "dataset_id": "USZ_kidney",
                "tissue": "kidney",
                "split": "test",
                "n_spots": 500,
            }
        )
    for sid in ("LC1", "LC2"):
        rows.append(
            {
                "sample_id": sid,
                "dataset_id": "USZ_lung",
                "tissue": "lung",
                "split": "test",
                "n_spots": 700,
            }
        )
    return pd.DataFrame(rows)


def cohort_registry():
    return {
        "LUAD": {
            "name": "Lung Cancer LUAD",
            "tissue": "lung",
            "donor_pattern": r"^(P\d+)-",
            "citation": {"accession": "E-MTAB-13530", "bibkey": "dezuani2024nsclc"},
        },
        "USZ_kidney": {
            "name": "TLS Visium USZ",
            "tissue": "kidney",
            "donor_pattern": None,
            "citation": {"accession": "Zenodo 14620362", "bibkey": "dawo2025tls"},
        },
        "USZ_lung": {
            "name": "TLS Visium USZ",
            "tissue": "lung",
            "donor_pattern": None,
            "citation": {"accession": "Zenodo 14620362", "bibkey": "dawo2025tls"},
        },
    }


def test_dataset_table_totals_match_the_cohort_table(tmp_path):
    """The counts in the table are the ones the run loaded."""
    cohort = cohort_frame()
    df = dataset_table(cohort, cohort_registry(), tmp_path)
    assert df["samples"].sum() == len(cohort)
    assert df["spots"].sum() == cohort["n_spots"].sum()
    assert set(df["dataset_id"]) == set(cohort["dataset_id"])


def test_dataset_table_splits_cohorts_by_whether_the_fit_ever_saw_them(tmp_path):
    """A cohort is Training when *any* slide sits in the fitted split."""
    df = dataset_table(cohort_frame(), cohort_registry(), tmp_path).set_index("dataset_id")
    assert df.loc["LUAD", "split"] == "Training"  # has train and validation
    assert df.loc["USZ_kidney", "split"] == "Held-out eval"
    assert df.loc["USZ_lung", "split"] == "Held-out eval"


def test_a_different_fit_split_relabels_the_cohorts(tmp_path):
    df = dataset_table(cohort_frame(), cohort_registry(), tmp_path, fit_split="test").set_index(
        "dataset_id"
    )
    assert df.loc["LUAD", "split"] == "Held-out eval"
    assert df.loc["USZ_kidney", "split"] == "Training"


def test_dataset_table_counts_patients_not_slides(tmp_path):
    """Five LUAD slides are three from P10 and two from P15 — two patients."""
    df = dataset_table(cohort_frame(), cohort_registry(), tmp_path).set_index("dataset_id")
    assert df.loc["LUAD", "donors"] == 2
    assert df.loc["LUAD", "samples"] == 5
    # No pattern: every slide is its own patient.
    assert df.loc["USZ_kidney", "donors"] == df.loc["USZ_kidney", "samples"] == 2


def test_a_donor_pattern_that_does_not_match_falls_back_to_the_slide(tmp_path):
    """A typo in the regex must not silently merge unrelated patients."""
    reg = cohort_registry()
    reg["LUAD"]["donor_pattern"] = r"^(Q\d+)-"
    df = dataset_table(cohort_frame(), reg, tmp_path).set_index("dataset_id")
    assert df.loc["LUAD", "donors"] == 5


def test_two_cohorts_sharing_a_name_are_disambiguated_by_organ(tmp_path):
    df = dataset_table(cohort_frame(), cohort_registry(), tmp_path).set_index("dataset_id")
    assert df.loc["USZ_kidney", "qualifier"] == "kidney"
    assert df.loc["USZ_lung", "qualifier"] == "lung"
    assert df.loc["LUAD", "qualifier"] == ""  # unique name, no qualifier

    body = (tmp_path / "table_datasets.tex").read_text()
    assert "TLS Visium USZ (kidney, Zenodo 14620362)" in body
    assert "TLS Visium USZ (lung, Zenodo 14620362)" in body


def test_dataset_table_groups_organs_and_orders_them_by_size(tmp_path):
    df = dataset_table(cohort_frame(), cohort_registry(), tmp_path)
    assert list(df["organ"]) == ["Lung", "Lung", "Kidney"]  # 6400 spots vs 1000
    # Training before held-out inside an organ, and the organ cell is not repeated.
    assert list(df["split"])[:2] == ["Training", "Held-out eval"]
    body = (tmp_path / "table_datasets.tex").read_text()
    assert body.count("Lung &") == 1
    assert r"\cmidrule(lr){1-6}" in body


def test_dataset_table_writes_a_csv_a_latex_fragment_and_a_total(tmp_path):
    df = dataset_table(cohort_frame(), cohort_registry(), tmp_path)
    assert (tmp_path / "table_datasets.csv").exists()
    body = (tmp_path / "table_datasets.tex").read_text()
    assert body.startswith(r"\begin{tabular}") and body.rstrip().endswith(r"\end{tabular}")
    assert r"\citep{dezuani2024nsclc}" in body
    assert r"\textbf{Total}" in body
    assert f"\\textbf{{{df['spots'].sum():,}}}" in body


def test_dataset_table_omits_a_citation_it_does_not_have(tmp_path):
    """TuPro has no public accession; the row must still render."""
    reg = cohort_registry()
    reg["LUAD"]["citation"] = {}
    body_df = dataset_table(cohort_frame(), reg, tmp_path)
    assert body_df.set_index("dataset_id").loc["LUAD", "accession"] == ""
    body = (tmp_path / "table_datasets.tex").read_text()
    assert "Lung Cancer LUAD & 2 &" in body  # bare name, no parens
    assert r"\citep{dezuani2024nsclc}" not in body


def test_dataset_table_of_an_empty_cohort_is_empty(tmp_path):
    assert dataset_table(pd.DataFrame(), cohort_registry(), tmp_path).empty


def test_tissue_labels_are_display_names_not_registry_keys():
    assert tissue_label("ovarian") == "Ovary"
    assert tissue_label("lymph-nodes") == "Lymph node"
    assert tissue_label("pancreas") == "Pancreas"  # unknown key degrades
    for key, label in TISSUE_LABELS.items():
        assert label == tex_escape(label), key  # LaTeX-safe verbatim


# ── Table 3's intervals ──────────────────────────────────────────────


def shuffle_bootstrap_frame():
    """Donor-level intervals for the same rows :func:`per_class_frame` covers."""
    rows = []
    for condition, lo, hi in (
        ("none", 0.41, 0.61),
        ("within-sample", 0.40, 0.60),
        ("global", 0.39, 0.59),
        ("gaussian", 0.38, 0.58),
    ):
        for seed in (42, 43):
            for scope, cls in (("melanoma", "TUM"), ("kidney", "TLS")):
                rows.append(
                    {
                        "protocol": "pooled_loso",
                        "level": "all",
                        "scope": scope,
                        "class": cls,
                        "condition": condition,
                        "seed": seed,
                        "ci_lo": lo,
                        "ci_hi": hi,
                    }
                )
    # Macro and global rows the per-class table must not pick up.
    rows.append(
        {
            "protocol": "pooled_loso",
            "level": "all",
            "scope": "melanoma",
            "class": "macro",
            "condition": "none",
            "seed": 42,
            "ci_lo": 0.0,
            "ci_hi": 1.0,
        }
    )
    rows.append(
        {
            "protocol": "pooled_loso",
            "level": "all",
            "scope": "global",
            "class": "TUM",
            "condition": "none",
            "seed": 42,
            "ci_lo": 0.0,
            "ci_hi": 1.0,
        }
    )
    return pd.DataFrame(rows)


def test_shuffle_table_carries_the_interval_its_caption_promises(tmp_path):
    """Table 3's caption claims a donor-level 95% CI; the table must show one."""
    df = shuffle_table(per_class_frame(), tmp_path, bootstrap=shuffle_bootstrap_frame())
    assert {"ci_lo", "ci_hi"} <= set(df.columns)

    body = (tmp_path / "table_patch_shuffle.tex").read_text()
    assert "[0.410, 0.610]" in body  # matched
    assert "[0.380, 0.580]" in body  # Gaussian
    # The macro and global rows carry 0..1 and would be unmistakable if picked up.
    assert "[0.000, 1.000]" not in body


def test_shuffle_table_without_a_bootstrap_still_builds(tmp_path):
    """`ablate` writes no bootstrap when eval.bootstrap_n is 0; that is not a defect."""
    df = shuffle_table(per_class_frame(), tmp_path)
    assert not df.empty
    body = (tmp_path / "table_patch_shuffle.tex").read_text()
    assert "0.510" in body and "[" not in body.split(r"\midrule")[1]


def test_the_table_orders_baselines_by_how_much_they_are_allowed_to_see(tmp_path):
    """Counts, then the gene FM, then morphology-trained, then the oracles."""
    rows = []
    for m in ("pca_oracle", "ae", "hvg_pca", "majority", "pca", "pca_oracle_matched"):
        rows.append(
            {
                "model": m,
                "seed": 42,
                "protocol": "heldout_donor",
                "level": "cross_donor",
                "scope": "global",
                "fold": "pooled",
                "f1_score": 0.4,
                "n_donors": 7,
            }
        )
    df = annotation_table(pd.DataFrame(rows), None, tmp_path)
    assert list(df["model"]) == [
        "hvg_pca",
        "pca",
        "ae",
        "pca_oracle_matched",
        "pca_oracle",
        "majority",
    ]


def test_every_registered_model_has_a_display_name():
    """A model missing from MODEL_LABELS would reach the table as its config key."""
    from vgtfm.models.base import MODEL_NAMES

    assert set(MODEL_NAMES) <= set(MODEL_LABELS)


# ── the fold hierarchy ───────────────────────────────────────────────


def _hierarchy_cohort():
    """Two TuPro donors (one with two regions), two USZ slides, one unannotated."""
    rows = [
        ("MACEGEJ-1-1", "10x_TuPro", "skin", "MACEGEJ", 1, 1, 100, 5),
        ("MACEGEJ-1-2", "10x_TuPro", "skin", "MACEGEJ", 1, 2, 100, 5),
        ("MACEGEJ-2-1", "10x_TuPro", "skin", "MACEGEJ", 2, 1, 100, 4),
        ("MAHEFOG-1-1", "10x_TuPro", "skin", "MAHEFOG", 1, 1, 100, 4),
        ("KC1", "TLS_VISIUM_USZ_kidney", "kidney", "KC1", -1, -1, 50, 4),
        ("LC1", "TLS_VISIUM_USZ_lung", "lung", "LC1", -1, -1, 50, 4),
        ("MW-B-001a-vis", "MOSAIC_Bladder", "bladder", "MW-B-001a-vis", -1, -1, 0, 0),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "sample_id",
            "dataset_id",
            "tissue",
            "donor",
            "region",
            "replicate",
            "n_annotated",
            "n_classes",
        ],
    )


def _hierarchy_folds():
    return {
        "cross_replicate": [{"eval_sample_ids": ["MACEGEJ-1-2"]}],
        "cross_region": [{"eval_sample_ids": ["MACEGEJ-2-1"]}],
        "cross_donor": [
            {"eval_sample_ids": ["MACEGEJ-1-1", "MACEGEJ-1-2", "MACEGEJ-2-1"]},
            {"eval_sample_ids": ["MAHEFOG-1-1"]},
            {"eval_sample_ids": ["KC1"]},
            {"eval_sample_ids": ["LC1"]},
        ],
    }


def test_the_hierarchy_table_covers_every_evaluated_cohort(tmp_path):
    df = fold_hierarchy_table(_hierarchy_cohort(), _hierarchy_folds(), tmp_path)
    assert set(df["tissue"]) == {"skin", "kidney", "lung"}
    # An unannotated slide trains the representation and defines no fold.
    assert "MW-B-001a-vis" not in set(df["sample_id"])


def test_the_hierarchy_table_says_which_levels_hold_a_slide_out(tmp_path):
    df = fold_hierarchy_table(_hierarchy_cohort(), _hierarchy_folds(), tmp_path).set_index(
        "sample_id"
    )
    assert df.loc["MACEGEJ-1-2", "eval_levels"] == "cross_replicate|cross_donor"
    assert df.loc["MACEGEJ-2-1", "eval_levels"] == "cross_region|cross_donor"
    # USZ carries no region or replicate, so it is held out at the patient level only.
    assert df.loc["KC1", "eval_levels"] == "cross_donor"
    assert df.loc["LC1", "n_eval_folds"] == 1


def test_a_cohort_without_regions_gets_a_dash_not_an_invented_region(tmp_path):
    """`-1` means the cohort captured one block per patient, not "unknown"."""
    df = fold_hierarchy_table(_hierarchy_cohort(), _hierarchy_folds(), tmp_path).set_index(
        "sample_id"
    )
    assert df.loc["KC1", "region_label"] == "--"
    assert df.loc["MACEGEJ-2-1", "region_label"] == "Region 2"


def test_the_hierarchy_latex_nests_donor_and_region(tmp_path):
    """The published table's shape: a donor spans its regions, a region its slides."""
    fold_hierarchy_table(_hierarchy_cohort(), _hierarchy_folds(), tmp_path)
    tex = (tmp_path / "table_fold_hierarchy.tex").read_text()
    assert "\\usepackage{booktabs,multirow}" in tex
    assert "\\multirow{3}{*}{MACEGEJ}" in tex  # 3 slides under one donor
    assert "\\multirow{2}{*}{Region 1}" in tex  # 2 replicates under one region
    # A donor with a single slide is a plain cell, not a one-row multirow.
    assert "\\multirow{1}" not in tex
    assert tex.count("MACEGEJ") == 4  # the label plus its 3 slides


def test_the_hierarchy_table_survives_a_missing_fold_map(tmp_path):
    """`figures` can run before `data` recorded folds.json; the nesting still holds."""
    df = fold_hierarchy_table(_hierarchy_cohort(), {}, tmp_path)
    assert len(df) == 6 and set(df["eval_levels"]) == {""}
    assert (tmp_path / "table_fold_hierarchy.csv").exists()


def test_an_empty_cohort_yields_no_hierarchy_table(tmp_path):
    assert fold_hierarchy_table(pd.DataFrame(), {}, tmp_path).empty
    assert not (tmp_path / "table_fold_hierarchy.csv").exists()


# ── per-organ and organ-balanced tables ──────────────────────────────


def _multi_organ_frames():
    """Three organs with very different scores, plus the organ-balanced summary."""
    per_organ = {"kidney": (0.20, 2), "lung": (0.45, 4), "skin": (0.50, 7)}
    rows, boot = [], []
    for model in ("pca", "ae"):
        bump = 0.0 if model == "pca" else 0.06
        for scope, (f1, n) in per_organ.items():
            rows.append(
                {
                    "model": model,
                    "seed": 42,
                    "protocol": "heldout_donor",
                    "level": "cross_donor",
                    "scope": scope,
                    "fold": "pooled",
                    "f1_score": f1 + bump,
                    "n_donors": n,
                }
            )
            boot.append(
                {
                    "model": model,
                    "protocol": "heldout_donor",
                    "level": "cross_donor",
                    "scope": scope,
                    "class": "macro",
                    "ci_lo": f1 + bump - 0.05,
                    "ci_hi": f1 + bump + 0.05,
                }
            )
        balanced = sum(f1 for f1, _ in per_organ.values()) / 3 + bump
        rows.append(
            {
                "model": model,
                "seed": 42,
                "protocol": "heldout_donor",
                "level": "cross_donor",
                "scope": TISSUE_BALANCED,
                "fold": "pooled",
                "f1_score": balanced,
                "n_donors": 13,
            }
        )
        boot.append(
            {
                "model": model,
                "protocol": "heldout_donor",
                "level": "cross_donor",
                "scope": TISSUE_BALANCED,
                "class": "macro",
                "ci_lo": balanced - 0.12,
                "ci_hi": balanced + 0.12,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(boot)


def test_per_organ_table_puts_every_organ_on_one_row_per_model(tmp_path):
    results, boot = _multi_organ_frames()
    df = per_organ_table(results, boot, tmp_path)
    assert list(df["model"]) == ["pca", "ae"]
    for organ, n in (("kidney", 2), ("lung", 4), ("skin", 7)):
        assert f"{organ}_macro_f1" in df.columns
        assert df[f"{organ}_n_donors"].max() == n
    assert df.loc[df.model == "pca", "kidney_macro_f1"].iloc[0] == pytest.approx(0.20)
    assert df.loc[df.model == "ae", "skin_macro_f1"].iloc[0] == pytest.approx(0.56)


def test_per_organ_table_carries_each_organs_own_interval(tmp_path):
    """A per-organ CI must come from that organ, not from the pooled row."""
    results, boot = _multi_organ_frames()
    df = per_organ_table(results, boot, tmp_path)
    r = df[df.model == "pca"].iloc[0]
    assert r["kidney_ci_lo"] == pytest.approx(0.15)
    assert r["kidney_ci_hi"] == pytest.approx(0.25)
    assert r["skin_ci_lo"] == pytest.approx(0.45)


def test_per_organ_table_appends_the_organ_balanced_column(tmp_path):
    results, boot = _multi_organ_frames()
    df = per_organ_table(results, boot, tmp_path)
    assert df.loc[df.model == "pca", "tissue_balanced_macro_f1"].iloc[0] == pytest.approx(
        (0.20 + 0.45 + 0.50) / 3
    )
    tex = (tmp_path / "table_annotation_by_organ_heldout_donor_cross_donor.tex").read_text()
    assert "Organ-balanced" in tex and "Kidney ($n$=2)" in tex


def test_per_organ_table_is_empty_without_organ_scopes(tmp_path):
    results, _ = _multi_organ_frames()
    only_global = results.assign(scope="global")
    assert per_organ_table(only_global, None, tmp_path).empty


def test_the_organ_balanced_table_drops_the_fold_range_column(tmp_path):
    """That scope averages over organs, so it has no per-fold rows to bracket."""
    results, boot = _multi_organ_frames()
    df = annotation_table(results, boot, tmp_path, scope=TISSUE_BALANCED, stem="t_balanced")
    assert df["fold_min"].isna().all()
    tex = (tmp_path / "t_balanced.tex").read_text()
    assert "fold range" not in tex
    assert "organ- and donor-level 95\\% CI" in tex
    # Four columns, not five: label, macro-F1, CI, n. Counting `&` over the whole
    # fragment would also catch the escaped ampersand inside the AE label.
    assert r"\begin{tabular}{lccr}" in tex
    header = [ln for ln in tex.splitlines() if "macro-F1" in ln][0]
    assert header.count("&") == 3


# ── the integration appendix table ───────────────────────────────────


def integration_frames():
    """One `integrate` run: four methods, one of them graph-only.

    Two organs, so the pooled `global` scope and the per-organ ones are different
    columns rather than the same number under two names.
    """

    def probe(scope, f1, lo, hi, n):
        return {
            f"f1/{scope}/cross_donor": f1,
            f"f1/{scope}/cross_donor_lo": lo,
            f"f1/{scope}/cross_donor_hi": hi,
            f"f1/{scope}/cross_donor_n_donors": n,
        }

    rows = [
        {
            "method": "none",
            "dim": 128,
            "batch/ilisi": 0.043,
            "batch/kbet": 0.190,
            "bio/clisi": 0.984,
            **probe("global", 0.445, 0.36, 0.52, 13),
            **probe("skin", 0.523, 0.46, 0.60, 7),
            **probe("lung", 0.402, 0.31, 0.48, 4),
        },
        {
            "method": "harmony",
            "dim": 128,
            "batch/ilisi": 0.090,
            "batch/kbet": 0.202,
            "bio/clisi": 0.975,
            **probe("global", 0.419, 0.33, 0.50, 13),
            **probe("skin", 0.470, 0.40, 0.55, 7),
            **probe("lung", 0.438, 0.35, 0.51, 4),
        },
        {
            "method": "bbknn",
            "batch/ilisi": 0.191,
            "batch/kbet": 0.137,
            "bio/clisi": 0.946,
            "note": "graph-only; no embedding to probe",
        },
        {
            "method": "scvi",
            "dim": 128,
            "batch/ilisi": 0.050,
            "batch/kbet": 0.224,
            "bio/clisi": 0.981,
            **probe("global", 0.485, 0.40, 0.55, 13),
            **probe("skin", 0.545, 0.48, 0.61, 7),
            **probe("lung", 0.451, 0.37, 0.53, 4),
        },
    ]
    deltas = [
        {
            "method": "harmony",
            "level": "cross_donor",
            "scope": "global",
            "delta_f1": -0.026,
            "ci_lo": -0.048,
            "ci_hi": -0.004,
            "p_two_sided": 0.02,
            "n_donors": 13,
        },
        # Harmony costs skin and buys lung; the pooled column is the average of the
        # two, which is the reason the per-organ table exists.
        {
            "method": "harmony",
            "level": "cross_donor",
            "scope": "skin",
            "delta_f1": -0.053,
            "ci_lo": -0.081,
            "ci_hi": -0.019,
            "p_two_sided": 0.01,
            "n_donors": 7,
        },
        {
            "method": "harmony",
            "level": "cross_donor",
            "scope": "lung",
            "delta_f1": 0.036,
            "ci_lo": -0.007,
            "ci_hi": 0.079,
            "p_two_sided": 0.11,
            "n_donors": 4,
        },
        {
            "method": "harmony",
            "level": "cross_region",
            "scope": "global",
            "delta_f1": 0.006,
            "ci_lo": -0.03,
            "ci_hi": 0.04,
            "p_two_sided": 0.7,
            "n_donors": 9,
        },
        {
            "method": "scvi",
            "level": "cross_donor",
            "scope": "global",
            "delta_f1": 0.039,
            "ci_lo": 0.011,
            "ci_hi": 0.068,
            "p_two_sided": 0.008,
            "n_donors": 13,
        },
    ]
    return pd.DataFrame(rows), pd.DataFrame(deltas)


def test_integration_table_pairs_each_batch_score_with_its_probe_score(tmp_path):
    df = integration_table(*integration_frames(), tmp_path)

    assert list(df["label"]) == ["Uncorrected", "Harmony", "BBKNN", "scVI"]
    tex = (tmp_path / "table_integration.tex").read_text()
    # Harmony's row carries the mixing it bought and the F1 it cost, side by side.
    harmony = [ln for ln in tex.splitlines() if ln.startswith("Harmony")][0]
    assert "0.090" in harmony and "-0.026" in harmony and "0.020" in harmony


def test_the_integration_table_reads_deltas_from_the_level_it_reports(tmp_path):
    """`integration_deltas.csv` carries every fold level; a table for one of them
    must not pick up another's interval."""
    df = integration_table(*integration_frames(), tmp_path, level="cross_region")

    # cross_region has a delta for harmony but no absolute F1 columns in this run,
    # so the delta arrives and the absolute score does not.
    harmony = df.set_index("method").loc["harmony"]
    assert harmony["delta_f1"] == pytest.approx(0.006)
    assert pd.isna(harmony["f1"])


def test_a_graph_only_method_keeps_its_row_with_the_probe_columns_empty(tmp_path):
    """The empty cells are the finding: BBKNN corrects the graph, so there is no
    corrected matrix for the probe to read."""
    integration, deltas = integration_frames()
    integration_table(integration, deltas, tmp_path)
    tex = (tmp_path / "table_integration.tex").read_text()

    bbknn = [ln for ln in tex.splitlines() if ln.startswith("BBKNN")][0]
    assert "0.191" in bbknn  # scored on the metrics a graph has
    assert bbknn.count("--") == 3  # macro-F1, delta and p are absent
    assert "corrects the neighbour graph" in tex


def test_the_integration_table_survives_a_run_made_before_the_intervals(tmp_path):
    """An older `integrate` wrote no deltas and no CI columns. The table is still
    the table; its interval brackets are simply absent."""
    integration, _ = integration_frames()
    bare = integration.drop(
        columns=[
            c
            for c in integration.columns
            if c.endswith(("_lo", "_hi", "_n_donors"))
            or c.startswith("f1/skin/")
            or c.startswith("f1/lung/")
        ]
    )
    df = integration_table(bare, None, tmp_path)

    tex = (tmp_path / "table_integration.tex").read_text()
    assert df["delta_f1"].isna().all()
    assert "0.445" in tex and r"\tiny" not in tex


def test_the_integration_table_reports_one_organ_when_asked_for_one(tmp_path):
    """The pooled scope averages (tissue, class) cells from every organ and
    bootstraps donors without regard to organ, so it can read as a null when two
    organs move in opposite directions. Asking for the organ is how you see that."""
    integration, deltas = integration_frames()
    skin = integration_table(integration, deltas, tmp_path, scope="skin")

    harmony = skin.set_index("method").loc["harmony"]
    assert harmony["f1"] == pytest.approx(0.470)  # skin, not the pooled 0.419
    assert harmony["delta_f1"] == pytest.approx(-0.053)
    assert harmony["n_donors"] == 7  # skin's donors, not 13

    # Named for its organ, so the pooled table it sits beside keeps the bare stem.
    tex = (tmp_path / "table_integration_skin.tex").read_text()
    assert "skin scope" in tex
    # The mixing half has no per-organ version, and the caption has to say so.
    assert "cohort-wide" in tex


def test_an_organ_with_no_delta_still_gets_its_absolute_column(tmp_path):
    """scVI has a pooled delta and no per-organ one in this fixture. Its skin row
    keeps the absolute score it does have rather than vanishing."""
    integration, deltas = integration_frames()
    skin = integration_table(integration, deltas, tmp_path, scope="skin")

    scvi = skin.set_index("method").loc["scvi"]
    assert scvi["f1"] == pytest.approx(0.545)
    assert pd.isna(scvi["delta_f1"])


def test_the_scopes_a_run_scored_are_read_off_its_columns():
    from vgtfm.figures.build import _integration_scopes

    integration, _ = integration_frames()
    # Pooled first, then organs alphabetically.
    assert _integration_scopes(integration) == ["global", "lung", "skin"]

    # A run made before the stage scored per organ carries `f1/<level>` and yields
    # the one scope it has, so an old run directory still builds its one table.
    old = integration.rename(columns=lambda c: c.replace("f1/global/", "f1/"))
    old = old.drop(columns=[c for c in old.columns if c.startswith(("f1/skin/", "f1/lung/"))])
    assert _integration_scopes(old) == ["global"]

    assert _integration_scopes(pd.DataFrame({"method": ["none"]})) == []


def test_an_absent_integration_stage_produces_no_table(tmp_path):
    assert integration_table(pd.DataFrame(), None, tmp_path).empty
    assert not (tmp_path / "table_integration.tex").exists()
