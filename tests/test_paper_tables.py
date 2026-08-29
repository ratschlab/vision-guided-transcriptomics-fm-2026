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


# ── the §4.3 enrichment tables ───────────────────────────────────────
#
# What is pinned is that the numbers cannot arrive stripped of what makes them
# checkable: a p column that is really an FDR, a caption describing a background
# only one of the three runs was scored on, or a row whose cells no longer line up
# with its column heads.


def _enrichment_run(root: Path, name: str, substrate: str, *, nes: float, sha: str = "abc") -> Path:
    """A run directory carrying just the enrichment artefacts and their sidecar."""
    scope_dir = root / name / "biosignal" / "skin"
    scope_dir.mkdir(parents=True)
    (root / name / "config.resolved.json").write_text(
        json.dumps({"run_name": name, "data": {"substrate": substrate}})
    )
    contrasts = ["guided_vs_frozen", "capacity_vs_frozen", "guided_vs_capacity"]
    sources = [*pt.SECTION_PATHWAYS, "HYPOXIA", "MYOGENESIS"]
    rows = []
    for i, contrast in enumerate(contrasts):
        for j, source in enumerate(sources):
            rows.append(
                {
                    "contrast": contrast,
                    "detrended": False,
                    "sample": f"cross_donor_{contrast}",
                    "source": source,
                    "set_size": 20 + j,
                    "norm": nes + 0.01 * i - 0.02 * j,
                    "pval": 0.0 if j == 0 else 0.001 * (j + 1),
                    "padj": 0.0001 * (j + 1),
                }
            )
    pd.DataFrame(rows).to_csv(scope_dir / "gsea_hallmark_cross_donor.csv", index=False)
    (scope_dir / "gsea_hallmark_cross_donor_meta.json").write_text(
        json.dumps(
            {
                "level": "cross_donor",
                "scope": "skin",
                "substrate": substrate,
                "fdr": 0.05,
                "collection": {
                    "collection": "MSigDB Hallmark (h.all)",
                    "citation": "Liberzon et al.",
                    "sha256": sha,
                    "n_sets": 50,
                    "n_genes": 6971,
                },
                "rankings": [
                    {
                        "contrast": c,
                        "detrended": False,
                        "n_sets_scored": len(sources),
                        "background_n_genes": 15944,
                        "min_set_size": 15,
                        "permutations": 10000,
                        "pval_resolution": 1e-4,
                        "seed": 42,
                        "coverage": 0.5676,
                        "multiple_testing": "Benjamini-Hochberg ...",
                    }
                    for c in contrasts
                ],
            }
        )
    )
    return root / name


def _enrichment_runs(root: Path, **kwargs) -> list:
    substrates = ["cancerfoundation", "scgpt", "geneformer"]
    for i, substrate in enumerate(substrates):
        _enrichment_run(root, f"run{i}", substrate, nes=-1.5 - 0.1 * i, **kwargs)
    return [pt.Run(root / f"run{i}") for i in range(len(substrates))]


def _cells(line: str) -> int:
    return line.count("&") + 1


def test_the_enrichment_table_rows_line_up_with_their_column_heads(tmp_path):
    """A miscounted cell shifts a q-value under a NES heading and still compiles."""
    tex, _ = pt.enrichment_table(
        _enrichment_runs(tmp_path),
        level="cross_donor",
        scope="skin",
        gene_set="hallmark",
        fdr=0.05,
        label="tab:enrichment",
    )
    body = [ln for ln in tex.splitlines() if ln.startswith(r"\quad ")]
    assert len(body) == 3 * len(pt.SECTION_PATHWAYS)
    # 1 label + (NES, p, q) + (NES, q) + (NES, q).
    assert {_cells(ln) for ln in body} == {8}
    header = next(ln for ln in tex.splitlines() if ln.startswith("Hallmark set"))
    assert _cells(header) == 8


def test_the_enrichment_table_reports_a_bound_not_a_zero_p_value(tmp_path):
    """The distinction the whole appendix turns on has to survive formatting."""
    tex, _ = pt.enrichment_table(
        _enrichment_runs(tmp_path),
        level="cross_donor",
        scope="skin",
        gene_set="hallmark",
        fdr=0.05,
        label="tab:enrichment",
    )
    assert "$<10^{-4}$" in tex
    assert "$0.0000$" not in tex
    # And the caption has to say what the reader is looking at.
    assert "permutation p-value" in tex and "Benjamini--Hochberg" in tex
    assert "15,944" in tex and "10,000" in tex


def test_the_enrichment_caption_counts_the_collection_rather_than_the_three(tmp_path):
    """Three significant sets out of fifty and three out of thirty-seven are
    different claims, and the caption has to say which it is."""
    tex, _ = pt.enrichment_table(
        _enrichment_runs(tmp_path),
        level="cross_donor",
        scope="skin",
        gene_set="hallmark",
        fdr=0.05,
        label="tab:enrichment",
    )
    assert "Across the whole collection" in tex
    assert "broad loss of per-gene predictivity" in tex


def test_the_enrichment_table_refuses_runs_scored_against_different_collections(tmp_path):
    """One caption speaks for all three blocks, so all three must share a method."""
    _enrichment_run(tmp_path, "run0", "cancerfoundation", nes=-1.5, sha="aaa")
    _enrichment_run(tmp_path, "run1", "scgpt", nes=-1.6, sha="bbb")
    runs = [pt.Run(tmp_path / "run0"), pt.Run(tmp_path / "run1")]
    with pytest.raises(SystemExit, match="collection_sha256"):
        pt.enrichment_table(
            runs, level="cross_donor", scope="skin", gene_set="hallmark", fdr=0.05, label="x"
        )


def test_the_enrichment_table_refuses_when_the_sidecar_is_missing(tmp_path):
    """Without it the table cannot state its own background or permutation count,
    and a caption written from memory is what produced the original defect."""
    run_dir = _enrichment_run(tmp_path, "run0", "cancerfoundation", nes=-1.5)
    (run_dir / "biosignal" / "skin" / "gsea_hallmark_cross_donor_meta.json").unlink()
    with pytest.raises(SystemExit, match="rescore_gsea"):
        pt.enrichment_table(
            [pt.Run(run_dir)],
            level="cross_donor",
            scope="skin",
            gene_set="hallmark",
            fdr=0.05,
            label="x",
        )


def test_the_supplementary_frame_spells_out_the_permutation_floor(tmp_path):
    """`pval` is a raw count and reads 0.0 at the floor; a spreadsheet column of
    zeros is how a bound gets cited as certainty."""
    runs = _enrichment_runs(tmp_path)
    frame = pt.enrichment_frame(runs, gene_set="hallmark", scope="skin")
    assert (frame["pval"] == 0).any()
    assert (frame["pval_upper_bound"] > 0).all()
    assert frame["pval_upper_bound"].min() == pytest.approx(1e-4)
    # Every row carries the method beside it rather than in a caption elsewhere.
    for column in ("background_n_genes", "permutations", "collection_sha256", "multiple_testing"):
        assert frame[column].notna().all(), column


# ── the §4.3 per-gene predictivity tables ────────────────────────────
#
# A run scored without the capacity control cannot silently produce a table that
# omits it, and the caption's claims are computed rather than asserted.


def _per_gene_run(root: Path, name: str, substrate: str, *, control: bool = True) -> Path:
    scope_dir = root / name / "biosignal" / "skin"
    scope_dir.mkdir(parents=True)
    (root / name / "config.resolved.json").write_text(
        json.dumps({"run_name": name, "data": {"substrate": substrate}})
    )
    rows, quartiles = [], []
    for level, scale in (("cross_replicate", 1.0), ("cross_donor", 2.0)):
        for i in range(8):
            frozen = -0.05 + 0.02 * i
            # Monotone decreasing in the frozen R^2: the shape the appendix claims.
            delta = scale * (0.004 - 0.002 * i)
            rows.append(
                {
                    "level": level,
                    "gene": f"G{i}",
                    "r2_frozen": frozen,
                    "r2_refined": frozen + delta,
                    "r2_pca_control": frozen + 0.5 * delta,
                    "delta_r2": delta,
                    "delta_r2_pca_control": 0.5 * delta,
                }
            )
        for q, i in zip(["Q1", "Q2", "Q3", "Q4"], range(4)):
            quartiles.append(
                {
                    "level": level,
                    "quartile": q,
                    "n_genes": 2,
                    "mean_delta_r2": scale * (0.004 - 0.003 * i),
                    "pct_improved": 100.0 - 25.0 * i,
                }
            )
    frame = pd.DataFrame(rows)
    if not control:
        frame = frame.drop(columns=["r2_pca_control", "delta_r2_pca_control"])
    frame.to_csv(scope_dir / "per_gene_r2.csv", index=False)
    q = pd.DataFrame(quartiles)
    for level in ("cross_replicate", "cross_donor"):
        q[q.level == level].to_csv(scope_dir / f"quartiles_{level}.csv", index=False)
    return root / name


def test_the_baseline_table_reports_both_levels_for_every_backbone(tmp_path):
    for i, sub in enumerate(["cancerfoundation", "scgpt", "geneformer"]):
        _per_gene_run(tmp_path, f"run{i}", sub)
    runs = [pt.Run(tmp_path / f"run{i}") for i in range(3)]
    tex, loaded = pt.delta_r2_baseline_table(runs, scope="skin", digits=4, label="tab:x")

    body = [ln for ln in tex.splitlines() if ln.startswith(r"\quad ")]
    assert len(body) == 3 * len(pt.QUARTILE_LABELS)
    # 1 label + (ΔR², % suppressed) per level.
    assert {_cells(ln) for ln in body} == {5}
    assert len(loaded) == 3


def test_the_baseline_caption_states_the_split_the_claim_is_about(tmp_path):
    """§4.3's sentence is about genes with negative $R^2$, so the caption answers
    in those terms rather than leaving the reader to derive it from quartiles."""
    for i, sub in enumerate(["cancerfoundation", "scgpt", "geneformer"]):
        _per_gene_run(tmp_path, f"run{i}", sub)
    runs = [pt.Run(tmp_path / f"run{i}") for i in range(3)]
    tex, loaded = pt.delta_r2_baseline_table(runs, scope="skin", digits=4, label="tab:x")

    assert "negative frozen $R^2$" in tex
    assert "monotone" in tex
    split = loaded[0].sign_split("cross_replicate")
    assert split["neg"]["n"] > 0 and split["pos"]["n"] > 0
    assert split["neg"]["mean"] > split["pos"]["mean"]


def test_the_baseline_table_refuses_a_run_scored_without_the_control(tmp_path):
    """Without it the loss cannot be separated from the change of width, which is
    the question the table exists to answer."""
    run_dir = _per_gene_run(tmp_path, "run0", "cancerfoundation", control=False)
    with pytest.raises(SystemExit, match="include_pca_control"):
        pt.delta_r2_baseline_table([pt.Run(run_dir)], scope="skin", digits=4, label="tab:x")


# ── the biological reading ───────────────────────────────────────────


def test_the_drivers_table_names_genes_from_the_scored_vocabulary(tmp_path):
    """A driver column is only worth printing if the genes in it are the genes the
    enrichment walked over. A set named from the collection but scored on a
    different vocabulary would print members that were never tested."""
    for i, sub in enumerate(["cancerfoundation", "scgpt", "geneformer"]):
        _enrichment_run(tmp_path, f"run{i}", sub, nes=-1.5 - 0.1 * i)
    runs = [pt.Run(tmp_path / f"run{i}") for i in range(3)]
    # The per-gene tables have to carry real Hallmark symbols for the join to mean
    # anything, so build them from the collection itself.
    members = pt._hallmark_members(str(pt.BiosignalConfig().resources_dir))
    symbols = sorted(set().union(*members.loc[list(pt.SECTION_PATHWAYS)]))[:60]
    for i in range(3):
        scope = tmp_path / f"run{i}" / "biosignal" / "skin"
        rows, quartiles = [], []
        for level in ("cross_replicate", "cross_donor"):
            for k, sym in enumerate(symbols):
                rows.append(
                    {
                        "level": level,
                        "gene": sym,
                        "r2_frozen": 0.4,
                        "r2_refined": 0.4 - 0.001 * k,
                        "r2_pca_control": 0.4,
                        "delta_r2": -0.001 * k,
                        "delta_r2_pca_control": 0.0,
                    }
                )
            for q, n in zip(["Q1", "Q2", "Q3", "Q4"], range(4)):
                quartiles.append(
                    {
                        "level": level,
                        "quartile": q,
                        "n_genes": 2,
                        "mean_delta_r2": -0.001 * n,
                        "pct_improved": 100.0 - 25.0 * n,
                    }
                )
        pd.DataFrame(rows).to_csv(scope / "per_gene_r2.csv", index=False)
        q = pd.DataFrame(quartiles)
        for level in ("cross_replicate", "cross_donor"):
            q[q.level == level].to_csv(scope / f"quartiles_{level}.csv", index=False)

    _, per_gene = pt.delta_r2_baseline_table(runs, scope="skin", digits=4, label="tab:x")
    loaded = [
        pt.Enrichment(r, gene_set="hallmark", level="cross_donor", scope="skin") for r in runs
    ]
    tex = pt.enrichment_drivers_table(
        loaded, per_gene, contrast="guided_vs_frozen", top_n=3, n_drivers=4, label="tab:d"
    )

    body = [ln for ln in tex.splitlines() if ln.strip().startswith(("1 &", "2 &", "3 &"))]
    assert body, tex
    assert {_cells(ln) for ln in body} == {5}
    named = {w.strip() for ln in body for w in ln.split("&")[-1].replace(r"\\", "").split(",")}
    named = {w.replace(r"\small", "").strip() for w in named}
    assert named <= set(symbols), sorted(named - set(symbols))
    # The caption has to warn that overlapping sets share drivers, or the table
    # reads as N independent findings.
    assert "not independent findings" in tex


def test_the_gene_family_table_refuses_panels_it_cannot_populate(tmp_path):
    """On a vocabulary the panels do not match -- Ensembl ids, or a different
    organism -- reporting a median over two genes would be an anecdote."""
    for i, sub in enumerate(["cancerfoundation", "scgpt", "geneformer"]):
        _per_gene_run(tmp_path, f"run{i}", sub)
    runs = [pt.Run(tmp_path / f"run{i}") for i in range(3)]
    _, per_gene = pt.delta_r2_baseline_table(runs, scope="skin", digits=4, label="tab:x")
    # `_per_gene_run` writes genes named G0..G7, which match no marker panel.
    with pytest.raises(SystemExit, match="marker panel"):
        pt.gene_family_table(per_gene, digits=3, label="tab:f")


def test_every_marker_panel_is_large_enough_to_report():
    """The panels are hand-made; what a test can pin is that none is too small for
    the median it will be summarised by."""
    for name, symbols in pt.GENE_FAMILIES.items():
        assert len(symbols) >= pt.MIN_FAMILY_MEMBERS, name
        assert len(set(symbols)) == len(symbols), f"{name} repeats a gene"
