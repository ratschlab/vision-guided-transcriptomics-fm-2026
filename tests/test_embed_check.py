"""The harness that runs the gene FMs against real checkpoints.

``scripts/embed_check.py`` is the only thing between a change to
:mod:`vgtfm.embed` and a cluster job, so its own failure modes matter: a harness
that reports success because it never ran anything, or that accepts parquets the
merge step would reject, is worse than no harness. Nothing here needs a
checkpoint, a GPU or a conda environment — the model invocation is the one part
that cannot be tested without one, and it is a single ``subprocess.run``.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))

import embed_check  # noqa: E402
from vgtfm.embed import cohort, gene_fm  # noqa: E402


def slide_h5ad(tmp_path, name, *, n_spots=8, n_genes=12, uns=False):
    """A slide in the 10x convention, optionally carrying an image in ``uns``."""
    ad = pytest.importorskip("anndata")
    sp = pytest.importorskip("scipy.sparse")
    rng = np.random.default_rng(abs(hash(name)) % 2**31)
    counts = sp.csr_matrix(rng.poisson(2.0, size=(n_spots, n_genes)).astype(np.float32))
    adata = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(index=pd.Index([f"{name}-BC{i}" for i in range(n_spots)])),
        var=pd.DataFrame(
            {"gene_ids": [f"ENSG{i:011d}" for i in range(n_genes)], "gene_name": "x"},
            index=pd.Index([f"G{i}" for i in range(n_genes)]),
        ),
    )
    adata.layers["counts"] = adata.X.copy()
    if uns:
        adata.uns["spatial"] = {"image": np.zeros((64, 64, 3), np.uint8)}
        adata.obsm["spatial"] = rng.random((n_spots, 2))
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"{name}.h5ad"
    adata.write_h5ad(path)
    return path


# ── which slides a --from-h5ad names ─────────────────────────────────


def test_a_directory_names_every_h5ad_in_it(tmp_path):
    slide_h5ad(tmp_path, "KC1")
    slide_h5ad(tmp_path, "LC1")
    assert [p.stem for p in embed_check.h5ad_paths(str(tmp_path))] == ["KC1", "LC1"]


def test_a_glob_separates_two_cohorts_that_share_a_directory(tmp_path):
    """The case this exists for: USZ kidney and lung are one directory of files."""
    for name in ("KC1", "KC2", "LC1"):
        slide_h5ad(tmp_path, name)
    assert [p.stem for p in embed_check.h5ad_paths(f"{tmp_path}/KC*.h5ad")] == ["KC1", "KC2"]
    assert [p.stem for p in embed_check.h5ad_paths(f"{tmp_path}/LC*.h5ad")] == ["LC1"]


def test_a_single_file_is_a_cohort_of_one(tmp_path):
    path = slide_h5ad(tmp_path, "KC1")
    assert embed_check.h5ad_paths(str(path)) == [path]


def test_a_path_that_matches_nothing_is_empty_rather_than_an_error(tmp_path):
    """Emptiness is reported by the caller as a Skip, with the pattern in it."""
    assert embed_check.h5ad_paths(f"{tmp_path}/nope*.h5ad") == []
    assert embed_check.h5ad_paths(f"{tmp_path}/nope.h5ad") == []


# ── building the fixture cohort ──────────────────────────────────────


def test_a_slide_is_stripped_to_counts_barcodes_and_gene_identity(tmp_path):
    ad = pytest.importorskip("anndata")
    adata = ad.read_h5ad(slide_h5ad(tmp_path, "KC1", uns=True))
    slim = embed_check.slim_slide(adata, "gene_ids")
    assert not slim.uns and not slim.obsm, "the tissue image is most of the file size"
    assert list(slim.obs_names) == list(adata.obs_names)
    assert "gene_ids" in slim.var.columns
    assert "counts" in slim.layers, "cohort._read_slide prefers the layer over X"


def test_a_slide_that_stores_counts_only_in_x_still_yields_a_counts_layer(tmp_path):
    ad = pytest.importorskip("anndata")
    adata = ad.read_h5ad(slide_h5ad(tmp_path, "KC1"))
    del adata.layers["counts"]
    assert "counts" in embed_check.slim_slide(adata, "gene_ids").layers


def test_real_slides_subsample_spots_and_keep_every_gene(tmp_path):
    ad = pytest.importorskip("anndata")
    for name in ("KC1", "KC2"):
        slide_h5ad(tmp_path / "src", name, n_spots=40, n_genes=30)
    slides = embed_check.real_slides(
        [("usz_kidney", f"{tmp_path / 'src'}/KC*.h5ad")],
        tmp_path / "out",
        n_slides=2,
        n_spots=10,
        verbose=False,
    )
    assert [s.source for s in slides] == ["usz_kidney__KC1", "usz_kidney__KC2"]
    written = ad.read_h5ad(slides[0].h5ad)
    assert written.n_obs == 10, "spots decide the runtime"
    assert written.n_vars == 30, "genes decide what a selection has to choose between"


def test_real_slides_stop_at_n_slides(tmp_path):
    for name in ("KC1", "KC2", "KC3"):
        slide_h5ad(tmp_path / "src", name)
    slides = embed_check.real_slides(
        [("usz", str(tmp_path / "src"))], tmp_path / "out", n_slides=2, n_spots=4, verbose=False
    )
    assert len(slides) == 2


def test_a_source_that_matches_no_slide_is_skipped_by_name(tmp_path):
    with pytest.raises(embed_check.Skip, match="nowhere"):
        embed_check.real_slides(
            [("usz", str(tmp_path / "nowhere"))],
            tmp_path / "out",
            n_slides=1,
            n_spots=4,
            verbose=False,
        )


def test_synthetic_slides_pool_and_carry_a_spot_with_no_counts(tmp_path):
    """The empty spot is the case that crashes upstream's binning."""
    pytest.importorskip("anndata")
    slides = embed_check.synthetic_slides(
        tmp_path, [f"G{i}" for i in range(40)], n_slides=4, n_spots=12, verbose=False
    )
    assert {s.dataset_id for s in slides} == {"cohort_a", "cohort_b"}
    pooled = cohort.pool(slides, verbose=False)
    assert pooled.n_obs == 48, "the same barcodes on every slide stay separate spots"
    per_spot = np.asarray(pooled.X.sum(axis=1)).ravel()
    assert (per_spot == 0).sum() == 4, "one empty spot per slide"


def test_synthetic_gene_names_come_from_a_checkpoint_vocabulary(tmp_path):
    (tmp_path / "cf" / "model" / "assets").mkdir(parents=True)
    (tmp_path / "cf" / "model" / "assets" / "vocab.json").write_text(
        json.dumps({"<pad>": 0, "TP53": 1, "EGFR": 2, "MYC": 3})
    )
    got = embed_check.vocabulary_symbols({"cancerfoundation": tmp_path / "cf"}, 3)
    assert got == ["TP53", "EGFR", "MYC"], "the specials are not genes"


def test_a_vocabulary_too_small_to_fill_the_request_is_not_used(tmp_path):
    (tmp_path / "vocab.json").write_text(json.dumps({"TP53": 0}))
    with pytest.raises(embed_check.Skip, match="vocabulary"):
        embed_check.vocabulary_symbols({"scgpt": tmp_path}, 10)


def test_the_floor_is_the_strictest_requested_strategys():
    """Measured on USZ slides: at 600 pooled spots global fits at 20, mixed at 30
    and per_slide needs 150, because each fits over a smaller unit than the last."""
    assert embed_check.default_min_cells(600, ["global"]) == 21
    assert embed_check.default_min_cells(600, ["mixed"]) == 30
    assert embed_check.default_min_cells(600, ["per_slide"]) == 150
    assert embed_check.default_min_cells(600, ["global", "per_slide"]) == 150


def test_the_floor_follows_the_fixture_size():
    """1600 pooled spots at per_slide: 400, over the 200 measured to suffice."""
    assert embed_check.default_min_cells(1600, ["per_slide"]) == 400


def test_a_tiny_fixture_does_not_drop_below_the_measured_minimum():
    assert embed_check.default_min_cells(100, ["global"]) == 20


def test_an_unrecognised_strategy_falls_back_to_the_global_fraction():
    assert embed_check.default_min_cells(600, []) == 21


def test_spots_are_read_from_the_slides_not_from_the_cap(tmp_path):
    pytest.importorskip("anndata")
    slides = embed_check.synthetic_slides(
        tmp_path, [f"G{i}" for i in range(10)], n_slides=4, n_spots=7, verbose=False
    )
    assert sum(embed_check.slide_spots(slides).values()) == 28


# ── what this machine can run ────────────────────────────────────────


def test_the_repositorys_own_environment_name_wins():
    envs = {"vgtfm-scgpt": "/a", "scgpt": "/b"}
    assert embed_check.resolve_env("scgpt", envs, None) == "vgtfm-scgpt"


def test_a_preexisting_environment_is_used_when_the_prefixed_one_is_absent():
    assert embed_check.resolve_env("scgpt", {"scgpt": "/b"}, None) == "scgpt"


def test_an_explicit_env_that_does_not_exist_is_refused_rather_than_guessed():
    with pytest.raises(embed_check.Skip, match="--env scgpt=mine"):
        embed_check.resolve_env("scgpt", {"scgpt": "/b"}, "mine")


def test_no_environment_at_all_names_the_command_that_makes_one():
    with pytest.raises(embed_check.Skip, match="envs/scgpt.yaml"):
        embed_check.resolve_env("scgpt", {}, None)


def test_a_model_that_needs_a_checkpoint_says_so(tmp_path):
    with pytest.raises(embed_check.Skip, match="--model-dir cancerfoundation"):
        embed_check.resolve_model_dir("cancerfoundation", None)


def test_geneformer_needs_no_checkpoint_because_it_reads_the_hf_cache():
    assert embed_check.resolve_model_dir("geneformer", None) is None


def test_a_checkpoint_is_checked_before_a_gpu_is_allocated(tmp_path):
    with pytest.raises(embed_check.Skip, match="does not exist"):
        embed_check.resolve_model_dir("scgpt", tmp_path / "gone")


# ── the command ──────────────────────────────────────────────────────


def command(model, **kw):
    kw = {
        "manifest": "/m.json",
        "out_dir": "/out",
        "model_dir": None,
        "device": "cuda",
        "batch_size": 8,
        "strategy": None,
        "max_length": None,
        "min_cells": None,
        "seed": 42,
        **kw,
    }
    return embed_check.gene_fm_command(model, "env", **kw)


def test_the_command_runs_the_module_the_embed_stage_prints():
    cmd = command("scgpt")
    assert cmd[:5] == ["conda", "run", "-n", "env", "--no-capture-output"]
    assert cmd[cmd.index("-m") + 1] == "vgtfm.embed.gene_fm"
    assert "--manifest" in cmd and "--out-dir" in cmd


def test_the_selection_flags_are_only_sent_to_the_models_that_select():
    """``--hvg-strategy`` on Geneformer is an argparse error, not a no-op."""
    pooled = command("scgpt", strategy="mixed", min_cells=3)
    assert pooled[pooled.index("--hvg-strategy") + 1] == "mixed"
    assert pooled[pooled.index("--min-cells") + 1] == "3"
    solo = command("geneformer", strategy=None, min_cells=3)
    assert "--hvg-strategy" not in solo and "--min-cells" not in solo


def test_a_checkpoint_is_passed_only_when_there_is_one():
    assert "--model-dir" not in command("geneformer")
    assert command("scgpt", model_dir="/ckpt")[-1] == "/ckpt"


# ── the parquet contract ─────────────────────────────────────────────


def parquet(tmp_path, ids, features):
    path = tmp_path / "s.parquet"
    gene_fm.write_parquet(path, list(ids), np.asarray(features, dtype=np.float32))
    return path


def test_the_writers_output_is_what_the_reader_expects(tmp_path):
    path = parquet(tmp_path, ["a", "b"], np.arange(6).reshape(2, 3))
    ids, features = embed_check.read_features(path)
    assert list(ids) == ["a", "b"]
    assert features.shape == (2, 3)


def test_a_parquet_without_barcodes_is_refused(tmp_path):
    path = tmp_path / "s.parquet"
    pd.DataFrame({"e0": [1.0], "e1": [2.0]}).to_parquet(path)
    with pytest.raises(AssertionError, match="no spot_id"):
        embed_check.read_features(path)


def test_feature_columns_with_a_gap_are_refused(tmp_path):
    """A renamed or missing column would silently narrow the merged features."""
    path = tmp_path / "s.parquet"
    pd.DataFrame({"spot_id": ["a"], "e0": [1.0], "e2": [2.0]}).to_parquet(path)
    with pytest.raises(AssertionError, match="not e0..e1"):
        embed_check.read_features(path)


# ── verifying a run ──────────────────────────────────────────────────


@pytest.fixture
def run(tmp_path):
    """A finished pooled run: two slides, their parquets, and a provenance file."""
    pytest.importorskip("anndata")
    slides = embed_check.synthetic_slides(
        tmp_path / "cohort", [f"G{i}" for i in range(20)], n_slides=2, n_spots=6, verbose=False
    )
    out = tmp_path / "out"
    rng = np.random.default_rng(0)
    for slide in slides:
        ids = [f"AAAC{i:04d}-1" for i in range(6)]
        gene_fm.write_parquet(
            out / slide.dataset_id / f"{slide.sample_id}.parquet",
            ids,
            rng.random((6, 4), dtype=np.float32),
        )
    cohort.write_provenance(
        out,
        {"hvg_strategy": "global", "seed": 42, "gene_sets": {"all": [f"G{i}" for i in range(5)]}},
    )
    return slides, out


def test_a_complete_run_passes(run):
    slides, out = run
    facts = embed_check.verify("cancerfoundation", out, slides, strategy="global")
    assert facts["dims"] == 4
    assert facts["spots"] == 12
    assert facts["dropped"] == 0
    assert facts["gene_sets"] == {"all": 5}


def test_a_missing_parquet_is_named(run):
    slides, out = run
    (out / slides[0].dataset_id / f"{slides[0].sample_id}.parquet").unlink()
    with pytest.raises(AssertionError, match="no parquet"):
        embed_check.verify("cancerfoundation", out, slides, strategy="global")


def test_a_barcode_the_input_never_had_is_a_misalignment(run):
    """The failure the barcode round-trip exists to catch: shifted rows."""
    slides, out = run
    path = out / slides[0].dataset_id / f"{slides[0].sample_id}.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "spot_id"] = "NOT-A-BARCODE-1"
    frame.to_parquet(path)
    with pytest.raises(AssertionError, match="rows are misaligned"):
        embed_check.verify("cancerfoundation", out, slides, strategy="global")


def test_slides_that_disagree_on_the_feature_width_are_refused(run):
    slides, out = run
    path = out / slides[1].dataset_id / f"{slides[1].sample_id}.parquet"
    gene_fm.write_parquet(path, [f"AAAC{i:04d}-1" for i in range(6)], np.zeros((6, 7), np.float32))
    with pytest.raises(AssertionError, match="disagree on the feature width"):
        embed_check.verify("cancerfoundation", out, slides, strategy="global")


def test_non_finite_features_are_refused(run):
    slides, out = run
    path = out / slides[0].dataset_id / f"{slides[0].sample_id}.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "e0"] = np.nan
    frame.to_parquet(path)
    with pytest.raises(AssertionError, match="non-finite"):
        embed_check.verify("cancerfoundation", out, slides, strategy="global")


def test_a_pooled_run_that_recorded_no_provenance_is_refused(run):
    slides, out = run
    (out / "provenance.json").unlink()
    with pytest.raises(AssertionError, match="no provenance"):
        embed_check.verify("scgpt", out, slides, strategy="global")


def test_provenance_that_names_another_strategy_is_refused(run):
    """The flag is only worth having if what it recorded is what it ran."""
    slides, out = run
    with pytest.raises(AssertionError, match="provenance says 'global'"):
        embed_check.verify("cancerfoundation", out, slides, strategy="mixed")


def test_a_per_slide_model_is_not_asked_for_provenance(run):
    slides, out = run
    (out / "provenance.json").unlink()
    assert embed_check.verify("geneformer", out, slides, strategy=None)["dims"] == 4


def test_spots_the_tokenizer_dropped_are_counted_not_refused(run):
    """Geneformer drops a spot whose genes are all out of vocabulary."""
    slides, out = run
    path = out / slides[0].dataset_id / f"{slides[0].sample_id}.parquet"
    pd.read_parquet(path).iloc[:4].to_parquet(path)
    assert embed_check.verify("geneformer", out, slides, strategy=None)["dropped"] == 2


def test_features_are_stacked_in_manifest_order_for_the_repeat_check(run):
    slides, out = run
    stacked = embed_check.features_of(out, slides)
    _, first = embed_check.read_features(
        out / slides[0].dataset_id / f"{slides[0].sample_id}.parquet"
    )
    assert stacked.shape == (12, 4)
    assert np.array_equal(stacked[:6], first)


# ── the command line ─────────────────────────────────────────────────


def test_a_bare_value_names_the_form_it_should_have_had():
    with pytest.raises(SystemExit, match="name=value"):
        embed_check.parse_pairs(["/some/path"], what="from-h5ad")


def test_an_unknown_model_is_rejected_before_anything_runs(tmp_path):
    with pytest.raises(SystemExit, match="unknown model"):
        embed_check.main(["--out", str(tmp_path), "--model-dir", "scGPT=/x"])


def test_nothing_this_machine_can_run_is_not_reported_as_success(tmp_path, monkeypatch):
    """The distinction the harness exists to keep: skipped is not passed."""
    pytest.importorskip("anndata")
    monkeypatch.setattr(embed_check, "conda_environments", dict)
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt" / "vocab.json").write_text(json.dumps({f"G{i}": i for i in range(20)}))
    code = embed_check.main(
        [
            "--out",
            str(tmp_path / "check"),
            "--model-dir",
            f"scgpt={tmp_path / 'ckpt'}",
            "--n-genes",
            "10",
            "--n-spots",
            "4",
            "--n-slides",
            "2",
        ]  # fmt: skip
    )
    assert code == 0, "a machine with no environments has not failed the check"
    assert (tmp_path / "check" / "cohort").exists(), "but the fixture was still built"


def test_the_makefile_offers_the_harness_as_a_target():
    text = (ROOT / "Makefile").read_text()
    assert "embed-check:" in text
    assert "scripts/embed_check.py" in text


def test_the_makefile_does_not_let_the_shell_expand_a_from_glob():
    """`FROM='usz=/data/KC*.h5ad'` names one cohort, not one per matching file.

    Unquoted, the shell expands it before argparse sees it and the extra paths
    arrive as positional arguments the parser has nowhere to put.
    """
    import shutil
    import subprocess

    if shutil.which("make") is None:
        pytest.skip("no make")
    out = subprocess.run(
        ["make", "-n", "embed-check", "FROM=usz=/data/KC*.h5ad"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "--from-h5ad 'usz=/data/KC*.h5ad'" in out
