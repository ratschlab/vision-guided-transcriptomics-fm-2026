"""The `embed` stage: spot-to-pixel alignment, patch cutting, and the parquet contract.

Nothing here needs a GPU, a slide image or a foundation model. What it pins is the
part of the stage that fails silently: an alignment that mirrors every patch about
the diagonal, a crop that shifts the grid by half a patch, or a barcode column that
comes back out of order all produce embeddings of the wrong tissue and no error.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vgtfm.config import load_config
from vgtfm.data.tables import h5ad_dirs, load_registry
from vgtfm.embed import cohort, gene_fm
from vgtfm.embed.alignment import ALIGNMENT_METHODS, affine_from_loupe, spot_pixels
from vgtfm.embed.build import embed_genes, gene_command, gene_manifest, read_parquet
from vgtfm.embed.cohort import Slide
from vgtfm.embed.midnight import PATCH_SIZE, cut_patch


def loupe(tmp_path, M, n=8, key="oligo", jitter=0.0):
    """Write a Loupe alignment JSON whose landmarks follow the affine *M*."""
    rng = np.random.default_rng(0)
    pts = []
    for i in range(n):
        col, row = float(i), float((i * 7) % 5)
        x, y = M @ np.array([col, row, 1.0])
        if jitter:
            x, y = x + rng.normal(0, jitter), y + rng.normal(0, jitter)
        pts.append({"col": col, "row": row, "imageX": x, "imageY": y})
    path = tmp_path / "align.json"
    path.write_text(json.dumps({key: pts}))
    return path


def obs_frame(**cols):
    n = len(next(iter(cols.values())))
    return types.SimpleNamespace(obs=pd.DataFrame(cols, index=[f"bc{i}" for i in range(n)]))


# ── the Loupe affine ─────────────────────────────────────────────────


def test_the_affine_is_recovered_exactly_from_clean_landmarks(tmp_path):
    M = np.array([[3.0, 0.5, 100.0], [-0.25, 2.0, -40.0]])
    assert np.allclose(affine_from_loupe(loupe(tmp_path, M)), M)


def test_fiducial_landmarks_are_accepted_as_well_as_oligo(tmp_path):
    M = np.array([[2.0, 0.0, 5.0], [0.0, 2.0, 7.0]])
    assert np.allclose(affine_from_loupe(loupe(tmp_path, M, key="fiducial")), M)


def test_a_noisy_alignment_is_still_recovered_by_least_squares(tmp_path):
    M = np.array([[3.0, 0.5, 100.0], [-0.25, 2.0, -40.0]])
    got = affine_from_loupe(loupe(tmp_path, M, n=60, jitter=0.05))
    assert np.allclose(got, M, atol=0.05)


def test_an_alignment_with_no_landmark_points_is_refused(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"something_else": []}))
    with pytest.raises(ValueError, match="neither 'oligo' nor 'fiducial'"):
        affine_from_loupe(path)


def test_too_few_usable_landmarks_are_refused(tmp_path):
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    path = loupe(tmp_path, M, n=2)
    with pytest.raises(ValueError, match="usable landmarks"):
        affine_from_loupe(path)


def test_landmarks_missing_a_key_do_not_count_toward_the_minimum(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(
        json.dumps(
            {
                "oligo": [
                    {"col": 0, "row": 0, "imageX": 0, "imageY": 0},
                    {"col": 1, "row": 0, "imageX": 1, "imageY": 0},
                    {"col": 2, "row": 0},  # no image coordinates
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="usable landmarks"):
        affine_from_loupe(path)


def test_collinear_landmarks_are_refused_rather_than_fitted(tmp_path):
    """Three points on a line determine no affine map; least squares returns one
    anyway, and every patch would be cut somewhere plausible and wrong."""
    path = tmp_path / "a.json"
    path.write_text(
        json.dumps(
            {
                "oligo": [
                    {"col": float(i), "row": float(i), "imageX": float(i), "imageY": float(i)}
                    for i in range(6)
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="collinear"):
        affine_from_loupe(path)


def test_a_singular_linear_part_is_refused(tmp_path):
    """A map that collapses the grid onto a line: rank-3 landmarks, det(M) == 0."""
    M = np.array([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0]])
    with pytest.raises(ValueError, match="determinant"):
        affine_from_loupe(loupe(tmp_path, M))


# ── spot -> pixel ────────────────────────────────────────────────────


def test_h5ad_pixels_are_transposed_on_purpose():
    """In these files ``y_pixel`` is the image *x* and ``x_pixel`` the image *y*.

    Reading them straight through mirrors every patch about the diagonal, which no
    downstream stage can detect.
    """
    adata = obs_frame(
        x_pixel=[10.0, 20.0], y_pixel=[300.0, 400.0], array_col=[1, 2], array_row=[3, 4]
    )
    out = spot_pixels(adata, method="h5ad_obs_pixels")

    assert out["image_x"].tolist() == [300.0, 400.0]
    assert out["image_y"].tolist() == [10.0, 20.0]
    assert out.index.tolist() == ["bc0", "bc1"]


def test_h5ad_pixels_name_the_columns_they_need():
    adata = obs_frame(x_pixel=[1.0], array_col=[0], array_row=[0])
    with pytest.raises(KeyError, match="y_pixel"):
        spot_pixels(adata, method="h5ad_obs_pixels")


def test_lstsq_maps_array_coordinates_through_the_recovered_affine(tmp_path):
    M = np.array([[3.0, 0.5, 100.0], [-0.25, 2.0, -40.0]])
    adata = obs_frame(array_col=[0, 4], array_row=[0, 2])
    out = spot_pixels(adata, method="lstsq_estimate", loupe_json_path=loupe(tmp_path, M))

    expected = np.array([[0, 0, 1], [4, 2, 1]], dtype=float) @ M.T
    assert np.allclose(out[["image_x", "image_y"]].to_numpy(), expected)


def test_lstsq_without_an_alignment_file_is_an_error():
    adata = obs_frame(array_col=[0], array_row=[0])
    with pytest.raises(ValueError, match="requires a Loupe alignment"):
        spot_pixels(adata, method="lstsq_estimate")


def test_the_alternate_column_spellings_are_accepted(tmp_path):
    """Some cohorts write ``y_array`` / ``x_array`` for the spot grid."""
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    adata = obs_frame(y_array=[5, 6], x_array=[7, 8])
    out = spot_pixels(adata, method="lstsq_estimate", loupe_json_path=loupe(tmp_path, M))

    assert out["array_col"].tolist() == [5, 6]
    assert out["array_row"].tolist() == [7, 8]
    assert np.allclose(out["image_x"].to_numpy(), [5, 6])


def test_a_grid_column_that_is_absent_everywhere_defaults_to_zero():
    adata = obs_frame(x_pixel=[1.0], y_pixel=[2.0])
    out = spot_pixels(adata, method="h5ad_obs_pixels")
    assert out["array_col"].tolist() == [0] and out["array_row"].tolist() == [0]


def test_an_unknown_alignment_method_names_the_known_ones():
    adata = obs_frame(x_pixel=[1.0], y_pixel=[2.0])
    with pytest.raises(ValueError) as e:
        spot_pixels(adata, method="guess")
    for name in ALIGNMENT_METHODS:
        assert name in str(e.value)


# ── patch cutting ────────────────────────────────────────────────────


def image(h=400, w=500):
    """An image whose every pixel encodes its own coordinates."""
    ys, xs = np.mgrid[0:h, 0:w]
    return np.stack(
        [
            (ys % 251).astype(np.uint8),
            (xs % 251).astype(np.uint8),
            np.zeros_like(xs, dtype=np.uint8),
        ],
        axis=-1,
    )


def test_a_patch_is_centred_on_the_spot():
    img = image()
    patch = cut_patch(img, cx=200.0, cy=150.0, size=8)

    assert patch.shape == (8, 8, 3)
    assert np.array_equal(patch, img[146:154, 196:204, :3])


def test_a_patch_keeps_its_size_and_dtype_at_the_default():
    patch = cut_patch(image(h=2000, w=2000), 1000.0, 1000.0)
    assert patch.shape == (PATCH_SIZE, PATCH_SIZE, 3)
    assert patch.dtype == np.uint8


def test_an_edge_spot_is_padded_with_white_rather_than_shifted():
    """Padding keeps the output row-aligned with the expression matrix; shifting the
    crop back inside the image would silently embed the neighbouring tissue."""
    img = image()
    patch = cut_patch(img, cx=1.0, cy=1.0, size=8)

    assert np.all(patch[:3, :3] == 255)  # the padded corner
    assert np.array_equal(patch[3:, 3:], img[0:5, 0:5, :3])


def test_a_spot_entirely_outside_the_image_is_all_white():
    patch = cut_patch(image(), cx=-500.0, cy=-500.0, size=8)
    assert patch.shape == (8, 8, 3)
    assert np.all(patch == 255)


def test_an_rgba_image_contributes_only_its_three_colour_channels():
    img = np.dstack([image(h=32, w=32), np.full((32, 32), 7, dtype=np.uint8)])
    patch = cut_patch(img, 16.0, 16.0, size=4)
    assert patch.shape == (4, 4, 3)


# ── the parquet contract between environments ────────────────────────


def test_a_gene_parquet_round_trips_with_its_barcodes(tmp_path):
    """The gene-side models run in their own conda environments and hand back one
    parquet per slide; only ``spot_id`` links it to the rest of the pipeline."""
    from vgtfm.embed.gene_fm import write_parquet

    features = np.arange(12, dtype=np.float32).reshape(4, 3)
    ids = ["AAAC-1", "AAAG-1", "AACC-1", "AAGG-1"]
    path = tmp_path / "sample.parquet"
    write_parquet(path, ids, features)

    got_ids, got = read_parquet(path)
    assert got_ids.tolist() == ids
    assert got.dtype == np.float32
    assert np.array_equal(got, features)
    assert list(pd.read_parquet(path).columns) == ["spot_id", "e0", "e1", "e2"]


def test_geneformer_output_keeps_the_barcodes_the_tokenizer_returned():
    """The tokenizer drops spots with no expressed genes. Carrying the barcodes
    through means those rows are dropped too, rather than shifting every later one.
    """
    from vgtfm.embed.gene_fm import _parse_geneformer_output

    barcodes = np.array(["a", "b", "c"])
    embs = pd.DataFrame({"spot_barcode": ["c", "a"], "e0": [1.0, 2.0], "e1": [3.0, 4.0]})
    ids, features = _parse_geneformer_output(embs, barcodes)

    assert ids.tolist() == ["c", "a"]
    assert features.shape == (2, 2) and features.dtype == np.float32


def test_geneformer_output_without_a_label_column_falls_back_to_the_input_order():
    from vgtfm.embed.gene_fm import _parse_geneformer_output

    barcodes = np.array(["a", "b", "c"])
    ids, features = _parse_geneformer_output(np.zeros((2, 4)), barcodes)
    assert ids.tolist() == ["a", "b"]
    assert features.shape == (2, 4)


def test_an_image_is_found_by_any_of_the_slide_extensions(tmp_path):
    from vgtfm.embed.build import _find_image

    assert _find_image(tmp_path, "S1") is None
    (tmp_path / "S1.ndpi").touch()
    assert _find_image(tmp_path, "S1").name == "S1.ndpi"
    (tmp_path / "S1.tif").touch()
    assert _find_image(tmp_path, "S1").name == "S1.tif", "the first extension wins"


# ── pooling the cohort ───────────────────────────────────────────────
#
# Everything below pins the route scGPT and CancerFoundation take: the slides are
# concatenated, one gene set is chosen over the whole cohort, and the rows are split
# back into per-slide parquets. The failure it guards against is silent — a spot's
# embedding written under another slide's name still merges, still trains, and is
# simply the wrong tissue.


def slide_h5ad(tmp_path, sample_id, *, barcodes, ensembl, symbols=None, counts=None):
    """Write one slide in the 10x convention: symbols in var_names, Ensembl in var."""
    ad = pytest.importorskip("anndata")
    sp = pytest.importorskip("scipy.sparse")
    n, g = len(barcodes), len(ensembl)
    X = np.arange(1, n * g + 1, dtype=np.float32).reshape(n, g) if counts is None else counts
    a = ad.AnnData(
        X=sp.csr_matrix(np.asarray(X, dtype=np.float32)),
        obs=pd.DataFrame(index=pd.Index(barcodes)),
        var=pd.DataFrame({"gene_ids": ensembl}, index=pd.Index(symbols or ensembl)),
    )
    path = tmp_path / f"{sample_id}.h5ad"
    a.write_h5ad(path)
    return Slide("COHORT", sample_id, str(path), "gene_ids")


def two_slides(tmp_path):
    """Two slides that share the same Visium barcodes and all but one gene."""
    barcodes = ["AAAC-1", "AAAG-1"]
    a = slide_h5ad(
        tmp_path,
        "S1",
        barcodes=barcodes,
        ensembl=["ENSG001.4", "ENSG002.1", "ENSG003"],
        symbols=["ALPHA", "BETA", "GAMMA"],
    )
    b = slide_h5ad(
        tmp_path,
        "S2",
        barcodes=barcodes,
        ensembl=["ENSG001", "ENSG002", "ENSG009"],
        symbols=["ALPHA", "BETA", "OMEGA"],
    )
    return [a, b]


# -- the manifest --


def test_a_manifest_round_trips(tmp_path):
    slides = two_slides(tmp_path)
    path = cohort.write_manifest(tmp_path / "m.json", "scgpt", slides)
    assert cohort.read_manifest(path) == slides


def test_a_manifest_naming_a_missing_slide_is_refused(tmp_path):
    """The alternative is a pooled gene selection made over a smaller cohort than
    the one the run believes it covered, which nothing downstream can see."""
    slides = two_slides(tmp_path) + [Slide("COHORT", "S3", str(tmp_path / "gone.h5ad"))]
    path = cohort.write_manifest(tmp_path / "m.json", "scgpt", slides)
    with pytest.raises(FileNotFoundError, match="1 of 3 manifest entries"):
        cohort.read_manifest(path)


def test_a_slide_names_itself_by_cohort_and_sample():
    assert Slide("10x_TuPro", "MACEGEJ-1-1", "x.h5ad").source == "10x_TuPro__MACEGEJ-1-1"


# -- pooling --


def test_slides_are_joined_on_version_stripped_ensembl_ids(tmp_path):
    pooled = cohort.pool(two_slides(tmp_path), verbose=False)

    assert pooled.var_names.tolist() == ["ENSG001", "ENSG002"], "the inner join, unversioned"
    assert pooled.var["gene_symbol"].tolist() == ["ALPHA", "BETA"]
    assert pooled.n_obs == 4


def test_the_same_barcode_on_two_slides_stays_two_spots(tmp_path):
    """Every Visium slide has an AAAC-1. Pooling on the barcode alone would collapse
    them, or worse, hand one slide's embedding back under the other's name."""
    pooled = cohort.pool(two_slides(tmp_path), verbose=False)

    assert pooled.obs_names.is_unique
    assert pooled.obs["spot_id"].tolist() == ["AAAC-1", "AAAG-1"] * 2
    assert pooled.obs["sample_id"].tolist() == ["S1", "S1", "S2", "S2"]
    assert pooled.obs["sample_source"].tolist() == ["COHORT__S1"] * 2 + ["COHORT__S2"] * 2


def test_pooling_carries_the_counts_of_each_slide(tmp_path):
    pooled = cohort.pool(two_slides(tmp_path), verbose=False)
    # S1 rows are [1,2,3],[4,5,6] over (ALPHA, BETA, GAMMA); GAMMA is not shared.
    assert pooled.X.toarray().tolist() == [[1, 2], [4, 5], [1, 2], [4, 5]]


def test_slides_that_share_no_genes_are_refused(tmp_path):
    slides = [
        slide_h5ad(tmp_path, "S1", barcodes=["A-1"], ensembl=["ENSG001"]),
        slide_h5ad(tmp_path, "S2", barcodes=["A-1"], ensembl=["ENSG777"]),
    ]
    with pytest.raises(ValueError, match="share no genes"):
        cohort.pool(slides, verbose=False)


def test_a_slide_without_an_ensembl_column_falls_back_to_its_var_names(tmp_path):
    ad = pytest.importorskip("anndata")
    sp = pytest.importorskip("scipy.sparse")
    a = ad.AnnData(
        X=sp.csr_matrix(np.ones((1, 2), dtype=np.float32)),
        obs=pd.DataFrame(index=pd.Index(["A-1"])),
        var=pd.DataFrame(index=pd.Index(["ENSG001.2", "ENSG002"])),
    )
    a.write_h5ad(tmp_path / "S1.h5ad")
    pooled = cohort.pool([Slide("C", "S1", str(tmp_path / "S1.h5ad"))], verbose=False)

    assert pooled.var_names.tolist() == ["ENSG001", "ENSG002"]
    assert pooled.var["gene_symbol"].tolist() == ["ENSG001", "ENSG002"]


def test_a_counts_layer_wins_over_x(tmp_path):
    """X is raw counts in every cohort here, but a cohort that ships both should be
    read from the layer that says so."""
    ad = pytest.importorskip("anndata")
    sp = pytest.importorskip("scipy.sparse")
    a = ad.AnnData(
        X=sp.csr_matrix(np.zeros((1, 1), dtype=np.float32)),
        obs=pd.DataFrame(index=pd.Index(["A-1"])),
        var=pd.DataFrame({"gene_ids": ["ENSG001"]}, index=pd.Index(["ALPHA"])),
        layers={"counts": sp.csr_matrix(np.array([[7.0]], dtype=np.float32))},
    )
    a.write_h5ad(tmp_path / "S1.h5ad")
    pooled = cohort.pool([Slide("C", "S1", str(tmp_path / "S1.h5ad"))], verbose=False)
    assert pooled.X.toarray().tolist() == [[7.0]]


# -- the QC filters --


def counts_adata(X, batches):
    ad = pytest.importorskip("anndata")
    sp = pytest.importorskip("scipy.sparse")
    return ad.AnnData(
        X=sp.csr_matrix(np.asarray(X, dtype=np.float32)),
        obs=pd.DataFrame({"dataset_id": batches}, index=[f"s{i}" for i in range(len(X))]),
        var=pd.DataFrame(index=pd.Index([f"g{j}" for j in range(len(X[0]))])),
    )


def test_a_gene_below_the_support_floor_is_dropped():
    a = counts_adata([[1, 0], [1, 0], [1, 1]], ["A"] * 3)
    assert cohort.filter_genes(a, min_cells=2, verbose=False).var_names.tolist() == ["g0"]


def test_a_gene_absent_from_one_batch_is_dropped():
    """Batch-aware HVG selection fits a dispersion per batch; a gene with no counts
    in one of them has none to fit, and scanpy raises rather than skipping it."""
    a = counts_adata([[1, 1], [1, 0]], ["A", "B"])
    kept = cohort.drop_batch_zero_genes(a, batch_key="dataset_id", verbose=False)
    assert kept.var_names.tolist() == ["g0"]


def test_the_filters_leave_an_untouched_cohort_alone():
    a = counts_adata([[1, 1], [1, 1]], ["A", "B"])
    assert cohort.filter_genes(a, min_cells=1, verbose=False) is a
    assert cohort.drop_batch_zero_genes(a, verbose=False) is a


# -- a degenerate mean-variance fit --


def test_a_singular_loess_names_its_cause_and_the_knob_that_moves_it(monkeypatch):
    """seurat_v3's message is a condition number and nothing else.

    Measured on four USZ slides at 600 spots: with min_cells=3 the fit is singular,
    at 10 it is not. Someone hitting this on a subset should not have to find that
    out by bisecting.
    """
    sc = pytest.importorskip("scanpy")
    a = counts_adata([[1, 1], [1, 1]], ["A", "B"])
    monkeypatch.setattr(
        sc.pp,
        "highly_variable_genes",
        lambda *args, **kw: (_ for _ in ()).throw(
            ValueError("b'reciprocal condition number  2.5833e-16'")
        ),
    )
    with pytest.raises(ValueError) as excinfo:
        cohort._hvg(a, n_top=2, flavor="seurat_v3", layer=None, batch_key=None)
    message = str(excinfo.value)
    assert "2 spots x 2 genes" in message, "the shape that was too small to fit"
    assert "--min-cells" in message, "the knob that moves it"
    assert "reciprocal condition number" in message, "the original, not swallowed"


def test_the_same_diagnosis_covers_cell_rangers_duplicate_bin_edges(monkeypatch):
    """The other flavour fails differently and for the same reason."""
    sc = pytest.importorskip("scanpy")
    a = counts_adata([[1, 1], [1, 1]], ["A", "B"])
    monkeypatch.setattr(
        sc.pp,
        "highly_variable_genes",
        lambda *args, **kw: (_ for _ in ()).throw(ValueError("Bin edges must be unique")),
    )
    with pytest.raises(ValueError, match="cell_ranger could not fit"):
        cohort._hvg(a, n_top=2, flavor="cell_ranger", layer=None, batch_key="dataset_id")


def test_the_diagnosis_does_not_hide_a_selection_that_works():
    pytest.importorskip("scanpy")
    a = grouped_cohort()
    genes = cohort._hvg(a, n_top=20, flavor="cell_ranger", layer=None, batch_key=None)
    assert len(genes) == 20


# -- splitting back out --


def test_pooled_rows_are_split_back_to_the_slide_they_came_from(tmp_path):
    pooled = cohort.pool(two_slides(tmp_path), verbose=False)
    features = np.arange(8, dtype=np.float32).reshape(4, 2)
    written = cohort.split_to_parquets(pooled.obs, features, tmp_path / "out")

    assert [p.relative_to(tmp_path / "out").as_posix() for p in written] == [
        "COHORT/S1.parquet",
        "COHORT/S2.parquet",
    ]
    ids, got = read_parquet(written[1])
    assert ids.tolist() == ["AAAC-1", "AAAG-1"], "the raw barcodes, not the pooled index"
    assert np.array_equal(got, features[2:])


def test_splitting_follows_the_rows_the_model_returned(tmp_path):
    """A model that drops a spot must drop it here too. Reindexing obs by the labels
    the embedder handed back is what makes that impossible to get wrong."""
    pooled = cohort.pool(two_slides(tmp_path), verbose=False)
    kept = [pooled.obs_names[0], pooled.obs_names[3]]
    features = np.array([[1.0], [2.0]], dtype=np.float32)
    written = cohort.split_to_parquets(pooled.obs.loc[kept], features, tmp_path / "out")

    assert [read_parquet(p)[0].tolist() for p in written] == [["AAAC-1"], ["AAAG-1"]]


def test_a_row_count_mismatch_is_refused_rather_than_broadcast(tmp_path):
    pooled = cohort.pool(two_slides(tmp_path), verbose=False)
    with pytest.raises(ValueError, match="4 rows of obs vs 3 embeddings"):
        cohort.split_to_parquets(pooled.obs, np.zeros((3, 2), np.float32), tmp_path / "out")


def test_a_slide_that_produced_no_parquet_is_named(tmp_path):
    slides = two_slides(tmp_path)
    written = [tmp_path / "COHORT" / "S1.parquet"]
    assert cohort.report_missing(slides, written) == ["COHORT__S2"]


# ── what each foundation model is handed ─────────────────────────────


def upstream_binning():
    """A stand-in for the ``binning()`` scGPT and CancerFoundation both carry.

    The defect is faithful: the row is converted to numpy on entry and the all-zero
    branch then calls ``torch.zeros_like`` on it, which is a ``TypeError``.
    """
    import torch

    module = types.ModuleType("fake_data_collator")

    def binning(row, n_bins):
        values = row.cpu().numpy() if isinstance(row, torch.Tensor) else row
        if values.max() == 0:  # ValueError on an empty row: max() has no identity
            return torch.zeros_like(values, dtype=values.dtype)
        return row * 2

    module.binning = binning
    return module


def test_a_spot_with_no_counts_left_crashes_the_unpatched_binning():
    torch = pytest.importorskip("torch")
    with pytest.raises(TypeError):
        upstream_binning().binning(torch.zeros(3), 51)


def test_the_patched_binning_returns_zeros_for_a_spot_with_no_counts():
    """A Visium spot on the tissue border can have nothing left after gene
    selection. An all-zero embedding is the honest answer; the alternative is the
    whole cohort's run dying in a dataloader worker, hours in."""
    torch = pytest.importorskip("torch")
    module = upstream_binning()
    gene_fm._patch_binning(module)

    out = module.binning(torch.zeros(3, dtype=torch.float32), 51)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32 and out.tolist() == [0.0, 0.0, 0.0]


def test_a_spot_whose_genes_were_all_dropped_arrives_empty_not_zero():
    """scGPT's dataset passes only a spot's *non-zero* genes, so a spot with no
    counts reaches binning as a zero-length vector rather than a zero-valued one and
    never reaches the all-zero branch at all. Found by running it: a real cohort has
    such spots on the tissue border, and the run died in a dataloader worker."""
    torch = pytest.importorskip("torch")
    empty = torch.zeros(0, dtype=torch.float32)
    with pytest.raises(ValueError, match="zero-size array"):
        upstream_binning().binning(empty, 51)

    module = upstream_binning()
    gene_fm._patch_binning(module)
    out = module.binning(empty, 51)
    assert isinstance(out, torch.Tensor) and out.numel() == 0


def test_the_patched_binning_still_defers_for_an_ordinary_spot():
    torch = pytest.importorskip("torch")
    module = upstream_binning()
    gene_fm._patch_binning(module)
    assert module.binning(torch.tensor([1.0, 2.0]), 51).tolist() == [2.0, 4.0]


def test_patching_binning_twice_does_not_stack():
    torch = pytest.importorskip("torch")
    module = upstream_binning()
    gene_fm._patch_binning(module)
    gene_fm._patch_binning(module)
    assert module.binning(torch.tensor([1.0]), 51).tolist() == [2.0]


def test_a_single_batch_key_is_dropped_before_it_reaches_scanpy():
    """scanpy's batched HVG path fits a dispersion per batch and ranks across them;
    with one batch there is nothing to rank and it raises."""
    a = counts_adata([[1, 1], [1, 1]], ["A", "A"])
    assert gene_fm._batch_key_or_none(a, "dataset_id") is None
    assert gene_fm._batch_key_or_none(a, "absent_column") is None

    b = counts_adata([[1, 1], [1, 1]], ["A", "B"])
    assert gene_fm._batch_key_or_none(b, "dataset_id") == "dataset_id"


def test_genes_are_handed_to_the_vocabularies_as_symbols(tmp_path):
    """Pooling joins on Ensembl ids; both vocabularies are keyed on HGNC symbols."""
    pooled = cohort.pool(two_slides(tmp_path), verbose=False)
    assert pooled.var_names.tolist() == ["ENSG001", "ENSG002"]
    assert gene_fm._symbols_as_var_names(pooled).var_names.tolist() == ["ALPHA", "BETA"]


def test_a_frame_without_symbols_is_left_as_it_is():
    a = counts_adata([[1, 1]], ["A"])
    assert gene_fm._symbols_as_var_names(a).var_names.tolist() == ["g0", "g1"]


# -- locating the checkpoints --


def cancerfoundation_checkout(tmp_path, *, assets=gene_fm.CANCERFOUNDATION_ASSETS):
    repo = tmp_path / "CancerFoundation"
    (repo / "model" / "assets").mkdir(parents=True)
    (repo / "model" / "embedding.py").touch()
    for name in assets:
        (repo / "model" / "assets" / name).touch()
    return repo


def test_cancerfoundation_is_found_from_the_checkout_or_from_its_assets(tmp_path):
    """Both are the obvious answer to "where is the model", and the checkout has to
    be located either way — it goes on sys.path, because there is no package to
    install."""
    repo = cancerfoundation_checkout(tmp_path)
    assert gene_fm._cancerfoundation_paths(str(repo)) == (repo, repo / "model" / "assets")
    assert gene_fm._cancerfoundation_paths(str(repo / "model" / "assets")) == (
        repo,
        repo / "model" / "assets",
    )


def test_cancerfoundation_names_the_weights_that_are_not_there(tmp_path):
    """The weights are a separate download from the repository, so a fresh clone
    reaches exactly here."""
    repo = cancerfoundation_checkout(tmp_path, assets=("vocab.json",))
    with pytest.raises(FileNotFoundError, match="model.pth"):
        gene_fm._cancerfoundation_paths(str(repo))


def test_assets_outside_a_checkout_are_refused(tmp_path):
    loose = tmp_path / "weights"
    loose.mkdir()
    for name in gene_fm.CANCERFOUNDATION_ASSETS:
        (loose / name).touch()
    with pytest.raises(FileNotFoundError, match="no model/embedding.py"):
        gene_fm._cancerfoundation_paths(str(loose))


def test_cancerfoundation_without_a_model_dir_says_what_to_clone():
    with pytest.raises(ValueError, match="github.com/BoevaLab/CancerFoundation"):
        gene_fm._cancerfoundation_paths(None)


def test_a_checkpoint_is_checked_before_the_gpu_is_taken(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing vocab.json, args.json"):
        gene_fm._check_checkpoint(tmp_path, gene_fm.SCGPT_CHECKPOINT_FILES, "scGPT", "--model-dir")
    for name in gene_fm.SCGPT_CHECKPOINT_FILES:
        (tmp_path / name).touch()
    assert gene_fm._check_checkpoint(tmp_path, gene_fm.SCGPT_CHECKPOINT_FILES, "scGPT", "-m")


# -- the two routes --


def test_every_pooled_model_declares_what_it_calls_a_batch():
    assert set(gene_fm.POOLED) == set(gene_fm.DEFAULT_BATCH_KEY)
    assert set(gene_fm.POOLED) <= set(gene_fm.MODELS)


def test_a_gene_selecting_model_is_not_run_on_one_slide_by_accident(capsys):
    """Per slide is a defensible choice — each slide gets the genes that describe it,
    and no small cohort is outvoted. It is not a defensible *accident*: fifty slides
    each with their own gene set merge, train and evaluate without a single error, and
    the cached artefacts this pipeline reads were pooled. So it has to be asked for."""
    with pytest.raises(SystemExit):
        gene_fm.main(["--model", "scgpt", "--h5ad", "a.h5ad", "--out", "a.parquet"])
    err = capsys.readouterr().err
    assert "chooses its genes from whatever it is given" in err
    assert "--allow-per-slide" in err


def test_geneformer_needs_no_such_opt_in(tmp_path, capsys):
    """It ranks against a fixed vocabulary, so a slide alone is a slide in company."""
    with pytest.raises(SystemExit):
        gene_fm.main(["--model", "geneformer", "--h5ad", str(tmp_path / "nope.h5ad")])
    assert "--h5ad and --out" in capsys.readouterr().err


def test_a_manifest_without_a_destination_is_refused(capsys):
    with pytest.raises(SystemExit):
        gene_fm.main(["--model", "scgpt", "--manifest", "m.json"])
    assert "--manifest needs --out-dir" in capsys.readouterr().err


# ── what the stage hands the other environments ──────────────────────
#
# The gene-side models cannot import this repository's config — they hold a
# different scanpy and a different torch, which is the whole reason they are
# separate environments. So the stage resolves the cohort and writes it down, and
# these tests pin what it writes.


def registry_json(tmp_path):
    path = tmp_path / "datasets.json"
    path.write_text(
        json.dumps(
            {
                "datasets": {
                    "cohort_a": {
                        "tissue": "skin",
                        "gene_ids_col": "gene_ids",
                        "subdirs": {"h5ad": "h5ad_preprocessed"},
                        "samples": [{"id": "S1", "split": "train"}, {"id": "S2", "split": "test"}],
                    },
                    "cohort_b": {
                        "tissue": "lung",
                        "subdirs": {"h5ad": "counts"},
                        "samples": [{"id": "T1", "split": "test"}],
                    },
                }
            }
        )
    )
    return path


def stage_cfg(tmp_path, **overrides):
    return load_config(
        None,
        {
            "paths.artifact_root": str(tmp_path / "artifacts"),
            "paths.datasets_json": str(registry_json(tmp_path)),
            "paths.raw_data_root": str(tmp_path / "raw"),
            **overrides,
        },
    )


def test_an_explicit_raw_path_wins_over_the_registry_layout(tmp_path):
    """A site profile names a cohort explicitly when the raw cohorts and the
    tokenizer's output are different filesystems, which on a cluster they are."""
    cfg = stage_cfg(tmp_path)
    cfg.paths.raw_h5ad["cohort_a"] = "/elsewhere/tupro"  # what a site profile writes
    dirs = h5ad_dirs(cfg, load_registry(cfg.paths.datasets_json))

    assert dirs["cohort_a"] == Path("/elsewhere/tupro")
    assert dirs["cohort_b"] == tmp_path / "raw" / "cohort_b" / "counts"


def test_the_manifest_lists_every_slide_this_site_actually_holds(tmp_path, capsys):
    cfg = stage_cfg(tmp_path)
    (tmp_path / "raw" / "cohort_a" / "h5ad_preprocessed").mkdir(parents=True)
    for sid in ("S1", "S2"):
        (tmp_path / "raw" / "cohort_a" / "h5ad_preprocessed" / f"{sid}.h5ad").touch()

    path, slides, absent = gene_manifest(cfg, "scgpt")

    assert [s.sample_id for s in slides] == ["S1", "S2"]
    assert absent == ["cohort_b/T1"], "the registry lists cohorts a site need not hold"
    assert path == Path(cfg.paths.artifact_root) / "_embed" / "scgpt" / "manifest.json"
    assert json.loads(path.read_text())["model"] == "scgpt"


def test_the_manifest_carries_each_cohorts_own_ensembl_column(tmp_path):
    """Pooling has to strip version suffixes off Ensembl ids, and where those live
    is a per-cohort fact the isolated environments have no other way to learn."""
    cfg = stage_cfg(tmp_path)
    for ds, sub, sid in (("cohort_a", "h5ad_preprocessed", "S1"), ("cohort_b", "counts", "T1")):
        d = tmp_path / "raw" / ds / sub
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{sid}.h5ad").touch()

    _, slides, _ = gene_manifest(cfg, "cancerfoundation")
    assert {s.dataset_id: s.gene_id_column for s in slides} == {
        "cohort_a": "gene_ids",
        "cohort_b": "gene_ids",  # the default, for a cohort that does not name one
    }


def test_the_stage_counts_the_parquets_the_other_environment_has_written(tmp_path, capsys):
    cfg = stage_cfg(tmp_path)
    d = tmp_path / "raw" / "cohort_a" / "h5ad_preprocessed"
    d.mkdir(parents=True)
    for sid in ("S1", "S2"):
        (d / f"{sid}.h5ad").touch()
    done = Path(cfg.paths.artifact_root) / "_embed" / "scgpt" / "cohort_a"
    done.mkdir(parents=True)
    (done / "S1.parquet").touch()

    _, have, total = embed_genes(cfg, "scgpt")
    assert (have, total) == (1, 2)
    out = capsys.readouterr().out
    assert "1/2 already embedded" in out
    assert "python -m vgtfm.embed.gene_fm --model scgpt" in out, "the command, ready to paste"


def test_the_printed_command_names_the_checkpoint_the_model_needs(tmp_path):
    cfg = stage_cfg(tmp_path)
    manifest = Path(cfg.paths.artifact_root) / "_embed" / "scgpt" / "manifest.json"

    assert "--model-dir /path/to/scGPT_human" in gene_command(cfg, "scgpt", manifest)
    assert "--model-dir" not in gene_command(cfg, "geneformer", manifest)
    assert "conda run -n vgtfm-scgpt" in gene_command(cfg, "scgpt", manifest)


def test_symbols_are_restored_before_the_filters_not_after(tmp_path):
    """Where the duplicate-symbol tie is broken decides which copy keeps the bare
    name, and only the bare name matches a vocabulary. `merged/precompute_*.py` broke
    it on the full gene set, so this does too — doing it after HVG selection would
    hand the model a different gene set from the one the cached artefacts used."""
    ad = pytest.importorskip("anndata")
    sp = pytest.importorskip("scipy.sparse")
    # ENSG001 and ENSG002 are both called ALPHA; ENSG002 is the rarer of the two.
    a = ad.AnnData(
        X=sp.csr_matrix(np.array([[5, 1, 5], [5, 0, 5], [5, 0, 5]], dtype=np.float32)),
        obs=pd.DataFrame(index=pd.Index(["b0", "b1", "b2"])),
        var=pd.DataFrame(
            {"gene_ids": ["ENSG001", "ENSG002", "ENSG003"]},
            index=pd.Index(["ALPHA", "ALPHA", "BETA"]),
        ),
    )
    a.write_h5ad(tmp_path / "S1.h5ad")
    slides = [Slide("C", "S1", str(tmp_path / "S1.h5ad"))]

    named = cohort.prepare_cohort(slides, min_cells=2, batch_key="dataset_id", verbose=False)
    assert named.var_names.tolist() == ["ALPHA", "BETA"], "ENSG002 lost the tie, then the filter"
    assert named.var["ensembl_id"].tolist() == ["ENSG001", "ENSG003"]

    # Naming after the filter would have left ENSG002 out and ENSG001 named ALPHA
    # too, but on a cohort where the filter drops the *first* copy the two orders
    # disagree about which gene the vocabulary sees.
    raw = cohort.pool(slides, verbose=False)
    assert raw.var_names.tolist() == ["ENSG001", "ENSG002", "ENSG003"], "pooled on Ensembl"


def test_the_collision_count_does_not_confuse_a_real_symbol_for_a_suffix(capsys):
    """Plenty of genuine symbols end in -<digits>. A diagnostic that counts those as
    collisions reports damage that never happened."""
    ad = pytest.importorskip("anndata")
    sp = pytest.importorskip("scipy.sparse")
    a = ad.AnnData(
        X=sp.csr_matrix(np.ones((2, 3), dtype=np.float32)),
        obs=pd.DataFrame(index=pd.Index(["b0", "b1"])),
        var=pd.DataFrame(
            {"gene_symbol": ["RP11-554J4-1", "MARCH-1", "TBCE"]},
            index=pd.Index(["ENSG001", "ENSG002", "ENSG003"]),
        ),
    )
    cohort.symbols_as_var_names(a, verbose=True)
    assert "0 collided" in capsys.readouterr().out


# ── where the genes are chosen ───────────────────────────────────────


def grouped_cohort(n_per_group=60, n_genes=240, seed=0):
    """Two groups whose variable genes barely overlap, so the strategies diverge.

    Wide enough for `cell_ranger`, which bins gene means into 20 quantiles and raises
    on duplicate edges — a handful of genes is not a cohort.
    """
    ad = pytest.importorskip("anndata")
    sp = pytest.importorskip("scipy.sparse")
    rng = np.random.default_rng(seed)
    rows, groups, sources = [], [], []
    base = rng.uniform(0.5, 12.0, size=n_genes)  # distinct means, so the bins are unique
    for g, (lo, hi) in enumerate([(0, 60), (120, 180)]):
        block = rng.poisson(base, size=(n_per_group, n_genes)).astype(np.float32)
        block[:, lo:hi] *= rng.uniform(1, 40, size=(n_per_group, hi - lo))  # this group's own
        rows.append(block)
        groups += [f"ds{g}"] * n_per_group
        sources += [f"ds{g}__s{i % 2}" for i in range(n_per_group)]
    X = np.vstack(rows)
    return ad.AnnData(
        X=sp.csr_matrix(X),
        obs=pd.DataFrame(
            {"dataset_id": groups, "sample_source": sources},
            index=[f"c{i}" for i in range(len(X))],
        ),
        var=pd.DataFrame(index=pd.Index([f"g{j}" for j in range(n_genes)])),
    )


def test_global_selection_hands_every_spot_the_same_genes():
    sets = cohort.select_genes(
        grouped_cohort(), strategy="global", n_genes=20, flavor="cell_ranger", verbose=False
    )
    assert list(sets) == ["all"]
    assert len(sets["all"]) >= 20


def test_a_global_gene_budget_is_a_floor_not_a_count():
    """scanpy turns `n_top_genes` into a dispersion cutoff and keeps everything at or
    above it, so ties overshoot. Worth knowing before reading `--n-hvg 2000` as 2000:
    the per-group strategies below slice to the budget, this one does not."""
    sets = cohort.select_genes(
        grouped_cohort(), strategy="global", n_genes=20, flavor="cell_ranger", verbose=False
    )
    assert len(sets["all"]) >= 20


def test_mixed_spends_the_shared_ratio_on_genes_every_group_gets():
    """The 70/30 split ported from merged/cancerfoundation_embed_mixed.py: most of the
    budget on genes that rank highly across groups, the rest on each group's own."""
    sets = cohort.select_genes(
        grouped_cohort(),
        strategy="mixed",
        n_genes=20,
        shared_ratio=0.70,
        flavor="cell_ranger",
        verbose=False,
    )
    assert sorted(sets) == ["ds0", "ds1"]
    shared = set(sets["ds0"]) & set(sets["ds1"])
    assert len(shared) >= 14, "at least the shared budget is common to both groups"
    assert set(sets["ds0"]) != set(sets["ds1"]), "and each keeps some of its own"


def test_per_slide_selection_shares_nothing_by_construction():
    sets = cohort.select_genes(
        grouped_cohort(), strategy="per_slide", n_genes=20, flavor="cell_ranger", verbose=False
    )
    assert sorted(sets) == ["ds0__s0", "ds0__s1", "ds1__s0", "ds1__s1"]
    assert all(len(v) <= 20 for v in sets.values()), "sliced to the budget, unlike global"
    assert set(sets["ds0__s0"]) != set(sets["ds1__s0"]), "different groups, different genes"


def test_a_group_too_small_to_estimate_dispersion_falls_back_to_the_shared_set(capsys):
    """Rather than ranking genes off a handful of spots and calling it a selection."""
    a = grouped_cohort(n_per_group=60)
    a.obs.loc[a.obs.index[:59], "dataset_id"] = "ds1"  # leaves ds0 with one spot
    sets = cohort.select_genes(
        a, strategy="mixed", n_genes=20, flavor="cell_ranger", min_group_size=50, verbose=True
    )
    assert "shared genes only" in capsys.readouterr().out
    assert set(sets["ds0"]) <= set(sets["ds1"])


def test_the_masks_line_up_with_the_gene_sets():
    a = grouped_cohort()
    for strategy in cohort.HVG_STRATEGIES:
        sets = cohort.select_genes(
            a, strategy=strategy, n_genes=20, flavor="cell_ranger", verbose=False
        )
        masks = cohort.group_masks(a, strategy, "dataset_id")
        assert set(masks) == set(sets), strategy
        assert sum(m.sum() for m in masks.values()) == a.n_obs, strategy


def test_an_unknown_strategy_names_the_known_ones():
    with pytest.raises(ValueError, match="per_slide"):
        cohort.select_genes(
            grouped_cohort(), strategy="guess", n_genes=8, flavor="cell_ranger", verbose=False
        )


def test_the_basis_is_recorded_beside_the_parquets(tmp_path):
    """Two runs under different strategies write indistinguishable parquets, so what
    separates them has to be on disk and not in someone's shell history."""
    cohort.write_provenance(
        tmp_path,
        {"model": "scgpt", "hvg_strategy": "mixed", "seed": 42, "gene_sets": {"a": ["G1"]}},
    )
    got = json.loads((tmp_path / "provenance.json").read_text())
    assert got["hvg_strategy"] == "mixed" and got["seed"] == 42


def test_scvi_is_not_offered_a_gene_selection_strategy(capsys):
    with pytest.raises(SystemExit):
        gene_fm.main(
            ["--model", "scvi", "--manifest", "m.json", "--out-dir", "o", "--hvg-strategy", "mixed"]
        )
    assert "scvi selects no genes" in capsys.readouterr().err


def test_an_embedder_that_reports_no_gene_set_still_unpacks():
    ids, feats, sets = gene_fm._unpack((np.array(["a"]), np.zeros((1, 2), np.float32)))
    assert sets == {}
    ids, feats, sets = gene_fm._unpack(
        (np.array(["a"]), np.zeros((1, 2), np.float32), {"g": ["X"]})
    )
    assert sets == {"g": ["X"]}
