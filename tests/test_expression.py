"""Count-level expression access, against real ``.h5ad`` files on disk.

:class:`~vgtfm.biosignal.expression.ExpressionIndex` is the only path from a spot in
the cohort table back to its raw counts, and both the count-level baseline
(``hvg_pca``) and the `biosignal` stage read every number they report through it. Its
failure mode is misalignment rather than an exception: a barcode that resolves to the
wrong row, or a gene column taken from the wrong slide's panel, produces a full
matrix of plausible values.
"""

from __future__ import annotations

import numpy as np
import pytest

from vgtfm.config import load_config
from vgtfm.biosignal.expression import ExpressionIndex, _column_index, _group_by_sample

ad = pytest.importorskip("anndata")


def write_slide(directory, sample_id, genes, counts, barcodes=None):
    import pandas as pd

    directory.mkdir(parents=True, exist_ok=True)
    counts = np.asarray(counts, dtype=np.float32)
    barcodes = barcodes or [f"{sample_id}-bc{i}" for i in range(len(counts))]
    a = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(index=pd.Index(barcodes, dtype=str)),
        var=pd.DataFrame(index=pd.Index(list(genes), dtype=str)),
    )
    a.write_h5ad(directory / f"{sample_id}.h5ad")
    return barcodes


@pytest.fixture
def cohort(tmp_path):
    """Two slides sharing ``A B C``; the first also has ``D``, the second ``E``.

    ``C`` is expressed in only one spot overall, so ``min_expressed`` has something
    to drop that the intersection alone would keep.
    """
    root = tmp_path / "raw" / "ds"
    s1 = write_slide(
        root, "S1", ["A", "B", "C", "D"], [[10, 1, 0, 5], [20, 2, 0, 5], [30, 3, 1, 5]]
    )
    s2 = write_slide(root, "S2", ["A", "B", "C", "E"], [[40, 4, 0, 9], [50, 5, 0, 9]])

    cfg = load_config(
        None,
        {
            "paths.data_root": str(tmp_path),
            "paths.raw_data_root": str(tmp_path / "raw"),
            "paths.raw_h5ad.10x_TuPro": "ds",
        },
    )
    keys = [("ds_key", "S1", b) for b in s1] + [("ds_key", "S2", b) for b in s2]
    return cfg, keys


@pytest.fixture
def index(cohort, monkeypatch):
    """An index whose one cohort key ``ds_key`` resolves to the slides above."""
    cfg, keys = cohort
    idx = ExpressionIndex(cfg)
    idx._paths = {"ds_key": idx.cfg.raw_h5ad_dir("10x_TuPro")}
    return idx, keys


def columns(keys):
    return (
        np.array([k[0] for k in keys]),
        np.array([k[1] for k in keys]),
        np.array([k[2] for k in keys]),
    )


# ── grouping helpers ─────────────────────────────────────────────────


def test_rows_are_grouped_by_slide_and_keep_their_original_positions():
    ds = np.array(["d", "d", "e", "d"])
    sid = np.array(["S1", "S2", "S1", "S1"])
    groups = _group_by_sample(ds, sid, np.array(["a", "b", "c", "d"]))

    assert groups[("d", "S1")].tolist() == [0, 3]
    assert groups[("d", "S2")].tolist() == [1]
    assert groups[("e", "S1")].tolist() == [2]


def test_a_gene_is_looked_up_by_name_not_by_position():
    names = np.array(["Z", "A", "M"])
    assert _column_index(names, np.array(["A", "M"])).tolist() == [1, 2]


# ── vocabulary ───────────────────────────────────────────────────────


def test_the_vocabulary_is_the_intersection_of_every_contributing_slide(index):
    idx, keys = index
    genes = idx.build_vocabulary(*columns(keys), min_expressed=1)
    assert genes.tolist() == ["A", "B", "C"], "D and E are slide-specific"


def test_min_expressed_drops_genes_seen_in_too_few_spots(index):
    idx, keys = index
    genes = idx.build_vocabulary(*columns(keys), min_expressed=2)
    assert genes.tolist() == ["A", "B"], "C is non-zero in one spot only"


def test_a_slide_with_no_h5ad_is_named_and_its_spots_dropped(index, capsys):
    idx, keys = index
    keys = keys + [("ds_key", "S9", "S9-bc0")]
    genes = idx.build_vocabulary(*columns(keys), min_expressed=1)

    assert "S9" in capsys.readouterr().out
    assert genes.tolist() == ["A", "B", "C"], "the absent slide narrows nothing"


def test_slides_with_no_shared_gene_at_all_stop_the_run(tmp_path):
    root = tmp_path / "raw" / "ds"
    b1 = write_slide(root, "S1", ["A"], [[1]])
    b2 = write_slide(root, "S2", ["B"], [[1]])
    cfg = load_config(
        None,
        {
            "paths.data_root": str(tmp_path),
            "paths.raw_data_root": str(tmp_path / "raw"),
            "paths.raw_h5ad.10x_TuPro": "ds",
        },
    )
    idx = ExpressionIndex(cfg)
    idx._paths = {"k": cfg.raw_h5ad_dir("10x_TuPro")}
    keys = [("k", "S1", b1[0]), ("k", "S2", b2[0])]

    with pytest.raises(SystemExit, match="no shared genes"):
        idx.build_vocabulary(*columns(keys), min_expressed=1)


def test_streaming_leaves_no_slide_in_the_cache(index):
    """120 dense count matrices at once is tens of GB, so the sweeping callers
    (``hvg_pca``'s vocabulary pass) ask for the slide to be dropped again."""
    idx, keys = index
    idx.build_vocabulary(*columns(keys), min_expressed=1, stream=True)
    assert idx._cache == {}

    idx.build_vocabulary(*columns(keys), min_expressed=1)
    assert set(idx._cache) == {"ds_key|S1", "ds_key|S2"}
    idx.drop_cache()
    assert idx._cache == {}


# ── the matrix ───────────────────────────────────────────────────────


def test_matrix_must_be_preceded_by_a_vocabulary(index):
    idx, keys = index
    with pytest.raises(RuntimeError, match="build_vocabulary"):
        idx.matrix(*columns(keys))


def test_raw_counts_come_back_exactly_as_stored(index):
    idx, keys = index
    idx.build_vocabulary(*columns(keys), min_expressed=1)
    Y, found = idx.matrix(*columns(keys), normalize=False)

    assert found.all()
    assert Y.tolist() == [[10, 1, 0], [20, 2, 0], [30, 3, 1], [40, 4, 0], [50, 5, 0]]


def test_normalisation_uses_the_full_panel_before_the_vocabulary_is_applied(index):
    """CP10k must divide by the spot's whole library, including the genes the shared
    vocabulary drops — otherwise a slide with an extra gene is scaled differently
    from one without it, and the difference reads as biology."""
    idx, keys = index
    idx.build_vocabulary(*columns(keys), min_expressed=1)
    Y, found = idx.matrix(*columns(keys))

    assert found.all()
    # First spot of S1: counts [10, 1, 0, 5], library 16 across all four genes.
    expected = np.log1p(np.array([10, 1, 0]) / 16 * 1e4)
    assert np.allclose(Y[0], expected, rtol=1e-5)


def test_matrix_can_stream_and_then_holds_no_slide(index):
    """``build_vocabulary(stream=True)`` alone does not bound the sweep: ``matrix``
    reloads every slide it is given whatever the vocabulary pass evicted, and holds
    them at the full panel while ``Y`` is materialised. The one-pass caller
    (``integrate``'s scVI fit) needs both halves streamed, and got OOM-killed inside
    this method when only the first was."""
    idx, keys = index
    idx.build_vocabulary(*columns(keys), min_expressed=1, stream=True)
    assert idx._cache == {}

    Y, found = idx.matrix(*columns(keys), normalize=False, stream=True)
    assert idx._cache == {}
    assert found.all()
    assert Y.tolist() == [[10, 1, 0], [20, 2, 0], [30, 3, 1], [40, 4, 0], [50, 5, 0]]


def test_streaming_the_matrix_changes_nothing_about_its_values(index):
    """Eviction is a memory decision, not a numerical one — the two paths differ
    only in what is still resident afterwards."""
    idx, keys = index
    idx.build_vocabulary(*columns(keys), min_expressed=1)
    kept, ok_kept = idx.matrix(*columns(keys))
    idx.drop_cache()
    streamed, ok_streamed = idx.matrix(*columns(keys), stream=True)

    assert np.array_equal(kept, streamed)
    assert ok_kept.tolist() == ok_streamed.tolist()
    assert idx._cache == {}


def test_a_slide_matching_no_requested_barcode_is_still_evicted(index):
    """The row-writing block is reached only when a barcode resolved, so eviction
    happens at load time: a slide that matches nothing takes an early `continue`
    and would otherwise sit in the cache for the rest of the sweep."""
    idx, keys = index
    idx.build_vocabulary(*columns(keys), min_expressed=1)
    idx.drop_cache()
    # S2 is asked for, but under a barcode it does not carry.
    mixed = [keys[0], ("ds_key", "S2", "not-a-barcode")]
    _, found = idx.matrix(*columns(mixed), normalize=False, stream=True)

    assert found.tolist() == [True, False]
    assert idx._cache == {}


def test_rows_keep_the_order_they_were_asked_for(index):
    """The caller indexes the result by its own row order, so a slide-grouped read
    must scatter its block back rather than concatenate it."""
    idx, keys = index
    idx.build_vocabulary(*columns(keys), min_expressed=1)
    interleaved = [keys[3], keys[0], keys[4], keys[1]]
    Y, found = idx.matrix(*columns(interleaved), normalize=False)

    assert found.all()
    assert Y[:, 0].tolist() == [40, 10, 50, 20]


def test_unresolvable_spots_are_zero_rows_flagged_in_the_mask(index):
    idx, keys = index
    idx.build_vocabulary(*columns(keys), min_expressed=1)
    mixed = [keys[0], ("ds_key", "S1", "not-a-barcode"), keys[1]]
    Y, found = idx.matrix(*columns(mixed), normalize=False)

    assert found.tolist() == [True, False, True]
    assert Y[1].tolist() == [0.0, 0.0, 0.0]
    assert Y[0][0] == 10 and Y[2][0] == 20


def test_a_counts_layer_is_preferred_over_x(tmp_path):
    """``X`` is raw counts in every cohort used here, but the layer is unambiguous."""
    import pandas as pd

    root = tmp_path / "raw" / "ds"
    root.mkdir(parents=True)
    a = ad.AnnData(
        X=np.array([[1.0, 2.0]], dtype=np.float32),
        obs=pd.DataFrame(index=pd.Index(["bc0"], dtype=str)),
        var=pd.DataFrame(index=pd.Index(["A", "B"], dtype=str)),
    )
    a.layers["counts"] = np.array([[70.0, 30.0]], dtype=np.float32)
    a.write_h5ad(root / "S1.h5ad")

    cfg = load_config(
        None,
        {
            "paths.data_root": str(tmp_path),
            "paths.raw_data_root": str(tmp_path / "raw"),
            "paths.raw_h5ad.10x_TuPro": "ds",
        },
    )
    idx = ExpressionIndex(cfg)
    idx._paths = {"k": cfg.raw_h5ad_dir("10x_TuPro")}
    keys = [("k", "S1", "bc0")]
    idx.build_vocabulary(*columns(keys), min_expressed=1)
    Y, _ = idx.matrix(*columns(keys), normalize=False)

    assert Y.tolist() == [[70.0, 30.0]]


def test_a_sparse_matrix_is_densified_transparently(tmp_path):
    import pandas as pd
    import scipy.sparse as sp

    root = tmp_path / "raw" / "ds"
    root.mkdir(parents=True)
    a = ad.AnnData(
        X=sp.csr_matrix(np.array([[7.0, 3.0]], dtype=np.float32)),
        obs=pd.DataFrame(index=pd.Index(["bc0"], dtype=str)),
        var=pd.DataFrame(index=pd.Index(["A", "B"], dtype=str)),
    )
    a.write_h5ad(root / "S1.h5ad")

    cfg = load_config(
        None,
        {
            "paths.data_root": str(tmp_path),
            "paths.raw_data_root": str(tmp_path / "raw"),
            "paths.raw_h5ad.10x_TuPro": "ds",
        },
    )
    idx = ExpressionIndex(cfg)
    idx._paths = {"k": cfg.raw_h5ad_dir("10x_TuPro")}
    keys = [("k", "S1", "bc0")]
    idx.build_vocabulary(*columns(keys), min_expressed=1)
    Y, found = idx.matrix(*columns(keys), normalize=False)

    assert found.tolist() == [True]
    assert Y.tolist() == [[7.0, 3.0]]


# ── locating the files ───────────────────────────────────────────────


def test_an_explicit_raw_h5ad_entry_beats_the_registry_subdir(tmp_path):
    cfg = load_config(
        None,
        {
            "paths.data_root": str(tmp_path),
            "paths.raw_data_root": str(tmp_path / "raw"),
            "paths.raw_h5ad.10x_TuPro": "somewhere/else",
        },
    )
    idx = ExpressionIndex(cfg)
    assert idx._paths["10x_TuPro"] == tmp_path / "raw" / "somewhere" / "else"


def test_an_absolute_raw_h5ad_entry_wins_outright(tmp_path):
    cfg = load_config(
        None,
        {
            "paths.raw_data_root": str(tmp_path / "raw"),
            "paths.raw_h5ad.10x_TuPro": "/mnt/scratch/tupro",
        },
    )
    idx = ExpressionIndex(cfg)
    assert str(idx._paths["10x_TuPro"]) == "/mnt/scratch/tupro"


def test_a_missing_slide_file_resolves_to_none(index):
    idx, _ = index
    assert idx.sample_path("ds_key", "S1") is not None
    assert idx.sample_path("ds_key", "nope") is None
    assert idx.sample_path("unregistered", "S1") is None
