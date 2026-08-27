"""Figure conventions: stable class colours and unlabelled spots drawn behind.

Both are invariants across stages rather than per-figure choices — a class that
changes colour between two panels makes them look like different data, and an
unlabelled spot drawn on top hides the annotated one under it while contributing to
no metric.
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vgtfm.plotting import (  # noqa: E402
    CLASS_COLORS,
    GRAY,
    class_color,
    model_color,
    palette_for,
    save,
    scatter_categorical_2d,
    scatter_labels_2d,
    set_pub_style,
)


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


# ── colours ──────────────────────────────────────────────────────────


def test_a_class_keeps_one_colour_wherever_it_is_drawn():
    for cls in CLASS_COLORS:
        assert class_color(cls, 0) == class_color(cls, 7) == CLASS_COLORS[cls]


def test_the_merged_tumour_spellings_share_a_colour():
    """`TUM` and `Tumor` are one class after canonicalisation, so a figure drawn
    from raw labels must not give them two."""
    assert class_color("TUM") == class_color("Tumor")


def test_an_unregistered_class_gets_a_stable_fallback_by_position():
    assert class_color("Unknown class", 0) != class_color("Unknown class", 1)
    assert class_color("Unknown class", 0) == class_color("Other class", 0)


def test_a_palette_covers_every_class_exactly_once():
    classes = ["Stroma", "Tumor", "TLS", "Mystery"]
    palette = palette_for(classes)

    assert set(palette) == set(classes)
    assert len(set(palette.values())) == len(classes)
    assert palette == palette_for(reversed(classes)), "order must not matter"


def test_model_colours_are_stable_and_fall_back_for_new_models():
    assert model_color("pca") == model_color("pca", 3)
    assert model_color("brand_new", 0) != model_color("brand_new", 1)


# ── scatters ─────────────────────────────────────────────────────────


def test_unlabelled_spots_are_drawn_first_in_grey(ax):
    XY = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    scatter_labels_2d(ax, XY, ["Tumor", "UNASSIGNED", "Stroma", None])

    first, *classes = ax.collections
    assert matplotlib.colors.to_hex(first.get_facecolor()[0]) == GRAY
    assert first.get_label() == "unannotated"
    assert [c.get_label() for c in classes] == ["Stroma", "Tumor"]
    assert [len(c.get_offsets()) for c in classes] == [1, 1]


def test_a_fully_annotated_panel_draws_no_grey_layer(ax):
    XY = np.array([[0.0, 0.0], [1.0, 1.0]])
    scatter_labels_2d(ax, XY, ["Tumor", "Stroma"])

    assert [c.get_label() for c in ax.collections] == ["Stroma", "Tumor"]


def test_class_series_use_the_shared_palette(ax):
    XY = np.array([[0.0, 0.0], [1.0, 1.0]])
    scatter_labels_2d(ax, XY, ["Tumor", "TLS"])

    for coll in ax.collections:
        expected = CLASS_COLORS[coll.get_label()]
        assert matplotlib.colors.to_hex(coll.get_facecolor()[0]) == expected


def test_a_categorical_scatter_draws_one_series_per_value(ax):
    XY = np.arange(8, dtype=float).reshape(4, 2)
    scatter_categorical_2d(ax, XY, ["S2", "S1", "S2", "S3"])

    assert [c.get_label() for c in ax.collections] == ["S1", "S2", "S3"]
    assert [len(c.get_offsets()) for c in ax.collections] == [1, 2, 1]


def test_a_categorical_legend_is_dropped_once_it_would_be_unreadable(ax):
    XY = np.zeros((30, 2))
    scatter_categorical_2d(ax, XY, [f"slide{i}" for i in range(30)], max_legend=12)
    assert ax.get_legend() is None


def test_scatters_carry_a_title_and_no_axis_ticks(ax):
    XY = np.array([[0.0, 0.0], [1.0, 1.0]])
    scatter_labels_2d(ax, XY, ["Tumor", "Stroma"], title="panel")

    assert ax.get_title() == "panel"
    assert list(ax.get_xticks()) == [] and list(ax.get_yticks()) == []


# ── saving ───────────────────────────────────────────────────────────


def test_saving_writes_a_vector_and_a_raster_copy_and_closes_the_figure(tmp_path):
    fig, _ = plt.subplots()
    save(fig, tmp_path / "sub" / "fig_thing")

    assert (tmp_path / "sub" / "fig_thing.pdf").exists()
    assert (tmp_path / "sub" / "fig_thing.png").exists()
    assert fig.number not in plt.get_fignums(), "figures must not accumulate"


def test_the_raster_copy_can_be_declined(tmp_path):
    fig, _ = plt.subplots()
    save(fig, tmp_path / "fig_thing", also_png=False)

    assert (tmp_path / "fig_thing.pdf").exists()
    assert not (tmp_path / "fig_thing.png").exists()


def test_the_publication_style_keeps_text_editable_and_is_idempotent():
    """Type-3 fonts turn every label into outlines, which a journal will reject."""
    set_pub_style()
    first = dict(matplotlib.rcParams)
    assert matplotlib.rcParams["pdf.fonttype"] == 42
    assert matplotlib.rcParams["ps.fonttype"] == 42

    set_pub_style()
    assert dict(matplotlib.rcParams) == first
