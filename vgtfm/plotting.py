"""Shared plotting helpers — publication style and stable class colours.

Two invariants hold in every figure this repo produces:

* unlabelled spots are drawn as faint grey dots *behind* the annotated classes, so
  they never obscure the signal (and are excluded from every metric, see
  :mod:`vgtfm.labels`);
* a pathology class keeps the same colour in every slide, UMAP, table and panel.
"""

from __future__ import annotations

import numpy as np

from .labels import UNLABELED, labeled_mask, real_classes

# Stable palette: a class -> the same colour in every figure and stage.
CLASS_COLORS = {
    "TUM": "#d62728",
    "Tumor": "#d62728",
    "TLS": "#2ca02c",
    "Normal lymphoid tissue": "#17becf",
    "LN": "#17becf",
    "NOR": "#ff7f0e",
    "Normal": "#ff7f0e",
    "INFL": "#1f77b4",
    "Stroma": "#9467bd",
    "Pigment": "#8c564b",
    "Blood and necrosis": "#e377c2",
}
_FALLBACK = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#393b79",
    "#637939",
    "#8c6d31",
]
GRAY = "#c9c9c9"

#: Point clouds at least this large are drawn as an embedded raster instead of one
#: vector marker per spot. A UMAP panel here carries up to
#: ``diagnostics.umap_max_spots`` points and there are six of them per figure,
#: which costs a couple of MB of PDF and makes the file slow to open in a viewer
#: and slow to typeset. Text, axes and legend stay vector, so nothing that has to
#: be read or searched is affected; below the threshold vector is smaller anyway.
RASTER_MIN_POINTS = 2_000

#: Colours for the model series, shared by every results figure.
MODEL_COLORS = {
    "pca": "#1f77b4",
    "pca_oracle": "#7f7f7f",
    "ae": "#2ca02c",
    "cdann": "#ff7f0e",
    "majority": "#4d4d4d",
}

DPI = 300


def set_pub_style() -> None:
    """Publication-ready matplotlib defaults. Idempotent."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.titleweight": "medium",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            # TrueType embedding keeps text editable in vector exports.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def class_color(cls: str, i: int = 0) -> str:
    """The fixed colour for an annotation class, or the *i*-th fallback.

    Fixed so a class keeps its colour across every figure in the paper; *i* only
    decides which fallback a class outside :data:`CLASS_COLORS` gets, which is why
    the caller passes its index in a *sorted* list.
    """
    return CLASS_COLORS.get(cls, _FALLBACK[i % len(_FALLBACK)])


def palette_for(classes) -> dict[str, str]:
    """Stable ``{class: colour}`` for a list of annotated classes."""
    return {c: class_color(c, i) for i, c in enumerate(sorted(classes))}


def model_color(name: str, i: int = 0) -> str:
    """:func:`class_color` for a model name, off :data:`MODEL_COLORS`."""
    return MODEL_COLORS.get(name, _FALLBACK[i % len(_FALLBACK)])


def scatter_labels_2d(
    ax,
    XY: np.ndarray,
    labels,
    *,
    size: float = 6.0,
    gray_alpha: float = 0.35,
    title=None,
    legend: bool = True,
) -> None:
    """2-D scatter coloured by annotation, unlabelled spots grey and behind."""
    lab = np.asarray(labels).astype(str)
    lbl = labeled_mask(lab)
    # One decision for the whole cloud, so the layers merge into a single image.
    raster = len(XY) >= RASTER_MIN_POINTS
    if (~lbl).any():
        ax.scatter(
            XY[~lbl, 0],
            XY[~lbl, 1],
            s=size,
            c=GRAY,
            alpha=gray_alpha,
            linewidths=0,
            label="unannotated",
            rasterized=raster,
        )
    for i, c in enumerate(real_classes(lab)):
        m = lab == c
        ax.scatter(
            XY[m, 0],
            XY[m, 1],
            s=size,
            c=class_color(c, i),
            linewidths=0,
            label=c,
            rasterized=raster,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(markerscale=3, fontsize=7, loc="best", frameon=False)


def scatter_categorical_2d(
    ax,
    XY: np.ndarray,
    values,
    *,
    size: float = 6.0,
    title=None,
    legend: bool = True,
    max_legend: int = 12,
) -> None:
    """Generic categorical 2-D scatter (sample_id / tissue / dataset_id)."""
    import matplotlib as mpl

    vals = np.asarray(values).astype(str)
    uniq = sorted(set(vals))
    cmap = mpl.colormaps["tab20"].resampled(max(len(uniq), 1))
    raster = len(XY) >= RASTER_MIN_POINTS
    for i, v in enumerate(uniq):
        m = vals == v
        ax.scatter(
            XY[m, 0],
            XY[m, 1],
            s=size,
            color=cmap(i),
            linewidths=0,
            label=str(v)[:16],
            rasterized=raster,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    if legend and len(uniq) <= max_legend:
        ax.legend(markerscale=3, fontsize=7, loc="best", frameon=False)


def save(fig, path, *, also_png: bool = True) -> None:
    """Write a figure as PDF (vector, for LaTeX) and optionally PNG (for review)."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), dpi=DPI, bbox_inches="tight")
    if also_png:
        fig.savefig(path.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


__all__ = [
    "CLASS_COLORS",
    "MODEL_COLORS",
    "GRAY",
    "DPI",
    "UNLABELED",
    "RASTER_MIN_POINTS",
    "set_pub_style",
    "class_color",
    "palette_for",
    "model_color",
    "scatter_labels_2d",
    "scatter_categorical_2d",
    "save",
]
