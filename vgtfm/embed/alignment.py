"""Placing Visium spots on the whole-slide image.

Each cohort records the correspondence between the spot grid and the slide image
differently, and getting it wrong produces plausible embeddings of the wrong
tissue. Two methods cover the cohorts used here, named in the registry's
``alignment_method`` field:

``h5ad_obs_pixels``   the ``.h5ad`` already carries pixel coordinates per spot.
                      Note the axis convention: in these files ``y_pixel`` is the
                      image *x* coordinate and ``x_pixel`` is the image *y* — a
                      transposition that is easy to miss and silently mirrors every
                      patch about the diagonal.

``lstsq_estimate``    the alignment was done manually in Loupe, which exports a set
                      of (array col/row -> image x/y) landmark points. The affine
                      map is recovered by least squares from those landmarks, with
                      explicit degeneracy checks: collinear landmarks or a singular
                      linear part mean the exported alignment is unusable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ALIGNMENT_METHODS = ("h5ad_obs_pixels", "lstsq_estimate")


def affine_from_loupe(loupe_json_path: str | Path) -> np.ndarray:
    """Recover the 2x3 affine mapping array coordinates to image pixels."""
    with open(loupe_json_path) as fh:
        data = json.load(fh)

    points = data.get("oligo") or data.get("fiducial")
    if not points:
        raise ValueError(f"{loupe_json_path} has neither 'oligo' nor 'fiducial' points")

    src, dst = [], []
    for p in points:
        if all(k in p for k in ("col", "row", "imageX", "imageY")):
            src.append([p["col"], p["row"], 1.0])
            dst.append([p["imageX"], p["imageY"]])
    if len(src) < 3:
        raise ValueError(f"{loupe_json_path} has {len(src)} usable landmarks; need 3+")

    M_T, _residuals, rank, _s = np.linalg.lstsq(np.array(src), np.array(dst), rcond=None)
    M = M_T.T
    if rank < 3:
        raise ValueError(
            f"degenerate alignment in {loupe_json_path}: landmark rank {rank} (collinear points)"
        )
    det = float(np.linalg.det(M[:, :2]))
    if np.isclose(det, 0.0):
        raise ValueError(f"degenerate alignment in {loupe_json_path}: determinant {det:.2e}")
    return M


def spot_pixels(adata, *, method: str, loupe_json_path: str | Path | None = None) -> pd.DataFrame:
    """Return ``image_x``, ``image_y``, ``array_col``, ``array_row`` per barcode."""
    obs = adata.obs
    out = pd.DataFrame(index=obs.index.astype(str))

    if method == "h5ad_obs_pixels":
        missing = {"x_pixel", "y_pixel"} - set(obs.columns)
        if missing:
            raise KeyError(f"h5ad_obs_pixels needs {sorted(missing)} in adata.obs")
        # Deliberate transposition; see the module docstring.
        out["image_x"] = obs["y_pixel"].to_numpy()
        out["image_y"] = obs["x_pixel"].to_numpy()
    elif method == "lstsq_estimate":
        if loupe_json_path is None:
            raise ValueError("lstsq_estimate requires a Loupe alignment JSON")
        col = _first_column(obs, ("array_col", "y_array"))
        row = _first_column(obs, ("array_row", "x_array"))
        coords = np.column_stack([col, row, np.ones(len(obs))])
        transformed = coords @ affine_from_loupe(loupe_json_path).T
        out["image_x"] = transformed[:, 0]
        out["image_y"] = transformed[:, 1]
    else:
        raise ValueError(f"unknown alignment method '{method}'. Known: {ALIGNMENT_METHODS}")

    out["array_col"] = _first_column(obs, ("array_col", "y_array"), default=0)
    out["array_row"] = _first_column(obs, ("array_row", "x_array"), default=0)
    return out


def _first_column(obs, names, default=None) -> np.ndarray:
    for name in names:
        if name in obs.columns:
            return obs[name].to_numpy()
    if default is None:
        raise KeyError(f"none of {names} present in adata.obs")
    return np.full(len(obs), default)
