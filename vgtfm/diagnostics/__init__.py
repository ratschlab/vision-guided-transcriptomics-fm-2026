"""Diagnostics shared setup."""

from __future__ import annotations

import numpy as np

# Re-exported: `model_seeds` is a reading of the config, not a diagnostic, and
# `figures` and `biosignal` answer the same question. It lives in one place.
from ..config import model_seeds  # noqa: F401


def split_rows(cfg, table) -> np.ndarray:
    """Rows of the cohort split the diagnostics run on (``all`` selects everything)."""
    split = cfg.diagnostics.split
    rows = np.arange(table.n) if split == "all" else np.where(table.col("split") == split)[0]
    if len(rows) == 0:
        raise SystemExit(
            f"diagnostics.split='{split}' selects no spots. "
            f"Available: {sorted(set(table.col('split')))}"
        )
    return rows
