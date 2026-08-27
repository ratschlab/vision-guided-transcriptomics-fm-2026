"""Alternative objectives, reported in the appendix.

Not part of the default model list; add them explicitly, e.g.::

    python run.py train --set models.names=pca,ae,dual_decoder,infonce
"""

from __future__ import annotations


def build_variant(name: str, cfg, seed: int = 42):
    """Return a variant model, or ``None`` if *name* is not a known variant."""
    from .heads import VARIANT_KINDS, build_head_variant

    if name in VARIANT_KINDS:
        return build_head_variant(name, cfg, seed)
    if name == "jepa":
        from .jepa import JEPAModel

        return JEPAModel(cfg, seed)
    return None
