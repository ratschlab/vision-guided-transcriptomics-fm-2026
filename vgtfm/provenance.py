"""Which source tree produced this output.

Every run stamps a fingerprint of its sources into its banner and manifest, and
``slurm/preflight.sh`` prints the same value. Two machines showing the same twelve
characters are running the same code.

This is a content hash, not a version number: it moves with any edit to any shipped
``.py`` and is blind to everything else — data, environment and config are recorded
separately in the same manifest.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

#: The repository root: the directory holding the ``vgtfm`` package.
ROOT = Path(__file__).resolve().parents[1]

#: What counts as "the code". Tests and configs are excluded: a config change is
#: already recorded in config.resolved.json, and a test cannot change a result.
_SOURCES = ("vgtfm/**/*.py", "run.py")


@lru_cache(maxsize=1)
def source_fingerprint(length: int = 12) -> str:
    """Short content hash over the shipped Python sources.

    Stable across machines and interpreter versions: paths are relative and
    hashed in sorted order, and file bytes go in unmodified.
    """
    h = hashlib.sha256()
    for pattern in _SOURCES:
        for path in sorted(ROOT.glob(pattern)):
            if "__pycache__" in path.parts:
                continue
            h.update(str(path.relative_to(ROOT)).encode())
            h.update(b"\0")
            h.update(path.read_bytes())
            h.update(b"\0")
    return h.hexdigest()[:length]


def array_fingerprint(Z, length: int = 12) -> str:
    """Short content hash of one array, for tying an artefact to the matrix it scored.

    Shape and dtype go in with the bytes, so a reshape or a precision change is a
    different fingerprint rather than a collision. Used to check that a cached
    prediction vector and the embedding a later stage loaded are the same
    representation — the two live in different files and nothing else relates them.
    """
    import hashlib as _h

    import numpy as _np

    a = _np.ascontiguousarray(Z)
    h = _h.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()[:length]
