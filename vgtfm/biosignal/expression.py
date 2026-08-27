"""Count-level expression for the spots in the cohort.

Asking *which genes* a refinement preserves or destroys needs the expression
itself, not the frozen embedding. This module reads it from the per-sample ``.h5ad``
files listed in the cohort registry, matching on
``(dataset_id, sample_id, spot_id)`` where ``spot_id`` is the raw Visium barcode
and ``var_names`` are already HGNC symbols.

Only a shared gene vocabulary is used: genes present in every sample that
contributes spots, further filtered to those expressed in enough spots to be
predictable at all. Fixing one vocabulary up front (rather than intersecting
fold-local ones afterwards) is what lets the per-gene R^2 be accumulated in a
streaming fashion instead of materialising a spots-by-genes matrix per fold.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data.tables import h5ad_dirs, load_registry


class ExpressionIndex:
    """Lazily-opened per-sample count matrices with one shared gene vocabulary."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.registry = load_registry(cfg.paths.datasets_json)
        self.data_root = Path(cfg.paths.data_root)
        self._paths = self._resolve_paths()
        self._cache: dict[str, tuple] = {}
        self.genes: np.ndarray | None = None

    # -- locating files ---------------------------------------------
    def _resolve_paths(self) -> dict[str, Path]:
        """``dataset_id -> directory of .h5ad files``.

        Shared with the ``embed`` stage, which reads the same files to build the
        features these counts are the target for — see :func:`..data.tables.h5ad_dirs`.
        """
        return h5ad_dirs(self.cfg, self.registry)

    def sample_path(self, dataset_id: str, sample_id: str) -> Path | None:
        base = self._paths.get(dataset_id)
        if base is None:
            return None
        p = base / f"{sample_id}.h5ad"
        return p if p.exists() else None

    # -- reading ----------------------------------------------------
    def _load_sample(self, dataset_id: str, sample_id: str):
        """Return ``(barcode -> row, counts, var_names)`` for one slide."""
        key = f"{dataset_id}|{sample_id}"
        if key in self._cache:
            return self._cache[key]

        import anndata as ad
        import scipy.sparse as sp

        path = self.sample_path(dataset_id, sample_id)
        if path is None:
            self._cache[key] = None
            return None
        a = ad.read_h5ad(path)
        # Prefer an explicit counts layer when present; X is raw counts in all
        # cohorts used here, but the layer is the unambiguous source.
        X = a.layers["counts"] if "counts" in a.layers else a.X
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        entry = (
            {b: i for i, b in enumerate(a.obs_names.astype(str))},
            np.asarray(X, dtype=np.float32),
            np.asarray(a.var_names.astype(str)),
        )
        self._cache[key] = entry
        return entry

    def drop_cache(self) -> None:
        self._cache.clear()

    # -- vocabulary -------------------------------------------------
    def build_vocabulary(
        self,
        dataset_ids,
        sample_ids,
        spot_ids,
        *,
        min_expressed: int = 10,
        max_spots: int = 40_000,
        seed: int = 42,
        stream: bool = False,
    ) -> np.ndarray:
        """Genes shared by every contributing slide and expressed widely enough.

        ``min_expressed`` counts spots with a non-zero count, evaluated on a random
        subsample so building the vocabulary does not require reading every slide's
        full matrix into memory at once.

        ``stream`` evicts each slide from the cache as soon as it has been read. The
        default keeps them, which suits callers that immediately ask for the same
        slides again (the biosignal folds); set it for a caller that sweeps all 120
        slides once, where holding every dense count matrix at once is tens of GB.
        """
        # Coerce to ndarray: positional indexing below is wrong for a pandas Series.
        spot_ids = np.asarray(spot_ids).astype(str)
        keys = _group_by_sample(dataset_ids, sample_ids, spot_ids)
        shared: set[str] | None = None
        for (ds, sid), _rows in keys.items():
            entry = self._load_sample(ds, sid)
            if entry is None:
                print(f"    no .h5ad for {ds}/{sid} — its spots are dropped")
                continue
            names = set(entry[2].tolist())
            shared = names if shared is None else (shared & names)
            if stream:
                self._cache.pop(f"{ds}|{sid}", None)
        if not shared:
            raise SystemExit("no shared genes across the requested slides")

        genes = np.array(sorted(shared))
        rng = np.random.default_rng(seed)
        expressed = np.zeros(len(genes), dtype=np.int64)
        n_seen = 0
        for (ds, sid), rows in keys.items():
            entry = self._load_sample(ds, sid)
            if entry is None:
                continue
            barcode_to_row, X, names = entry
            take = [barcode_to_row[b] for b in spot_ids[rows] if b in barcode_to_row]
            if not take:
                continue
            if len(take) > max_spots // max(len(keys), 1):
                take = rng.choice(take, size=max_spots // max(len(keys), 1), replace=False)
            col = _column_index(names, genes)
            block = X[np.asarray(take)][:, col]
            expressed += (block > 0).sum(axis=0)
            n_seen += len(take)
            if stream:
                self._cache.pop(f"{ds}|{sid}", None)

        keep = expressed >= min_expressed
        self.genes = genes[keep]
        print(
            f"    gene vocabulary: {len(self.genes):,} of {len(genes):,} shared genes "
            f"expressed in >= {min_expressed} of {n_seen:,} sampled spots"
        )
        return self.genes

    def matrix(
        self, dataset_ids, sample_ids, spot_ids, *, normalize: bool = True, stream: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expression for the requested spots.

        Returns ``(Y, found)`` where ``Y`` has one row per *found* spot, in the
        original order, and ``found`` is the boolean mask of resolvable spots.

        With ``normalize`` (the default) the values are CP10k followed by
        ``log1p``, computed on the full gene panel before restricting to the
        vocabulary so library sizes are correct. With ``normalize=False`` they are
        the raw counts. Both are needed by the same caller: HVG selection models
        the mean-variance trend of counts, which library-size normalisation
        changes, while everything downstream of it wants the log-normalised
        values.

        ``stream`` evicts each slide once its rows have been written, as the flag of
        the same name does in :meth:`build_vocabulary`. The loop below visits every
        slide exactly once, so this is never wrong — but it is only worth asking
        for when the caller will not want those slides again, and it costs the
        biosignal folds a re-read if they do. Streaming *both* calls is what a
        one-pass caller needs: this method reloads every slide it was given
        regardless of what ``build_vocabulary`` streamed away, so dropping the cache
        between the two moves the peak here rather than removing it. Held at the
        full gene panel rather than the vocabulary, that cache is tens of GB, and it
        is resident at exactly the moment ``Y`` is fully materialised.
        """
        if self.genes is None:
            raise RuntimeError("call build_vocabulary() before matrix()")
        spot_ids = np.asarray(spot_ids).astype(str)
        n = len(spot_ids)
        Y = np.zeros((n, len(self.genes)), dtype=np.float32)
        found = np.zeros(n, dtype=bool)

        for (ds, sid), rows in _group_by_sample(dataset_ids, sample_ids, spot_ids).items():
            entry = self._load_sample(ds, sid)
            # Evicted here, not after the row-writing below: two of the paths through
            # this body `continue` before reaching the end of it, and a slide that
            # took one of them would stay cached for the rest of the sweep. `entry`
            # holds the arrays alive for this iteration either way, so one slide is
            # resident at a time and the cache ends the call empty.
            if stream:
                self._cache.pop(f"{ds}|{sid}", None)
            if entry is None:
                continue
            barcode_to_row, X, names = entry
            local, target = [], []
            for r in rows:
                idx = barcode_to_row.get(spot_ids[r])
                if idx is not None:
                    local.append(idx)
                    target.append(r)
            if not local:
                continue
            counts = X[np.asarray(local)]
            if normalize:
                totals = counts.sum(axis=1, keepdims=True)
                totals[totals <= 0] = 1.0
                counts = np.log1p(counts / totals * 1e4)
            Y[np.asarray(target)] = counts[:, _column_index(names, self.genes)]
            found[np.asarray(target)] = True

        return Y, found


def _group_by_sample(dataset_ids, sample_ids, spot_ids) -> dict[tuple[str, str], np.ndarray]:
    ds = np.asarray(dataset_ids).astype(str)
    sid = np.asarray(sample_ids).astype(str)
    out: dict[tuple[str, str], list[int]] = {}
    for i in range(len(sid)):
        out.setdefault((ds[i], sid[i]), []).append(i)
    return {k: np.asarray(v) for k, v in out.items()}


def _column_index(names: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """Positions of *wanted* inside *names*, assuming every gene is present."""
    lookup = {g: i for i, g in enumerate(names)}
    return np.array([lookup[g] for g in wanted], dtype=np.int64)
