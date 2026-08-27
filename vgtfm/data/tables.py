"""Canonical spot table: features + metadata for one gene-FM substrate.

The cached HuggingFace ``DatasetDict``s produced by the embedding stage store one
row per Visium spot with columns::

    sample_id, spot_id, array_row, array_col, dataset_id, annotation,
    gene_features (1, Dg), patch_features (1, Dp)

Arrow is a poor fit for the access pattern downstream (repeated full-column reads
of a 3072-wide float32 matrix), so the first thing every run does is materialise
the two feature matrices as ``.npy`` next to a parquet metadata frame. That cache
is a pure function of (merged dataset, filters) and is therefore shared across
runs and seeds — it lives under ``<artifact_root>/_cache/<substrate>/``.

``tissue`` comes from the cohort registry, and TuPro sample ids are parsed into
``donor / region / replicate`` so the hierarchical folds and the donor-level
bootstrap have a patient key rather than a slide name.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..labels import BELOW_FLOOR, canonical_labels, class_floor_mask, labeled_mask

FEATURE_COLUMNS = ("gene_features", "patch_features")
META_COLUMNS = ("sample_id", "spot_id", "array_row", "array_col", "dataset_id", "annotation")

#: Rows written per chunk when materialising a feature matrix.
_CHUNK = 8192

#: Bumped whenever the *shape* of the cached metadata frame changes, so a cache
#: written by an older version is rebuilt rather than reused with missing or stale
#: columns. The row count and source path alone cannot detect that.
#:   1 -> base columns
#:   2 -> annotation canonicalised (vgtfm.labels.CANONICAL), raw kept as
#:        `annotation_raw`
_CACHE_SCHEMA = 2


# ── sample-id parsing ────────────────────────────────────────────────


@dataclass(frozen=True)
class TuProSampleID:
    donor: str
    region: int
    replicate: int
    raw: str


def parse_tupro_sample_id(s: str) -> TuProSampleID | None:
    """Parse a TuPro slide id like ``MACEGEJ-1-2`` into donor/region/replicate.

    Returns ``None`` for ids from the other cohorts, which have no replicate
    structure and therefore cannot enter the hierarchical folds.
    """
    m = re.match(r"^(.+)-(\d+)-(\d+)$", str(s))
    if not m:
        return None
    return TuProSampleID(
        donor=m.group(1), region=int(m.group(2)), replicate=int(m.group(3)), raw=str(s)
    )


# ── cohort registry ──────────────────────────────────────────────────


@dataclass
class Registry:
    """Cohort metadata loaded from ``configs/datasets.json``."""

    datasets: dict[str, dict]

    @property
    def sample_tissue(self) -> dict[str, str]:
        return {s["id"]: d["tissue"] for d in self.datasets.values() for s in d.get("samples", [])}

    @property
    def sample_dataset(self) -> dict[str, str]:
        return {s["id"]: name for name, d in self.datasets.items() for s in d.get("samples", [])}

    @property
    def sample_split(self) -> dict[str, str]:
        return {
            s["id"]: s.get("split", "")
            for d in self.datasets.values()
            for s in d.get("samples", [])
        }

    def annotated_cohorts(self) -> list[str]:
        return [k for k, d in self.datasets.items() if d.get("annotation_col")]

    @property
    def sample_donor_pattern(self) -> dict[str, str | None]:
        return {
            s["id"]: d.get("donor_pattern")
            for name, d in self.datasets.items()
            for s in d.get("samples", [])
        }

    @property
    def dataset_donor_pattern(self) -> dict[str, str | None]:
        return {name: d.get("donor_pattern") for name, d in self.datasets.items()}

    def patient_keys(self, sample_ids, dataset_ids=None) -> np.ndarray:
        """Patient key per slide, per each cohort's registry ``donor_pattern``.

        Not :attr:`SpotTable.donor`. That one drives the folds and the donor-level
        bootstrap, parses only TuPro ids, and so counts De Zuani's six ``P10-*``
        slides as six patients. It is nonetheless the right key *there*: every
        annotated cohort either encodes its donor in the id (TuPro) or has one slide
        per patient (USZ), so the two agree wherever a fold or an interval is built.

        They part company on the unannotated training cohorts, which is exactly
        where the train-split diagnostics look — 76 slides but 56 patients, because
        De Zuani contributes six patients with two to six slides each. Reporting a
        "between-donor" fraction off the slide-level key there makes it identical to
        the between-slide one by construction, and makes the patient-leakage gap
        unmeasurable rather than merely small.
        """
        by_sample = self.sample_donor_pattern
        by_dataset = self.dataset_donor_pattern
        # A slide the registry enumerates carries its own pattern. One it does not
        # still belongs to a cohort, and that cohort's pattern is the right answer —
        # without this a new slide added to an existing cohort would silently become
        # its own patient, which is the exact failure this key exists to remove.
        datasets = (
            [None] * len(np.asarray(sample_ids)) if dataset_ids is None else list(dataset_ids)
        )
        return np.array(
            [
                patient_key(s, by_sample.get(str(s)) or by_dataset.get(str(d)))
                for s, d in zip(sample_ids, datasets)
            ],
            dtype=object,
        )


def load_registry(path: str | Path) -> Registry:
    with open(path) as fh:
        raw = json.load(fh)
    return Registry(datasets=raw.get("datasets", {}))


def h5ad_dirs(cfg, registry: "Registry | None" = None) -> dict[str, Path]:
    """``dataset_id -> directory of raw per-sample .h5ad files``.

    ``paths.raw_h5ad`` wins where a site profile names a cohort explicitly, because
    on a cluster the raw cohorts and the tokenizer's output are usually different
    filesystems; the registry's ``subdirs.h5ad`` under ``raw_root()`` covers the rest.

    Two stages need this and must agree on it: ``biosignal`` reads counts as its
    regression target, and ``embed`` reads the same files to produce the features
    everything else is fitted on. A profile that pointed them at different copies
    would put the ridge R² of one cohort against the embeddings of another.
    """
    registry = registry or load_registry(cfg.paths.datasets_json)
    out = {name: cfg.raw_h5ad_dir(name) for name in cfg.paths.raw_h5ad}
    for name, spec in registry.datasets.items():
        sub = (spec.get("subdirs") or {}).get("h5ad")
        if name not in out and sub:
            out[name] = cfg.raw_root() / name / sub
    return out


def patient_key(sample_id, pattern: str | None) -> str:
    """Patient key for one slide, per its cohort's registry ``donor_pattern``.

    ``None`` — and a pattern that does not match — means the slide is its own
    patient, which is the right answer for MOSAIC and USZ, where each slide is a
    different person.
    """
    if not pattern:
        return str(sample_id)
    m = re.match(pattern, str(sample_id))
    return m.group(1) if m else str(sample_id)


def infer_tissue(sample_id: str, sample_tissue: dict[str, str]) -> str:
    """Tissue for a slide, preferring the registry over name heuristics.

    The heuristics only fire for slides absent from the registry (``KC1`` -> ``KC``,
    ``MW-B-001a-vis`` -> ``MW-B``); they exist so an unregistered cohort degrades to
    a sensible grouping key instead of crashing the per-tissue evaluation.
    """
    name = str(sample_id)
    if name in sample_tissue:
        return sample_tissue[name]
    m = re.match(r"^(MW-[A-Za-z])-\d", name)
    if m:
        return m.group(1).upper()
    m = re.match(r"^([A-Za-z]+)\d", name)
    if m:
        return m.group(1).upper()
    return name


# ── spot table ───────────────────────────────────────────────────────


@dataclass
class SpotTable:
    """Feature matrices plus a row-aligned metadata frame."""

    gene: np.ndarray  # (N, Dg) float32
    patch: np.ndarray  # (N, Dp) float32
    meta: pd.DataFrame  # N rows, see META_COLUMNS + derived columns
    substrate: str

    def __post_init__(self) -> None:
        n = len(self.meta)
        if self.gene.shape[0] != n or self.patch.shape[0] != n:
            raise ValueError(
                f"row mismatch: gene={self.gene.shape[0]} patch={self.patch.shape[0]} meta={n}"
            )

    # -- shape ------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.meta)

    @property
    def gene_dim(self) -> int:
        return int(self.gene.shape[1])

    @property
    def patch_dim(self) -> int:
        return int(self.patch.shape[1])

    # -- columns as arrays -------------------------------------------
    def col(self, name: str) -> np.ndarray:
        return self.meta[name].to_numpy()

    @property
    def sample_id(self) -> np.ndarray:
        return self.col("sample_id")

    @property
    def annotation(self) -> np.ndarray:
        """Canonical class name — what every metric in the pipeline scores."""
        return self.col("annotation")

    @property
    def annotation_raw(self) -> np.ndarray:
        """The cohort's own spelling, before :data:`vgtfm.labels.CANONICAL`.

        Falls back to the canonical column for a frame cached before schema 2, so
        a stale cache degrades to "no merge happened" rather than raising.
        """
        if "annotation_raw" not in self.meta.columns:
            return self.col("annotation")
        return self.col("annotation_raw")

    @property
    def tissue(self) -> np.ndarray:
        return self.col("tissue")

    @property
    def donor(self) -> np.ndarray:
        """Patient key: the TuPro donor where parseable, else the slide id.

        Slides that are their own donor (USZ, MOSAIC, De Zuani) still get a
        well-defined key, so the donor-level bootstrap never silently treats two
        slides from one patient as independent.
        """
        return self.col("donor")

    @property
    def labeled(self) -> np.ndarray:
        return labeled_mask(self.annotation)

    def features(self, kind: str) -> np.ndarray:
        if kind in ("gene", "gene_features"):
            return self.gene
        if kind in ("patch", "patch_features"):
            return self.patch
        raise KeyError(f"unknown feature kind '{kind}'")

    # -- selection ---------------------------------------------------
    def rows_for_samples(self, sample_ids) -> np.ndarray:
        want = set(map(str, sample_ids))
        return np.where(np.isin(self.sample_id.astype(str), list(want)))[0]

    def subset(self, idx: np.ndarray) -> "SpotTable":
        idx = np.asarray(idx)
        return SpotTable(
            gene=np.ascontiguousarray(self.gene[idx]),
            patch=np.ascontiguousarray(self.patch[idx]),
            meta=self.meta.iloc[idx].reset_index(drop=True),
            substrate=self.substrate,
        )

    def describe(self) -> str:
        n_lab = int(self.labeled.sum())
        return (
            f"{self.substrate}: {self.n:,} spots  gene={self.gene_dim} "
            f"patch={self.patch_dim}  slides={self.meta.sample_id.nunique()}  "
            f"donors={self.meta.donor.nunique()}  labelled={n_lab:,}"
        )


# ── cache building ───────────────────────────────────────────────────


def _cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "gene": cache_dir / "gene_features.npy",
        "patch": cache_dir / "patch_features.npy",
        "meta": cache_dir / "meta.parquet",
        "info": cache_dir / "cache_info.json",
    }


def _extract_matrix(ds, column: str, out_path: Path) -> np.ndarray:
    """Materialise one Array2D feature column into a ``.npy`` on disk.

    Chunked so peak memory stays at ``_CHUNK`` rows rather than the whole column
    (the full patch matrix is several GB).
    """
    n = len(ds)
    probe = np.asarray(ds[0][column], dtype=np.float32).reshape(-1)
    dim = probe.shape[0]
    arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n, dim))
    fmt = ds.with_format("numpy")
    for start in range(0, n, _CHUNK):
        stop = min(start + _CHUNK, n)
        block = np.asarray(fmt[start:stop][column], dtype=np.float32)
        arr[start:stop] = block.reshape(stop - start, dim)
        if (start // _CHUNK) % 20 == 0:
            print(f"    {column}: {stop:,}/{n:,}", flush=True)
    arr.flush()
    return arr


def _annotate(meta: pd.DataFrame) -> pd.DataFrame:
    """Add the canonical annotation column, preserving the cohort's own spelling.

    ``annotation`` becomes the canonical name every downstream stage scores and
    plots; ``annotation_raw`` keeps what the cohort wrote, so each cohort's own
    vocabulary is still reportable.
    """
    raw = meta["annotation"].to_numpy()
    meta["annotation_raw"] = raw
    meta["annotation"] = canonical_labels(raw)
    return meta


def _migrate_meta(paths: dict[str, Path], schema: int) -> bool:
    """Bring an older cached metadata frame up to :data:`_CACHE_SCHEMA` in place.

    Returns ``False`` when the gap cannot be closed without the source dataset, in
    which case the caller rebuilds. Migration is preferred wherever it is exact,
    since the feature matrices are several GB.

    Schema 1 -> 2 is exact without the source: a schema-1 frame's ``annotation``
    column *is* the raw cohort spelling that schema 2 stores as ``annotation_raw``.
    """
    if schema != 1:
        return False
    meta = pd.read_parquet(paths["meta"])
    if "annotation" not in meta.columns:
        return False
    meta.to_parquet(paths["meta"].with_suffix(".parquet.bak"), index=False)
    _annotate(meta).to_parquet(paths["meta"], index=False)
    return True


def build_cache(cfg, substrate: str | None = None, *, force: bool = False) -> Path:
    """Extract the merged dataset into ``.npy`` + ``.parquet``; return the cache dir.

    Idempotent: an existing cache whose ``cache_info.json`` matches the current
    source path and row count is reused.
    """
    from datasets import concatenate_datasets, load_from_disk

    substrate = substrate or cfg.data.substrate
    src = cfg.data_path(substrate)
    cache_dir = cfg.cache_dir(substrate)
    paths = _cache_paths(cache_dir)

    if not src.exists():
        raise SystemExit(
            f"Merged dataset for substrate '{substrate}' not found at {src}.\n"
            f"Either run the embed stage or point paths.merged_datasets['{substrate}'] "
            f"at an existing cached dataset."
        )

    if paths["info"].exists() and not force:
        info = json.loads(paths["info"].read_text())
        if info.get("source") == str(src) and all(
            paths[k].exists() for k in ("gene", "patch", "meta")
        ):
            schema = int(info.get("schema", 1))
            if schema == _CACHE_SCHEMA:
                print(f"  cache hit: {cache_dir} ({info['n_spots']:,} spots)")
                return cache_dir
            if _migrate_meta(paths, schema):
                info["schema"] = _CACHE_SCHEMA
                paths["info"].write_text(json.dumps(info, indent=2))
                print(
                    f"  cache hit: {cache_dir} ({info['n_spots']:,} spots, "
                    f"migrated schema {schema} -> {_CACHE_SCHEMA})"
                )
                return cache_dir
            print(
                f"  cache at {cache_dir} is schema {schema} and cannot be "
                f"migrated to {_CACHE_SCHEMA} — rebuilding"
            )

    print(f"  building feature cache for '{substrate}' from {src}")
    dsd = load_from_disk(str(src))
    split_names = [s for s in cfg.data.splits if s in dsd]
    if not split_names:
        raise SystemExit(
            f"None of splits {list(cfg.data.splits)} present in {src} (found {list(dsd.keys())})"
        )
    parts, split_col = [], []
    for name in split_names:
        parts.append(dsd[name])
        split_col.extend([name] * len(dsd[name]))
    ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]

    missing = [c for c in (*META_COLUMNS, *FEATURE_COLUMNS) if c not in ds.column_names]
    if missing:
        raise SystemExit(f"{src} is missing required columns: {missing}")

    meta = pd.DataFrame({c: ds[c] for c in META_COLUMNS})
    meta["split"] = split_col
    _annotate(meta)

    registry = load_registry(cfg.paths.datasets_json)
    tissue_map = registry.sample_tissue
    meta["tissue"] = [infer_tissue(s, tissue_map) for s in meta["sample_id"]]

    parsed = [parse_tupro_sample_id(s) for s in meta["sample_id"]]
    meta["donor"] = [p.donor if p else s for p, s in zip(parsed, meta["sample_id"])]
    meta["region"] = [p.region if p else -1 for p in parsed]
    meta["replicate"] = [p.replicate if p else -1 for p in parsed]

    _extract_matrix(ds, "gene_features", paths["gene"])
    _extract_matrix(ds, "patch_features", paths["patch"])
    meta.to_parquet(paths["meta"], index=False)

    gene_dim = int(np.load(paths["gene"], mmap_mode="r").shape[1])
    patch_dim = int(np.load(paths["patch"], mmap_mode="r").shape[1])
    paths["info"].write_text(
        json.dumps(
            {
                "substrate": substrate,
                "source": str(src),
                "splits": split_names,
                "schema": _CACHE_SCHEMA,
                "n_spots": int(len(meta)),
                "gene_dim": gene_dim,
                "patch_dim": patch_dim,
                "n_samples": int(meta.sample_id.nunique()),
            },
            indent=2,
        )
    )
    print(f"  cached {len(meta):,} spots (gene {gene_dim}d, patch {patch_dim}d) -> {cache_dir}")
    return cache_dir


def select_rows(cfg, meta: pd.DataFrame) -> np.ndarray:
    """Rows kept after the config's sample filters and optional spot cap.

    The cap is applied proportionally per slide rather than uniformly at random,
    so a smoke run keeps every slide (and therefore every fold) instead of
    dropping the small ones entirely.
    """
    keep = np.ones(len(meta), dtype=bool)
    if cfg.data.include_samples:
        keep &= meta["sample_id"].isin(list(cfg.data.include_samples)).to_numpy()
    if cfg.data.exclude_samples:
        keep &= ~meta["sample_id"].isin(list(cfg.data.exclude_samples)).to_numpy()
    idx = np.where(keep)[0]

    cap = int(cfg.data.max_spots or 0)
    if cap and len(idx) > cap:
        rng = np.random.default_rng(cfg.folds.seed)
        sub = meta.iloc[idx]
        frac = cap / len(idx)
        picks = []
        for _, rows in sub.groupby("sample_id", sort=True).groups.items():
            rows = np.asarray(rows)
            take = max(1, int(round(len(rows) * frac)))
            picks.append(rng.choice(rows, size=min(take, len(rows)), replace=False))
        idx = np.sort(np.concatenate(picks))
    return idx


def apply_class_floor(cfg, meta: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    """Demote classes under ``data.min_class_{spots,slides}`` to unlabelled, in place.

    Applied after :func:`select_rows`, so the counts describe the cohort this config
    actually keeps rather than the whole cache: a smoke run that includes four
    donors must judge its classes on those four donors. ``annotation_raw`` is left
    alone, so the original name is still recoverable from any artefact.
    """
    drop, dropped = class_floor_mask(
        meta["annotation"].to_numpy(),
        meta["tissue"].to_numpy(),
        meta["sample_id"].to_numpy(),
        min_spots=int(cfg.data.min_class_spots),
        min_slides=int(cfg.data.min_class_slides),
    )
    # Always present, so a consumer can test it without knowing whether the floor
    # bit — and so a class that was dropped is still nameable in the appendix.
    meta["annotation_floored"] = ""
    if not dropped:
        return meta
    meta.loc[drop, "annotation_floored"] = meta.loc[drop, "annotation"]
    meta.loc[drop, "annotation"] = BELOW_FLOOR
    if verbose:
        floor = (
            f"< {cfg.data.min_class_spots} spots or "
            f"< {cfg.data.min_class_slides} slides in their tissue"
        )
        print(
            f"  classes: {len(dropped)} below the support floor ({floor}) "
            f"— {int(drop.sum()):,} spot(s) demoted to unlabelled"
        )
        for d in dropped:
            print(
                f"    dropped {d['tissue']} | {d['annotation']}: "
                f"{d['n_spots']:,} spot(s) on {d['n_slides']} slide(s)"
            )
    return meta


def load_meta(cfg, substrate: str | None = None) -> pd.DataFrame:
    """Metadata only, with the same row filtering :func:`load` applies.

    Stages that need labels and slide ids but not the features — figures, cohort
    summaries — use this instead of materialising several GB of float32.
    """
    substrate = substrate or cfg.data.substrate
    cache_dir = build_cache(cfg, substrate, force=cfg.data.force_rebuild)
    meta = pd.read_parquet(_cache_paths(cache_dir)["meta"])
    idx = select_rows(cfg, meta)
    if len(idx) != len(meta):
        meta = meta.iloc[idx]
    # Quiet: the stages that load metadata alone call this repeatedly, and `load`
    # already reports the floor once per stage.
    return apply_class_floor(cfg, meta.reset_index(drop=True), verbose=False)


def load(cfg, substrate: str | None = None) -> SpotTable:
    """Build (if needed) and load the spot table for one substrate."""
    substrate = substrate or cfg.data.substrate
    cache_dir = build_cache(cfg, substrate, force=cfg.data.force_rebuild)
    paths = _cache_paths(cache_dir)

    meta = pd.read_parquet(paths["meta"])
    idx = select_rows(cfg, meta)

    gene_mm = np.load(paths["gene"], mmap_mode="r")
    patch_mm = np.load(paths["patch"], mmap_mode="r")
    full = len(idx) == len(meta)
    gene = np.ascontiguousarray(gene_mm) if full else np.ascontiguousarray(gene_mm[idx])
    patch = np.ascontiguousarray(patch_mm) if full else np.ascontiguousarray(patch_mm[idx])
    meta = meta if full else meta.iloc[idx]
    meta = apply_class_floor(cfg, meta.reset_index(drop=True))

    table = SpotTable(gene=gene, patch=patch, meta=meta, substrate=substrate)
    print(f"  {table.describe()}")
    return table
