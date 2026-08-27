"""``embed`` stage — build a merged dataset from raw slides and foundation models.

Optional and heavy. The default entry point of this repository is a cached merged
dataset; this stage is what produces one.

It writes one parquet per (slide, modality) under ``<artifact_root>/_embed/``, then
joins them on ``spot_id`` into the HuggingFace ``DatasetDict`` every other stage
reads. Splitting it that way means a failed or newly added slide is re-embedded on
its own, and that the expensive gene-side models can run in separate environments:

* **Midnight** (H&E) runs here, in the main environment.
* **scVI / sysVI** run here as well — they are trained on the counts in this repo.
* **Geneformer, scGPT, CancerFoundation** have mutually incompatible dependencies
  and are therefore invoked as separate processes with their own conda environments
  (``envs/*.yaml``); see :mod:`vgtfm.embed.gene_fm` for the contract they satisfy.

This stage is not needed to reproduce the analysis from cached embeddings — only to
reproduce the cached embeddings themselves.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.tables import h5ad_dirs, load_registry
from ..degraded import Incomplete, pending, refuse


def _slide_parquet(cfg, modality: str, dataset_id: str, sample_id: str) -> Path:
    d = Path(cfg.paths.artifact_root) / "_embed" / modality / dataset_id
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{sample_id}.parquet"


def _modality_dir(cfg, modality: str) -> Path:
    return Path(cfg.paths.artifact_root) / "_embed" / modality


def gene_manifest(cfg, model: str):
    """Enumerate every registered slide that has counts, for one gene-side model.

    The gene-side environments cannot read this repository's config — they hold a
    different scanpy and a different torch, and asking them to import
    :mod:`vgtfm.config` would couple three dependency sets that exist precisely to
    stay apart. So the stage resolves the cohort here and writes the answer to a
    file they can read with nothing but the standard library.

    Returns ``(manifest_path, slides, absent)``; *absent* names the registered slides
    this site does not hold, which is expected — the registry lists every cohort the
    project knows about and a given machine has a subset.
    """
    from .cohort import Slide, write_manifest

    registry = load_registry(cfg.paths.datasets_json)
    dirs = h5ad_dirs(cfg, registry)
    slides, absent = [], []
    for dataset_id, spec in registry.datasets.items():
        base = dirs.get(dataset_id)
        gene_col = spec.get("gene_ids_col") or "gene_ids"
        for sample in spec.get("samples", []):
            sid = sample["id"]
            path = (base / f"{sid}.h5ad") if base else None
            if path is None or not path.exists():
                absent.append(f"{dataset_id}/{sid}")
                continue
            slides.append(Slide(dataset_id, sid, str(path), gene_col))

    target = _modality_dir(cfg, model) / "manifest.json"
    write_manifest(target, model, slides)
    return target, slides, absent


def embed_patches(cfg, *, batch_size: int = 32, overwrite: bool = False) -> list[Path]:
    """Run Midnight over every registered slide that has an image."""
    import anndata as ad

    from . import midnight
    from .alignment import spot_pixels

    registry = load_registry(cfg.paths.datasets_json)
    data_root = Path(cfg.paths.data_root)
    # The same .h5ad the gene side reads: a profile that pointed the two at
    # different copies would align patches to one cohort and counts to another.
    dirs = h5ad_dirs(cfg, registry)
    model = transform = device = None
    written: list[Path] = []
    cached: list[str] = []
    absent: list[str] = []

    for dataset_id, spec in registry.datasets.items():
        subdirs = spec.get("subdirs") or {}
        h5ad_dir = dirs.get(dataset_id, data_root / dataset_id / (subdirs.get("h5ad") or ""))
        tif_dir = data_root / dataset_id / (subdirs.get("tif") or "")
        method = spec.get("alignment_method", "h5ad_obs_pixels")

        for sample in spec.get("samples", []):
            sid = sample["id"]
            target = _slide_parquet(cfg, "patch", dataset_id, sid)
            if target.exists() and not overwrite:
                cached.append(f"{dataset_id}/{sid}")
                continue
            h5ad_path = h5ad_dir / f"{sid}.h5ad"
            image_path = _find_image(tif_dir, sid)
            if not h5ad_path.exists() or image_path is None:
                what = "h5ad" if not h5ad_path.exists() else "slide image"
                absent.append(f"{dataset_id}/{sid} (no {what})")
                continue

            if model is None:
                model, transform, device = midnight.load_model(device=cfg.perf.device)
                print(f"    Midnight loaded on {device}")

            adata = ad.read_h5ad(h5ad_path)
            loupe = tif_dir / f"{sid}_manual_loupe_alignment.json"
            coords = spot_pixels(
                adata, method=method, loupe_json_path=loupe if loupe.exists() else None
            )
            print(f"    {dataset_id}/{sid}: {len(coords):,} spots")
            features = midnight.embed_slide(
                image_path,
                coords,
                model=model,
                transform=transform,
                device=device,
                batch_size=batch_size,
            )
            _write_parquet(target, coords.index.to_numpy(), features)
            written.append(target)

    # The registry lists every cohort the project knows about and a given site holds
    # a subset, so slides without raw data are expected. They are named rather than
    # counted, so a misconfigured path is recognisable as one.
    print(
        f"\n  patches: {len(written)} slide(s) embedded, {len(cached)} already "
        f"cached, {len(absent)} without raw data"
    )
    for item in absent:
        print(f"    absent  {item}")
    if not written and not cached:
        refuse(
            "patch embeddings for any slide",
            f"none of the {len(absent)} registered slides has both an .h5ad and "
            f"a slide image under {data_root}",
            hint="check paths.data_root and paths.raw_h5ad for this site profile",
        )
    return written


def _find_image(folder: Path, sample_id: str) -> Path | None:
    for ext in (".tif", ".tiff", ".svs", ".ndpi"):
        candidate = folder / f"{sample_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def _write_parquet(path: Path, spot_ids, features: np.ndarray) -> None:
    df = pd.DataFrame(features, columns=[f"e{i}" for i in range(features.shape[1])])
    df.insert(0, "spot_id", np.asarray(spot_ids).astype(str))
    df.to_parquet(path, index=False)


def read_parquet(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(path)
    spot_ids = df["spot_id"].to_numpy().astype(str)
    features = df.drop(columns=["spot_id"]).to_numpy(dtype=np.float32)
    return spot_ids, features


def merge(cfg, *, gene_modality: str, out_name: str | None = None) -> Path:
    """Join the per-slide gene and patch parquets into one ``DatasetDict``.

    Only spots present in *both* modalities survive: a spot without a patch cannot
    train the morphology objective, and a patch without expression has nothing to
    predict. The split assignment comes from the cohort registry, so it is a
    property of the data rather than of the run.
    """
    import anndata as ad
    from datasets import Array2D, Dataset, DatasetDict, Features, Value

    registry = load_registry(cfg.paths.datasets_json)
    data_root = Path(cfg.paths.data_root)
    dirs = h5ad_dirs(cfg, registry)
    rows_by_split: dict[str, list[dict]] = {}
    gene_dim = patch_dim = None

    for dataset_id, spec in registry.datasets.items():
        subdirs = spec.get("subdirs") or {}
        h5ad_dir = dirs.get(dataset_id, data_root / dataset_id / (subdirs.get("h5ad") or ""))
        anno_col = spec.get("annotation_col")
        for sample in spec.get("samples", []):
            sid, split = sample["id"], sample.get("split", "train")
            gene_path = _slide_parquet(cfg, gene_modality, dataset_id, sid)
            patch_path = _slide_parquet(cfg, "patch", dataset_id, sid)
            if not (gene_path.exists() and patch_path.exists()):
                continue

            gene_ids, gene = read_parquet(gene_path)
            patch_ids, patch = read_parquet(patch_path)
            shared = np.intersect1d(gene_ids, patch_ids)
            if len(shared) == 0:
                print(f"    {dataset_id}/{sid}: no shared spots between modalities")
                continue
            gi = {b: i for i, b in enumerate(gene_ids)}
            pi = {b: i for i, b in enumerate(patch_ids)}

            annotation = np.full(len(shared), "UNASSIGNED", dtype=object)
            array_row = np.zeros(len(shared), dtype=np.int32)
            array_col = np.zeros(len(shared), dtype=np.int32)
            h5ad_path = h5ad_dir / f"{sid}.h5ad"
            if h5ad_path.exists():
                obs = ad.read_h5ad(h5ad_path).obs
                obs.index = obs.index.astype(str)
                present = obs.reindex(shared)
                if anno_col and anno_col in obs.columns:
                    annotation = present[anno_col].astype(str).fillna("UNASSIGNED").to_numpy()
                for name, target in (("array_row", array_row), ("array_col", array_col)):
                    alt = {"array_row": "x_array", "array_col": "y_array"}[name]
                    col = name if name in obs.columns else (alt if alt in obs.columns else None)
                    if col:
                        target[:] = present[col].fillna(0).to_numpy().astype(np.int32)

            gene_dim = gene.shape[1] if gene_dim is None else gene_dim
            patch_dim = patch.shape[1] if patch_dim is None else patch_dim
            for k, barcode in enumerate(shared):
                rows_by_split.setdefault(split, []).append(
                    {
                        "sample_id": sid,
                        "spot_id": str(barcode),
                        "array_row": int(array_row[k]),
                        "array_col": int(array_col[k]),
                        "dataset_id": dataset_id,
                        "annotation": str(annotation[k]),
                        "gene_features": gene[gi[barcode]][None, :].tolist(),
                        "patch_features": patch[pi[barcode]][None, :].tolist(),
                    }
                )
            print(f"    {dataset_id}/{sid} -> {len(shared):,} spots ({split})")

    if not rows_by_split:
        raise SystemExit("no slide had both gene and patch parquets; nothing to merge")

    features = Features(
        {
            "sample_id": Value("string"),
            "spot_id": Value("string"),
            "array_row": Value("int32"),
            "array_col": Value("int32"),
            "dataset_id": Value("string"),
            "annotation": Value("string"),
            "gene_features": Array2D(shape=(1, gene_dim), dtype="float32"),
            "patch_features": Array2D(shape=(1, patch_dim), dtype="float32"),
        }
    )
    dsd = DatasetDict(
        {split: Dataset.from_list(rows, features=features) for split, rows in rows_by_split.items()}
    )
    out = Path(cfg.paths.data_root) / (out_name or f"vgtfm_{gene_modality}_midnight")
    dsd.save_to_disk(str(out / "merged_dataset"))
    print(f"  merged dataset -> {out / 'merged_dataset'}")
    return out / "merged_dataset"


def gene_command(cfg, model: str, manifest: Path) -> str:
    """The exact command that fills in the gene side, for the stage to print.

    Two environments and a checkpoint path stand between this stage and its input,
    and every one of them is a place to mistype something. Printing the command with
    this run's paths already substituted is the difference between a copy-paste and
    a reading of ``gene_fm.py``'s docstring.
    """
    checkpoint = {
        "scgpt": " --model-dir /path/to/scGPT_human",
        "cancerfoundation": " --model-dir /path/to/CancerFoundation",
    }.get(model, "")
    return (
        f"conda run -n vgtfm-{model} python -m vgtfm.embed.gene_fm --model {model} "
        f"--manifest {manifest} --out-dir {manifest.parent}{checkpoint}"
    )


def embed_genes(cfg, model: str) -> tuple[Path, int, int]:
    """Write the manifest for one gene-side model and report what it already has.

    This stage does not *run* the model: Geneformer, scGPT and CancerFoundation each
    need a Python environment this one cannot be, so the work crosses a process
    boundary and this side prepares the input and counts the output.
    """
    manifest, slides, absent = gene_manifest(cfg, model)
    have = sum(1 for s in slides if _slide_parquet(cfg, model, s.dataset_id, s.sample_id).exists())
    print(f"    manifest: {len(slides)} slide(s) with counts -> {manifest}")
    for item in absent:
        print(f"    absent  {item} (no .h5ad at this site)")
    print(f"    {have}/{len(slides)} already embedded under {_modality_dir(cfg, model)}")
    if have < len(slides):
        print(f"\n    run, in the {model} environment:\n      {gene_command(cfg, model, manifest)}")
    return manifest, have, len(slides)


def run(cfg) -> None:
    out = cfg.sub("embed")
    substrate = cfg.data.substrate
    print("  [1/3] H&E patches (Midnight)")
    written = embed_patches(cfg)
    print(f"    {len(written)} slide(s) embedded")

    print(f"\n  [2/3] gene features ({substrate})")
    manifest, have, total = embed_genes(cfg, substrate)

    print("\n  [3/3] merge")
    try:
        path = merge(cfg, gene_modality=substrate)
    except Incomplete:
        raise  # one of ours: a defect, not the two-phase wait
    except SystemExit as e:
        # The gene-side models run in their own environments, so the first pass of
        # this stage legitimately has nothing to merge yet. Which of the two it is —
        # not run at all, or run and produced nothing — is the difference between
        # waiting and debugging, so the count goes in the message.
        pending(
            "the merged dataset",
            f"{e}; {have}/{total} gene parquets exist. Run the command above in the "
            f"{substrate} environment, then re-run `embed`",
        )
        return
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "substrate": substrate,
                "merged_dataset": str(path),
                "n_slides_patched": len(written),
                "n_slides_gene": have,
                "gene_manifest": str(manifest),
            },
            indent=2,
        )
    )
