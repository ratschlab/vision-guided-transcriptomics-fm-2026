"""Gene-expression foundation model embeddings.

Geneformer, scGPT and CancerFoundation have mutually incompatible dependency
pinnings — different torch, transformers and scanpy versions, and for scGPT a
flash-attention build tied to its torch pin — so they cannot share a Python
environment. Rather than hide that behind a plugin system, each runs as a separate
process in its own conda environment (``envs/*.yaml``) and communicates through the
filesystem.

The contract is narrow. Each embedder writes one parquet per slide::

    <artifact_root>/_embed/<model>/<dataset_id>/<sample_id>.parquet
    columns: spot_id, e0 .. e{D-1}

``vgtfm.embed.build.merge`` joins those with the morphology parquets on ``spot_id``.
Nothing downstream knows or cares which environment produced them.

**Two routes to the same parquets.** Geneformer ranks each spot's genes against a
fixed vocabulary, so a slide embedded alone gives the answer it gives in company:

    conda run -n vgtfm-geneformer python -m vgtfm.embed.gene_fm \\
        --model geneformer --h5ad .../MACEGEJ-1-1.h5ad \\
        --out artifacts/_embed/geneformer/10x_TuPro/MACEGEJ-1-1.parquet

scGPT, CancerFoundation and scVI depend on the unit they are run over: the first two
select highly variable genes before tokenising and the third is *fitted* on the
counts. Each is run once over a manifest of every slide, and the rows are split back
out afterwards. Where the two HVG models choose their genes is ``--hvg-strategy``
(``global``, the cached artefacts' route, ``mixed`` or ``per_slide``), recorded in
``provenance.json`` beside the parquets; :mod:`vgtfm.embed.cohort` has what each
costs, because none of the three is the neutral option:

    python run.py embed                       # writes _embed/<model>/manifest.json
    conda run -n vgtfm-scgpt python -m vgtfm.embed.gene_fm \\
        --model scgpt --manifest artifacts/_embed/scgpt/manifest.json \\
        --out-dir artifacts/_embed/scgpt --model-dir /path/to/scGPT_human
    python run.py embed                       # merges, now that the parquets exist

The ``embed`` stage prints that middle command with this run's paths substituted, so
it does not have to be reconstructed from here.

Each model is used with the preprocessing its authors specify, which differs between
them and is the reason the substrates are not comparable as *inputs* — only their
downstream behaviour is. Where this repository departs from an upstream default, or
repairs an upstream bug, the reason is in the function that does it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ("geneformer", "scgpt", "cancerfoundation", "scvi")

#: Models whose output for one slide depends on which other slides were present —
#: because they choose genes from whatever they are given, or are fitted on it. Run
#: per slide they still return the same 512 or 256 dimensions; what changes is which
#: genes each slide's embedding summarises. ``--hvg-strategy`` makes that a choice.
POOLED = ("scgpt", "cancerfoundation", "scvi")

#: What each pooled model treats as a batch, absent ``--batch-key``. These are the
#: values ``merged/precompute_*.py`` used, i.e. what the cached artefacts encode —
#: cohort for the two HVG models, slide for scVI, whose covariate removes the slide.
#: The key is not a neutral knob: it decides whose genes get a vote, and the two HVG
#: flavours count those votes by opposite rules. :mod:`vgtfm.embed.cohort` has both.
DEFAULT_BATCH_KEY = {
    "scgpt": "dataset_id",
    "cancerfoundation": "dataset_id",
    "scvi": "sample_source",
}


def cohort_strategies() -> tuple[str, ...]:
    """The gene-selection strategies, named where argparse can reach them."""
    from .cohort import HVG_STRATEGIES

    return HVG_STRATEGIES


def write_parquet(path: str | Path, spot_ids, features: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    features = np.asarray(features, dtype=np.float32)
    df = pd.DataFrame(features, columns=[f"e{i}" for i in range(features.shape[1])])
    df.insert(0, "spot_id", np.asarray(spot_ids).astype(str))
    df.to_parquet(path, index=False)
    print(f"wrote {path} ({df.shape[0]:,} spots, {features.shape[1]} dims)")


# ── upstream repairs, shared by scGPT and CancerFoundation ───────────


def _patch_binning(module) -> None:
    """Repair the empty-row handling of a scGPT-lineage ``binning()``.

    A Visium spot on the tissue border can have no counts at all among the genes a
    model selected. Both scGPT and CancerFoundation carry the same ``binning()`` and
    the same three defects on exactly that input, and each kills the run in a
    dataloader worker rather than raising anything a caller could act on:

    * The row is converted to numpy on entry, and the zero branch then returns
      ``torch.zeros_like(row, ...)`` on that numpy array — a ``TypeError``.
    * CancerFoundation's copy calls a module-level ``logger`` it never defines, so
      that branch raises ``NameError`` before it gets there.
    * scGPT's dataset passes only a spot's *non-zero* genes, so an empty spot arrives
      as a zero-length vector and never reaches the zero branch at all: ``row.max()``
      raises ``ValueError: zero-size array to reduction operation maximum``.

    An empty or all-zero row bins to itself, so this repairs the branch rather than
    dropping the spot — the encoder then sees ``<cls>`` and padding, which is the
    honest input for a spot with nothing in it.
    """
    if getattr(module, "_vgtfm_binning_patched", False):
        return
    import logging

    import torch

    if not hasattr(module, "logger"):
        module.logger = logging.getLogger(module.__name__)
    original = module.binning

    def binning(row, n_bins):
        is_tensor = isinstance(row, torch.Tensor)
        values = row.cpu().numpy() if is_tensor else row
        if values.size == 0 or values.max() == 0:
            zeros = np.zeros_like(values)
            return torch.from_numpy(zeros).to(row.dtype) if is_tensor else zeros
        return original(row, n_bins)

    module.binning = binning
    module._vgtfm_binning_patched = True


def _patch_sparse_matrix_A() -> None:
    """Restore ``scipy.sparse.spmatrix.A`` for CancerFoundation's ``embed()``.

    ``embed()`` densifies with ``count_matrix.A``, an alias scipy removed in 1.14.
    Pre-densifying instead is not an option — ``embed()`` reads ``adata.X`` only
    *after* its own vocabulary and HVG reduction, so the matrix we would have to hand
    it dense is the full cohort by every gene. The alternative to this shim is
    re-implementing the authors' gene selection here, where it could drift from
    theirs without anything noticing; a three-line alias cannot.
    """
    import scipy.sparse as sp

    if not hasattr(sp.csr_matrix, "A"):
        sp.spmatrix.A = property(lambda self: self.toarray())


def _seed_tokenisation(seed: int) -> None:
    """Make a scGPT-lineage forward pass reproducible.

    Tokenisation is stochastic in both models, in two places that are easy to miss
    because neither is documented as random:

    * ``_digitize`` spreads values across tied quantile bins with an unseeded
      ``np.random.rand``. Count data ties constantly — most genes in a spot are at 1
      or 2 counts — so this fires for essentially every spot.
    * ``DataCollator(sampling=True)`` draws which genes to keep, with
      ``torch.randperm``, for any spot longer than the context.

    Both run inside dataloader workers, which PyTorch seeds — torch *and* numpy —
    from a base seed drawn off the default generator, so seeding torch here covers
    them; numpy is seeded as well for the ``num_workers=0`` path, where the workers'
    seeding never happens and the main process's global RNG is what gets used.

    Without this the same cohort embeds differently on every run — by about 1e-2 per
    coordinate, which is small, invisible, and enough that no result downstream can
    be reproduced exactly.
    """
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)


def _batch_key_or_none(adata, batch_key: str | None) -> str | None:
    """Drop a batch key that names one batch — scanpy's batched HVG path needs two."""
    if not batch_key or batch_key not in adata.obs:
        return None
    return batch_key if adata.obs[batch_key].nunique() > 1 else None


def _symbols_as_var_names(adata):
    """Ensure genes are indexed by HGNC symbol, which is what both vocabularies use.

    The pooled route has already done this — :func:`vgtfm.embed.cohort.prepare_cohort`
    restores symbols on the full gene set, before any filtering, because where the
    duplicate-symbol tie is broken changes which genes match a vocabulary. This is the
    single-slide route's safeguard, and a no-op when the names are already right.
    """
    if "gene_symbol" not in adata.var.columns:
        return adata  # a raw 10x .h5ad: var_names are already symbols
    if adata.var_names.equals(pd.Index(adata.var["gene_symbol"].astype(str))):
        return adata
    adata.var_names = pd.Index(adata.var["gene_symbol"].astype(str))
    adata.var_names_make_unique()
    return adata


# ── Geneformer ───────────────────────────────────────────────────────


def prepare_for_geneformer(adata, *, gene_id_column: str = "gene_ids"):
    """Put an AnnData into the shape Geneformer's tokenizer expects.

    Geneformer encodes each spot as a *rank ordering* of its expressed genes, so it
    needs raw counts, Ensembl ids (version suffixes stripped) in
    ``var['ensembl_id']``, and a per-spot ``n_counts``. Barcodes are copied into an
    obs column because tokenisation drops spots with no expressed genes and would
    otherwise silently break row alignment.
    """
    import scipy.sparse as sp

    adata = adata.copy()
    if adata.raw is not None:
        raw = adata.raw.to_adata()
        raw.obs = adata.obs.copy()
        adata = raw
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)  # the tokenizer calls .tocsc()

    if "n_counts" not in adata.obs.columns:
        counts = adata.X.sum(axis=1)
        adata.obs["n_counts"] = counts.A1 if hasattr(counts, "A1") else np.ravel(counts)

    if gene_id_column not in adata.var.columns:
        raise ValueError(
            f"Geneformer needs Ensembl ids; '{gene_id_column}' is not in "
            f"adata.var (have {list(adata.var.columns)})"
        )
    stripped = [str(g).split(".")[0] for g in adata.var[gene_id_column]]
    adata.var["ensembl_id"] = stripped
    adata.var_names = pd.Index(stripped)
    adata.var_names_make_unique()
    adata.obs["spot_barcode"] = adata.obs_names.astype(str)
    return adata


def embed_geneformer(
    adata,
    *,
    model_dir: str | None = None,
    batch_size: int = 64,
    model_version: str = "V2",
    emb_layer: int = -1,
    nproc: int = 4,
    gene_id_column: str = "gene_ids",
    **_,
):
    """Geneformer spot embeddings via the official tokenizer and extractor.

    Runs in the ``geneformer`` environment. Barcodes are carried through
    tokenisation as a custom attribute and returned alongside the embeddings, so a
    spot dropped by the tokenizer is dropped from the output too rather than
    shifting every subsequent row.
    """
    import tempfile
    from pathlib import Path as _Path

    from datasets import load_from_disk
    from geneformer import EmbExtractor, TranscriptomeTokenizer

    adata = prepare_for_geneformer(adata, gene_id_column=gene_id_column)
    model_dir = model_dir or _resolve_hf_cache("ctheodoris/Geneformer")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = _Path(tmp)
        input_dir, token_dir, emb_dir = tmp / "in", tmp / "tok", tmp / "emb"
        for d in (input_dir, token_dir, emb_dir):
            d.mkdir()
        adata.write_h5ad(input_dir / "sample.h5ad")

        tokenizer = TranscriptomeTokenizer(
            custom_attr_name_dict={"spot_barcode": "spot_barcode"},
            nproc=nproc,
            model_input_size=_model_input_size(model_dir, model_version),
            model_version=model_version,
        )
        tokenizer.tokenize_data(str(input_dir), str(token_dir), "sample", file_format="h5ad")

        tokenized_path = next(iter(sorted(token_dir.glob("**/dataset_info.json"))), None)
        if tokenized_path is None:
            raise RuntimeError(f"tokenizer produced no dataset under {token_dir}")
        tokenized_path = tokenized_path.parent
        tokenized = load_from_disk(str(tokenized_path))
        if "spot_barcode" not in tokenized.column_names:
            raise RuntimeError(
                "tokenized dataset lost the spot_barcode column; row alignment cannot be guaranteed"
            )
        barcodes = np.asarray(tokenized["spot_barcode"], dtype=str)

        extractor = EmbExtractor(
            model_type="Pretrained",
            num_classes=0,
            emb_layer=emb_layer,
            emb_label=["spot_barcode"],
            forward_batch_size=batch_size,
            nproc=nproc,
            model_version=model_version,
            max_ncells=None,
        )
        embs = extractor.extract_embs(str(model_dir), str(tokenized_path), str(emb_dir), "emb")

    ids, features = _parse_geneformer_output(embs, barcodes)
    if len(ids) != len(features):
        raise RuntimeError(f"{len(ids)} barcodes vs {len(features)} embeddings")
    return ids, features


def _parse_geneformer_output(embs, barcodes):
    """Split the extractor's frame into (barcodes, features)."""
    if isinstance(embs, pd.DataFrame):
        label = "spot_barcode"
        if label in embs.columns:
            ids = embs[label].to_numpy().astype(str)
            features = embs.drop(columns=[label]).to_numpy(dtype=np.float32)
        else:
            ids = barcodes[: len(embs)]
            features = embs.to_numpy(dtype=np.float32)
        return ids, features
    return barcodes[: len(embs)], np.asarray(embs, dtype=np.float32)


def _model_input_size(model_dir, model_version: str) -> int:
    try:
        from transformers import AutoConfig

        return getattr(AutoConfig.from_pretrained(str(model_dir)), "max_position_embeddings", 4096)
    except Exception as e:
        # This is the tokenisation length. Reading it wrong does not raise — it
        # produces embeddings of the wrong context, so the fallback says so.
        fallback = {"V1": 2048, "V2": 4096}.get(model_version, 4096)
        print(
            f"    WARNING could not read max_position_embeddings from {model_dir} "
            f"({type(e).__name__}: {e}); using the {model_version} default "
            f"{fallback}, which sets the tokenisation length"
        )
        return fallback


def _resolve_hf_cache(repo_id: str) -> str:
    """Locate a model in the local HuggingFace cache (compute nodes are offline)."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_id, local_files_only=True)


# ── scGPT ────────────────────────────────────────────────────────────

SCGPT_CHECKPOINT_FILES = ("vocab.json", "args.json", "best_model.pt")


def embed_scgpt(
    adata,
    *,
    model_dir: str | None = None,
    batch_size: int = 64,
    n_hvg: int = 2000,
    max_length: int = 1200,
    batch_key: str | None = "dataset_id",
    device: str = "cuda",
    seed: int = 42,
    hvg_strategy: str = "global",
    group_key: str = "dataset_id",
    shared_ratio: float = 0.70,
    **_,
):
    """scGPT cell embeddings, 512-d, from the pooled cohort.

    Follows the authors' zero-shot embedding tutorial: highly variable genes are
    selected with ``flavor="seurat_v3"`` — the one flavour that wants raw counts, and
    the one scGPT's own ``Preprocessor`` defaults to — and ``scgpt.tasks.embed_data``
    is handed the result, keyed on gene symbol. It takes the ``<cls>`` token and
    L2-normalises it.

    Two choices are worth naming:

    * **HVG selection is batch-aware and cohort-wide.** The tutorial embeds a single
      dataset and passes no ``batch_key``; here ten cohorts are pooled, and a
      selection made without one ranks genes that separate cohorts above genes that
      vary within them — handing the model the batch effect as its feature set.
    * **No cell filtering**, which the tutorial also does none of, but which
      ``Preprocessor`` offers and an earlier in-house version of this code did
      (``filter_cells(min_genes=200)``). A Visium spot is a fixed piece of tissue,
      not a cell that either worked or did not, and a depth floor drops the tissue
      border preferentially — the spots the morphology side has the most to say about.

    ``adata.X`` is normalised and log1p'd before the model sees it, which the
    tutorial does not bother with, because it cannot matter: ``DataCollator`` bins
    each row against its own non-zero quantiles, and quantiles commute with a
    monotone per-row transform. It is done because the authors' ``Preprocessor``
    does, and because the value in ``X`` would otherwise be read as a count by anyone
    inspecting the object.

    Tokenisation is stochastic — see :func:`_seed_tokenisation` for what is random
    and how it is pinned.
    """
    import scanpy as sc
    from scgpt import data_collator
    from scgpt.tasks import embed_data

    from . import cohort

    model_dir = _check_checkpoint(model_dir, SCGPT_CHECKPOINT_FILES, "scGPT", "--model-dir")
    _patch_binning(data_collator)

    adata = _symbols_as_var_names(adata.copy())
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    selection = cohort.select_genes(
        adata,
        strategy=hvg_strategy,
        n_genes=n_hvg,
        flavor="seurat_v3",  # the one flavour that wants counts, hence the layer
        layer="counts",
        batch_key=_batch_key_or_none(adata, batch_key),
        group_key=group_key,
        shared_ratio=shared_ratio,
    )
    del adata.layers["counts"]

    fast = _flash_attention_available(device)
    if not fast:
        print("  scGPT: flash-attn unavailable, using the standard attention path")
    _seed_tokenisation(seed)

    masks = cohort.group_masks(adata, hvg_strategy, group_key)
    features = None
    for group, genes in selection.items():
        rows = masks[group]
        print(f"  scGPT [{group}]: {int(rows.sum()):,} spots x {len(genes):,} genes")
        out = embed_data(
            adata[rows][:, genes].copy(),
            model_dir=str(model_dir),
            gene_col="index",
            max_length=max_length,
            batch_size=batch_size,
            device=device,
            use_fast_transformer=fast,
            return_new_adata=False,
        )
        block = np.asarray(out.obsm["X_scGPT"], dtype=np.float32)
        if features is None:
            features = np.zeros((adata.n_obs, block.shape[1]), dtype=np.float32)
        features[rows] = block
    return adata.obs_names.to_numpy().astype(str), features, selection


def _flash_attention_available(device: str) -> bool:
    """Whether scGPT's fast attention path can run: it needs flash-attn and a GPU.

    scGPT falls back on its own with a warning, but only for the missing-package
    case; asking for it on CPU raises inside the forward pass instead.
    """
    import torch

    if not (device == "cuda" or str(device).startswith("cuda")) or not torch.cuda.is_available():
        return False
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True


# ── CancerFoundation ─────────────────────────────────────────────────

CANCERFOUNDATION_ASSETS = ("vocab.json", "args.json", "model.pth")


def _cancerfoundation_paths(model_dir: str | None) -> tuple[Path, Path]:
    """Locate the repository root and its assets, from either one.

    CancerFoundation ships as a repository with no package metadata — there is
    nothing to ``pip install`` — so the checkout has to go on ``sys.path`` for
    ``model.embedding`` to import. ``--model-dir`` may name either the checkout or
    the ``model/assets`` directory inside it, because both are the obvious answer to
    "where is the model".
    """
    if not model_dir:
        raise ValueError(
            "CancerFoundation needs --model-dir: a checkout of "
            "https://github.com/BoevaLab/CancerFoundation with the pretrained "
            "weights unpacked into model/assets/"
        )
    given = Path(model_dir).expanduser().resolve()
    candidates = [(given, given / "model" / "assets"), (given.parent.parent, given)]
    for repo, assets in candidates:
        if all((assets / f).exists() for f in CANCERFOUNDATION_ASSETS):
            if not (repo / "model" / "embedding.py").exists():
                raise FileNotFoundError(
                    f"found the assets under {assets} but no model/embedding.py under "
                    f"{repo}; --model-dir must point inside a CancerFoundation checkout"
                )
            return repo, assets
    raise FileNotFoundError(
        f"no CancerFoundation assets under {given}: expected "
        f"{', '.join(CANCERFOUNDATION_ASSETS)} in {given / 'model' / 'assets'}. "
        "The weights are a separate download from the repository; see its README."
    )


def embed_cancerfoundation(
    adata,
    *,
    model_dir: str | None = None,
    batch_size: int = 64,
    max_length: int = 1200,
    batch_key: str | None = "dataset_id",
    device: str = "cuda",
    seed: int = 42,
    hvg_strategy: str = "global",
    group_key: str = "dataset_id",
    shared_ratio: float = 0.70,
    **_,
):
    """CancerFoundation (CancerGPT) cell embeddings, 256-d, from the pooled cohort.

    Almost everything happens inside the authors' ``model.embedding.embed``: it
    filters to its vocabulary, selects ``max_length - 1`` highly variable genes with
    ``flavor="cell_ranger"``, bins, runs the encoder, takes the ``<cls>`` token and
    L2-normalises. This function's job is to hand it the cohort in the shape its own
    tutorial and zero-shot benchmark script hand it theirs — raw counts, gene
    symbols as ``var_names``, and a batch key for the HVG step — and to repair the
    two places where that code no longer runs on a current scipy (see
    :func:`_patch_binning` and :func:`_patch_sparse_matrix_A`).

    One upstream choice is worth naming, because it is not what scanpy documents:
    ``flavor="cell_ranger"`` expects logarithmised data, and ``embed()`` applies it
    to raw counts. That changes which genes are selected — not what the model is fed,
    since the binning downstream is rank-based. Both the CancerFoundation tutorial
    and its zero-shot integration script pass raw counts, so raw counts are what this
    passes: the embeddings then match the ones the model's authors would produce, and
    a substrate that quietly used a different gene set from everyone else's
    CancerFoundation would be the worse failure.

    Unlike scGPT no genes are *sampled* — the selection leaves exactly
    ``max_length - 1`` of them and the ``<cls>`` token makes ``max_length``, so the
    collator never has to choose. The binning is still stochastic, though, which is
    why this is seeded too; :func:`_seed_tokenisation` has the detail.
    """
    repo, assets = _cancerfoundation_paths(model_dir)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    import model.data_collator as cf_collator  # the checkout's top-level `model` package
    from model.embedding import embed

    _patch_binning(cf_collator)
    _patch_sparse_matrix_A()
    _seed_tokenisation(seed)

    from . import cohort

    adata = _symbols_as_var_names(adata.copy())
    print(f"  CancerFoundation: {adata.n_obs:,} spots x {adata.n_vars:,} genes in, assets {assets}")

    if hvg_strategy == "global":
        # embed() runs its own cell_ranger selection over max_length-1 genes; letting
        # it is what the upstream `merged/precompute_cancerfoundation.py` did, so the
        # cached artefacts encode exactly this call.
        out = embed(
            adata_or_file=adata,
            model_dir=str(assets),
            batch_key=_batch_key_or_none(adata, batch_key),
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )
        features = np.asarray(out.obsm["CancerGPT"], dtype=np.float32)
        return out.obs_names.to_numpy().astype(str), features, {"all": list(out.var_names)}

    # Per-group gene sets. Selection happens here rather than inside embed(), which
    # only knows how to choose one set; embed() then re-selects over exactly the genes
    # it is handed, which keeps all of them and leaves the authors' encoder path the
    # single source of truth for everything after the choice.
    selection = cohort.select_genes(
        adata,
        strategy=hvg_strategy,
        n_genes=max_length - 1,  # embed() reserves one slot for <cls>
        flavor="cell_ranger",  # what embed() itself uses, on raw counts
        group_key=group_key,
        shared_ratio=shared_ratio,
    )
    masks = cohort.group_masks(adata, hvg_strategy, group_key)
    features = None
    for group, genes in selection.items():
        rows = masks[group]
        print(f"  CancerFoundation [{group}]: {int(rows.sum()):,} spots x {len(genes):,} genes")
        out = embed(
            adata_or_file=adata[rows][:, genes].copy(),
            model_dir=str(assets),
            batch_key=None,  # one group is one batch
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )
        block = np.asarray(out.obsm["CancerGPT"], dtype=np.float32)
        if features is None:
            features = np.zeros((adata.n_obs, block.shape[1]), dtype=np.float32)
        features[rows] = block
    return adata.obs_names.to_numpy().astype(str), features, selection


# ── scVI ─────────────────────────────────────────────────────────────


def embed_scvi(
    adata,
    *,
    n_latent: int = 50,
    batch_key: str = "sample_source",
    max_epochs: int = 200,
    seed: int = 42,
    device: str = "cuda",
    **_,
):
    """scVI latent representation, trained on raw counts with the slide as batch.

    Unlike the frozen transformer models this one is *fitted on this cohort*, so it
    is not zero-shot, and it is the only gene-side representation here that is
    batch-aware by construction. It is therefore also pooled: a per-slide fit would
    give every slide its own latent space and a constant batch covariate, which is
    both incomparable across slides and not what scVI is for.
    """
    import scvi

    scvi.settings.seed = seed
    adata = adata.copy()
    if batch_key not in adata.obs:
        adata.obs[batch_key] = "all"
    scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key)
    model = scvi.model.SCVI(adata, n_latent=n_latent)
    model.train(max_epochs=max_epochs, accelerator="auto" if device == "cuda" else device)
    features = model.get_latent_representation().astype(np.float32)
    return adata.obs_names.to_numpy().astype(str), features


EMBEDDERS = {
    "geneformer": embed_geneformer,
    "scgpt": embed_scgpt,
    "cancerfoundation": embed_cancerfoundation,
    "scvi": embed_scvi,
}


def _unpack(result):
    """``(ids, features, gene_sets)`` from an embedder that may not report genes.

    Only the two gene-selecting models have a gene set to report; Geneformer ranks
    against a fixed vocabulary and scVI is fitted on every gene it is given, so for
    those the answer is legitimately "no selection was made", not an empty one.
    """
    if len(result) == 3:
        return result
    ids, features = result
    return ids, features, {}


def _check_checkpoint(model_dir, files, name: str, flag: str) -> Path:
    """Fail on a missing checkpoint here, not five minutes into a GPU allocation."""
    if not model_dir:
        raise ValueError(f"{name} needs {flag}: the directory holding {', '.join(files)}")
    path = Path(model_dir).expanduser().resolve()
    missing = [f for f in files if not (path / f).exists()]
    if missing:
        raise FileNotFoundError(f"{name} checkpoint at {path} is missing {', '.join(missing)}")
    return path


# ── the two routes ───────────────────────────────────────────────────


def run_pooled(
    model: str,
    manifest: str | Path,
    out_dir: str | Path,
    *,
    min_cells: int = 100,
    batch_key: str | None = None,
    **kwargs,
) -> list[Path]:
    """Embed every slide in *manifest* in one pass and split the rows back out."""
    from . import cohort

    batch_key = batch_key or DEFAULT_BATCH_KEY[model]
    # The zero-gene filter exists to make a *batched HVG selection* legal, so it has
    # to group the way that selection will. scVI does no gene selection, so it keeps
    # the cohort-level grouping and the same gene universe as its neighbours.
    qc_key = "dataset_id" if model == "scvi" else batch_key

    slides = cohort.read_manifest(manifest)
    print(f"  {model}: pooling {len(slides)} slide(s) from {manifest}")
    adata = cohort.prepare_cohort(slides, min_cells=min_cells, batch_key=qc_key)
    ids, features, gene_sets = _unpack(EMBEDDERS[model](adata, batch_key=batch_key, **kwargs))
    # .loc raises on a label the pooled frame never had, which is the only way a
    # model can hand back rows that belong to no slide.
    obs = adata.obs.loc[list(ids)]
    written = cohort.split_to_parquets(obs, features, out_dir)

    cohort.write_provenance(
        out_dir,
        {
            "model": model,
            "manifest": str(manifest),
            "n_slides": len(slides),
            "n_spots": int(len(ids)),
            "dims": int(features.shape[1]),
            "batch_key": batch_key,
            "qc_batch_key": qc_key,
            "min_cells": min_cells,
            "hvg_strategy": kwargs.get("hvg_strategy", "global"),
            "group_key": kwargs.get("group_key"),
            "shared_ratio": kwargs.get("shared_ratio"),
            "seed": kwargs.get("seed"),
            "n_hvg": kwargs.get("n_hvg"),
            "max_length": kwargs.get("max_length"),
            "gene_sets": {g: list(genes) for g, genes in gene_sets.items()},
        },
    )

    absent = cohort.report_missing(slides, written)
    print(f"\n  {model}: {len(written)} parquet(s) under {out_dir}")
    for name in absent:
        print(f"    WARNING no spots survived for {name}")
    return written


def run_per_slide(
    model: str,
    manifest: str | Path,
    out_dir: str | Path,
    *,
    overwrite: bool = False,
    **kwargs,
) -> list[Path]:
    """Embed each slide in *manifest* on its own, one parquet at a time.

    Cached slides are skipped rather than recomputed, so a job that dies at slide 40
    of 50 resumes where it stopped — which is worth having when each slide is a
    GPU-minute and the allocation is finite.
    """
    import anndata as ad

    from . import cohort

    slides = cohort.read_manifest(manifest)
    written: list[Path] = []
    for i, slide in enumerate(slides, 1):
        target = Path(out_dir) / slide.dataset_id / f"{slide.sample_id}.parquet"
        if target.exists() and not overwrite:
            print(f"  [{i}/{len(slides)}] {slide.source}: cached")
            written.append(target)
            continue
        print(f"  [{i}/{len(slides)}] {slide.source}")
        ids, features, _ = _unpack(
            EMBEDDERS[model](
                ad.read_h5ad(slide.h5ad), gene_id_column=slide.gene_id_column, **kwargs
            )
        )
        write_parquet(target, ids, features)
        written.append(target)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--manifest", help="slides to embed, from the `embed` stage")
    ap.add_argument(
        "--out-dir", help="destination root: <out-dir>/<dataset_id>/<sample_id>.parquet"
    )
    ap.add_argument("--h5ad", help="one slide of raw counts (instead of --manifest)")
    ap.add_argument("--out", help="destination parquet for --h5ad")
    ap.add_argument("--model-dir", default=None, help="checkpoint directory or model checkout")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-hvg", type=int, default=2000, help="scGPT: genes kept before tokenising")
    ap.add_argument("--max-length", type=int, default=1200, help="scGPT/CancerFoundation context")
    ap.add_argument(
        "--batch-key",
        default=None,
        help=f"obs column the pooled HVG selection and scVI treat as the batch "
        f"(default per model: {DEFAULT_BATCH_KEY})",
    )
    ap.add_argument("--min-cells", type=int, default=100, help="pooled: gene support floor")
    ap.add_argument(
        "--hvg-strategy",
        default="global",
        choices=cohort_strategies(),
        help="where gene selection is made: one set for the cohort (global, what the "
        "cached artefacts used), shared+specific per group (mixed), or one set per "
        "slide (per_slide). See vgtfm/embed/cohort.py for what each costs",
    )
    ap.add_argument(
        "--group-key",
        default="dataset_id",
        help="obs column --hvg-strategy mixed selects genes within",
    )
    ap.add_argument(
        "--shared-ratio",
        type=float,
        default=0.70,
        help="mixed: fraction of the gene budget spent on genes shared across groups",
    )
    ap.add_argument(
        "--gene-id-column", default="gene_ids", help="--h5ad: .var column of Ensembl ids"
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--allow-per-slide",
        action="store_true",
        help="run a gene-selecting model on one slide; its gene set is then that "
        "slide's own, which is a different basis from every other slide's",
    )
    args = ap.parse_args(argv)

    shared = {
        "model_dir": args.model_dir,
        "batch_size": args.batch_size,
        "device": args.device,
        "seed": args.seed,
    }
    hvg = {
        "n_hvg": args.n_hvg,
        "max_length": args.max_length,
        "hvg_strategy": args.hvg_strategy,
        "group_key": args.group_key,
        "shared_ratio": args.shared_ratio,
    }
    if args.model == "scvi" and args.hvg_strategy != "global":
        ap.error("scvi selects no genes; --hvg-strategy applies to scgpt and cancerfoundation")

    if args.manifest:
        if not args.out_dir:
            ap.error("--manifest needs --out-dir")
        if args.model in POOLED:
            run_pooled(
                args.model,
                args.manifest,
                args.out_dir,
                min_cells=args.min_cells,
                batch_key=args.batch_key,
                **shared,
                **hvg,
            )
        else:
            run_per_slide(
                args.model, args.manifest, args.out_dir, overwrite=args.overwrite, **shared
            )
        return 0

    if not (args.h5ad and args.out):
        ap.error("give either --manifest and --out-dir, or --h5ad and --out")
    if args.model in POOLED and not args.allow_per_slide:
        ap.error(
            f"{args.model} chooses its genes from whatever it is given, so a slide "
            f"embedded alone gets that slide's own basis and not the cohort's. The "
            f"cached artefacts were pooled (see vgtfm/embed/cohort.py for what each "
            f"unit costs). Use --manifest, or --allow-per-slide to mean it."
        )

    import anndata as ad

    kwargs = {**shared, "gene_id_column": args.gene_id_column}
    if args.model in POOLED:
        kwargs.update(hvg)
        kwargs["batch_key"] = args.batch_key  # None: one slide is one batch
    spot_ids, features, _ = _unpack(EMBEDDERS[args.model](ad.read_h5ad(args.h5ad), **kwargs))
    write_parquet(args.out, spot_ids, features)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
