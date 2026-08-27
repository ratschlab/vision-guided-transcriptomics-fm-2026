"""End-to-end check of the gene-side foundation models, on one GPU in minutes.

``make test`` exercises every line of :mod:`vgtfm.embed.cohort` and
:mod:`vgtfm.embed.gene_fm` without loading a checkpoint; a cluster run exercises the
checkpoints but is not something anyone repeats while changing code. This is the rung
between the two — a real checkpoint, a real forward pass, and every assertion the
cross-environment parquet contract rests on, over a handful of slides and a few
hundred spots.

    make embed-check                                    # synthetic slides
    make embed-check FROM='usz_kidney=/data/TLS_VISIUM_USZ/h5ad_preprocessed/KC*.h5ad'

A model whose conda environment or checkpoint is not on this machine is named and
skipped rather than passed over in silence: the summary lists what ran, what did not
and why, and the exit status counts only models that ran *and* failed. That
distinction is the whole point — "the check passed" and "the check never executed"
are the two outcomes a harness like this must never conflate.

Synthetic slides make the code path run; they say nothing about whether the
embeddings mean anything, because the counts are Poisson noise over gene names
borrowed from a checkpoint's own vocabulary. ``--from-h5ad`` points the same harness
at real slides, which is the only mode in which an HVG selection, a vocabulary
overlap or a spot-dropout rate is worth reading — and the only mode Geneformer runs
in at all, since it looks its genes up by Ensembl id and invented ids resolve to
nothing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vgtfm.embed import cohort  # noqa: E402
from vgtfm.embed.cohort import Slide  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

#: model -> the environment names to look for, in order. The first is what
#: ``envs/<model>.yaml`` creates; the bare name is the fallback, because a machine
#: that has run these models before this repository existed generally has one, and
#: rebuilding a working scGPT environment to satisfy a naming convention is a poor
#: trade. Which one was used is printed, and ``--env`` overrides both.
ENVS = {
    "geneformer": ("vgtfm-geneformer", "geneformer"),
    "scgpt": ("vgtfm-scgpt", "scgpt"),
    "cancerfoundation": ("vgtfm-cancerfoundation", "cancerfoundation"),
}

#: Models that need a ``--model-dir``. Geneformer is the exception: it reads the
#: HuggingFace cache, so on a machine that has pulled the weights it needs nothing.
NEEDS_MODEL_DIR = {"geneformer": False, "scgpt": True, "cancerfoundation": True}

#: Models that select genes from the whole manifest and so take the pooled route.
#: These are the ones ``--hvg-strategy`` and ``provenance.json`` apply to.
POOLED = ("scgpt", "cancerfoundation")


#: Gene support floor as a fraction of the *pooled* fixture, per strategy.
#:
#: Empirical, and deliberately not a formula. Both HVG flavours give up when too
#: many genes share a near-zero mean in the unit being fitted, but where that
#: happens depends on the tie structure of one slide's gene means, not on any ratio
#: that generalises: measured on USZ slides, ``cell_ranger`` fits one slide happily
#: at 12,604 genes and refuses another at 1,904. So these are the floors observed to
#: work on every slide of a real cohort at two fixture sizes, with the strictest of
#: the two flavours setting each:
#:
#:     pooled  strategy    seurat_v3  cell_ranger   fraction used
#:     600     global      >= 10      >= 20         0.035
#:     600     mixed       >= 30      >= 30         0.05
#:     600     per_slide   >= 100     >= 150        0.25
#:     1600    per_slide   >= 40      >= 200        0.25 (400, over the 200 needed)
#:
#: ``per_slide`` is expensive here for a reason worth knowing: a slide-sized unit
#: needs so hard a filter that the surviving gene space drops below the token budget
#: — 923 genes at pooled 600 — so the check runs but selects from less than it would
#: on a real cohort. :func:`vgtfm.embed.cohort._hvg` carries the mechanism.
FLOOR_FRACTION = {"global": 0.035, "mixed": 0.05, "per_slide": 0.25}

#: No fixture is small enough to justify going below this; two flavours both fit at
#: 20 over 600 pooled spots, and under it the gene space stops being a cohort.
MIN_FLOOR = 20


def default_min_cells(pooled_spots: int, strategies) -> int:
    """The floor the strictest requested strategy needs on a fixture this size.

    ``filter_genes`` applies its floor to the pooled cohort, but the HVG fit happens
    over whatever unit the strategy selects within — every spot for ``global``, one
    group for ``mixed``, one slide for ``per_slide`` — so the same cohort needs a
    very different floor depending on which is asked for.
    """
    wanted = [s for s in strategies if s in FLOOR_FRACTION] or ["global"]
    return max(MIN_FLOOR, max(round(FLOOR_FRACTION[s] * pooled_spots) for s in wanted))


class Skip(Exception):
    """This machine cannot run this model. Not a failure of the code under test."""


# ── the fixture cohort ───────────────────────────────────────────────


def _anndata():
    import anndata as ad

    return ad


def slim_slide(adata, gene_id_column: str):
    """Strip a slide to what the embedders read: counts, barcodes, gene identity.

    Real Visium h5ads carry the full-resolution tissue image in ``uns['spatial']``,
    which is most of the file size and none of the gene-side input. Dropping it is
    what makes a fixture slide a few MB instead of a few hundred.
    """
    ad = _anndata()
    X = adata.layers["counts"] if "counts" in adata.layers else adata.X
    keep = [c for c in (gene_id_column, "gene_name", "feature_types") if c in adata.var.columns]
    out = ad.AnnData(
        X=X.copy(),
        obs=pd.DataFrame(index=adata.obs_names.copy()),
        var=adata.var[keep].copy(),
    )
    # Counts in the layer as well as in X, so the fixture exercises the branch of
    # cohort._read_slide that prefers the layer over a possibly-normalised X.
    out.layers["counts"] = out.X.copy()
    return out


def h5ad_paths(spec: str) -> list[Path]:
    """The slides one ``--from-h5ad`` value names: a directory, a glob, or a file.

    A glob is not a convenience. Cohorts that differ biologically often share a
    directory — the USZ kidney and lung slides are ``KC*.h5ad`` and ``LC*.h5ad``
    side by side — and a harness that could only take whole directories would put
    both under one ``dataset_id``, which is the grouping ``--hvg-strategy mixed``
    and every batch key are defined over.
    """
    path = Path(spec).expanduser()
    if any(ch in spec for ch in "*?["):
        return sorted(path.parent.glob(path.name))
    if path.is_dir():
        return sorted(path.glob("*.h5ad"))
    return [path] if path.exists() else []


def real_slides(
    sources: list[tuple[str, str]],
    out: Path,
    *,
    n_slides: int,
    n_spots: int,
    gene_id_column: str = "gene_ids",
    seed: int = 0,
    verbose: bool = True,
) -> list[Slide]:
    """Subsample real slides into a fixture cohort, keeping the full gene space.

    Spots are subsampled and genes are not, on purpose: the number of spots decides
    how long the forward pass takes, and the gene space decides what an HVG
    selection has to choose between and how much of it a model's vocabulary covers.
    Cutting genes would make the parts of the run this harness exists to check
    cheaper *and* meaningless.
    """
    ad = _anndata()
    rng = np.random.default_rng(seed)
    slides: list[Slide] = []
    for dataset_id, spec in sources:
        found = h5ad_paths(spec)
        if not found:
            raise Skip(f"no .h5ad matching {spec}")
        for path in found[:n_slides]:
            adata = ad.read_h5ad(path, backed="r")
            if adata.n_obs > n_spots:
                idx = np.sort(rng.choice(adata.n_obs, n_spots, replace=False))
                adata = adata[idx]
            adata = slim_slide(adata.to_memory(), gene_id_column)
            target = out / dataset_id / f"{path.stem}.h5ad"
            target.parent.mkdir(parents=True, exist_ok=True)
            adata.write_h5ad(target)
            if verbose:
                print(f"    {dataset_id}/{path.stem}: {adata.n_obs} spots x {adata.n_vars} genes")
            slides.append(Slide(dataset_id, path.stem, str(target), gene_id_column))
    return slides


def vocabulary_symbols(model_dirs: dict[str, Path], n: int) -> list[str]:
    """Gene symbols a checkpoint will recognise, for the synthetic cohort.

    Invented gene names would be dropped by every vocabulary, leaving nothing to
    tokenise, so the synthetic slides borrow real symbols from whichever checkpoint
    is on this machine. It makes the fixture depend on a checkpoint, which is
    acceptable here: with no checkpoint at all there is nothing for it to feed.
    """
    for path in model_dirs.values():
        for candidate in (path / "vocab.json", path / "model" / "assets" / "vocab.json"):
            if candidate.exists():
                vocab = json.loads(candidate.read_text())
                genes = [g for g in vocab if not str(g).startswith("<")]
                if len(genes) >= n:
                    return genes[:n]
    raise Skip("no checkpoint vocabulary to draw synthetic gene names from")


def synthetic_slides(
    out: Path,
    symbols: list[str],
    *,
    n_slides: int = 4,
    n_spots: int = 60,
    seed: int = 0,
    verbose: bool = True,
) -> list[Slide]:
    """A cohort of Poisson counts over real gene symbols, in the 10x convention.

    Two dataset_ids so ``--hvg-strategy mixed`` and the batch keys have groups to
    work with, the same barcodes on every slide so the pooled index has collisions
    to resolve, version-suffixed Ensembl ids on a third of the genes, and one spot
    per slide with no counts at all — the case that crashes upstream's ``binning``
    and the reason :func:`vgtfm.embed.gene_fm._patch_binning` exists.
    """
    ad = _anndata()
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    ensembl = [f"ENSG{i:011d}" + (".3" if i % 3 == 0 else "") for i in range(len(symbols))]
    barcodes = [f"AAAC{i:04d}-1" for i in range(n_spots)]

    slides: list[Slide] = []
    for i in range(n_slides):
        dataset_id = f"cohort_{'ab'[i % 2]}"
        sample_id = f"S{i + 1}"
        counts = rng.poisson(rng.uniform(0.1, 4.0, size=len(symbols)), size=(n_spots, len(symbols)))
        counts[0, :] = 0
        adata = ad.AnnData(
            X=sp.csr_matrix(counts.astype(np.float32)),
            obs=pd.DataFrame(index=pd.Index(barcodes)),
            var=pd.DataFrame({"gene_ids": ensembl}, index=pd.Index(symbols)),
        )
        adata.layers["counts"] = adata.X.copy()
        target = out / dataset_id / f"{sample_id}.h5ad"
        target.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(target)
        slides.append(Slide(dataset_id, sample_id, str(target), "gene_ids"))
    if verbose:
        print(f"    {n_slides} synthetic slides, {n_spots} spots x {len(symbols)} genes")
    return slides


# ── what this machine can run ────────────────────────────────────────


def slide_spots(slides: list[Slide]) -> dict[str, int]:
    """Spots per slide, read back rather than assumed.

    ``--n-spots`` is a cap, not a count: a slide with fewer spots contributes what
    it has, and every figure derived from cohort size has to follow the files.
    """
    ad = _anndata()
    return {s.source: ad.read_h5ad(s.h5ad, backed="r").n_obs for s in slides}


def conda_environments() -> dict[str, Path]:
    """Environment name -> prefix. Empty when there is no conda on this machine.

    A broken conda and an absent one both end here, and both would otherwise read
    downstream as "no environment for this model" — the right conclusion for the
    second and a misleading one for the first, so the first says what happened.
    """
    if shutil.which("conda") is None:
        return {}
    proc = subprocess.run(
        ["conda", "env", "list", "--json"], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        print(f"  WARNING `conda env list` exited {proc.returncode}: {proc.stderr.strip()[:200]}")
        return {}
    return {Path(p).name: Path(p) for p in json.loads(proc.stdout).get("envs", [])}


def resolve_env(model: str, available: dict[str, Path], override: str | None) -> str:
    """The conda environment to run *model* in, or a Skip naming what to create."""
    if override is not None:
        if override not in available:
            raise Skip(f"--env {model}={override} is not a conda environment here")
        return override
    for candidate in ENVS[model]:
        if candidate in available:
            return candidate
    raise Skip(
        f"no conda environment {' or '.join(ENVS[model])} (conda env create -f envs/{model}.yaml)"
    )


def resolve_model_dir(model: str, given: Path | None) -> Path | None:
    """Check the checkpoint is where it was said to be, before a GPU is allocated."""
    if given is None:
        if NEEDS_MODEL_DIR[model]:
            raise Skip(f"no --model-dir {model}=... and no default")
        return None
    given = Path(given).expanduser()
    if not given.exists():
        raise Skip(f"--model-dir {model}={given} does not exist")
    return given


# ── running one model ────────────────────────────────────────────────


def gene_fm_command(
    model: str,
    env: str,
    *,
    manifest: Path,
    out_dir: Path,
    model_dir: Path | None,
    device: str,
    batch_size: int,
    strategy: str | None,
    max_length: int | None,
    min_cells: int | None,
    seed: int,
) -> list[str]:
    """The command the ``embed`` stage prints, with this harness's paths in it."""
    cmd = [
        "conda", "run", "-n", env, "--no-capture-output",
        "python", "-m", "vgtfm.embed.gene_fm",
        "--model", model,
        "--manifest", str(manifest),
        "--out-dir", str(out_dir),
        "--device", device,
        "--batch-size", str(batch_size),
        "--seed", str(seed),
    ]  # fmt: skip
    if model_dir is not None:
        cmd += ["--model-dir", str(model_dir)]
    if strategy is not None:
        cmd += ["--hvg-strategy", strategy]
    if max_length is not None:
        cmd += ["--max-length", str(max_length)]
    if min_cells is not None and model in POOLED:
        cmd += ["--min-cells", str(min_cells)]
    return cmd


def run_model(cmd: list[str], log: Path, *, verbose: bool) -> tuple[int, float]:
    """Run one model, teeing its output to *log*. Returns (exit status, seconds)."""
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log.open("w") as fh:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, check=False)
    elapsed = time.time() - started
    if proc.returncode != 0 and verbose:
        print(f"    exit {proc.returncode}; last lines of {log}:")
        for line in log.read_text().splitlines()[-12:]:
            print(f"      {line}")
    return proc.returncode, elapsed


# ── checking what it produced ────────────────────────────────────────


def read_features(path: Path) -> tuple[pd.Index, np.ndarray]:
    """Split one parquet into its barcodes and its feature matrix.

    Enforces the column contract the merge step depends on: ``spot_id`` plus
    ``e0..e{D-1}``, contiguous and in order. A gap or a rename here would surface
    downstream as a silently narrower feature matrix.
    """
    frame = pd.read_parquet(path)
    if "spot_id" not in frame.columns:
        raise AssertionError(f"{path}: no spot_id column, columns are {list(frame.columns)[:6]}")
    dims = [c for c in frame.columns if c != "spot_id"]
    expected = [f"e{i}" for i in range(len(dims))]
    if dims != expected:
        raise AssertionError(f"{path}: feature columns are not e0..e{len(dims) - 1}: {dims[:6]}")
    return pd.Index(frame["spot_id"]), frame[dims].to_numpy()


def verify(model: str, out_dir: Path, slides: list[Slide], *, strategy: str | None) -> dict:
    """Every property the rest of the pipeline assumes about these parquets.

    Geneformer may return fewer spots than it was given — the tokenizer drops a spot
    whose genes are all outside the vocabulary — so a shortfall is reported as a
    dropout rate rather than treated as a fault. What is a fault is a barcode that
    was never in the input, which is what a row-alignment bug looks like.
    """
    ad = _anndata()
    dims: set[int] = set()
    total_in = total_out = 0
    for slide in slides:
        path = out_dir / slide.dataset_id / f"{slide.sample_id}.parquet"
        if not path.exists():
            raise AssertionError(f"{slide.source}: no parquet at {path}")
        ids, features = read_features(path)
        if not np.isfinite(features).all():
            raise AssertionError(f"{slide.source}: {(~np.isfinite(features)).sum()} non-finite")
        if ids.has_duplicates:
            raise AssertionError(f"{slide.source}: duplicate spot_ids in one slide")
        source = ad.read_h5ad(slide.h5ad, backed="r")
        unknown = set(ids) - set(source.obs_names)
        if unknown:
            raise AssertionError(
                f"{slide.source}: {len(unknown)} barcode(s) the input never had, "
                f"e.g. {sorted(unknown)[:3]} — rows are misaligned"
            )
        dims.add(features.shape[1])
        total_in += source.n_obs
        total_out += len(ids)
    if len(dims) != 1:
        raise AssertionError(f"{model}: slides disagree on the feature width: {sorted(dims)}")

    facts = {"dims": dims.pop(), "spots": total_out, "dropped": total_in - total_out}
    provenance = out_dir / "provenance.json"
    if model in POOLED:
        if not provenance.exists():
            raise AssertionError(f"{model}: pooled run wrote no provenance.json")
        payload = json.loads(provenance.read_text())
        if strategy is not None and payload.get("hvg_strategy") != strategy:
            raise AssertionError(
                f"{model}: ran --hvg-strategy {strategy}, provenance says "
                f"{payload.get('hvg_strategy')!r}"
            )
        sets = payload.get("gene_sets") or {}
        if not sets or not all(sets.values()):
            raise AssertionError(f"{model}: provenance records no genes for {list(sets)}")
        facts["gene_sets"] = {k: len(v) for k, v in sets.items()}
        facts["shared_genes"] = len(set.intersection(*(set(v) for v in sets.values())))
        facts["seed"] = payload.get("seed")
    return facts


def features_of(out_dir: Path, slides: list[Slide]) -> np.ndarray:
    """Every slide's features, stacked in manifest order, for a repeat comparison."""
    parts = [
        read_features(out_dir / s.dataset_id / f"{s.sample_id}.parquet")[1]
        for s in slides
        if (out_dir / s.dataset_id / f"{s.sample_id}.parquet").exists()
    ]
    return np.concatenate(parts) if parts else np.zeros((0, 0))


# ── the harness ──────────────────────────────────────────────────────


def parse_pairs(values: list[str], *, what: str) -> list[tuple[str, str]]:
    """``name=value`` arguments. A bare value is refused rather than guessed at."""
    out = []
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--{what} wants name=value, got {item!r}")
        name, _, value = item.partition("=")
        out.append((name, value))
    return out


def build_cohort(args, model_dirs: dict[str, Path], out: Path) -> tuple[list[Slide], str]:
    """The fixture slides, and a word for where their genes came from."""
    if args.from_h5ad:
        return real_slides(
            parse_pairs(args.from_h5ad, what="from-h5ad"),
            out,
            n_slides=args.n_slides,
            n_spots=args.n_spots,
            gene_id_column=args.gene_id_column,
            seed=args.seed,
        ), "real"
    symbols = vocabulary_symbols(model_dirs, args.n_genes)
    return synthetic_slides(
        out, symbols, n_slides=args.n_slides, n_spots=args.n_spots, seed=args.seed
    ), "synthetic"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out",
        default="artifacts/_embed_check",
        help="working directory for the fixture, the parquets and the logs. Under "
        "artifacts/ so a check does not leave untracked files in the checkout",
    )
    ap.add_argument(
        "--from-h5ad",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="subsample real slides instead of generating counts; PATH is a "
        "directory, a glob or one file, and DATASET is the dataset_id they get; "
        "repeatable, and the way to give two cohorts that share a directory "
        "their own group",
    )
    ap.add_argument(
        "--model-dir",
        action="append",
        default=[],
        metavar="MODEL=DIR",
        help="checkpoint for one model, e.g. scgpt=/data/scGPT_human; repeatable",
    )
    ap.add_argument("--models", nargs="+", default=list(ENVS), choices=list(ENVS))
    ap.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="MODEL=NAME",
        help="conda environment to run one model in, when it is not named "
        "vgtfm-<model> or <model>; repeatable",
    )
    ap.add_argument("--strategies", nargs="+", default=["global"], help="pooled models only")
    ap.add_argument("--n-slides", type=int, default=4, help="slides per --from-h5ad directory")
    ap.add_argument("--n-spots", type=int, default=60)
    ap.add_argument("--n-genes", type=int, default=1500, help="synthetic cohorts only")
    ap.add_argument(
        "--gene-id-column",
        default="gene_ids",
        help=".var column of Ensembl ids in the --from-h5ad slides",
    )
    ap.add_argument("--max-length", type=int, default=None, help="scGPT/CancerFoundation context")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--batch-size-for",
        action="append",
        default=[],
        metavar="MODEL=N",
        help="forward batch size for one model. Geneformer's context is 4096 "
        "tokens against scGPT's 1200, so one figure does not fit all three on a "
        "small card; repeatable",
    )
    ap.add_argument(
        "--min-cells",
        type=int,
        default=None,
        help="gene support floor for the pooled models; default scales with the "
        "fixture, because neither the production 100 nor a permissive floor lets "
        "a few-hundred-spot cohort select genes at all (see default_min_cells)",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--repeat",
        action="store_true",
        help="run each model twice and require the two outputs to be identical",
    )
    ap.add_argument("--keep", action="store_true", help="do not delete a previous --out")
    args = ap.parse_args(argv)

    out = Path(args.out).expanduser().resolve()
    if out.exists() and not args.keep:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    envs = conda_environments()
    model_dirs: dict[str, Path] = {}
    ran: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    failed: dict[str, str] = {}

    for name, path in parse_pairs(args.model_dir, what="model-dir"):
        if name not in ENVS:
            raise SystemExit(f"--model-dir names an unknown model {name!r}; known: {list(ENVS)}")
        model_dirs[name] = Path(path).expanduser()

    batch_sizes = {m: int(n) for m, n in parse_pairs(args.batch_size_for, what="batch-size-for")}
    unknown = set(batch_sizes) - set(ENVS)
    if unknown:
        raise SystemExit(f"--batch-size-for names unknown model(s) {sorted(unknown)}")

    env_overrides = dict(parse_pairs(args.env, what="env"))
    unknown = set(env_overrides) - set(ENVS)
    if unknown:
        raise SystemExit(f"--env names unknown model(s) {sorted(unknown)}; known: {list(ENVS)}")

    print(f"== fixture cohort  ({out / 'cohort'})")
    try:
        slides, provenance = build_cohort(args, model_dirs, out / "cohort")
    except Skip as e:
        print(f"  cannot build a cohort: {e}")
        return 1
    pooled = sum(slide_spots(slides).values())
    min_cells = args.min_cells or default_min_cells(pooled, args.strategies)
    strictest = max(args.strategies, key=lambda s: FLOOR_FRACTION.get(s, 0))
    print(
        f"  {len(slides)} slide(s), {pooled:,} spots; gene support floor "
        f"{min_cells} (set by --hvg-strategy {strictest})"
    )

    if provenance == "synthetic" and "geneformer" in args.models:
        args.models = [m for m in args.models if m != "geneformer"]
        skipped["geneformer"] = "synthetic Ensembl ids resolve to nothing; use --from-h5ad"

    for model in args.models:
        print(f"\n== {model}")
        try:
            env = resolve_env(model, envs, env_overrides.get(model))
            model_dir = resolve_model_dir(model, model_dirs.get(model))
            print(f"  environment {env}")
        except Skip as e:
            print(f"  skipped: {e}")
            skipped[model] = str(e)
            continue

        manifest = cohort.write_manifest(out / model / "manifest.json", model, slides)
        strategies = args.strategies if model in POOLED else [None]
        for strategy in strategies:
            label = model if strategy is None else f"{model} [{strategy}]"
            runs = ["a", "b"] if args.repeat else ["a"]
            outputs = []
            status = None
            for run in runs:
                target = out / model / (strategy or "run") / run
                cmd = gene_fm_command(
                    model,
                    env,
                    manifest=manifest,
                    out_dir=target,
                    model_dir=model_dir,
                    device=args.device,
                    batch_size=batch_sizes.get(model, args.batch_size),
                    strategy=strategy,
                    max_length=args.max_length,
                    min_cells=min_cells,
                    seed=args.seed,
                )
                code, elapsed = run_model(
                    cmd, out / "logs" / f"{model}-{strategy}-{run}.log", verbose=True
                )
                if code != 0:
                    status = f"exit {code}"
                    break
                outputs.append(target)
                print(f"  {label} run {run}: {elapsed:.0f}s")
            if status is not None:
                failed[label] = status
                continue
            try:
                facts = verify(model, outputs[0], slides, strategy=strategy)
                if args.repeat:
                    first, second = (features_of(d, slides) for d in outputs)
                    if first.shape != second.shape:
                        # Not a drifting coordinate: two runs kept different spots,
                        # so the seed is not reaching whatever decides that either.
                        raise AssertionError(
                            f"two runs at seed {args.seed} returned {first.shape} and "
                            f"{second.shape} — the runs disagree on which spots survived"
                        )
                    if not np.array_equal(first, second):
                        drift = float(np.abs(first - second).max())
                        differing = int((first != second).any(axis=1).sum())
                        raise AssertionError(
                            f"two runs at seed {args.seed} differ by up to {drift:.3g} "
                            f"over {differing}/{len(first)} spots; the tokenisation "
                            f"seed is not reaching every source of randomness"
                        )
                    facts["reproducible"] = True
            except AssertionError as e:
                print(f"  FAIL {e}")
                failed[label] = str(e)
                continue
            print(f"  ok  {facts}")
            ran[label] = facts

    print(f"\n== {len(ran)} ran, {len(failed)} failed, {len(skipped)} skipped")
    for label, facts in ran.items():
        detail = ", ".join(f"{k} {v}" for k, v in facts.items())
        print(f"  ok      {label}: {detail}")
    for label, why in failed.items():
        print(f"  FAIL    {label}: {why}")
    for model, why in skipped.items():
        print(f"  skipped {model}: {why}")
    if not ran and not failed:
        print("\n  nothing ran: this machine has no gene-FM environment or checkpoint.")
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
