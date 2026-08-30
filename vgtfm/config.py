"""Typed configuration — one dataclass tree loaded from a single YAML file.

Every knob that changes a reported number lives here, and the resolved tree is
dumped to ``<out_dir>/config.resolved.json`` on every run so any figure can be
traced back to the settings that produced it.

Override any field from the CLI with dotted ``--set key.sub=value``
(parsed in ``run.py``), e.g. ``--set models.pca_components=50 seeds=42,43,44``.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

# Three levels above this file. `data/` sits beside the checkout rather than inside
# it, because it is far too large to live under version control.
PKG_ROOT = Path(__file__).resolve().parent  # <checkout>/vgtfm
PROJECT_ROOT = PKG_ROOT.parent  # <checkout>
REPO_ROOT = PROJECT_ROOT.parent  # the directory holding it

#: Site profiles (paths that differ per machine). See environments.yaml.
ENVIRONMENTS_FILE = PROJECT_ROOT / "environments.yaml"

#: Profile keys that name a filesystem location and may be given relative to the
#: repository root. Everything else in a profile is consumed by the shell scripts.
_ENV_PATH_KEYS = ("data_root", "raw_data_root", "artifact_root", "datasets_json")


@dataclass
class PathsConfig:
    """Where inputs are read from and outputs are written to."""

    # Root holding the raw per-dataset directories and the cached merged datasets.
    data_root: str = str(REPO_ROOT / "data")
    # Everything this pipeline writes lands under <artifact_root>/<run_name>/.
    artifact_root: str = str(PROJECT_ROOT / "artifacts")
    # Cohort registry: sample_id -> tissue, split, citation.
    datasets_json: str = str(PROJECT_ROOT / "configs" / "datasets.json")
    # Cached HuggingFace DatasetDicts holding precomputed FM features, one per
    # gene-FM substrate. Paths are relative to ``data_root`` unless absolute.
    # Each must expose: sample_id, spot_id, array_row, array_col, annotation,
    # dataset_id, gene_features (1,Dg), patch_features (1,Dp).
    merged_datasets: dict[str, str] = field(
        default_factory=lambda: {
            "geneformer": "none_midnight_geneformer_none_none/merged_dataset",
            "scgpt": "none_midnight_scgpt_merged_none_none/merged_dataset",
            "cancerfoundation": "none_midnight_cancerfoundation_merged_none_none/merged_dataset",
            "sysvi": "none_midnight_sysvi_hvg_merged_none_none/merged_dataset",
        }
    )
    # Base for raw per-cohort directories, when they do not sit beside the merged
    # datasets. Empty = use ``data_root``. On a cluster these are usually different
    # filesystems: ``data_root`` holds the tokenizer output, this holds the cohorts.
    raw_data_root: str = ""
    # dataset_id -> directory of raw per-sample .h5ad files, used only by the
    # biosignal (per-gene ridge R^2) stage, which needs count-level targets.
    # Relative entries resolve under ``raw_data_root``; absolute ones win outright.
    raw_h5ad: dict[str, str] = field(
        default_factory=lambda: {
            "10x_TuPro": "10x_TuPro/out_all_genes/data/h5ad",
            "TLS_VISIUM_USZ_kidney": "TLS_VISIUM_USZ/h5ad_preprocessed",
            "TLS_VISIUM_USZ_lung": "TLS_VISIUM_USZ/h5ad_preprocessed",
        }
    )


@dataclass
class DataConfig:
    """Which spots enter the pipeline."""

    # Key into paths.merged_datasets. Also names the feature cache directory.
    substrate: str = "geneformer"
    annotation_col: str = "annotation"
    # Restrict to these HF splits (the cached datasets ship train/validation/test).
    splits: tuple[str, ...] = ("train", "validation", "test")
    # Optional whitelist / blacklist of sample_ids (empty = no filtering).
    include_samples: tuple[str, ...] = ()
    exclude_samples: tuple[str, ...] = ()
    # Cap total spots (0 = all). Used by the smoke config; stratified by sample.
    max_spots: int = 0
    # Rebuild the .npy feature cache even when it already exists.
    force_rebuild: bool = False
    # Support floor for a pathology class, counted *within its own tissue* over the
    # spots this config actually keeps. A class under either bound is demoted to
    # unlabelled — it leaves the probe's training set, its test rows, and the pooled
    # macro average alike. Every probe is fitted within one tissue, so a class on a
    # single slide there cannot be held out and scored: it contributes a structural
    # zero that no representation can move, and drags every model's macro down by
    # the same constant. The bounds are independent — 0 switches one off — and 0
    # for both scores every class as annotated.
    # At the defaults this drops `lung | LN` (33 spots, 1 slide) and
    # `skin | Pigment` (18 spots); the next-smallest class, `kidney | INFL`, keeps
    # 164 spots on 2 slides.
    min_class_spots: int = 50
    min_class_slides: int = 2


@dataclass
class FoldsConfig:
    """Hierarchical held-out splits. See :mod:`vgtfm.data.folds` for what each holds out.

    ``cross_replicate`` and ``cross_region`` need TuPro's ``DONOR-REGION-REPLICATE``
    ids; ``cross_donor`` and ``cross_slide`` need only the cohort's own donor and
    tissue columns, so every annotated cohort enters them.
    """

    levels: tuple[str, ...] = ("cross_replicate", "cross_region", "cross_donor")
    # Restrict fold generation to these organs (empty = every annotated organ).
    # Only the folds are filtered: the representation is still fitted on the whole
    # `train` split. Set it to reproduce a published melanoma-only protocol, or to
    # score one cohort on its own; leave it empty for the headline run.
    tissues: tuple[str, ...] = ()
    # Cap on leave-one-donor-out folds, applied *per tissue* so a large cohort
    # cannot crowd out a small one: 7 TuPro skin + 4 USZ lung + 2 USZ kidney = 13
    # folds, all uncapped.
    max_cross_donor_splits: int = 10
    seed: int = 42


@dataclass
class AEConfig:
    """The gene -> morphology encoder-decoder.

    Encoder: MLP with BatchNorm + GELU, projecting gene features to ``latent_dim``.
    Decoder mirrors it back to the morphology dimension. MSE loss on the raw patch
    features; AdamW, cosine-annealed; early stopping on a random validation split.
    At inference the decoder is discarded and the deployed embedding is
    ``LayerNorm(Encoder(gene))``.

    ``enc_layers`` / ``dec_layers`` count Linear layers. The default of 2 gives
    encoder 1152 -> 512 -> 128 and decoder 128 -> 512 -> 3072 for
    Geneformer + Midnight.
    """

    latent_dim: int = 128
    enc_layers: int = 2
    dec_layers: int = 2
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 4096
    num_epochs: int = 200
    patience: int = 20
    val_frac: float = 0.1
    # Patch-target transform for the correspondence ablation:
    # none | within-sample | global | gaussian.
    patch_transform: str = "none"
    # Seed for the patch permutation, kept separate from the model-init seed so
    # matched and shuffled conditions share identical initialisation.
    shuffle_seed: int = 42


@dataclass
class VariantsConfig:
    """Alternative objectives for the appendix table.

    Each reuses the autoencoder's encoder and training schedule and changes only
    what the latent is asked to do, so the comparison isolates the objective rather
    than confounding it with capacity or optimisation settings.
    """

    # dual_decoder: also reconstruct the gene input, weighting the gene head by
    # this factor relative to the morphology head.
    gene_recon_weight: float = 1.0
    # infonce: temperature of the contrastive alignment between gene and patch.
    temperature: float = 0.07
    # jepa: weight of the variance-preservation term that stops the predictor
    # collapsing to a constant.
    variance_weight: float = 1.0


@dataclass
class CDANNConfig:
    """Conditional domain-adversarial alignment.

    Symmetric InfoNCE between the gene and morphology projections, plus a
    gradient-reversed discriminator that predicts slide identity conditioned on
    tissue. Deployed embedding = the gene projector alone.
    """

    latent_dim: int = 128
    hidden_dim: int = 512
    disc_hidden_dim: int = 256
    disc_layers: int = 3
    dropout: float = 0.1
    # The discriminator learns much faster than the projectors it fights, so it is
    # given a far smaller learning rate.
    projector_lr: float = 1e-4
    discriminator_lr: float = 1e-5
    weight_decay: float = 1e-2
    batch_size: int = 256
    num_epochs: int = 50
    label_smoothing: float = 0.1
    max_grl_lambda: float = 0.5


@dataclass
class HVGConfig:
    """The count-level baseline (``hvg_pca``): scanpy's standard workflow.

    CP10k -> log1p -> top-``n_top_genes`` HVG -> per-gene scaling -> PCA at
    ``models.pca_components``, so it lands at the same width as every other
    representation and the kNN probe stays comparable.
    """

    # Genes kept, ranked by seurat_v3 dispersion on the fit split only.
    n_top_genes: int = 2000
    # Zero-mean/unit-variance per gene before the PCA, as `sc.pp.scale` does.
    # Without it the leading PCs follow a handful of very highly expressed genes.
    scale: bool = True
    # A gene must be detected in at least this many sampled spots to enter the
    # vocabulary at all. Matches the biosignal stage's filter.
    min_expressed: int = 10
    # Spots sampled when deciding the vocabulary, and when ranking dispersion.
    # Both are estimates over genes; neither needs the full 193k fit spots, and
    # the full spots-by-genes matrix at the full vocabulary would be ~12 GB.
    vocabulary_max_spots: int = 40_000
    rank_max_spots: int = 50_000
    # Slides read before the per-slide count cache is dropped. Bounds the resident
    # set: the raw .h5ad files are dense-expanded and 120 of them do not fit in
    # what the train stage is allocated.
    slides_per_chunk: int = 4


@dataclass
class ModelsConfig:
    # The majority-class baseline is not listed here: it ignores the embedding
    # entirely, so the evaluation stage emits it automatically for every protocol.
    names: tuple[str, ...] = ("pca", "pca_oracle", "ae")
    # Components for the gene-only PCA baseline, the count-level HVG baseline and
    # the H&E PCA oracle alike, so no comparison between them is a comparison of
    # dimensionality.
    pca_components: int = 128
    # Components for `pca_oracle_matched`, the capacity control. At 128 components
    # the oracle has ~2.5x the effective rank of the gene PCAs (43.0 against
    # 15.6-22.4); refitting it where those coincide separates its lead from its
    # capacity.
    pca_oracle_matched_components: int = 32
    # Dimensionality reduction applied to the gene features *before* a
    # `<correction>_<model>` arm corrects them; 0 corrects at full width. Harmony and
    # ComBat on 328k x 1152 float32 are the expensive part of that arm, and setting
    # this to `pca_components` corrects in the same 128-d space `integrate` does, in
    # minutes rather than hours. It changes what the inner model consumes, so an arm
    # run this way is a narrower model and does not belong in a column with the
    # reported one. See `vgtfm/models/corrected.py`.
    correction_input_dim: int = 0
    ae: AEConfig = field(default_factory=AEConfig)
    cdann: CDANNConfig = field(default_factory=CDANNConfig)
    hvg: HVGConfig = field(default_factory=HVGConfig)
    variants: "VariantsConfig" = field(default_factory=lambda: VariantsConfig())


@dataclass
class TrainConfig:
    """How the representation is fitted."""

    # Cohort split whose spots fit the representation. The annotated cohorts
    # (TuPro, USZ) sit in `test` and are never seen during representation training.
    fit_split: str = "train"
    # Legacy behaviour: after fitting on `fit_split`, refit the model on each
    # fold's training slides before scoring that fold. Costs one fit per fold and
    # makes the evaluation no longer zero-shot, so it is off by default.
    refit_per_fold: bool = False
    save_embeddings: bool = True


@dataclass
class EvalConfig:
    """Annotation-prediction probes on top of the learned embeddings."""

    # See vgtfm/evaluate/protocol.py. In decreasing strictness:
    #   heldout_donor        fit on the fold's training donors, score held-out donors
    #   loso_donor           leave one donor out within tissue, no folds
    #   pooled_loso          leave one slide out within tissue (leaks the replicate)
    #   pooled_loso_on_fold  the legacy protocol, reproduced exactly
    # The default reports the first three, so the cost of each leak is visible.
    protocols: tuple[str, ...] = ("heldout_donor", "loso_donor", "pooled_loso")
    knn_k: int = 5
    # Standardise embeddings before the kNN probe. Unstandardised kNN silently
    # favours whichever embedding has the larger leading-eigenvalue spread.
    standardize: bool = True
    # Cap on probe training spots; stratified by class.
    max_train_samples: int = 10_000
    # Fixed independently of the model seed, so every model and seed subsamples the
    # same probe training rows.
    subsample_seed: int = 42
    # Donor-level bootstrap for confidence intervals.
    bootstrap_n: int = 2000
    bootstrap_seed: int = 42
    bootstrap_ci: float = 0.95


@dataclass
class DiagnosticsConfig:
    """Batch-effect and collapse diagnostics."""

    # Which cached feature columns to score.
    columns: tuple[str, ...] = ("gene_features", "patch_features")
    split: str = "train"
    # Effective rank is bootstrapped over random subsamples of this size.
    n_samples: int = 8000
    n_iters: int = 5
    # Controls that make an effective rank of ~29/1152 interpretable.
    include_controls: bool = True
    # CCA spectral ceiling.
    cca_max_spots: int = 30_000
    cca_pca_dim: int = 256
    cca_n_permutations: int = 50
    cca_ridge: float = 1e-4
    # scIB panel. Turning it off is a recorded decision, not a silent skip: the
    # stage refuses to run without scib-metrics unless this says otherwise, and
    # config.resolved.json carries the choice next to the results.
    run_scib: bool = True
    scib_max_spots: int = 10_000
    # Spots drawn for the UMAP figures (UMAP is O(n log n) but plots saturate).
    umap_max_spots: int = 30_000
    integration_methods: tuple[str, ...] = ("harmony", "bbknn", "combat", "scvi")
    # Methods listed here run in a forked subprocess, so a fault in compiled code is
    # a recorded failure rather than the death of the whole stage.
    #
    # Empty, deliberately, and the default worth leaving alone. The SIGILL this was
    # built for cannot happen any more (`correct_bbknn` asks bbknn for
    # `computation="cKDTree"`), and forking a process already holding a CUDA context,
    # an OpenBLAS pool and JAX's threads can hang the *parent* inside `os.fork()`,
    # where no deadline reaches it — in practice the guard has cost more runs than the
    # fault it guards against. `_isolated` in `vgtfm/diagnostics/integration.py` has
    # the mechanism. List a method only to contain a native crash that is actually
    # happening, and never one that returns a graph (`_isolated` carries back an array
    # or None) or one that touches the GPU.
    isolate_methods: tuple[str, ...] = ()
    # Wall-clock budget for one isolated method, in seconds, so a child that
    # deadlocks instead of faulting becomes the same recorded row a crash produces
    # rather than an 8h SLURM kill. 0 disables the deadline.
    isolate_timeout_s: float = 1800.0
    # The *algorithm* seed, and only that: which spots the scIB panel draws, which
    # permutations the CCA null uses, which subsample the effective rank is measured
    # on, which correction a stochastic method converges to. Fixed independently of
    # the model seed for the same reason `eval.subsample_seed` and
    # `eval.bootstrap_seed` are: every representation has to be scored on the same
    # spots, or a difference between two representations is confounded with a
    # difference in which spots were drawn.
    #
    # It is *not* "which fit to score". That is `seeds` below, and these stages loop
    # over it like every other stage — see `model_seeds`.
    seed: int = 42
    # Fits to score, when a run wants fewer than `seeds`. Empty means all of them,
    # which is the default and what makes `diagnose` and `integrate` the same
    # experiment as `eval`. Narrow it only to trade a published statistic for
    # wall time, and note that a single seed cannot be compared with a table that
    # averages three.
    model_seeds: tuple[int, ...] = ()


@dataclass
class AblationConfig:
    """Gene/morphology correspondence ablation."""

    model: str = "ae"
    # Conditions to train, in reporting order. See vgtfm/ablations/patch_shuffle.py.
    conditions: tuple[str, ...] = ("none", "within-sample", "global", "gaussian")


@dataclass
class BiosignalConfig:
    """Per-gene ridge R^2 of frozen vs refined embeddings, plus GSEA."""

    # Only the fallback once `ridge_alpha_search` is on: the fitted penalty comes
    # from the grid below. Left here so a fixed-penalty run stays expressible, and
    # so the published figure remains reproducible by setting the search off.
    ridge_alpha: float = 1.0
    # Fit the penalty per fold, per representation and per gene by generalised
    # cross-validation on the training rows. Off, this stage compares widths rather
    # than embeddings: at alpha=1 a 1152-d design with ~4.3k training spots is
    # unregularised, and the p/n it pays in held-out R^2 exceeds any plausible
    # difference in information between the representations being compared.
    ridge_alpha_search: bool = True
    # Log-spaced search grid as (lo, hi, num). Wide enough to reach both ends of
    # the design: alpha << min(s^2) leaves least squares untouched, alpha >> max(s^2)
    # predicts the training mean.
    ridge_alpha_grid: tuple[float, float, int] = (1e-1, 1e7, 33)
    # Drop genes expressed in fewer than this many spots.
    min_expressed: int = 10
    # Cap on ridge training spots per fold (0 = all). A closed-form ridge with at
    # most ~1150 features does not need 70k rows, and the targets matrix is the
    # memory bottleneck: 70k spots x 11k genes is ~3 GB.
    max_train_spots: int = 30_000
    # Organs this stage scores. Distinct from `folds.tissues`, which is global and
    # would restrict the annotation probe too: the levels are not comparable across
    # organs, because only TuPro carries replicate and region ids. `cross_replicate`
    # and `cross_region` are TuPro skin whatever this says, so leaving USZ in only
    # made `cross_donor` a different cohort from the two levels above it. Empty
    # scores every organ together, which pools fold residuals against a grand mean
    # no fold ever saw — kept expressible, not recommended.
    tissues: tuple[str, ...] = ("skin",)
    # Build the gene vocabulary from every annotated spot rather than from the spots
    # this scope's folds happen to touch, so per-organ runs stay comparable gene by
    # gene. A gene silent in the scope scores SS_tot = 0 and drops out as NaN.
    shared_vocabulary: bool = True
    gene_sets: tuple[str, ...] = ("hallmark", "progeny")
    fdr: float = 0.05
    # Permutations behind each enrichment p-value, which is the resolution of the
    # test rather than a speed knob: nothing below 1/gsea_permutations can be
    # resolved. At 1,000 a third of the Hallmark sets came back at exactly p=0 and no
    # bound tighter than 0.001 could be reported; at 10,000 the bound is 1e-4 and the
    # whole panel costs ~9 s per ranking.
    gsea_permutations: int = 10_000
    # Fewest members a set must have *in the ranked background* to be scored at all.
    # Sets below it are dropped rather than tested, which is the family the
    # Benjamini-Hochberg adjustment then runs over.
    gsea_min_set_size: int = 15
    resources_dir: str = str(PKG_ROOT / "biosignal" / "resources")
    # Capacity control: also score PCA(frozen, k=latent_dim) so the frozen-vs-refined
    # comparison is not confounded by dimensionality alone.
    include_pca_control: bool = True
    # Which differences of per-gene R^2 the enrichment is computed on. Ranking by
    # `guided_vs_frozen` alone cannot attribute a degraded pathway to morphology:
    # the guided embedding is also the narrower one, and `capacity_vs_frozen` — the
    # same width change with no morphology in it — is what says how much of the
    # enrichment that width buys. `guided_vs_capacity` puts both sides at the same
    # width, so it is the contrast that isolates guidance. Names index
    # `run_biosignal.GSEA_CONTRASTS`; those needing the control are skipped when
    # `include_pca_control` is off.
    gsea_contrasts: tuple[str, ...] = (
        "guided_vs_frozen",
        "capacity_vs_frozen",
        "guided_vs_capacity",
    )
    # Bootstrap resamples behind the per-set interval the dot plot draws. Resampling
    # is over *member genes*, so the interval is a spread rather than a test and is
    # not over donors.
    setmean_bootstrap: int = 10_000
    setmean_ci: float = 0.95
    # Equal-count bins of the frozen R^2 the baseline-matched permutation null shuffles
    # within. Change in R^2 is strongly anti-correlated with the frozen R^2 (Spearman
    # ~-0.6 per gene, ~-0.8 per set) and curated sets are built from well-predicted
    # genes, so a set clears the plain background test on composition alone: 41 of the
    # 50 Hallmark sets do, and 3 survive this null. Below 2 switches it off and leaves
    # the matched columns NaN.
    setmean_baseline_bins: int = 20
    # Fold levels the per-gene R^2 scatter draws, in order. The stage scores every
    # level the cohort supports; the manuscript's figure reads within-donor against
    # across-donor, and the middle level is a run-review view. Empty draws every
    # level scored.
    per_gene_r2_levels: tuple[str, ...] = ("cross_replicate", "cross_donor")
    # Sensitivity analysis, off by default. Every contrast is additionally scored on a
    # ranking centred within this many equal-count bins of the frozen R^2, which sweeps
    # out the shrinkage trend `quartile_table` tabulates. Worth running when a result
    # comes back one-sided across nearly every set — that is the signature of a ranking
    # ordered mostly by baseline predictivity — but it is an aggressive correction that
    # also removes real signal wherever the effect genuinely tracks how well a gene
    # started out, so it informs the headline rather than being it.
    gsea_detrend_bins: int = 0


@dataclass
class PerfConfig:
    device: str = "auto"  # auto | cpu | cuda
    amp: bool = True
    amp_dtype: str = "auto"  # auto (bf16 when supported) | bf16 | fp16
    compile: bool = False
    tf32: bool = True
    cudnn_benchmark: bool = True
    matmul_precision: str = "high"
    num_threads: int = 0  # 0 = leave torch's default
    gpu_batch_size: int = 0  # >0 overrides batch size on CUDA


@dataclass
class Config:
    run_name: str = "default"
    # Name of the site profile from environments.yaml that supplied the paths.
    # Recorded rather than configured: set it with --env, not in a config file.
    env: str = ""
    # Every stage that has randomness is repeated once per seed and reported as
    # mean +/- spread across seeds. That includes `diagnose` and `integrate`, which
    # score one row per fit through `vgtfm.diagnostics.model_seeds` — for a release
    # they scored a single fit named by `diagnostics.seed` while `eval` averaged
    # three, which made the integration table's uncorrected row and Table 1's PCA row
    # two different statistics separated by a few thousandths.
    seeds: tuple[int, ...] = (42, 43, 44)
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    folds: FoldsConfig = field(default_factory=FoldsConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    biosignal: BiosignalConfig = field(default_factory=BiosignalConfig)
    perf: PerfConfig = field(default_factory=PerfConfig)

    # ── derived paths ────────────────────────────────────────────────
    @property
    def out_dir(self) -> Path:
        return Path(self.paths.artifact_root) / self.run_name

    def sub(self, *parts: str) -> Path:
        """Return (and create) a subdirectory of the run's output directory."""
        p = self.out_dir.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def data_path(self, substrate: str | None = None) -> Path:
        """Absolute path of the cached merged dataset for a substrate."""
        key = substrate or self.data.substrate
        try:
            rel = self.paths.merged_datasets[key]
        except KeyError:
            raise SystemExit(
                f"Unknown substrate '{key}'. Known: {sorted(self.paths.merged_datasets)}"
            ) from None
        p = Path(rel)
        return p if p.is_absolute() else Path(self.paths.data_root) / p

    def raw_root(self) -> Path:
        """Base directory for raw cohorts, falling back to ``data_root``."""
        return Path(self.paths.raw_data_root or self.paths.data_root)

    def raw_h5ad_dir(self, dataset_id: str) -> Path:
        """Absolute directory of raw ``.h5ad`` files for one cohort.

        Entries of ``paths.raw_h5ad`` are relative to ``data_root`` unless
        absolute — a site profile uses absolute paths when the raw cohorts do not
        live beside the merged datasets, which on a cluster they usually do not.
        """
        p = Path(self.paths.raw_h5ad[dataset_id])
        return p if p.is_absolute() else self.raw_root() / p

    def cache_dir(self, substrate: str | None = None) -> Path:
        """Where the extracted feature matrices for a substrate are cached.

        Lives outside the per-run directory: the extraction is a pure function of
        (merged dataset, data filters), so runs share it.
        """
        key = substrate or self.data.substrate
        p = Path(self.paths.artifact_root) / "_cache" / key
        p.mkdir(parents=True, exist_ok=True)
        return p


# ── YAML loading + dotted overrides ─────────────────────────────────


def _from_dict(cls, data: dict[str, Any]):
    if not is_dataclass(cls):
        return data
    kwargs: dict[str, Any] = {}
    # ``from __future__ import annotations`` stringifies field types; resolve them.
    type_by_name = get_type_hints(cls)
    valid = {f.name for f in fields(cls)}
    for key, val in data.items():
        if key not in valid:
            raise KeyError(f"Unknown config key '{key}' for {cls.__name__}")
        ftype = type_by_name.get(key)
        if is_dataclass(ftype) and isinstance(val, dict):
            kwargs[key] = _from_dict(ftype, val)
        elif isinstance(val, list):
            kwargs[key] = tuple(val)  # dataclass tuples <- yaml lists
        else:
            kwargs[key] = val
    return cls(**kwargs)


def load_environments(path: str | Path | None = None) -> dict[str, Any]:
    """Parse ``environments.yaml``. Returns ``{}`` when the file is absent."""
    f = Path(path) if path is not None else ENVIRONMENTS_FILE
    if not f.exists():
        return {}
    return yaml.safe_load(f.read_text()) or {}


def resolve_env_name(name: str | None = None, path: str | Path | None = None) -> str | None:
    """Pick the site profile: explicit name, then ``$VGTFM_ENV``, then the file's
    ``default:``. ``None`` means "no profile" — the dataclass defaults stand."""
    if name:
        return name
    env = os.environ.get("VGTFM_ENV")
    if env:
        return env
    return (load_environments(path) or {}).get("default")


def environment(name: str, path: str | Path | None = None) -> dict[str, Any]:
    """One site profile, by name. Unknown names are an error, not a silent default."""
    envs = (load_environments(path) or {}).get("environments") or {}
    if name not in envs:
        hint = ""
        if path is None and not ENVIRONMENTS_FILE.exists():
            hint = (
                f"\n{ENVIRONMENTS_FILE.name} does not exist yet — it is not tracked. "
                f"Copy {ENVIRONMENTS_FILE.name.replace('.yaml', '.example.yaml')} "
                f"and fill in the paths for this machine."
            )
        raise SystemExit(
            f"--env: unknown environment '{name}' "
            f"(known: {sorted(envs)}; defined in {ENVIRONMENTS_FILE}){hint}"
        )
    return envs[name] or {}


def apply_environment(cfg: Config, name: str, path: str | Path | None = None) -> None:
    """Overlay a site profile's paths onto ``cfg`` in place.

    Only the keys the profile actually states are touched, so a profile is a diff
    against the defaults rather than a second copy of them. Applied before the
    ``--set`` overrides, which therefore still win.
    """
    prof = environment(name, path)
    for key in _ENV_PATH_KEYS:
        val = prof.get(key)
        if not val:
            continue
        q = Path(str(val))
        setattr(cfg.paths, key, str(q if q.is_absolute() else PROJECT_ROOT / q))
    # Per-cohort mappings merge into the defaults instead of replacing them, so a
    # profile can relocate one cohort without restating the others.
    for key in ("merged_datasets", "raw_h5ad"):
        for k, v in (prof.get(key) or {}).items():
            getattr(cfg.paths, key)[k] = str(v)
    cfg.env = name


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    env: str | None = None,
    environments_file: str | Path | None = None,
) -> Config:
    """Load :class:`Config` from YAML (falling back to defaults), overlay the site
    profile, then apply ``--set`` overrides — in that order of precedence."""
    data: dict[str, Any] = {}
    if path is not None:
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
    cfg = _from_dict(Config, data)
    name = resolve_env_name(env, environments_file)
    if name:
        apply_environment(cfg, name, environments_file)
    for dotted, value in (overrides or {}).items():
        _apply_override(cfg, dotted, value)
    return cfg


_MISSING = object()


def _coerce_like(current: Any, value: Any) -> Any:
    """Coerce a string CLI value to the type of the field it is replacing."""
    if isinstance(current, bool):
        return str(value).lower() in {"1", "true", "yes"}
    if isinstance(current, tuple):
        return tuple(_coerce_scalar(v) for v in str(value).split(",") if v != "")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def _apply_override(cfg: Any, dotted: str, value: Any) -> None:
    """Apply one dotted ``--set`` override in place.

    Descends dataclass attributes *and* dict-valued fields, so the per-dataset
    entries of ``paths.merged_datasets`` and ``paths.raw_h5ad`` are addressable
    as ``paths.raw_h5ad.10x_TuPro=/abs/path`` — needed on a cluster, where the
    raw cohorts do not all live under one ``data_root``. In both cases the key
    must already exist, so a typo is an error rather than a silent no-op.
    """
    *parents, leaf = dotted.split(".")
    obj = cfg
    for p in parents:
        nxt = obj.get(p, _MISSING) if isinstance(obj, dict) else getattr(obj, p, _MISSING)
        if nxt is _MISSING:
            raise SystemExit(f"--set: unknown config key '{dotted}' (no '{p}')")
        obj = nxt
    if isinstance(obj, dict):
        if leaf not in obj:
            raise SystemExit(f"--set: unknown config key '{dotted}' (known: {sorted(obj)})")
        obj[leaf] = _coerce_like(obj[leaf], value)
        return
    if not hasattr(obj, leaf):
        raise SystemExit(f"--set: unknown config key '{dotted}'")
    setattr(obj, leaf, _coerce_like(getattr(obj, leaf), value))


def _coerce_scalar(s: str) -> Any:
    for caster in (int, float):
        try:
            return caster(s)
        except (TypeError, ValueError):
            continue
    return s


def to_dict(cfg: Any) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


# ── which fits an experiment scores ──────────────────────────────────


def model_seeds(cfg) -> tuple[int, ...]:
    """The fits ``diagnose`` and ``integrate`` score, one row of output each.

    ``cfg.seeds`` — the same replicate seeds ``train``, ``eval`` and ``ablate`` use —
    unless ``diagnostics.model_seeds`` narrows it. These two stages once scored a
    single fit named by ``diagnostics.seed`` while ``eval`` averaged three, so the
    integration table's uncorrected row and Table 1's PCA row were one seed against a
    mean of three: close enough to read as a discrepancy and not comparable as a
    statistic. They are the same experiment now.

    ``diagnostics.seed`` keeps its other job, which is not this one: it seeds the
    subsamples and permutations *inside* a diagnostic, and stays fixed across fits so
    every representation is scored on the same spots.
    """
    chosen = tuple(cfg.diagnostics.model_seeds) or tuple(cfg.seeds)
    unknown = [s for s in chosen if s not in tuple(cfg.seeds)]
    if unknown:
        raise SystemExit(
            f"diagnostics.model_seeds={list(chosen)} names seed(s) {unknown} that "
            f"`train` was not run with (seeds={list(cfg.seeds)}), so no embedding "
            f"exists for them."
        )
    return chosen


def headline_seed(cfg) -> int:
    """The single fit an artefact shows when it can only show one.

    A UMAP is a picture of one embedding and a per-gene R^2 scatter is a picture of
    one fit; three of either would be three figures, not one statistic. This is the
    *only* sanctioned reason to look at one seed, and naming it is the point:
    ``diagnose`` and ``integrate`` drifted apart from ``eval`` precisely by each
    spelling "which fit" for itself, one as ``seeds[0]`` and one as
    ``diagnostics.seed``. Anything that reports a *number* uses
    :func:`model_seeds` instead.
    """
    return int(cfg.seeds[0])
