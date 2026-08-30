# vgtfm — vision-guided transcriptomics foundation models

Code for *Batch effects limit histology-guided supervision of transcriptomic foundation models*.

Spatial transcriptomics pairs spatially resolved gene expression with tissue morphology,
providing complementary molecular and histological views of the same tissue.
Transcriptomic foundation models provide general-purpose representations of gene
expression, but these can capture batch effects that obscure biological signal. We test
whether matched H&E histology can improve these representations through training-time
supervision, with morphology used only during training and discarded at inference.
Across three transcriptomic foundation models, histology-guided supervision provides
little improvement in cross-donor annotation transfer, even though histology alone
transfers the same annotations far more accurately. We find that gene expression
embeddings from spot-based spatial transcriptomics are low-dimensional and strongly
structured by slide identity, which limits the biological information available for
cross-modal transfer. These results indicate that cross-modal supervision can reorganize
information already encoded in a frozen representation, but its effectiveness is
constrained by batch structure in the source representation.

![Gene counts and the matched H&E patch at each Visium spot are encoded by frozen foundation models; an encoder-decoder is trained to reconstruct the morphology embedding from the gene embedding, and its bottleneck is the refined representation read at inference.](docs/overview.png)

---

## Installation

Python 3.11. The `train`, `ablate` and `integrate` stages use a CUDA GPU where available
and otherwise run on CPU; the analysis stages are CPU-only.

```bash
conda create -n vgtfm python=3.11 && conda activate vgtfm   # or: python -m venv .venv
pip install -r requirements.txt
make test                                                   # ~1 min, no data required
```

`requirements.txt` pins the versions used for the reported results. The `torch` pin
assumes CUDA 12.6; install `torch==2.10.0` from the default index on a CPU-only machine.

The gene-side foundation models are not installed here. Their dependencies conflict with
each other and with `requirements.txt`, so each has a separate conda environment under
`envs/`. They are required only by the optional `embed` stage.

## Quick start

```bash
cp environments.example.yaml environments.yaml   # set data_root to the merged datasets
make smoke                                       # end-to-end on a subset, minutes
make all                                         # data → train → eval → diagnose → figures
make all CONFIG=configs/scgpt.yaml ENV=cluster
```

`environments.yaml` holds every machine-specific path; see
[Configuration](#configuration). `make smoke` skips itself when the cached merged dataset
is absent. Output lands under `<artifact_root>/<run_name>/`, and `make help` lists the
stages.

Every target wraps `run.py` and accepts the same overrides:

| Variable | Effect |
| --- | --- |
| `CONFIG=` | which config to run (default `configs/default.yaml`) |
| `ENV=` | which site profile from `environments.yaml` |
| `RUN=` | the run name, and so the output directory |
| `SET=` | any config field, dotted: `SET="models.pca_components=50"` |

### On a cluster

```bash
cp environments.example.yaml environments.yaml
bash slurm/preflight.sh          # validate data, environment and partitions; submits nothing
bash slurm/submit_all.sh         # every stage, dependency-chained
```

[`slurm/README.md`](slurm/README.md) documents the job graph, per-stage resources and
single-stage submission.

## Data

Ten Visium cohorts, 120 slides, registered in `configs/datasets.json` with tissue, split,
annotation column, accession and citation.

| Cohort | Tissue | Slides | Role | Access |
| --- | --- | ---: | --- | --- |
| 10x TuPro (melanoma) | skin | 18 | evaluation | controlled (Tumor Profiler consortium) |
| TLS Visium USZ | lung | 4 | evaluation | Zenodo 14620362 |
| TLS Visium USZ | kidney | 2 | evaluation | Zenodo 14620362 |
| De Zuani LUAD | lung | 22 | representation training | ArrayExpress E-MTAB-13530 |
| De Zuani LUSC | lung | 14 | representation training | ArrayExpress E-MTAB-13530 |
| MOSAIC GBM | brain | 10 | representation training | controlled (EGA EGAS50000000689) |
| MOSAIC DLBCL | lymph node | 10 | representation training | controlled (EGA EGAS50000000689) |
| MOSAIC bladder | bladder | 15 | representation training | controlled (EGA EGAS50000000689) |
| MOSAIC mesothelioma | mesothelium | 10 | representation training | controlled (EGA EGAS50000000689) |
| MOSAIC ovarian | ovarian | 15 | representation training | controlled (EGA EGAS50000000689) |

Representations are fitted on the unannotated cohorts (`train.fit_split`, default
`train`). The annotated cohorts sit in the `test` split and are unseen during training.
Evaluation folds are set by `folds.levels`: `cross_replicate` and `cross_region` require
TuPro's `DONOR-REGION-REPLICATE` slide ids; `cross_donor` covers all annotated cohorts.

The pipeline reads a cached merged dataset of frozen foundation-model features under
`data_root`. See [Computing embeddings from raw data](#computing-embeddings-from-raw-data)
for its schema and how to build one. The raw `.h5ad` files are additionally required by
the `hvg_pca` baseline and the `biosignal` stage.

## Pipeline

```
cached FM embeddings ──data──► spot table + folds ──train──► embeddings/<model>.npy
                                                        ├──eval──────► results, CIs
                                                        ├──diagnose──► rank, variance, CCA, scIB
                                                        ├──ablate────► shuffle controls
                                                        ├──biosignal─► per-gene R², GSEA
                                                        ├──results───► one joined score table
                                                        └──figures───► tables + plots
raw .h5ad + WSI ──embed──► cached FM embeddings                (optional, see below)
```

| Stage | What it does | Cost |
| --- | --- | --- |
| `data` | Materialise the feature cache, describe the cohort, enumerate the folds | ~1 min |
| `train` | Fit each representation on the unannotated cohorts, embed all spots; batch-correct the gene features for the `harmony_*`/`combat_*` arms | minutes–hours |
| `eval` | Annotation probes, per-class F1, donor-level bootstrap CIs | ~15 min |
| `diagnose` | Effective rank, variance decomposition, CCA ceiling, scIB panel | ~1 h |
| `ablate` | Retrain with permuted morphology targets | ~15 min |
| `biosignal` | Per-gene ridge R², GSEA and the per-set mean ΔR² tests, one organ at a time (`biosignal.tissues`) | ~15 min |
| `integrate` | Harmony / BBKNN / ComBat / scVI scored on batch and biology; requires `train` and `eval` | ~2 h |
| `results` | Join every stage's scores into one long-format table and check they agree | seconds |
| `figures` | Regenerate every table and figure from the artefacts above | ~2 min |
| `embed` | Raw slides → foundation-model features (optional) | hours, GPU |

`make all` runs `data → train → eval → diagnose → figures`; the rest are opt-in. Each
stage writes its resolved config beside its outputs, plus
`logs/manifest-<stage>-<ts>.json` recording host, device, wall time and status. Each run
also stamps a `src=` fingerprint, a content hash of the shipped `.py` files, into its
banner and manifest; `preflight.sh` prints the same value.

Several stages score overlapping quantities: `eval` scores the `pca` baseline,
`integrate` reports it as its uncorrected row, and `ablate` scores a matched condition
against it. `make results` joins every stage's scores into `results/scores.csv` and fails
when two artefacts report different values for one cell (`results/conflicts.csv`), or
when an expected cross-check never ran. `scripts/paper_tables.py` applies both checks
before writing LaTeX. Run `make eval integrate results`, in that order.

## Reproducing the manuscript

```bash
make all CONFIG=configs/default.yaml            # Geneformer + Midnight
make all CONFIG=configs/scgpt.yaml
make all CONFIG=configs/cancerfoundation.yaml
```

| Manuscript artefact | Command | Output |
| --- | --- | --- |
| Table 4 — dataset composition | `make data figures` | `figures/table_datasets.{csv,tex}`, `data/cohort.csv` |
| Table 1 — cross-donor annotation F1 | `make all` | `figures/table_annotation_heldout_donor_cross_donor_organ_balanced.{csv,tex}` |
| Table 7 — per-organ breakdown of Table 1 | `make all` | `figures/table_annotation_by_organ_heldout_donor_cross_donor.{csv,tex}` |
| Table 6 — fold hierarchy | `make all` | `figures/table_fold_hierarchy.{csv,tex}` |
| Table 2 — effective rank | `make diagnose figures` | `figures/table_effective_rank.{csv,tex}` |
| Table 3 — patch-shuffle ablation | `make ablate figures` | `figures/table_patch_shuffle.{csv,tex}`, `ablation/paired_deltas.csv` |
| Figure 3 — per-gene R² | `make biosignal figures` | `figures/fig_per_gene_r2.pdf`, `figures/fig_per_gene_r2_vs_control.pdf` |
| Per-gene R², other organs | `make biosignal SET="biosignal.tissues=lung"` | `biosignal/lung/per_gene_r2.csv` |
| Figure 4 — Hallmark dot plot | `make biosignal figures` | `biosignal/skin/setmean_hallmark_<level>.{csv,json}`, `figures/fig_setmean_hallmark_<level>.pdf` |
| Figures 5–7 — UMAPs | `make figures` | `figures/fig_umap.pdf` |
| §4.2 — batch effects | `make diagnose` | `diagnostics/{summary.json,variance_decomposition.csv,cca_ceiling.json,scib_panel.csv}` |
| §4.2 — integration methods | `make integrate results figures` | `figures/table_integration[_<organ>].{csv,tex}`, `integration/{integration.csv,integration_deltas.csv}` |
| Appendix — alternative architectures | `make train eval SET="models.names=pca,pca_oracle,ae,cdann,dual_decoder,gene_ae,infonce,jepa"` | `eval/results.csv` |
| Table 8 — batch correction under guidance | `make all integrate` then `make paper-tables` | `paper_tables/table_batch_correction.tex`, `eval/deltas.csv` |

`make paper-tables` builds the cross-run tables covering all three backbones. No stage
rebuilds them, so rerun it after any change to a run's `eval`, `ablate` or `integrate`
outputs; it refuses if the stages disagree.

Table 1 reports two baselines alongside the gene-side representations. `hvg_pca` applies
CP10k, `log1p`, 2000 highly-variable genes, per-gene scaling and PCA to the raw Visium
counts, so it requires `paths.raw_h5ad` to resolve for every cohort in the fit split.
`pca_oracle_matched` refits the H&E oracle at the width where its effective rank matches
the gene PCAs' (`models.pca_oracle_matched_components`); Table 2 reports the measured
ranks.

`seeds` lists the fits each experiment scores. Stages with randomness repeat over it and
report the mean with an interval pooled over the seeds' bootstrap replicates.
`diagnostics.seed`, `eval.subsample_seed` and `eval.bootstrap_seed` are held fixed across
representations. `config.model_seeds(cfg)` and `config.headline_seed(cfg)` determine which
fit an artefact reads.

## Computing embeddings from raw data

The pipeline's entry point is a cached merged dataset of frozen features: one HuggingFace
`DatasetDict` per gene-side backbone, holding both modalities per spot.

```
<data_root>/<combo>/merged_dataset/       dataset_dict.json, train/, validation/, test/
```

| Column | Type |
| --- | --- |
| `sample_id`, `spot_id`, `annotation`, `dataset_id` | string |
| `array_row`, `array_col` | int32 |
| `gene_features` | Array2D `(1, D_gene)` — 1152 Geneformer, 512 scGPT, 256 CancerFoundation |
| `patch_features` | Array2D `(1, 3072)` — Midnight |

If these already exist, point `data_root` at them and skip this section. The `embed` stage
produces them from raw slides. Midnight cuts a 224×224 patch at each spot's pixel centre
and returns CLS ⊕ mean-patch-token; the gene-side models run as separate processes in
their own environments and communicate through parquet.

### Prerequisites

1. **Raw per-slide counts**, one `<sample_id>.h5ad` per slide, resolved as
   `<raw_data_root>/<dataset_id>/<subdirs.h5ad>` from `configs/datasets.json`, with
   per-cohort exceptions under the profile's `raw_h5ad`.
2. **Whole-slide images**, resolved as
   `<data_root>/<dataset_id>/<subdirs.tif>/<sample_id>.<ext>`, `ext` one of `.tif`,
   `.tiff`, `.svs`, `.ndpi`. This resolves under `data_root`, not `raw_data_root`.
   Cohorts on the `lstsq_estimate` alignment (both USZ) also require
   `<sample_id>_manual_loupe_alignment.json` beside the image; the others carry
   `x_pixel`/`y_pixel` in `.obs`.
3. **Checkpoints.** `scGPT_human` from the scGPT repository, and for CancerFoundation a
   checkout plus the weights its README links, both passed with `--model-dir`. Midnight
   and the Geneformer weights are fetched from HuggingFace, the former on first use and
   the latter by the install script below.
4. **The gene-side environment** for the substrate being embedded:

   ```bash
   conda env create -f envs/scgpt.yaml            # or envs/cancerfoundation.yaml
   conda env create -f envs/geneformer.yaml
   conda run -n vgtfm-geneformer python envs/geneformer_install.py
   ```

   Geneformer requires the extra step because it ships as a HuggingFace model repository;
   `envs/geneformer.yaml` records the details. Environments are located as `vgtfm-<model>`
   first, then bare `<model>`.

### Running it

Three commands, since the second runs in another environment:

```bash
# 1. cut H&E patches with Midnight, write the gene-side manifest
python run.py embed --config configs/scgpt.yaml --env <profile>

# 2. embed the gene side. Step 1 prints this line with your paths substituted.
conda run -n vgtfm-scgpt python -m vgtfm.embed.gene_fm \
    --model scgpt --manifest <artifact_root>/_embed/scgpt/manifest.json \
    --out-dir <artifact_root>/_embed/scgpt --model-dir /path/to/scGPT_human

# 3. the same command as step 1; the parquets now exist, so it merges
python run.py embed --config configs/scgpt.yaml --env <profile>
```

On a cluster, steps 1 and 2 are `sbatch slurm/embed_midnight.sbatch` and
`sbatch slurm/embed_gene.sbatch <model>`. Each writes one parquet per (slide, modality)
under `<artifact_root>/_embed/<modality>/<dataset_id>/<sample_id>.parquet`. The merge
keeps only spots present in both modalities and takes the split assignment from the
cohort registry.

**Point the config at the result.** The merge writes to
`<data_root>/vgtfm_<substrate>_midnight/merged_dataset`, which differs from the shipped
default. Add the override to your site profile:

```yaml
environments:
  cluster:
    merged_datasets:
      scgpt: vgtfm_scgpt_midnight/merged_dataset
```

or pass it per run:

```bash
make all CONFIG=configs/scgpt.yaml \
     SET="paths.merged_datasets.scgpt=vgtfm_scgpt_midnight/merged_dataset"
```

`bash slurm/preflight.sh` then confirms the dataset resolves and carries the required
columns.

## Configuration

Every machine-specific path lives in `environments.yaml`, an untracked file holding named
site profiles. Start from `environments.example.yaml`.

| Profile key | Purpose |
| --- | --- |
| `data_root` | holds the cached merged datasets, `<combo>/merged_dataset` |
| `artifact_root` | run outputs and the shared feature cache |
| `merged_datasets` | per-substrate overrides of the merged-dataset directory names |
| `raw_data_root`, `raw_h5ad` | raw per-slide `.h5ad`, used by `hvg_pca`, `biosignal` and `embed` |
| `python_env`, `hf_home` | interpreter and offline model cache for cluster jobs |
| `slurm_*` | partitions, GPU request syntax, account/QOS flags |

Select a profile with `--env <name>`, `VGTFM_ENV`, or `make ... ENV=`. Precedence runs
dataclass defaults (`vgtfm/config.py`) < `configs/<name>.yaml` < the profile <
`--set key=value`. Only `data_root` is required. Each run writes its resolved
configuration to `config.resolved.json`.

## Repository layout

```
run.py                     stage dispatch, logging, run manifest
environments.example.yaml  template for the untracked environments.yaml
configs/                   one YAML per substrate, plus datasets.json, smoke.yaml
                           and regression_check.yaml
scripts/                   paper_tables.py (cross-run tables), embed_check.py
slurm/                     cluster entry points: preflight.sh, submit_all.sh, submit.sh
vgtfm/config.py            the whole config surface, as one dataclass tree
vgtfm/degraded.py          how a stage refuses, and how it reports an artefact whose
                           producing stage has not run yet
vgtfm/provenance.py        the `src=` fingerprint stamped into every banner
vgtfm/labels.py            annotated-spot rules, class-name normalisation, support floor
vgtfm/results.py           the joined score table and its cross-stage checks
vgtfm/data/                feature cache, cohort table, hierarchical folds
vgtfm/models/              hvg_pca, pca, pca_oracle(_matched), ae, cdann
                           + variants/ for the appendix
vgtfm/models/nn.py         the training loop and blocks every neural model shares
vgtfm/evaluate/            probes, protocols, donor bootstrap, reporting, eval stage
vgtfm/diagnostics/         effective rank, variance, CCA, scIB, integration
vgtfm/ablations/           patch-shuffle transforms and the ablation stage
vgtfm/biosignal/           expression index, ridge R², GSEA, per-set mean ΔR² tests
vgtfm/figures/             tables (CSV + LaTeX) and plots
vgtfm/embed/               optional: raw slides → foundation-model features
tests/                     unit tests, stage tests on a synthetic cohort, and an
                           end-to-end smoke test on the real data
```

The feature cache at `<artifact_root>/_cache/<substrate>/` holds memory-mapped `.npy`
matrices and a metadata parquet, roughly 5 GB per substrate. The first stage that needs it
builds it; runs on the same substrate share it. It is keyed on the source path and
regenerable, so it should not be copied between machines.

## Testing

```bash
make test        # unit suite: ~1 min, no data required
make smoke       # end-to-end on a small subset of the cohort (minutes)
make embed-check # gene-side models against real checkpoints, one GPU
make lint        # ruff rules, then a formatting check
make format      # apply the formatter
```

The suite concentrates on the statistics and the splits. `test_protocols.py` asserts each
protocol's documented leakage on synthetic data with a known answer. `test_models.py`
hashes the embeddings of all six torch models on a fixed fixture, so any change to the
optimiser, schedule, validation split or batching fails the test and the digests must be
updated deliberately.

## Citation

If you use this code, please cite:

> *Batch effects limit histology-guided supervision of transcriptomic foundation models.*

Per-cohort citations and accessions are in the `citation` field of each entry in
`configs/datasets.json`, reproduced by the generated `figures/table_datasets.tex`.
