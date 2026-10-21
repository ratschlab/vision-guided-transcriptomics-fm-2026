# vgtfm — vision-guided transcriptomics foundation models

Code for *Slide-level batch structure limits histology-guided supervision of transcriptomic
foundation models* ([bioRxiv](https://doi.org/10.64898/2026.09.30.755585)).

Spatial transcriptomics pairs spatially resolved gene expression with tissue morphology in
the same tissue section. Transcriptomic foundation models encode such expression profiles
into general-purpose representations, but these representations can retain slide- and
cohort-specific variation that obscures biological signal. Here, we test whether matched
H&E histology can improve these representations as a training-time supervisory signal
that is discarded at inference. Across three transcriptomic foundation models,
histology-guided supervision provides no consistent aggregate improvement in cross-donor
annotation transfer, despite substantially stronger transfer from histology alone. We find
that gene-expression embeddings from spot-based spatial transcriptomics are
low-dimensional and strongly structured by slide identity, limiting the shared
gene–morphology signal available for cross-modal transfer. Guidance benefits some
morphologically distinctive classes but reduces held-out gene predictivity broadly across
the transcriptome. These results suggest that cross-modal supervision can reorganize
information already encoded in a frozen representation but cannot recover information the
representation does not retain, highlighting the importance of diagnosing slide-specific
structure before applying such supervision.

![Gene counts and the matched H&E patch at each Visium spot are encoded by frozen foundation models; an encoder-decoder is trained to reconstruct the morphology embedding from the gene embedding, and its bottleneck is the refined representation read at inference.](docs/overview.png)

---

## Installation

Python 3.11. Training uses a CUDA GPU if available; everything also runs on CPU.

```bash
conda create -n vgtfm python=3.11 && conda activate vgtfm   # or: python -m venv .venv
pip install -r requirements.txt
make test                                                   # ~1 min, no data required
```

The gene-side foundation models have their own conda environments under `envs/` and are
needed only by the optional `embed` stage.

## Quick start

```bash
cp environments.example.yaml environments.yaml   # set data_root to the merged datasets
make smoke                                       # end-to-end on a subset, minutes
make all                                         # data → train → eval → diagnose → figures
make all CONFIG=configs/scgpt.yaml ENV=cluster
```

Output lands under `<artifact_root>/<run_name>/`; `make help` lists the stages. Every
target accepts:

| Variable | Effect |
| --- | --- |
| `CONFIG=` | config to run (default `configs/default.yaml`) |
| `ENV=` | site profile from `environments.yaml` |
| `RUN=` | run name, and so the output directory |
| `SET=` | any config field, e.g. `SET="models.pca_components=50"` |

### Applying the trained model to your own data

[`notebooks/pbmc3k_embeddings.ipynb`](notebooks/pbmc3k_embeddings.ipynb) downloads the
trained Geneformer autoencoder from
[umcconnell/vgtfm](https://huggingface.co/umcconnell/vgtfm), embeds the PBMC 3k dataset
and evaluates the result. It runs on CPU in a few minutes and also shows how to compute
Geneformer embeddings for your own data.

The released weights were exported with

```bash
python scripts/export_ae_checkpoint.py --config configs/default.yaml --env <profile>
```

### On a cluster

```bash
cp environments.example.yaml environments.yaml
bash slurm/preflight.sh          # validate data, environment and partitions; submits nothing
bash slurm/submit_all.sh         # every stage, dependency-chained
```

See [`slurm/README.md`](slurm/README.md) for the job graph and single-stage submission.

## Data

Ten Visium cohorts, 120 slides, registered in `configs/datasets.json`.

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

Representations are fitted on the unannotated cohorts and evaluated on the annotated ones.
The pipeline reads cached foundation-model features under `data_root` (see
[Computing embeddings from raw data](#computing-embeddings-from-raw-data)); the `hvg_pca`
baseline and the `biosignal` stage also need the raw `.h5ad` files.

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
| `data` | Build the feature cache, describe the cohort, enumerate the folds | ~1 min |
| `train` | Fit each representation and embed all spots | minutes–hours |
| `eval` | Annotation probes, per-class F1, donor-level bootstrap CIs | ~15 min |
| `diagnose` | Effective rank, variance decomposition, CCA ceiling, scIB panel | ~1 h |
| `ablate` | Retrain with permuted morphology targets | ~15 min |
| `biosignal` | Per-gene ridge R², GSEA, per-set mean ΔR² tests | ~15 min |
| `integrate` | Harmony / BBKNN / ComBat / scVI on batch and biology; needs `train` and `eval` | ~2 h |
| `results` | Join all stages' scores into one table and check they agree | seconds |
| `figures` | Regenerate every table and figure | ~2 min |
| `embed` | Raw slides → foundation-model features (optional) | hours, GPU |

`make all` runs `data → train → eval → diagnose → figures`; the rest are opt-in. Run
`make eval integrate results` in that order.

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

`make paper-tables` builds the tables spanning all three backbones; rerun it after
changing a run's `eval`, `ablate` or `integrate` outputs.

## Computing embeddings from raw data

The pipeline reads one HuggingFace `DatasetDict` per gene-side backbone:

```
<data_root>/<combo>/merged_dataset/       dataset_dict.json, train/, test/
```

| Column | Type |
| --- | --- |
| `sample_id`, `spot_id`, `annotation`, `dataset_id` | string |
| `array_row`, `array_col` | int32 |
| `gene_features` | Array2D `(1, D_gene)` — 1152 Geneformer, 512 scGPT, 256 CancerFoundation |
| `patch_features` | Array2D `(1, 3072)` — Midnight |

If these exist, point `data_root` at them and skip this section.

### Prerequisites

1. **Raw counts**, one `<sample_id>.h5ad` per slide, at
   `<raw_data_root>/<dataset_id>/<subdirs.h5ad>` (see `configs/datasets.json`), with
   per-cohort exceptions under the profile's `raw_h5ad`.
2. **Whole-slide images** at `<data_root>/<dataset_id>/<subdirs.tif>/<sample_id>.<ext>`
   (`.tif`, `.tiff`, `.svs` or `.ndpi`). The USZ cohorts also need
   `<sample_id>_manual_loupe_alignment.json` beside the image.
3. **Checkpoints.** `scGPT_human` and the CancerFoundation weights, passed with
   `--model-dir`. Midnight and Geneformer are fetched from HuggingFace.
4. **The gene-side environment:**

   ```bash
   conda env create -f envs/scgpt.yaml            # or envs/cancerfoundation.yaml
   conda env create -f envs/geneformer.yaml
   conda run -n vgtfm-geneformer python envs/geneformer_install.py
   ```

### Running it

```bash
# 1. cut H&E patches with Midnight, write the gene-side manifest
python run.py embed --config configs/scgpt.yaml --env <profile>

# 2. embed the gene side (step 1 prints this command with your paths)
conda run -n vgtfm-scgpt python -m vgtfm.embed.gene_fm \
    --model scgpt --manifest <artifact_root>/_embed/scgpt/manifest.json \
    --out-dir <artifact_root>/_embed/scgpt --model-dir /path/to/scGPT_human

# 3. merge
python run.py embed --config configs/scgpt.yaml --env <profile>
```

On a cluster, steps 1 and 2 are `sbatch slurm/embed_midnight.sbatch` and
`sbatch slurm/embed_gene.sbatch <model>`. The merge writes
`<data_root>/vgtfm_<substrate>_midnight/merged_dataset`; point the config at it in your
site profile:

```yaml
environments:
  cluster:
    merged_datasets:
      scgpt: vgtfm_scgpt_midnight/merged_dataset
```

or per run with `SET="paths.merged_datasets.scgpt=vgtfm_scgpt_midnight/merged_dataset"`.

## Configuration

Machine-specific paths live in `environments.yaml` (untracked; start from
`environments.example.yaml`).

| Profile key | Purpose |
| --- | --- |
| `data_root` | cached merged datasets, `<combo>/merged_dataset` |
| `artifact_root` | run outputs and the shared feature cache |
| `merged_datasets` | per-substrate overrides of the merged-dataset directory names |
| `raw_data_root`, `raw_h5ad` | raw per-slide `.h5ad`, for `hvg_pca`, `biosignal` and `embed` |
| `python_env`, `hf_home` | interpreter and offline model cache for cluster jobs |
| `slurm_*` | partitions, GPU request syntax, account/QOS flags |

Select a profile with `--env <name>`, `VGTFM_ENV` or `make ... ENV=`. Precedence:
`vgtfm/config.py` defaults < `configs/<name>.yaml` < profile < `--set key=value`. Only
`data_root` is required.

## Repository layout

```
run.py              stage dispatch
configs/            one YAML per substrate, plus datasets.json and smoke.yaml
scripts/            cross-run tables, embedding checks, weight export
notebooks/          the trained model on new data
slurm/              cluster entry points
vgtfm/data/         feature cache, cohort table, folds
vgtfm/models/       representations (hvg_pca, pca, pca_oracle, ae, cdann, appendix variants)
vgtfm/evaluate/     probes, protocols, bootstrap
vgtfm/diagnostics/  effective rank, variance, CCA, scIB, integration
vgtfm/ablations/    patch-shuffle ablation
vgtfm/biosignal/    per-gene R², GSEA
vgtfm/figures/      tables and plots
vgtfm/embed/        raw slides → foundation-model features
tests/              unit, stage and smoke tests
```

## Testing

```bash
make test        # unit suite: ~1 min, no data required
make smoke       # end-to-end on a small subset of the cohort (minutes)
make embed-check # gene-side models against real checkpoints, one GPU
make lint        # ruff rules, then a formatting check
make format      # apply the formatter
```

## Citation

```bibtex
@article{McConnell2026.09.30.755585,
  author    = {McConnell, Ulysse and Nonchev, Kalin and Koelzer, Viktor H and R{\"a}tsch, Gunnar},
  title     = {Slide-level batch structure limits histology-guided supervision of transcriptomic foundation models},
  journal   = {bioRxiv},
  year      = {2026},
  elocation-id = {2026.09.30.755585},
  doi       = {10.64898/2026.09.30.755585},
  publisher = {Cold Spring Harbor Laboratory},
  URL       = {https://www.biorxiv.org/content/early/2026/10/05/2026.09.30.755585}
}
```

Per-cohort citations are in `configs/datasets.json`.
