# Cluster runs

```bash
cp environments.example.yaml environments.yaml   # the site profile: every path lives here
bash slurm/preflight.sh          # cheap, read-only, safe on a login node
bash slurm/submit_all.sh         # the whole reproduction, dependency-chained
```

`preflight.sh` answers "will this run?" before anything costs an allocation: conda
environment, writable artifact root, that each config's merged dataset exists and
carries the eight required columns, whether the feature cache is already built, and
whether the raw counts `biosignal` needs are reachable. It exits non-zero if not.

## Where the paths are

All of them are in `../environments.yaml`, as named profiles. That file is not
tracked, so a checkout never carries one site's layout to another — start from
[`../environments.example.yaml`](../environments.example.yaml). Nothing under `slurm/`
contains a path: the scripts resolve every location by asking the pipeline's own config
code, so what they check is what the run will use.

| Profile key | Feeds | Notes |
| --- | --- | --- |
| `data_root` | `paths.data_root` | holds `<combo>/merged_dataset` — the tokenizer output |
| `artifact_root` | `paths.artifact_root` | run outputs *and* `_cache/<substrate>/` |
| `raw_data_root` | `paths.raw_data_root` | base for raw cohorts; empty = `data_root` |
| `raw_h5ad` | `paths.raw_h5ad` | per-cohort exceptions, absolute; merged into the defaults |
| `datasets_json` | `paths.datasets_json` | cohort registry, if not the shipped one |
| `python_env` | `slurm/env.sh` | interpreter holding `requirements.txt` — name, conda prefix, or venv path |
| `hf_home` | `$HF_HOME` | offline model cache for compute nodes |
| `slurm_cpu_partition` | CPU jobs | `--partition=` for `data`, `diagnose`, `figures` |
| `slurm_gpu_partition` | GPU jobs | `--partition=` for everything else |
| `slurm_gpu_args` | GPU jobs | how this site asks for a GPU — `--gres=gpu:1` or `--gpus=1` |
| `slurm_extra_args` | every `sbatch` | account, QOS, nice |

Select a profile with `--env cluster`, `VGTFM_ENV=cluster`, or `make ... ENV=cluster`.
Precedence runs dataclass defaults < `configs/<name>.yaml` < the profile < `--set`, so
a one-off override on the command line always wins. The active profile name is written
into `config.resolved.json` and each run manifest, so a figure records the machine
layout that produced it.

Adding a site means copying the commented template at the bottom of the example. A profile
only states what differs from the defaults; `raw_h5ad` and `merged_datasets` merge into
them per key rather than replacing them wholesale.

`slurm/env.sh` is only a bootstrap: it picks the profile name, finds the interpreter
(the one thing needed before python can read the YAML), exports `HF_HOME`, and pins
the BLAS/JAX thread counts to the allocation.

### `python_env` takes a path, not just a name

| Value | Resolved as |
| --- | --- |
| `""` | whatever `python` is on `PATH` |
| `vgtfm` | `conda run -n vgtfm` |
| `/path/to/conda/envs/vgtfm` | `conda run -p /path/...` (detected by `conda-meta/`) |
| `/path/to/venv` | `/path/to/venv/bin/python` directly — no conda needed |
| `/path/to/conda/bin/activate myenv` | `source`d in a subshell, then `python` |

The last form is the one to reach for on a compute node: unlike `conda run` it needs
no `conda` on `PATH`, which batch shells frequently lack. It is detected by the first
token's basename being `activate`, and sourced in a subshell under `set +eu` (conda's
activate scripts read unset variables, and this file runs under `set -euo pipefail`).

Otherwise the form is decided by shape — a leading `/`, `./`, `../` or `~` means a path
— never by what exists in the working directory: the conda environment and this
repository's package directory are both called `vgtfm`, so a `-d` test would misread
the name.

If the profile names an interpreter that cannot be found, the scripts warn on stderr
and fall back to `PATH` python rather than substituting one silently, and
`preflight.sh` turns that warning into a `FAIL`.

`preflight.sh` also checks each pinned dependency by distribution name and version,
and reports which *stage* a gap blocks.

Override for one run without editing the file: `VGTFM_PYTHON_ENV=/path/to/venv`.

## There is no conversion step

The pipeline reads the embeddings exactly as the embedding stage wrote them. The keys
of `PathsConfig.merged_datasets` in `vgtfm/config.py` are the directory names of the
tokenizer output:

```
<data_root>/                              <- tokenization_results/
  none_midnight_geneformer_none_none/
    merged_dataset/                       <- a HuggingFace DatasetDict
      dataset_dict.json
      train/ validation/ test/
  none_midnight_scgpt_merged_none_none/merged_dataset/
  none_midnight_cancerfoundation_merged_none_none/merged_dataset/
```

So pointing `paths.data_root` at that directory is the entire setup. Nothing is
copied, rewritten or re-embedded, and the `embed` stage stays unused unless you are
regenerating features from raw slides.

The required columns, checked by `preflight.sh` and enforced in
`vgtfm/data/tables.py`:

| column | type |
| --- | --- |
| `sample_id`, `spot_id`, `annotation`, `dataset_id` | string |
| `array_row`, `array_col` | int32 |
| `gene_features` | Array2D `(1, D_gene)` — 1152 for Geneformer |
| `patch_features` | Array2D `(1, 3072)` — Midnight |

## What the "shared feature cache" is, and why you never build it by hand

The merged dataset is an Arrow store of ~328k rows, each holding a 1152-d and a
3072-d nested array. Reading a column out of it row-by-row is slow, and every stage
needs the whole matrix. So the first `data` stage materialises it once:

```
<artifact_root>/_cache/<substrate>/
  gene_features.npy      float32 (n_spots, D_gene)     memory-mapped
  patch_features.npy     float32 (n_spots, 3072)       memory-mapped
  meta.parquet           sample_id, spot_id, annotation, donor, tissue, split, ...
  cache_info.json        substrate, source path, dims, spot count
```

It lives **outside** the per-run directory because the extraction is a pure function
of (merged dataset, data filters) — so all runs on one substrate share it, and a
`run_name` change costs nothing. It is 4-5 GB per substrate.

Three things follow, and they are the reason `submit_all.sh` looks the way it does:

* **You never invoke a cache build.** Any stage calling `tables.load` builds it if
  absent and reuses it otherwise (`cache hit: ...` in the log).
* **Cold parallel jobs race.** Four jobs starting at once on an unbuilt cache all
  write the same `.npy`. `submit_all.sh` therefore submits one `data` job per
  substrate first and hangs everything else off it with `--dependency=afterok`.
* **It is keyed on the source path.** `cache_info.json` records the absolute path it
  was built from; copying a cache from a laptop invalidates it and it rebuilds. Do
  not rsync `artifacts/` between machines — it is regenerable.

## The job graph

`submit_all.sh` submits this per substrate, then a final `figures` pass:

```
data ──┬── all (train → eval → diagnose → figures) ──┬── biosignal ──┐
       │                                             ├── integrate ──┤
       │                                             └──────────────┤
       └── ablate ──────────────────────────────────────────────────┴── figures ── results
```

`ablate` depends only on the cache — it fits its own autoencoders — so it runs in
parallel with the main job. `biosignal` and `integrate` do not. `biosignal` reads the
trained autoencoder embedding; `integrate` corrects the `pca` baseline `train` fitted
and reuses the prediction vector `eval` cached for it, rather than fitting a PCA of
its own over the whole cohort. Both therefore wait for `all`, the only job that runs
`train` and `eval`. The trailing `figures` job is not redundant: the figures stage
reports an artefact whose stage has not run as pending, so Table 3 (patch shuffle)
and Figure 2 (per-gene R²) only appear on a pass that runs after `ablate` and
`biosignal` have landed.

`results` is the last job: it joins every stage of the run into one long-format table
and fails if two of them computed one quantity and got two answers, or if a
correspondence that should have been checked was never exercised at all. It reads
artefacts and writes no input to anything, so it hangs off the same `afterany`
dependency as `figures`.

`submit_all.sh` exits non-zero if any config it was given did not reach the queue,
and prints which and why. A half-submitted graph is the failure worth catching: the
jobs that did go in run to completion, so the missing substrate looks like one nobody
asked for.

```bash
bash slurm/submit_all.sh --dry-run     # print the sbatch lines, submit nothing
bash slurm/submit_all.sh --core        # skip ablate/biosignal/integrate
bash slurm/submit_all.sh configs/regression_check.yaml --core
```

## Single stages

```bash
bash slurm/submit.sh <stage> [config] [key=value ...] [--sbatch-flag ...]

bash slurm/submit.sh biosignal configs/default.yaml
bash slurm/submit.sh train configs/default.yaml \
       models.names=pca,pca_oracle,ae,cdann,dual_decoder,gene_ae,infonce,jepa
bash slurm/submit.sh integrate --mem=256G --time=16:00:00   # flags win over the defaults
bash slurm/submit.sh data --dry-run
```

`submit.sh` knows which stages need a GPU and picks walltime, cpus and memory per
stage; anything starting with `--` is handed to `sbatch` verbatim and, coming last,
overrides those defaults. Use `--cpu` / `--gpu` to force the placement.

Do not run `sbatch slurm/stage.sbatch` directly. It intentionally carries no
`--partition` and no GPU request — see below.

Extra overrides are **bare `key=value` tokens**, never a second `--set`: `run.py`
declares `--set` with `nargs="*"`, so a repeated flag replaces the earlier group and
would silently drop every path in `env.sh`. `vgtfm_run` rejects arguments starting
with `--` for that reason.

## Three things worth knowing before submitting

**A job needs a partition, and the scripts do not hardcode one.** Submitting with
no `--partition` on a cluster whose default partition cannot host the request gets
you exactly one line:

```
sbatch: error: Batch job submission failed: Requested node configuration is not available
```

which names neither the partition nor the resource at fault. `submit.sh` and
`submit_all.sh` read `slurm_cpu_partition` / `slurm_gpu_partition` /
`slurm_gpu_args` out of the profile and pass them per job, so CPU-only stages never
sit on a GPU node and GPU stages ask for the GPU the way this SLURM expects — a
SLURM older than 19.05 wants `--gres=gpu:1`, newer installations take `--gpus=1`. `stage.sbatch` therefore
holds no `#SBATCH --partition` or `--gres` line at all: a directive there could not
be cancelled by a submitter that wanted a CPU node. `preflight.sh` checks the named
partitions against `sinfo` and prints the largest node in each.

**Submit from the repository root.** SLURM runs a *copy* of the batch script out of
`/var/spool` on the compute node, so `$0` inside a job says nothing about where the
checkout is; the scripts use the directory you submitted from (`SLURM_SUBMIT_DIR`)
and refuse with a readable message if that is not a vgtfm checkout. It is also where
`#SBATCH --output=logs/...` is resolved, so `logs/` has to exist there or the job
dies at launch with nowhere to report why — `logs/.gitkeep` is committed,
`submit_all.sh` creates it, and `preflight.sh` checks it.

**`biosignal` needs raw counts, which are not under `data_root`.** `data_root` is the
tokenizer output; the source cohorts are elsewhere, across two unrelated roots. The
`cluster` profile sets `raw_data_root` for the common case and lists the exceptions
under `raw_h5ad` (a cohort on a different filesystem, or one whose directory is
spelled differently from its dataset id, as `MOSAIC_DLBLC`'s is). Without these the stage still runs, but silently covers
fewer cohorts — `preflight.sh` counts the `.h5ad` files it can actually see.

## The `embed` jobs

Only for regenerating features from raw slides; the merged datasets above already
contain the published ones.

```bash
sbatch slurm/embed_midnight.sbatch          # H&E, GPU, once per cohort
sbatch slurm/embed_gene.sbatch geneformer   # gene side, in envs/geneformer.yaml
```

## Which code produced this run?

Compare fingerprints. Every run's banner starts with one and `preflight.sh` prints
it, so the laptop and the cluster can be checked against each other in one line:

```bash
bash slurm/preflight.sh | grep fingerprint      # on both machines
head -1 logs/vgtfm-all-*.out                    # or read it off a job's banner
```

Same twelve characters, same code. Different, and one copy is stale however recent
its timestamps look — resync before reading any traceback from it, since line numbers
in a stale checkout point at code that has since moved.

## Checking on a run

```bash
squeue -u $USER -o '%.10i %.24j %.9T %.10M %R'
cat <artifact_root>/<run_name>/logs/manifest-*.json   # host, device, env, timing, status
```

`run.py` records a failed stage in the manifest rather than exiting quietly, so the
manifest is the first thing to read when a job comes back.
