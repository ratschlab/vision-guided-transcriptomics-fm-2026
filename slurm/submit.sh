#!/usr/bin/env bash
# Submit ONE stage, with the placement and resources that stage actually needs.
#
#   bash slurm/submit.sh biosignal
#   bash slurm/submit.sh train configs/scgpt.yaml
#   bash slurm/submit.sh eval configs/default.yaml models.pca_components=50
#   bash slurm/submit.sh integrate --mem=256G --time=16:00:00
#   bash slurm/submit.sh data --dry-run
#
# This is the wrapper `sbatch slurm/stage.sbatch` deliberately is not: stage.sbatch
# carries no --partition and no --gres, because those are site-specific. They come
# from the active profile in environments.yaml (slurm_cpu_partition,
# slurm_gpu_partition, slurm_gpu_args), which is also what stops the
#
#     sbatch: error: Batch job submission failed:
#             Requested node configuration is not available
#
# you get on such a site from submitting with no partition at all.
#
# Arguments:
#   <stage>            one of the run.py stages, or `all`
#   [config]           a path ending in .yaml       (default configs/default.yaml)
#   [key=value ...]    config overrides, passed through to run.py --set
#   [--anything]       passed straight to sbatch, and wins over the defaults below
#   --cpu | --gpu      force the placement instead of using the table
#   --dry-run          print the sbatch command, submit nothing
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/env.sh

# stage -> placement, walltime, cpus, memory. Times are generous; a job that ends
# early costs nothing, one that hits the limit at hour 11 costs the whole run.
#
# `integrate` fits scVI once per seed in a single process, and each fit touches the
# whole cohort's expression matrix (~14G at 328k spots x 11k genes). It used to hold
# that several times over plus a per-slide dense count cache, and three seeds of it
# OOM-killed the stage on every substrate at 96G -- `sacct` put the last sampled RSS
# at 68-84 GiB with the cache still resident. `_scvi_embedding` now releases the
# cache and keeps one copy of the matrix, which puts the peak nearer 20G, so 64G
# here is headroom over the fixed behaviour and not over the old one.
#
# Not more than that on purpose: these GPU nodes carry roughly 60G of RAM per GPU
# (484G/8, 240G/4, 114G/2), so a single-GPU job asking much above 64G competes for
# memory with the other GPUs on the same node and waits for it.
#
# Changing any figure below: preflight.sh checks the partitions against the largest
# --mem asked for on each placement, and restates those two numbers. Keep it in step.
_defaults() {
    case "$1" in
        data)      echo "cpu 4:00:00 8 96G"  ;;
        embed)     echo "gpu 24:00:00 8 128G" ;;
        train)     echo "gpu 12:00:00 8 64G"  ;;
        eval)      echo "gpu 8:00:00 8 64G"   ;;
        diagnose)  echo "cpu 8:00:00 8 96G"   ;;
        integrate) echo "gpu 8:00:00 8 64G"   ;;
        ablate)    echo "gpu 24:00:00 8 64G"  ;;
        biosignal) echo "gpu 8:00:00 8 96G"   ;;
        figures)   echo "cpu 1:00:00 4 32G"   ;;
        results)   echo "cpu 0:30:00 2 32G"   ;;
        all)       echo "gpu 12:00:00 8 64G"  ;;
        *)         return 1 ;;
    esac
}

STAGE="${1:-}"
[ -n "$STAGE" ] || { sed -n '2,25p' "$0"; exit 2; }
shift

_spec=$(_defaults "$STAGE" || true)
if [ -z "$_spec" ]; then
    echo "submit.sh: unknown stage '$STAGE'" >&2
    echo "  known: data embed train eval diagnose integrate results" >&2
    echo "         ablate biosignal figures all" >&2
    exit 2
fi
read -r KIND WALL CPUS MEM <<<"$_spec"

CONFIG=configs/default.yaml
DRY=0
PASSTHRU=()      # extra sbatch flags, appended last so they override the defaults
OVERRIDES=()     # bare key=value, forwarded to run.py
for a in "$@"; do
    case "$a" in
        --dry-run) DRY=1 ;;
        --cpu)     KIND=cpu ;;
        --gpu)     KIND=gpu ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        --*)       PASSTHRU+=("$a") ;;
        *.yaml|*.yml) CONFIG="$a" ;;
        *=*)       OVERRIDES+=("$a") ;;
        *)         echo "submit.sh: don't know what to do with '$a'" >&2; exit 2 ;;
    esac
done
[ -f "$CONFIG" ] || { echo "submit.sh: no such config '$CONFIG'" >&2; exit 2; }

mkdir -p logs

PLACE=()
while IFS= read -r a; do [ -n "$a" ] && PLACE+=("$a"); done < <(vgtfm_slurm_res "$KIND")
EXTRA=()
while IFS= read -r a; do [ -n "$a" ] && EXTRA+=("$a"); done < <(vgtfm_slurm_args)

run=$(vgtfm_cfg "$CONFIG" run_name)
name="vgtfm-${STAGE}-${run:-run}"

CMD=(sbatch
     --job-name="$name"
     --output="logs/${name}-%j.out"
     --error="logs/${name}-%j.err"
     --time="$WALL" --cpus-per-task="$CPUS" --mem="$MEM"
     ${PLACE[@]+"${PLACE[@]}"}
     ${EXTRA[@]+"${EXTRA[@]}"}
     ${PASSTHRU[@]+"${PASSTHRU[@]}"}
     slurm/stage.sbatch "$STAGE" "$CONFIG" ${OVERRIDES[@]+"${OVERRIDES[@]}"})

echo "profile '$VGTFM_ENV', $KIND job: ${PLACE[*]:-(no partition named)}"
if [ "$DRY" -eq 1 ]; then
    printf '%q ' "${CMD[@]}"; echo
    echo "dry run — nothing submitted"
else
    "${CMD[@]}"
fi
