#!/usr/bin/env bash
#SBATCH --job-name=vgtfm-gene
#SBATCH --output=logs/vgtfm-gene-%j.out
#SBATCH --error=logs/vgtfm-gene-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#
# One gene-side foundation model over every registered slide. The model runs in its
# own conda environment (envs/<model>.yaml) because their dependency pins conflict
# with each other and with requirements.txt; they communicate with the rest of the
# pipeline only through parquet files.
#
#   sbatch slurm/embed_gene.sbatch geneformer
#
# Optional: the merged datasets already contain the published features. This is for
# regenerating them from raw slides.
set -euo pipefail
# Under sbatch, $0 is a /var/spool copy and `dirname "$0"` lands outside the
# checkout; a batch job starts where it was submitted. See stage.sbatch.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    cd "${SLURM_SUBMIT_DIR:-.}"
else
    cd "$(dirname "$0")/.."
fi
[ -r slurm/env.sh ] || {
    echo "$(basename "$0"): $PWD is not a vgtfm checkout;" \
         "submit from the repo root, or use slurm/submit.sh" >&2
    exit 2
}
source slurm/env.sh

MODEL="${1:?usage: sbatch embed_gene.sbatch <geneformer|scgpt|cancerfoundation|scvi>}"
CONFIG="${2:-configs/default.yaml}"
ENV_NAME="vgtfm-${MODEL}"

# Enumerate slides with the same path resolution the pipeline uses, so the raw
# cohort locations come from the active site profile rather than from this script.
SLIDES=$(vgtfm_py - "$CONFIG" "$VGTFM_ENV" <<'PY'
import json, sys
from vgtfm.config import load_config

cfg = load_config(sys.argv[1], env=sys.argv[2])
registry = json.load(open(cfg.paths.datasets_json))["datasets"]
for dataset_id, spec in registry.items():
    if dataset_id in cfg.paths.raw_h5ad:
        base = cfg.raw_h5ad_dir(dataset_id)
    else:
        sub = (spec.get("subdirs") or {}).get("h5ad")
        if not sub:
            continue
        base = cfg.raw_root() / dataset_id / sub
    for sample in spec.get("samples", []):
        print(dataset_id, sample["id"], base)
PY
)

ART_ROOT=$(vgtfm_cfg "$CONFIG" paths.artifact_root)

while read -r dataset_id sample_id base; do
    [ -z "$dataset_id" ] && continue
    h5ad="$base/$sample_id.h5ad"
    out="$ART_ROOT/_embed/$MODEL/$dataset_id/$sample_id.parquet"
    [ -f "$h5ad" ] || { echo "skip $dataset_id/$sample_id (no h5ad)"; continue; }
    [ -f "$out" ]  && { echo "skip $dataset_id/$sample_id (done)"; continue; }
    mkdir -p "$(dirname "$out")"
    conda run -n "$ENV_NAME" --no-capture-output \
        python -m vgtfm.embed.gene_fm --model "$MODEL" --h5ad "$h5ad" --out "$out"
done <<< "$SLIDES"
