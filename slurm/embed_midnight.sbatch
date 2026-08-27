#!/usr/bin/env bash
#SBATCH --job-name=vgtfm-midnight
#SBATCH --output=logs/vgtfm-midnight-%j.out
#SBATCH --error=logs/vgtfm-midnight-%j.err
# Whole-slide images are large; the memory request is for the image, not the model.
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
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
vgtfm_run embed "${1:-configs/default.yaml}"
