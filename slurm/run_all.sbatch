#!/usr/bin/env bash
#SBATCH --job-name=vgtfm
#SBATCH --output=logs/vgtfm-%j.out
#SBATCH --error=logs/vgtfm-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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
vgtfm_run all "${1:-configs/default.yaml}"
