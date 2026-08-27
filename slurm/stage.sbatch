#!/usr/bin/env bash
#SBATCH --job-name=vgtfm-stage
#SBATCH --output=logs/vgtfm-%x-%j.out
#SBATCH --error=logs/vgtfm-%x-%j.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#
# Run any single stage.
#
# Deliberately requests no partition and no GPU: those are site-specific, they
# live in environments.yaml, and a #SBATCH line here could not be cancelled by a
# submitter that wanted a CPU-only node. Submit through the wrappers, which read
# the profile and add the right placement:
#
#   bash slurm/submit.sh biosignal configs/default.yaml
#   bash slurm/submit.sh train configs/default.yaml models.pca_components=50
#   bash slurm/submit_all.sh                      # the whole graph
#
# A bare `sbatch slurm/stage.sbatch <stage>` runs CPU-only on the default
# partition, which on most clusters is not what you want.
#
# Extra overrides are BARE key=value tokens (see vgtfm_run in env.sh).
set -euo pipefail
# Under sbatch, $0 is a copy of this script in /var/spool, not the file in the
# checkout: `cd "$(dirname "$0")/.."` lands in the spool directory and the source
# below fails with "slurm/env.sh: No such file or directory". A batch job already
# starts in the directory it was submitted from, and every wrapper here submits
# from the repo root; $0 is only usable when the file is run directly.
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
STAGE="${1:?usage: sbatch slurm/stage.sbatch <stage> [config] [key=value ...]}"
CONFIG="${2:-configs/default.yaml}"
shift 2 || shift 1
vgtfm_run "$STAGE" "$CONFIG" "$@"
