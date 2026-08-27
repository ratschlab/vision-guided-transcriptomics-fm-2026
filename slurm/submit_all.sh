#!/usr/bin/env bash
# Submit the whole reproduction as one dependency-chained set of jobs.
#
#   bash slurm/submit_all.sh                          # the three paper substrates
#   VGTFM_ENV=cluster bash slurm/submit_all.sh        # pick the site profile
#   bash slurm/submit_all.sh configs/default.yaml     # just one
#   bash slurm/submit_all.sh --core configs/*.yaml    # skip the opt-in stages
#   bash slurm/submit_all.sh --dry-run                # print, submit nothing
#   bash slurm/submit_all.sh diagnostics.run_scib=false   # override, every stage
#
# A bare key=value argument is a config override passed to every job. Use it for a
# deliberate omission — the value lands in each run's config.resolved.json, so a
# figure always carries the settings it was built under.
#
# Shape of the graph, per substrate:
#
#   data ──┬── all (train,eval,diagnose,figures) ──┬── biosignal ──┐
#          │                                        ├── integrate ──┤
#          │                                        └───────────────┤
#          └── ablate ─────────────────────────────────────────────┴── figures ── results
#
# `biosignal` and `integrate` hang off `all`, not off `data`, because both read what
# `all` writes. `biosignal` loads the trained autoencoder embedding; `integrate`
# corrects the `pca` baseline `train` fitted and reuses the prediction vector `eval`
# cached for it, rather than refitting a PCA of its own over the whole cohort — which
# is what once put a transductive basis behind the integration table's uncorrected
# row while Table 1 reported an inductive one. Given only `data` it now starts hours
# too early and exits with "cannot produce the uncorrected reference embedding".
# `ablate` fits its own autoencoders, so it still needs nothing beyond `data`.
#
# `results` is last: it joins every stage of a run into one frame and fails the run
# if two of them computed one quantity and got two answers.
#
# `data` is split out and shared: the feature cache lives at
# <artifact_root>/_cache/<substrate>/ and is shared by every run using that
# substrate, so building it once up front stops parallel jobs racing to write it.
# The final `figures` pass is what puts Table 3 and Figure 2 in place: the figures
# stage lists artefacts whose stage has not run yet as pending rather than failing
# on them, and builds them on a later pass. It waits on the opt-in stages with
# `afterany`, so one of them failing costs its own figure and not the whole pass.
#
# On top of the per-substrate graph, one `regression_check` job reproduces the
# published protocol as a guard against the reproduction drifting.
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/env.sh

CORE_ONLY=0; DRY=0; CONFIGS=(); OVERRIDES=()
for a in "$@"; do
    case "$a" in
        --core)    CORE_ONLY=1 ;;
        --dry-run) DRY=1 ;;
        -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
        --*)       echo "unknown option '$a'" >&2; exit 2 ;;
        *=*)       OVERRIDES+=("$a") ;;
        *)         CONFIGS+=("$a") ;;
    esac
done
if [ ${#CONFIGS[@]} -eq 0 ]; then
    CONFIGS=(configs/default.yaml configs/scgpt.yaml configs/cancerfoundation.yaml)
fi

mkdir -p logs

# Partition / account / QOS come from the profile, not from the job scripts.
EXTRA_SBATCH=()
while IFS= read -r a; do [ -n "$a" ] && EXTRA_SBATCH+=("$a"); done < <(vgtfm_slurm_args)
CPU_SBATCH=(); GPU_SBATCH=()
while IFS= read -r a; do [ -n "$a" ] && CPU_SBATCH+=("$a"); done < <(vgtfm_slurm_res cpu)
while IFS= read -r a; do [ -n "$a" ] && GPU_SBATCH+=("$a"); done < <(vgtfm_slurm_res gpu)

# submit <name> <dependency-or-empty> <cpu|gpu> <resources...> -- <stage args...>
#
# The cpu/gpu kind picks the partition and the GPU request out of the profile.
# Naming the partition is not optional on every cluster: some sites reject a job
# with no --partition as "Requested node configuration is not available".
submit() {
    local name="$1" dep="$2" kind="$3"; shift 3
    local res=(); while [ "$1" != "--" ]; do res+=("$1"); shift; done; shift
    local place=(); case "$kind" in
        cpu) place=(${CPU_SBATCH[@]+"${CPU_SBATCH[@]}"}) ;;
        gpu) place=(${GPU_SBATCH[@]+"${GPU_SBATCH[@]}"}) ;;
    esac
    local args=(--job-name="$name" --output="logs/${name}-%j.out"
                --error="logs/${name}-%j.err" "${res[@]}"
                ${place[@]+"${place[@]}"}
                ${EXTRA_SBATCH[@]+"${EXTRA_SBATCH[@]}"})
    # A dependency that already names its type is used verbatim; a bare job-id list
    # means `afterok`, which is what every link in the required chain wants.
    if [ -n "$dep" ]; then
        case "$dep" in
            after*) args+=(--dependency="$dep" --kill-on-invalid-dep=yes) ;;
            *)      args+=(--dependency="afterok:${dep}" --kill-on-invalid-dep=yes) ;;
        esac
    fi
    if [ "$DRY" -eq 1 ]; then
        echo "  sbatch ${args[*]} slurm/stage.sbatch $* ${OVERRIDES[*]-}" >&2
        echo "DRY"
    else
        sbatch --parsable "${args[@]}" slurm/stage.sbatch "$@" \
               ${OVERRIDES[@]+"${OVERRIDES[@]}"}
    fi
}

SKIPPED=()                      # configs that could not be submitted, and why

# Resolve every config before submitting any of it. A half-submitted graph is
# worse than one that was never submitted: the jobs that did go in run to
# completion and the missing substrate looks like a substrate nobody asked for.
# `vgtfm_cfg` prints why a config would not load.
USABLE=()
for cfg in "${CONFIGS[@]}"; do
    if [ ! -f "$cfg" ]; then
        SKIPPED+=("$cfg: no such file"); continue
    fi
    if ! vgtfm_cfg "$cfg" run_name >/dev/null; then
        SKIPPED+=("$cfg: config did not resolve (see above)"); continue
    fi
    USABLE+=("$cfg")
done
if [ ${#USABLE[@]} -eq 0 ]; then
    echo "no usable config among: ${CONFIGS[*]}" >&2
    for s in ${SKIPPED[@]+"${SKIPPED[@]}"}; do echo "  $s" >&2; done
    exit 1
fi

ART_ROOT=$(vgtfm_cfg "${USABLE[0]}" paths.artifact_root)
echo "site profile '$VGTFM_ENV' -> artifacts under $ART_ROOT"
echo

declare -A DATA_JOB=()          # substrate -> job id, so the cache is built once
ALL_IDS=()

for cfg in "${USABLE[@]}"; do
    sub=$(vgtfm_cfg "$cfg" data.substrate)
    run=$(vgtfm_cfg "$cfg" run_name)
    src=$(vgtfm_cfg "$cfg" "data_path()")
    if [ ! -f "$src/dataset_dict.json" ]; then
        SKIPPED+=("$cfg [$sub]: no merged dataset at $src")
        continue
    fi

    echo "== $cfg  (run=$run, substrate=$sub)"

    # One data job per substrate. No GPU: it is a pure disk-to-disk extraction.
    if [ -z "${DATA_JOB[$sub]:-}" ]; then
        DATA_JOB[$sub]=$(submit "vgtfm-data-$sub" "" cpu \
            --time=4:00:00 --cpus-per-task=8 --mem=96G -- data "$cfg")
        echo "   data       ${DATA_JOB[$sub]}"
    else
        echo "   data       ${DATA_JOB[$sub]} (shared, already submitted)"
    fi
    d="${DATA_JOB[$sub]}"

    optional=()
    allj=$(submit "vgtfm-all-$run" "$d" gpu \
        --time=12:00:00 --cpus-per-task=8 --mem=64G -- all "$cfg")
    echo "   all        $allj"

    if [ "$CORE_ONLY" -eq 0 ]; then
        # Four conditions x three seeds = twelve autoencoder fits.
        id=$(submit "vgtfm-ablate-$run" "$d" gpu \
            --time=24:00:00 --cpus-per-task=8 --mem=64G -- ablate "$cfg")
        echo "   ablate     $id"; optional+=("$id")

        # After `all`, which is the job that runs `train`. One job per organ scope:
        # only TuPro carries replicate and region ids, so `cross_replicate` and
        # `cross_region` are skin whatever the cohort is. Scoring every organ into
        # one pooled R^2 made `cross_donor` a different cohort from the two levels
        # above it, and pooled fold residuals against a grand mean no fold saw.
        # skin is the headline (7 donors) and the one the figures read; lung (4
        # donors) is the generalisation check; kidney (2) is an anecdote and is
        # labelled one wherever it is quoted.
        for scope in skin lung kidney; do
            id=$(submit "vgtfm-biosignal-$scope-$run" "$allj" gpu \
                --time=8:00:00 --cpus-per-task=8 --mem=96G -- biosignal "$cfg" \
                biosignal.tissues="$scope")
            echo "   biosignal  $id  ($scope)"; optional+=("$id")
        done

        # After `all`: `integrate` loads train's `pca` embedding and eval's cached
        # predictions for it. See the graph comment at the top.
        #
        # 64G is enough only because `_scvi_embedding` releases the per-slide
        # count cache and keeps one copy of the cohort expression matrix; before
        # that, three seeds in one process OOM-killed this stage at 96G on all
        # three substrates. Keep in step with `_defaults` in submit.sh, which
        # carries the reasoning and the per-GPU memory these nodes are built with.
        id=$(submit "vgtfm-integrate-$run" "$allj" gpu \
            --time=8:00:00 --cpus-per-task=8 --mem=64G -- integrate "$cfg")
        echo "   integrate  $id"; optional+=("$id")
    fi

    # `afterok` on the required chain, `afterany` on the opt-in stages.
    #
    # The asymmetry matters: `afterok` on every leaf means one failed opt-in stage
    # makes the dependency unsatisfiable, and --kill-on-invalid-dep then cancels the
    # figures pass outright — losing the figures of the stages that did succeed. The
    # figures stage treats a missing artefact as pending rather than as an error,
    # which is the contract `afterany` needs. `all` stays `afterok`: without eval
    # output there is no table to draw.
    dep="afterok:$allj"
    if [ ${#optional[@]} -gt 0 ]; then
        dep="$dep,afterany:$(IFS=:; echo "${optional[*]}")"
    fi
    id=$(submit "vgtfm-figures-$run" "$dep" cpu \
        --time=1:00:00 --cpus-per-task=4 --mem=32G -- figures "$cfg")
    echo "   figures    $id"
    ALL_IDS+=("$id")

    # The cross-stage consistency check, on the same `afterany` dependency: it reads
    # artefacts rather than producing any, and a stage that failed simply is not in
    # the frame. Cheap, and it is what says whether two stages of this run computed
    # one quantity twice and disagreed — the failure `scripts/paper_tables.py`
    # would otherwise only meet at the point of writing the manuscript.
    id=$(submit "vgtfm-results-$run" "$dep" cpu \
        --time=0:30:00 --cpus-per-task=2 --mem=32G -- results "$cfg")
    echo "   results    $id"
    ALL_IDS+=("$id")
done

# The regression check: one seed, cross-donor only, reproducing the *published*
# protocol. It is what certifies that the corrected protocol is what moved the
# numbers rather than a difference in reimplementation, so it belongs in every
# batch and not in a drawer. Independent of the substrate loop above.
if [ "$CORE_ONLY" -eq 0 ] && [ -f configs/regression_check.yaml ]; then
    rsub=$(vgtfm_cfg configs/regression_check.yaml data.substrate)
    if [ -n "${DATA_JOB[$rsub]:-}" ]; then
        id=$(submit "vgtfm-regression-check" "${DATA_JOB[$rsub]}" gpu \
            --time=4:00:00 --cpus-per-task=8 --mem=64G -- all \
            configs/regression_check.yaml)
        echo "== configs/regression_check.yaml  (substrate=$rsub)"
        echo "   all        $id"
        ALL_IDS+=("$id")
    else
        SKIPPED+=("configs/regression_check.yaml: no run uses substrate '$rsub', "\
"so its data cache is not built by this batch")
    fi
fi

echo
# A config that could not be submitted is the failure this script exists to
# prevent: the graph looks submitted, one substrate is simply absent from it, and
# nothing says so until the paper is missing a column. Report and exit non-zero.
if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo "NOT SUBMITTED (${#SKIPPED[@]} of ${#CONFIGS[@]} configs):" >&2
    for s in "${SKIPPED[@]}"; do echo "  $s" >&2; done
    echo "  check data_root in the '$VGTFM_ENV' profile of environments.yaml" >&2
fi
if [ ${#ALL_IDS[@]} -eq 0 ]; then
    echo "nothing was submitted" >&2
    exit 1
fi

if [ "$DRY" -eq 1 ]; then
    echo "dry run — nothing submitted"
else
    echo "submitted. watch:   squeue -u \$USER -o '%.10i %.24j %.9T %.10M %R'"
    echo "results under:      $ART_ROOT/<run_name>/figures/"
    echo "per-run manifest:   $ART_ROOT/<run_name>/logs/manifest-*.json"
fi
[ ${#SKIPPED[@]} -eq 0 ]
