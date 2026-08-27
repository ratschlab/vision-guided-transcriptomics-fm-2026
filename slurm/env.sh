#!/usr/bin/env bash
# Bootstrap for the cluster job scripts.
#
# There are no paths in this file. Every machine-specific location lives in
# ../environments.yaml and is read by run.py itself via `--env`. What remains
# here is only what the shell needs before python is available, plus the runtime
# tuning that belongs to the allocation rather than to the science.
# Strict mode for scripts only. Sourcing this into an interactive shell must not
# leave `set -e` armed there, or a single failing grep closes the session.
case $- in *i*) ;; *) set -euo pipefail ;; esac

# Which profile in environments.yaml the jobs run under.
export VGTFM_ENV="${VGTFM_ENV:-cluster}"

# Read one scalar out of the active profile. Used for the two settings the shell
# needs before it can call into the package (chicken-and-egg: reading the YAML
# from inside the conda env would require already knowing which env that is).
_profile_get() {
    python3 -c "
import yaml
d = yaml.safe_load(open('environments.yaml')) or {}
print((d.get('environments', {}).get('$VGTFM_ENV') or {}).get('$1', '') or '')
" 2>/dev/null || true
}

# The interpreter holding requirements.txt. Accepts three forms, so a site can use
# whichever it has (see `python_env` in environments.yaml):
#   ""                       whatever python is on PATH
#   vgtfm                    a named conda environment      -> conda run -n
#   /path/to/env             a conda prefix or a venv       -> conda run -p / bin/python
export VGTFM_PYTHON_ENV="${VGTFM_PYTHON_ENV:-$(_profile_get python_env)}"

# Offline HuggingFace cache (compute nodes usually have no network).
_hf="$(_profile_get hf_home)"
[ -n "$_hf" ] && export HF_HOME="${HF_HOME:-$_hf}"
export HF_HUB_OFFLINE=1

# Keep BLAS from oversubscribing the allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

# JAX (used by scib-metrics) stays on CPU so it does not fight torch for VRAM.
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# numba's OpenMP threading layer is not fork-safe, and says so by killing the
# child: omppool.so registers a pthread_atfork handler that prints "Terminating:
# fork() called from a process already using GNU OpenMP" and raises SIGTERM in any
# process forked after a parallel region has run. `diagnostics.isolate_methods`
# forks exactly such a process to contain bbknn, and bbknn imports pynndescent,
# which imports numba — so the guard fires on the isolation itself and the method
# is recorded as "killed by signal 15" without ever having run. `workqueue` is
# numba's own pool and carries no such handler; `tbb` is faster and also fork-safe,
# but is a separate package the site environment may not have.
export NUMBA_THREADING_LAYER="${NUMBA_THREADING_LAYER:-workqueue}"

# Resolve `python_env` to a command prefix, once per shell. Prints the decision to
# stderr when it cannot honour the profile, because a silently-substituted
# interpreter is the kind of thing that produces plausible wrong numbers.
_VGTFM_PY_CMD=()
_vgtfm_resolve_py() {
    [ ${#_VGTFM_PY_CMD[@]} -gt 0 ] && return 0
    local e="${VGTFM_PYTHON_ENV:-}"
    local first="${e%% *}"
    # An activate script, optionally followed by an environment name — exactly what
    # you would type by hand when conda is not initialised in the shell:
    #   source /path/to/conda/bin/activate myenv
    # Handled before the cases below because it needs no `conda` on PATH at all.
    if [ -n "$e" ] && [ "${first##*/}" = "activate" ]; then
        if [ -r "$first" ]; then
            _VGTFM_ACTIVATE="$e"
            _VGTFM_PY_CMD=(_vgtfm_activated_python)
        else
            echo "vgtfm: activate script '$first' is not readable;" \
                 "falling back to PATH python" >&2
            _VGTFM_PY_CMD=(python)
        fi
        return 0
    fi
    # Otherwise decided by the *shape* of the value, never by what happens to exist
    # in the working directory: this repository contains a `vgtfm/` package
    # directory, and the conda environment is also called `vgtfm`.
    case "$e" in
        "")
            _VGTFM_PY_CMD=(python)
            ;;
        /*|./*|../*|~*)
            # A path: a conda prefix (conda-meta/) or a plain venv (bin/python).
            if [ -d "$e/conda-meta" ]; then
                _VGTFM_PY_CMD=(conda run -p "$e" --no-capture-output python)
            elif [ -x "$e/bin/python" ]; then
                _VGTFM_PY_CMD=("$e/bin/python")
            else
                echo "vgtfm: python_env '$e' is not a conda prefix or a venv" \
                     "(no conda-meta/, no bin/python); falling back to PATH python" >&2
                _VGTFM_PY_CMD=(python)
            fi
            ;;
        *)
            if conda env list 2>/dev/null | grep -qE "^${e}[[:space:]]"; then
                _VGTFM_PY_CMD=(conda run -n "$e" --no-capture-output python)
            else
                echo "vgtfm: conda environment '$e' not found;" \
                     "falling back to PATH python" >&2
                _VGTFM_PY_CMD=(python)
            fi
            ;;
    esac
}

# Run python behind `source <activate> [env]`, in a subshell so the activation does
# not leak into the caller. `set +eu` because conda's activate scripts read unset
# variables and would trip the `set -euo pipefail` this file runs under.
_vgtfm_activated_python() {
    ( set +eu
      # shellcheck disable=SC1090  # unquoted: carries an optional env-name argument
      . $_VGTFM_ACTIVATE
      exec python "$@" )
}

# The interpreter the profile asked for, as a printable string (for preflight).
vgtfm_py_cmd() {
    _vgtfm_resolve_py
    if [ "${_VGTFM_PY_CMD[0]}" = "_vgtfm_activated_python" ]; then
        echo "source $_VGTFM_ACTIVATE && python"
    else
        printf '%s ' "${_VGTFM_PY_CMD[@]}"; echo
    fi
}

# Python inside the project environment.
vgtfm_py() {
    _vgtfm_resolve_py
    "${_VGTFM_PY_CMD[@]}" "$@"
}

# Read one field out of a resolved config, so the shell never has to guess what a
# config means. The second argument is an expression on the Config object `c`:
#   vgtfm_cfg configs/default.yaml data.substrate
#   vgtfm_cfg configs/default.yaml "paths.merged_datasets['geneformer']"
# A missing or malformed config must not come back as an empty string: under
# `set -o pipefail` the caller would then die at the assignment with no output at
# all. Say what went wrong, and return non-zero.
vgtfm_cfg() {
    local out err rc=0
    # stderr to a file, never merged into stdout: the interpreter resolver above
    # warns on stderr, and folding that into the value would return a config field
    # with a diagnostic glued to the front of it.
    err=$(mktemp)
    out=$(vgtfm_py -c "
from vgtfm.config import load_config
c = load_config('$1', env='$VGTFM_ENV')
print(eval('c.$2'))
" 2>"$err") || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "vgtfm_cfg: cannot read '$2' from '$1' (site profile '$VGTFM_ENV'):" >&2
        tail -3 "$err" | sed 's/^/    /' >&2
        rm -f "$err"
        return 1
    fi
    rm -f "$err"
    printf '%s' "$out" | tr -d '\r'
}

# One profile field, printed one element per line (a scalar prints as one line,
# an empty or absent field prints nothing).
vgtfm_profile() {
    vgtfm_py -c "
from vgtfm.config import environment
v = environment('$VGTFM_ENV').get('$1')
for a in (v if isinstance(v, list) else ([v] if v else [])):
    print(a)
" 2>/dev/null | tr -d '\r'
}

# Extra sbatch arguments applied to every job (account, qos, ...), one per line.
vgtfm_slurm_args() { vgtfm_profile slurm_extra_args; }

# The sbatch arguments that place a job on the right nodes, one per line.
#
#   vgtfm_slurm_res cpu   ->  --partition=compute
#   vgtfm_slurm_res gpu   ->  --partition=gpu
#                             --gres=gpu:1
#
# Both halves come from the profile because both are site-specific: partition
# names differ everywhere, and whether a GPU is requested with --gres or --gpus
# depends on the SLURM version. A profile that names no partition prints
# nothing, which is correct for a single-partition cluster.
vgtfm_slurm_res() {
    local kind="${1:-cpu}" part
    case "$kind" in
        cpu) part=$(vgtfm_profile slurm_cpu_partition) ;;
        gpu) part=$(vgtfm_profile slurm_gpu_partition) ;;
        *)   echo "vgtfm_slurm_res: expected 'cpu' or 'gpu', got '$kind'" >&2
             return 2 ;;
    esac
    [ -n "$part" ] && echo "--partition=$part"
    [ "$kind" = gpu ] && vgtfm_profile slurm_gpu_args
    return 0
}

# Run one stage under the project environment and the active site profile.
#
# Usage: vgtfm_run <stage> <config> [key=value ...]
#
# Extra overrides are BARE key=value tokens, not a second `--set`: run.py declares
# `--set` with nargs="*", so a repeated flag replaces the earlier group instead of
# extending it. The guard turns that mistake into an error.
vgtfm_run() {
    local stage="$1" config="$2"; shift 2
    local extra
    for extra in "$@"; do
        case "$extra" in
            --*) echo "vgtfm_run: pass overrides as bare key=value, not '$extra'" >&2
                 return 2 ;;
        esac
    done
    if [ $# -gt 0 ]; then
        vgtfm_py run.py "$stage" --config "$config" --env "$VGTFM_ENV" --set "$@"
    else
        vgtfm_py run.py "$stage" --config "$config" --env "$VGTFM_ENV"
    fi
}
