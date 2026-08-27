#!/usr/bin/env bash
# Check everything a run needs, before any of it costs an allocation.
# Cheap and read-only: safe on a login node.
#
#   bash slurm/preflight.sh [config ...]          (default: configs/default.yaml)
#   VGTFM_ENV=local bash slurm/preflight.sh
#
# Every path below comes from the active profile in environments.yaml, resolved by
# the same code the pipeline uses — so this checks the real thing, not a copy of it.
cd "$(dirname "$0")/.."
source slurm/env.sh
set +e                                   # report every problem, not just the first

CONFIGS=("${@:-configs/default.yaml}")
ok=0; bad=0
pass() { printf '  \033[32mok\033[0m    %s\n' "$1"; ok=$((ok+1)); }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; bad=$((bad+1)); }
note() { printf '        %s\n' "$1"; }

echo "== site profile '$VGTFM_ENV'  (environments.yaml)"
DATA_ROOT=$(vgtfm_cfg "${CONFIGS[0]}" paths.data_root)
ART_ROOT=$(vgtfm_cfg "${CONFIGS[0]}" paths.artifact_root)
if [ -z "$DATA_ROOT" ]; then
    fail "profile did not resolve — is '$VGTFM_ENV' defined in environments.yaml?"
    echo; echo "== $ok ok, $bad problem(s)"; exit 1
fi
note "python_env $(printf '%s' "${VGTFM_PYTHON_ENV:-(PATH python)}")"
note "resolves to $(vgtfm_py_cmd)"
note "data       $DATA_ROOT"
note "artifacts  $ART_ROOT"
[ -n "${HF_HOME:-}" ] && note "hf_home    $HF_HOME"

echo "== source tree"
# The one thing a traceback cannot tell you: whether this copy is the code you
# fixed. Compare with the `src=` in a run's banner, or with the other machine.
SRC=$(vgtfm_py -c "from vgtfm.provenance import source_fingerprint as f; print(f())" \
      2>/dev/null)
if [ -n "$SRC" ]; then
    note "fingerprint $SRC   (same value elsewhere = same code)"
else
    fail "cannot fingerprint the sources — is this a vgtfm checkout?"
fi

echo "== submission directory"
[ -d logs ] && pass "logs/ exists (SLURM resolves --output relative to \$PWD)" \
            || fail "logs/ missing — run: mkdir -p logs"
[ -f run.py ] && pass "submitting from the repository root" \
              || fail "run.py not found — submit from the repository root"

echo "== environment"
# vgtfm_py warns on stderr when it could not honour the profile; catch that here
# rather than letting a substituted interpreter through.
warn=$(vgtfm_py -c "pass" 2>&1 >/dev/null)
if [ -n "$warn" ]; then
    fail "$warn"
    note "profile key: python_env"
else
    pass "interpreter: $(vgtfm_py -c 'import sys; print(sys.executable)' 2>/dev/null)"
fi
# Name the packages that are missing or mispinned, per stage, rather than just
# saying "imports failed": an environment built for a sibling project usually covers
# most of them, and which ones it misses decides which stages can run.
miss=$(vgtfm_py - <<'DEPS' 2>/dev/null
import importlib.metadata as md

# distribution name, required pin, stage that needs it
need = [
    ("torch",        "2.10.0",  "train"),
    ("scanpy",       "1.11.5",  "data"),
    ("anndata",      "0.12.10", "data"),
    ("datasets",     "4.5.0",   "data"),
    ("scikit-learn", "1.8.0",   "eval"),
    ("scib-metrics", "0.5.9",   "diagnose, unless diagnostics.run_scib=false"),
    ("decoupler",    "2.1.6",   "biosignal"),
    ("harmonypy",    "2.0.0",   "integrate"),
    ("bbknn",        "1.6.0",   "integrate"),
    ("scvi-tools",   "1.4.2",   "integrate"),
    ("umap-learn",   "0.5.11",  "figures"),
    ("matplotlib",   "3.10.8",  "figures"),
]
for dist, want, stage in need:
    try:
        have = md.version(dist).split("+")[0]
    except md.PackageNotFoundError:
        print(f"MISSING {dist} (needs {want}) -> blocks stage '{stage}'")
        continue
    if have != want:
        print(f"VERSION {dist} {have}, pinned {want} -> affects stage '{stage}'")
DEPS
)
if [ -z "$miss" ]; then
    pass "all pinned dependencies present at the required versions"
else
    while IFS= read -r line; do fail "$line"; done <<< "$miss"
    note "pip install -r requirements.txt into python_env, or point it at a new env"
fi

echo "== writable output root"
if mkdir -p "$ART_ROOT" 2>/dev/null && [ -w "$ART_ROOT" ]; then
    pass "artifact root writable"
    df -h "$ART_ROOT" 2>/dev/null | awk 'NR==2{printf "        %s free of %s\n",$4,$2}'
else
    fail "artifact root not writable: $ART_ROOT  (profile key: artifact_root)"
fi

echo "== precomputed embeddings"
for cfg in "${CONFIGS[@]}"; do
    sub=$(vgtfm_cfg "$cfg" data.substrate)
    src=$(vgtfm_cfg "$cfg" "data_path()")
    [ -z "$sub" ] && { fail "$cfg: config did not resolve"; continue; }
    if [ -f "$src/dataset_dict.json" ]; then
        pass "$cfg [$sub]"
        note "$src"
        missing=$(vgtfm_py - "$src" <<'PY' 2>/dev/null
import json, sys
from pathlib import Path
need = {"sample_id","spot_id","array_row","array_col","annotation",
        "dataset_id","gene_features","patch_features"}
info = json.loads((Path(sys.argv[1]) / "train" / "dataset_info.json").read_text())
print(",".join(sorted(need - set(info.get("features", {})))))
PY
)
        [ -z "$missing" ] && pass "  schema has all eight required columns" \
                          || fail "  missing columns: $missing"
    else
        fail "$cfg [$sub]: no dataset_dict.json at $src"
        note "check data_root in the '$VGTFM_ENV' profile"
    fi
done

echo "== feature cache"
for cfg in "${CONFIGS[@]}"; do
    sub=$(vgtfm_cfg "$cfg" data.substrate)
    c="$ART_ROOT/_cache/$sub/cache_info.json"
    if [ -f "$c" ]; then
        note "$sub: built — $(vgtfm_py -c "
import json; d=json.load(open('$c'))
print(f\"{d['n_spots']:,} spots, gene {d['gene_dim']}d, patch {d['patch_dim']}d\")" 2>/dev/null)"
    else
        note "$sub: not built yet — the first 'data' stage builds it (~6 GB, 10-30 min)"
    fi
done

echo "== raw counts (biosignal only)"
while read -r ds dir; do
    [ -z "$ds" ] && continue
    n=$(ls -1 "$dir"/*.h5ad 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] && pass "$ds: $n .h5ad" || fail "$ds: no .h5ad in $dir"
done < <(vgtfm_py -c "
from vgtfm.config import load_config
c = load_config('${CONFIGS[0]}', env='$VGTFM_ENV')
for k in c.paths.raw_h5ad:
    print(k, c.raw_h5ad_dir(k))
" 2>/dev/null)

echo "== slurm partitions"
if ! command -v sinfo >/dev/null; then
    note "no sinfo here — partition names unchecked (expected off-cluster)"
else
    # `sbatch: Requested node configuration is not available` is what a cluster
    # says when nothing matches the request. Usually that is a missing or wrong
    # --partition, occasionally a --mem no node can satisfy. Catch both here,
    # where the answer is one line of output rather than a dead job id.
    known=$(sinfo -h -o '%R' 2>/dev/null | sort -u)
    if [ -z "$known" ]; then
        note "sinfo returned nothing — cannot check partitions"
    else
        for kind in cpu gpu; do
            key="slurm_${kind}_partition"
            part=$(vgtfm_profile "$key")
            if [ -z "$part" ]; then
                if [ "$(echo "$known" | wc -l)" -gt 1 ]; then
                    fail "$key unset, but this cluster has partitions: $(echo $known)"
                    note "  set it in the '$VGTFM_ENV' profile of environments.yaml"
                else
                    pass "$key unset (single-partition cluster)"
                fi
                continue
            fi
            if ! echo "$known" | grep -qx "$part"; then
                fail "$key=$part is not a partition here (have: $(echo $known))"
                continue
            fi
            # Largest node in the partition, so an impossible --mem shows up now.
            # Compared against the biggest --mem `_defaults` in submit.sh asks for
            # on *this* placement, not a single figure for both: the cpu stages top
            # out at `data`/`diagnose`, the gpu ones at `embed`, which reads every
            # slide's patch features to write the frozen embedding.
            case "$kind" in
                cpu) want_mb=98304  want="96G"  by="data/diagnose" ;;
                gpu) want_mb=131072 want="128G" by="embed"         ;;
            esac
            # `-N`, node-oriented: without it sinfo groups nodes by configuration
            # and a partition whose nodes differ collapses to one row carrying a
            # single node's figure with a `+` appended. `sort -n` then reads that
            # `55847+` as 55847 and the check warns about a limit no node here
            # actually has. One line per node has nothing to aggregate.
            maxmem=$(sinfo -h -N -p "$part" -o '%m' 2>/dev/null | sort -n | tail -1)
            maxcpu=$(sinfo -h -N -p "$part" -o '%c' 2>/dev/null | sort -n | tail -1)
            pass "$key=$part  (largest node: ${maxcpu:-?} cpus, $(( ${maxmem:-0} / 1024 ))G)"
            [ -n "$maxmem" ] && [ "$maxmem" -lt "$want_mb" ] && \
                note "  $kind jobs ask for up to $want ($by) — override with --mem= if that does not fit"
        done
        gpuargs=$(vgtfm_profile slurm_gpu_args | tr '\n' ' ')
        [ -n "$gpuargs" ] && pass "gpu requested as: $gpuargs" \
                          || note "slurm_gpu_args unset — GPU stages will get no GPU"
    fi
fi

echo "== gpu"
command -v nvidia-smi >/dev/null && nvidia-smi -L 2>/dev/null | sed 's/^/        /' \
    || note "no GPU on this node (expected on a login node)"

echo
echo "== $ok ok, $bad problem(s)"
[ "$bad" -eq 0 ] && echo "ready: bash slurm/submit_all.sh" || echo "fix the FAILs above first"
exit $(( bad > 0 ))
