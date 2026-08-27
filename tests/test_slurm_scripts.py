"""The cluster launch scripts.

Static checks, plus one execution of ``stage.sbatch`` the way SLURM runs it. Errors
in the shell layer surface as a job that dies in its first second, hours after
submission, so the two failure modes seen so far are pinned directly: a GPU
requested as ``--gpus=1`` on a SLURM that knows only ``--gres``, and a repository
root derived from ``$0`` inside a job whose ``$0`` is a copy under ``/var/spool``.

Nothing here needs a cluster: ``sbatch`` is never called.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest

from conftest import ROOT

SLURM = ROOT / "slurm"
SBATCH = sorted(SLURM.glob("*.sbatch"))
SCRIPTS = sorted(SLURM.glob("*.sh")) + SBATCH

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_every_launch_script_parses(path):
    p = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


@pytest.mark.parametrize("path", SBATCH, ids=lambda p: p.name)
def test_a_job_script_does_not_find_the_repository_through_argv0(path):
    """``$0`` inside a job is ``/var/spool/slurm/job<id>/slurm_script``.

    sbatch copies the script to the node, so ``cd "$(dirname "$0")/.."`` lands in
    the spool directory and every relative path afterwards is wrong. The
    directory the job was submitted from is what SLURM preserves instead.
    """
    text = path.read_text()
    if 'dirname "$0"' not in text:
        return
    assert "SLURM_SUBMIT_DIR" in text, (
        f"{path.name} derives the repo root from $0 with no SLURM_SUBMIT_DIR "
        f"branch; under sbatch that resolves to /var/spool"
    )


@pytest.mark.parametrize("path", SBATCH, ids=lambda p: p.name)
def test_a_job_script_names_no_partition_it_cannot_cancel(path):
    """Placement lives in ``environments.yaml``, not in ``#SBATCH`` lines.

    Exception: the four single-purpose GPU scripts are entry points in their own
    right and may hardcode it. ``stage.sbatch`` may not — it is submitted for CPU
    and GPU stages alike by ``submit_all.sh``, and a directive here cannot be
    cancelled from the command line.
    """
    if path.name != "stage.sbatch":
        return
    for bad in ("--partition", "--gres", "--gpus"):
        assert bad not in path.read_text(), (
            f"stage.sbatch pins {bad}, which submit_all.sh cannot override"
        )


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_no_launch_script_requests_a_gpu_with_gpus(path):
    """``--gpus`` arrived in SLURM 19.05; older installations want ``--gres``.

    Which flag a site understands is a profile question (``slurm_gpu_args``), so
    the scripts should not answer it themselves.
    """
    assert "--gpus=" not in path.read_text()


def test_stage_sbatch_finds_the_repository_when_run_from_the_spool(tmp_path):
    """The regression test for the failure above, end to end.

    Runs a *copy* of the script from elsewhere, with the environment SLURM sets,
    and asks for a stage that does not exist: reaching ``run.py``'s usage message
    proves the repository, ``slurm/env.sh`` and the interpreter were all found,
    while argparse rejects the stage before anything is read or written.
    """
    spool = tmp_path / "job1234"
    spool.mkdir()
    copy = spool / "slurm_script"
    copy.write_bytes((SLURM / "stage.sbatch").read_bytes())

    p = subprocess.run(
        ["bash", str(copy), "no-such-stage"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "SLURM_JOB_ID": "1234",
            "SLURM_SUBMIT_DIR": str(ROOT),
            "VGTFM_ENV": "local",
        },
    )
    out = p.stdout + p.stderr
    assert "No such file or directory" not in out, out
    assert "invalid choice: 'no-such-stage'" in out, out


def test_stage_sbatch_says_so_when_submitted_from_the_wrong_directory(tmp_path):
    p = subprocess.run(
        ["bash", str(SLURM / "stage.sbatch"), "no-such-stage"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "SLURM_JOB_ID": "1234",
            "SLURM_SUBMIT_DIR": str(tmp_path),
            "VGTFM_ENV": "local",
        },
    )
    assert p.returncode == 2
    assert "not a vgtfm checkout" in p.stderr


def test_biosignal_waits_for_the_job_that_runs_train():
    """``biosignal`` loads the trained autoencoder embedding and exits if it is
    absent, so it cannot be a sibling of the job that writes it — and ``all`` is the
    only job in the graph that runs ``train``. Hung off the ``data`` job instead, it
    starts hours early and dies with "missing .../embeddings/ae/seed-42.npy".
    """
    text = (SLURM / "submit_all.sh").read_text()

    producer = re.search(r'(\w+)=\$\(submit "vgtfm-all-', text)
    assert producer, "submit_all.sh no longer submits an `all` job"
    consumer = re.search(r'submit "vgtfm-biosignal-[^"]*" "\$\{?(\w+)', text)
    assert consumer, "submit_all.sh no longer submits a `biosignal` job"

    assert consumer.group(1) == producer.group(1), (
        f"biosignal depends on ${consumer.group(1)}, but train runs in "
        f"${producer.group(1)} (the `all` job)"
    )


def test_integrate_waits_for_the_job_that_runs_train_and_eval():
    """`integrate` no longer fits a PCA of its own.

    It loads the `pca` baseline `train` fitted and reuses the prediction vector
    `eval` cached for it — which is what makes its uncorrected row Table 1's row
    rather than a second opinion on it. Hung off the `data` job, as it was while it
    recomputed everything itself, it starts hours early and dies on the first line
    with "cannot produce the uncorrected reference embedding".
    """
    text = (SLURM / "submit_all.sh").read_text()

    producer = re.search(r'(\w+)=\$\(submit "vgtfm-all-', text)
    assert producer, "submit_all.sh no longer submits an `all` job"
    consumer = re.search(r'submit "vgtfm-integrate-[^"]*" "\$\{?(\w+)', text)
    assert consumer, "submit_all.sh no longer submits an `integrate` job"

    assert consumer.group(1) == producer.group(1), (
        f"integrate depends on ${consumer.group(1)}, but train and eval run in "
        f"${producer.group(1)} (the `all` job)"
    )


def test_every_stage_run_py_accepts_can_be_submitted():
    """A stage `run.py` knows and `submit.sh` does not is a stage that can only be
    run by hand: the wrapper exits 2 with "unknown stage" before reaching sbatch, and
    that is how `results` spent its first day unsubmittable."""
    import run as run_py

    known = (SLURM / "submit.sh").read_text()
    missing = [s for s in run_py._STAGES if f"\n        {s})" not in known]
    assert not missing, f"submit.sh has no resource line for {missing}"


def test_the_consistency_check_is_submitted_with_the_graph():
    """`results` is what says whether two stages of a run computed one quantity and
    got two answers. Left out of the graph it is only ever run by hand, and the
    disagreement surfaces at the point of writing the manuscript instead."""
    text = (SLURM / "submit_all.sh").read_text()
    assert re.search(r'submit "vgtfm-results-[^"]*"', text), (
        "submit_all.sh submits no `results` job"
    )


# ── the shell layer must not fail silently either ────────────────────


def _bash(script: str, *args, cwd=None):
    return subprocess.run(
        ["bash", script, *args],
        cwd=cwd or ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "VGTFM_ENV": "local"},
    )


def test_reading_a_config_that_does_not_load_says_so_and_fails():
    """A config that will not load must not come back as an empty string with
    status 0: under `set -o pipefail` the caller then dies at the assignment with no
    output at all, which is the worst version of a silent failure."""
    p = subprocess.run(
        ["bash", "-c", "source slurm/env.sh; vgtfm_cfg configs/nope.yaml run_name"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "VGTFM_ENV": "local"},
    )

    assert p.returncode != 0
    assert "cannot read" in p.stderr
    assert "configs/nope.yaml" in p.stderr


def test_a_config_value_never_carries_a_diagnostic_with_it():
    """The interpreter resolver warns on stderr when it cannot honour the profile.
    Folding that into stdout would return `data.substrate` with a warning glued to
    the front, and every path built from it would be wrong."""
    p = subprocess.run(
        ["bash", "-c", "source slurm/env.sh; vgtfm_cfg configs/smoke.yaml data.substrate"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "VGTFM_ENV": "local"},
    )

    assert p.returncode == 0
    assert p.stdout.strip() == "geneformer"


def test_submitting_nothing_is_an_error_not_a_quiet_success():
    """A graph that looks submitted but is missing a substrate is exactly the
    failure this script exists to prevent."""
    p = _bash("slurm/submit_all.sh", "--dry-run", "configs/nope.yaml")

    assert p.returncode != 0
    assert "no usable config" in p.stderr
    assert "no such file" in p.stderr


def test_a_dry_run_of_a_usable_config_still_succeeds():
    """The guard above must not turn every run into a failure."""
    p = _bash("slurm/submit_all.sh", "--dry-run", "--core", "configs/smoke.yaml")

    assert p.returncode == 0, p.stderr
    assert "slurm/stage.sbatch data configs/smoke.yaml" in p.stderr  # the sbatch lines
    assert "dry run — nothing submitted" in p.stdout


def test_a_deliberate_omission_travels_with_every_job():
    """`diagnostics.run_scib=false` is the recorded way to skip work. It has to
    reach the jobs, or the only way to use it is one `submit.sh` at a time."""
    p = _bash(
        "slurm/submit_all.sh",
        "--dry-run",
        "--core",
        "configs/smoke.yaml",
        "diagnostics.run_scib=false",
    )

    assert p.returncode == 0, p.stderr
    lines = [ln for ln in p.stderr.splitlines() if "stage.sbatch" in ln]
    assert lines, p.stderr
    for line in lines:
        assert line.rstrip().endswith("diagnostics.run_scib=false"), line


def test_a_mistyped_option_is_rejected_rather_than_read_as_a_config():
    """A mistyped flag must not land in CONFIGS and be skipped as a missing file."""
    p = _bash("slurm/submit_all.sh", "--dryrun")
    assert p.returncode == 2
    assert "unknown option" in p.stderr
