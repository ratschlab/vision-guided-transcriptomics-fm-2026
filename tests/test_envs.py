"""The isolated conda environments the gene-side models run in.

Every assertion here corresponds to a way one of these files failed when it was
actually created, each of which surfaced late — during the pip install, or on the
first forward pass, never at solve time. They are static checks: creating three
environments takes half an hour and several GB, so what is pinned here is the
knowledge, not the install.
"""

from __future__ import annotations

import pytest
import yaml

from conftest import ROOT

ENVS = sorted((ROOT / "envs").glob("*.yaml"))


def spec(name):
    return yaml.safe_load((ROOT / "envs" / f"{name}.yaml").read_text())


def pip_requirements(payload) -> list[str]:
    """The pip section, flattened. This is the list that actually resolves."""
    for item in payload["dependencies"]:
        if isinstance(item, dict) and "pip" in item:
            return [str(r) for r in item["pip"]]
    return []


def conda_requirements(payload) -> list[str]:
    return [str(i) for i in payload["dependencies"] if not isinstance(i, dict)]


def requires(reqs: list[str], package: str) -> str | None:
    for req in reqs:
        head = req.split("==")[0].split(">=")[0].split("<")[0].strip()
        if head == package:
            return req
    return None


@pytest.mark.parametrize("path", ENVS, ids=lambda p: p.name)
def test_every_environment_file_parses_and_is_named_for_this_repository(path):
    payload = yaml.safe_load(path.read_text())
    assert payload["name"].startswith("vgtfm-"), (
        "the name is what scripts/embed_check.py and the README's conda run lines "
        "look for; an unprefixed one collides with a user's own environment"
    )


@pytest.mark.parametrize("path", ENVS, ids=lambda p: p.name)
def test_no_environment_pip_installs_from_a_huggingface_git_url(path):
    """`pip install git+https://huggingface.co/...` cannot work, twice over.

    pip clones with `--filter=blob:none` and the HF git server does not serve the
    follow-up promisor fetch (`fatal: expected 'packfile'`); and a clone that skips
    git-lfs leaves the package's own data files as pointer files, which surfaces
    only when something unpickles one. envs/geneformer_install.py goes through
    huggingface_hub instead.
    """
    reqs = pip_requirements(yaml.safe_load(path.read_text()))
    offenders = [r for r in reqs if "huggingface.co" in r and r.startswith("git+")]
    assert not offenders, offenders


@pytest.mark.parametrize("name", ["geneformer", "scgpt"])
def test_an_anndata_that_predates_pandas_3_pins_pandas(name):
    """anndata 0.10.9 registers no writer for the pandas 3 string dtype.

    Unpinned, pip resolves pandas 3 and the run gets as far as writing the
    temporary h5ad before failing with `No method registered for writing
    ArrowStringArray`.
    """
    reqs = pip_requirements(spec(name))
    if requires(reqs, "anndata") != "anndata==0.10.9":
        pytest.skip(f"{name} no longer pins anndata 0.10.9")
    assert requires(reqs, "pandas") == "pandas<3"


def test_scgpt_pins_numpy_in_the_section_that_resolves_it():
    """A conda-level `numpy<2` does not constrain the pip section.

    pip runs after the conda solve, scanpy and anndata pull numpy 2, and torch
    2.1.2 is then importable but broken — `Failed to initialize NumPy: _ARRAY_API
    not found` is a warning, so nothing fails until a tensor is built from an array.
    """
    payload = spec("scgpt")
    assert requires(conda_requirements(payload), "numpy"), "conda pin (not sufficient alone)"
    assert requires(pip_requirements(payload), "numpy") == "numpy<2"


def test_scgpt_declares_the_ipython_that_scgpt_itself_does_not():
    """`scgpt.tasks`, where embed_data lives, imports IPython at module level."""
    assert requires(pip_requirements(spec("scgpt")), "ipython")


def test_scgpt_declares_the_loess_backend_seurat_v3_needs():
    """scanpy's seurat_v3 flavour is implemented on skmisc and raises without it."""
    assert requires(pip_requirements(spec("scgpt")), "scikit-misc")


def test_cancerfoundation_declares_the_torchtext_its_vocabulary_needs():
    """model/embedding.py imports torchtext; upstream's environment.yml omits it,
    so the import fails before any weight is read."""
    assert requires(pip_requirements(spec("cancerfoundation")), "torchtext")


def test_cancerfoundation_holds_torch_below_the_weights_only_flip():
    """`embed()` calls torch.load without weights_only=False, and 2.6 flipped that
    default, so a newer torch refuses the checkpoint instead of loading it."""
    assert requires(pip_requirements(spec("cancerfoundation")), "torch") == "torch==2.3.1"


def test_geneformer_has_an_install_script_because_pip_cannot_do_it():
    script = ROOT / "envs" / "geneformer_install.py"
    assert script.exists()
    text = script.read_text()
    assert "snapshot_download" in text, "huggingface_hub resolves lfs without git-lfs"
    assert "--no-deps" in text, "the install must not undo the pins above it"
    assert "geneformer/**" in text, "the gene dictionaries are lfs blobs, like the weights"
