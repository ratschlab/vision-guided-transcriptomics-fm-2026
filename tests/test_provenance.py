"""Which source tree produced a result.

This repository is edited on one machine and run on another, and the two copies
drift. Line numbers in a traceback cannot settle which code ran — they move for
unrelated reasons, and a stale checkout raises a plausible error — so every banner
and manifest carries a content fingerprint of the sources instead.
"""

from __future__ import annotations

import pytest

from conftest import ROOT

from vgtfm.provenance import source_fingerprint


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A miniature source tree the fingerprint can be pointed at."""
    import vgtfm.provenance as prov

    (tmp_path / "vgtfm").mkdir()
    (tmp_path / "vgtfm" / "config.py").write_text("A = 1\n")
    (tmp_path / "run.py").write_text("print('hi')\n")
    monkeypatch.setattr(prov, "ROOT", tmp_path)

    def fingerprint():
        source_fingerprint.cache_clear()  # the cache holds one value per process
        return source_fingerprint()

    yield tmp_path, fingerprint
    # Leave no cached value from the miniature tree behind: `run.py` calls this and
    # would otherwise stamp a temp directory's hash into a later test's manifest.
    source_fingerprint.cache_clear()


def test_the_same_sources_always_fingerprint_the_same(tree):
    _, fingerprint = tree
    assert fingerprint() == fingerprint()


def test_the_fingerprint_is_short_enough_to_read_off_two_terminals(tree):
    _, fingerprint = tree
    value = fingerprint()
    assert len(value) == 12
    assert all(c in "0123456789abcdef" for c in value)


def test_editing_any_shipped_source_changes_it(tree):
    path, fingerprint = tree
    before = fingerprint()
    (path / "vgtfm" / "config.py").write_text("A = 2\n")
    assert fingerprint() != before


def test_adding_a_source_changes_it(tree):
    path, fingerprint = tree
    before = fingerprint()
    (path / "vgtfm" / "newmodule.py").write_text("B = 1\n")
    assert fingerprint() != before


def test_renaming_a_source_changes_it_even_with_the_same_contents(tree):
    """Paths go into the hash, not just bytes: two files that swapped names are a
    different tree and must not fingerprint alike."""
    path, fingerprint = tree
    before = fingerprint()
    (path / "vgtfm" / "config.py").rename(path / "vgtfm" / "renamed.py")
    assert fingerprint() != before


def test_data_and_caches_do_not_change_it(tree):
    """It answers one question — which code — so a rebuilt cache or a new figure
    must leave it alone, or it stops being comparable between machines."""
    path, fingerprint = tree
    before = fingerprint()
    (path / "artifacts").mkdir()
    (path / "artifacts" / "results.csv").write_text("model,f1\nae,0.4\n")
    (path / "notes.md").write_text("scratch\n")
    (path / "vgtfm" / "__pycache__").mkdir()
    (path / "vgtfm" / "__pycache__" / "config.cpython-311.pyc").write_bytes(b"\x00\x01")
    assert fingerprint() == before


def test_the_real_tree_fingerprints():
    assert len(source_fingerprint()) == 12


# That every run stamps the fingerprint into its banner and its manifest is asserted
# against a real `main()` call in test_cli.py.


def test_preflight_prints_the_same_value_so_two_machines_can_be_compared():
    text = (ROOT / "slurm" / "preflight.sh").read_text()
    assert "source_fingerprint" in text
