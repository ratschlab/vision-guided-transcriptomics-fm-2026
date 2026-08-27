"""The stage dispatcher in ``run.py``: overrides, logging and the run manifest.

The manifest is the only record of which code, on which machine, produced a set of
artefacts, and it is written from a ``finally`` block so that a failed stage is
recorded rather than vanishing. Both of those are asserted here against a real
``main()`` call with a stubbed stage, not against the source text.
"""

from __future__ import annotations

import json

import pytest

from conftest import ROOT

import run as cli  # noqa: E402


@pytest.fixture
def artifacts(tmp_path):
    """Argument list pinning every output under *tmp_path*."""
    return ["--set", f"paths.artifact_root={tmp_path}", "run_name=t"]


def manifests(tmp_path):
    return sorted((tmp_path / "t" / "logs").glob("manifest-*.json"))


# ── --set parsing ────────────────────────────────────────────────────


def test_set_parses_dotted_key_value_pairs():
    assert cli._parse_sets(["a.b=1", " c = two "]) == {"a.b": "1", "c": "two"}


def test_a_set_token_without_an_equals_sign_is_an_error():
    with pytest.raises(SystemExit, match="expects key=value"):
        cli._parse_sets(["models.names"])


def test_a_value_may_itself_contain_an_equals_sign():
    assert cli._parse_sets(["paths.data_root=/a=b"]) == {"paths.data_root": "/a=b"}


def test_an_unknown_stage_is_rejected_by_the_parser():
    with pytest.raises(SystemExit):
        cli.main(["diagnoze"])


# ── the tee ──────────────────────────────────────────────────────────


def test_the_tee_writes_to_every_sink_and_reports_the_length(tmp_path):
    import io

    a, b = io.StringIO(), io.StringIO()
    tee = cli._Tee(a, b)
    assert tee.write("hello") == 5
    tee.flush()
    assert a.getvalue() == b.getvalue() == "hello"


def test_closing_the_tee_leaves_its_sinks_usable():
    """absl calls close() on the replaced stream at interpreter shutdown.

    It must not close the console or the logfile, and must not raise — either would
    put a traceback after the run's own status line.
    """
    import io

    a = io.StringIO()
    tee = cli._Tee(a)
    tee.close()
    assert a.closed is False
    a.write("still open")
    assert tee.isatty() is False


# ── the manifest ─────────────────────────────────────────────────────


def test_a_successful_run_records_the_stage_and_its_provenance(tmp_path, artifacts, monkeypatch):
    from vgtfm.provenance import source_fingerprint

    monkeypatch.setattr(cli, "_run_stage", lambda stage, cfg: None)
    assert cli.main(["data", *artifacts]) == 0

    (path,) = manifests(tmp_path)
    m = json.loads(path.read_text())
    assert m["status"] == "ok"
    assert [s["stage"] for s in m["stages"]] == ["data"]
    assert m["stages"][0]["status"] == "ok"
    assert m["source_fingerprint"] == source_fingerprint()
    assert m["host"] and m["run_name"] == "t"


def test_the_resolved_config_is_written_next_to_the_outputs(tmp_path, artifacts, monkeypatch):
    monkeypatch.setattr(cli, "_run_stage", lambda stage, cfg: None)
    cli.main(["data", *artifacts, "models.pca_components=7"])

    resolved = json.loads((tmp_path / "t" / "config.resolved.json").read_text())
    assert resolved["models"]["pca_components"] == 7
    assert resolved["run_name"] == "t"


def test_a_failing_stage_is_recorded_as_an_error_and_still_raises(tmp_path, artifacts, monkeypatch):
    """A run that stopped early must never be recorded in the manifest as a success."""

    def boom(stage, cfg):
        raise SystemExit("no merged dataset")

    monkeypatch.setattr(cli, "_run_stage", boom)
    with pytest.raises(SystemExit):
        cli.main(["data", *artifacts])

    (path,) = manifests(tmp_path)
    m = json.loads(path.read_text())
    assert m["status"] == "error"
    assert m["stages"][0]["status"] == "error"
    assert "no merged dataset" in m["stages"][0]["error"]


def test_all_runs_the_default_chain_and_records_every_stage(tmp_path, artifacts, monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "_run_stage", lambda stage, cfg: seen.append(stage))
    cli.main(["all", *artifacts])

    assert seen == cli._ALL_STAGES
    (path,) = manifests(tmp_path)
    m = json.loads(path.read_text())
    assert [s["stage"] for s in m["stages"]] == cli._ALL_STAGES
    assert m["stage_arg"] == "all"


def test_two_stages_started_in_the_same_second_keep_separate_manifests(
    tmp_path, artifacts, monkeypatch
):
    """`_stamp()` has one-second resolution, so the stage has to be in the filename.

    Without it, two jobs submitted together overwrite each other's provenance record
    and the one that finishes first leaves nothing behind.
    """
    monkeypatch.setattr(cli, "_run_stage", lambda stage, cfg: None)
    monkeypatch.setattr(cli, "_stamp", lambda: "20260101-000000")

    cli.main(["data", *artifacts])
    cli.main(["ablate", *artifacts])

    names = [p.name for p in manifests(tmp_path)]
    assert names == ["manifest-ablate-20260101-000000.json", "manifest-data-20260101-000000.json"]


def test_the_banner_and_the_logfile_carry_the_source_fingerprint(
    tmp_path, artifacts, monkeypatch, capsys
):
    from vgtfm.provenance import source_fingerprint

    monkeypatch.setattr(cli, "_run_stage", lambda stage, cfg: None)
    cli.main(["data", *artifacts])

    assert f"src={source_fingerprint()}" in capsys.readouterr().out
    (log,) = (tmp_path / "t" / "logs").glob("data-*.log")
    assert f"src={source_fingerprint()}" in log.read_text()


def test_no_log_leaves_stdout_alone_and_writes_no_logfile(tmp_path, artifacts, monkeypatch):
    import sys

    monkeypatch.setattr(cli, "_run_stage", lambda stage, cfg: None)
    before = sys.stdout
    cli.main(["data", "--no-log", *artifacts])

    assert sys.stdout is before
    assert not list((tmp_path / "t" / "logs").glob("*.log"))
    assert manifests(tmp_path), "the manifest is written either way"


def test_an_unwritable_artifact_root_names_the_profile_and_the_way_out(monkeypatch):
    with pytest.raises(SystemExit, match="artifact_root"):
        cli.main(["data", "--set", "paths.artifact_root=/proc/nope/artifacts"])


# ── stage dispatch ───────────────────────────────────────────────────


def test_every_declared_stage_dispatches_to_a_real_module():
    """A stage in the CLI's choices with no branch in `_run_stage` would parse and
    then fail at runtime, after the manifest and logfile were already open."""
    import ast

    tree = ast.parse((ROOT / "run.py").read_text())
    fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_run_stage"
    )
    handled = {
        c.value
        for node in ast.walk(fn)
        if isinstance(node, ast.Compare)
        for c in node.comparators
        if isinstance(c, ast.Constant)
    }
    assert set(cli._STAGES) - {"all"} == handled
