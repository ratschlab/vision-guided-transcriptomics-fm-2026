"""What the pipeline does when it cannot do what it was asked.

A stage that skips a step it could not perform finishes green, and a figure that
was never drawn looks identical from the outside to one whose input was silently
empty. :mod:`vgtfm.degraded` fixes the policy in one place; these tests hold the
codebase to it, including an AST guard against new swallowed exceptions.
"""

from __future__ import annotations

import ast
import types

import pandas as pd
import pytest

from conftest import ROOT

from vgtfm.degraded import Incomplete, pending, refuse, report_pending


def _stub_table(n: int = 10):
    """The scIB helpers touch only the row count before they import."""
    return types.SimpleNamespace(n=n)


# ── the policy itself ────────────────────────────────────────────────


def test_refusing_is_a_systemexit_so_the_manifest_records_it():
    """``run.py`` catches ``BaseException`` per stage and writes the manifest from
    it. A refusal has to travel that path, or a stopped run is filed as a success."""
    with pytest.raises(SystemExit) as e:
        refuse("the widget", "no widgets")
    assert isinstance(e.value, Incomplete)
    assert "cannot produce the widget" in str(e.value)
    assert "no widgets" in str(e.value)


def test_a_refusal_carries_the_way_out():
    with pytest.raises(Incomplete, match="install it"):
        refuse("the panel", "it is not importable", hint="install it")


def test_pending_is_not_an_error_and_comes_back_for_the_summary(capsys):
    note = pending("the ablation table", "`ablate` has not run")
    assert "the ablation table" in note
    assert "pending" in capsys.readouterr().out

    report_pending([note], stage="figures")
    out = capsys.readouterr().out
    assert "did not build yet" in out and "the ablation table" in out


def test_report_pending_says_nothing_when_nothing_is_pending(capsys):
    report_pending([], stage="figures")
    assert capsys.readouterr().out == ""


# ── the figures stage: "not yet" versus "broken" ─────────────────────
#
# This stage is runnable before its inputs exist — the `all` job draws what it can,
# a second pass runs after `ablate` and `biosignal` land — so it is the one place
# where a missing input is ambiguous. The producing stage's directory disambiguates.


def figures_source(tmp_path, **kwargs):
    from vgtfm.figures.build import _source

    notes: list[str] = []
    df = _source(
        tmp_path / "ablation",
        "per_class.csv",
        what="the shuffle table",
        stage="ablate",
        notes=notes,
        **kwargs,
    )
    return df, notes


def test_an_upstream_stage_that_has_not_run_is_pending_not_fatal(tmp_path):
    df, notes = figures_source(tmp_path)
    assert df.empty
    assert len(notes) == 1 and "has not produced anything" in notes[0]


def test_an_empty_directory_from_a_crashed_stage_is_pending_not_fatal(tmp_path):
    """``cfg.sub()`` creates the directory before the stage does any work, so a
    stage that died on its first line leaves an empty one behind — and that job
    already failed. Treating the leftover as a defect would make a later `--core`
    run refuse over a crash it was told to skip."""
    (tmp_path / "ablation").mkdir()
    df, notes = figures_source(tmp_path)
    assert df.empty
    assert len(notes) == 1 and "has not produced anything" in notes[0]


def test_an_upstream_stage_that_ran_and_came_up_short_is_fatal(tmp_path):
    """Other outputs are there, this one is not: the stage worked and still did not
    produce what the figure needs."""
    (tmp_path / "ablation").mkdir()
    (tmp_path / "ablation" / "training_history.csv").write_text("epoch,loss\n1,0.5\n")
    with pytest.raises(Incomplete, match="did not write"):
        figures_source(tmp_path)


def test_an_upstream_output_with_no_rows_is_fatal(tmp_path):
    (tmp_path / "ablation").mkdir()
    (tmp_path / "ablation" / "per_class.csv").write_text("model,f1\n")  # header only
    with pytest.raises(Incomplete, match="has no rows"):
        figures_source(tmp_path)


def test_an_optional_output_is_announced_rather_than_demanded(tmp_path):
    """``eval`` writes no deltas.csv without a reference model, and no bootstrap.csv
    when ``eval.bootstrap_n`` is 0. Absent on purpose is still absent out loud."""
    (tmp_path / "ablation").mkdir()
    (tmp_path / "ablation" / "results.csv").write_text("a\n1\n")
    df, notes = figures_source(tmp_path, required=False)
    assert df.empty
    assert len(notes) == 1 and "wrote no per_class.csv" in notes[0]


def test_an_upstream_output_with_rows_is_returned(tmp_path):
    (tmp_path / "ablation").mkdir()
    pd.DataFrame([{"model": "ae", "f1": 0.4}]).to_csv(
        tmp_path / "ablation" / "per_class.csv", index=False
    )
    df, notes = figures_source(tmp_path)
    assert len(df) == 1 and notes == []


# ── the diagnose stage ───────────────────────────────────────────────


def test_an_embedding_of_the_wrong_length_is_never_scored(tmp_path, monkeypatch):
    """A row-count mismatch means the file was written against another cohort.
    Scoring it pairs each spot with another spot's vector — wrong, and plausible."""
    import numpy as np

    from vgtfm.config import Config
    from vgtfm.diagnostics import report

    cfg = Config()
    cfg.models.names = ("ae",)
    p = tmp_path / "seed-42.npy"
    np.save(p, np.zeros((7, 4), dtype=np.float32))
    monkeypatch.setattr("vgtfm.models.train.embedding_path", lambda *a, **k: p)

    with pytest.raises(Incomplete, match="has 7 rows"):
        report._learned_embeddings(cfg, n_spots=99)


def test_an_embedding_that_does_not_exist_yet_is_pending(tmp_path, monkeypatch, capsys):
    from vgtfm.config import Config
    from vgtfm.diagnostics import report

    cfg = Config()
    cfg.models.names = ("ae",)
    monkeypatch.setattr(
        "vgtfm.models.train.embedding_path", lambda *a, **k: tmp_path / "absent.npy"
    )

    assert report._learned_embeddings(cfg, n_spots=99) == {}
    assert "pending" in capsys.readouterr().out


def test_omitting_the_scib_panel_is_a_config_field_not_an_import_failure(tmp_path, monkeypatch):
    """scib-metrics is the dependency a cluster environment is most likely to lack.
    A missing install must stop the run and name the config field that omits the
    panel on purpose, so the decision lands in config.resolved.json rather than
    scrolling past in a log."""
    from vgtfm.config import Config
    from vgtfm.diagnostics import report

    assert Config().diagnostics.run_scib is True, "the default must be to run it"

    def no_scib(*a, **k):
        raise ImportError("No module named 'scib_metrics'")

    monkeypatch.setattr("vgtfm.diagnostics.scib.run", no_scib)
    cfg = Config()
    cfg.models.names = ()
    with pytest.raises(Incomplete, match="run_scib=false"):
        report._scib_panel(cfg, tmp_path, _stub_table())

    cfg.diagnostics.run_scib = False
    report._scib_panel(cfg, tmp_path, _stub_table())  # the recorded opt-out
    assert not (tmp_path / "scib_panel.csv").exists()


def test_a_scib_panel_that_scored_nothing_is_not_written(tmp_path, monkeypatch):
    from vgtfm.config import Config
    from vgtfm.diagnostics import report

    monkeypatch.setattr("vgtfm.diagnostics.scib.run", lambda *a, **k: pd.DataFrame())
    cfg = Config()
    cfg.models.names = ()
    with pytest.raises(Incomplete, match="scored no representation"):
        report._scib_panel(cfg, tmp_path, _stub_table())
    assert not (tmp_path / "scib_panel.csv").exists()


# ── the guard that keeps all of the above true ───────────────────────

#: Exception handlers that neither re-raise nor ``refuse``, and why each is allowed.
#: Keyed by (path relative to the repo root, enclosing function) so it survives
#: edits that move code around. Everything here is a backend or formatting detail
#: that cannot change a reported number.
ALLOWED_SWALLOWS = {
    ("run.py", "flush"): "teeing to a closed logfile during interpreter shutdown",
    ("run.py", "_device_info"): "a description string for the manifest; the stage "
    "that needs torch fails on its own",
    ("vgtfm/perf.py", "configure"): "an optional matmul-precision hint",
    ("vgtfm/perf.py", "amp"): "GradScaler moved modules between torch releases",
    ("vgtfm/perf.py", "free_cuda"): "best-effort VRAM release",
    ("vgtfm/models/nn.py", "__init__"): "a free-VRAM probe choosing resident vs "
    "pinned tensors; performance only",
    ("vgtfm/config.py", "_coerce_scalar"): "type probing for --set values",
    ("vgtfm/embed/gene_fm.py", "_model_input_size"): "documented per-version "
    "default, printed when used",
    ("vgtfm/embed/gene_fm.py", "_flash_attention_available"): "a capability probe, "
    "not a failure: flash-attn is an optional throughput build and the standard "
    "attention path is the same computation. Its absence is printed by the caller",
    ("vgtfm/biosignal/ridge.py", "quartile_table"): "degenerate input returns an "
    "empty frame by contract; the "
    "caller refuses on it",
    ("vgtfm/embed/build.py", "run"): "the gene models run in another environment, "
    "so the first pass has nothing to merge",
    (
        "vgtfm/diagnostics/integration.py",
        "_correct_or_crash",
    ): "defers rather than swallows: a native crash becomes a value so the "
    "remaining methods still run, and integration.run() refuses on the "
    "collected crashes once every method has had its turn",
}


def _swallows(handler: ast.ExceptHandler) -> bool:
    """True if the handler neither re-raises nor routes through ``refuse``."""
    for node in ast.walk(handler):
        if isinstance(node, ast.Raise):
            return False
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "refuse":
            return False
    return True


def _enclosing_function(tree: ast.AST, node: ast.AST) -> str:
    best = None
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if fn.lineno <= node.lineno <= (fn.end_lineno or fn.lineno):
                if best is None or fn.lineno > best.lineno:
                    best = fn
    return best.name if best else "<module>"


def _handlers():
    """Every swallowing ``except`` and ``suppress`` in the shipped code."""
    files = [p for p in sorted((ROOT / "vgtfm").rglob("*.py")) if "__pycache__" not in str(p)] + [
        ROOT / "run.py"
    ]
    for path in files:
        rel = str(path.relative_to(ROOT))
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and _swallows(node):
                what = ast.unparse(node.type) if node.type else "bare except"
                yield rel, _enclosing_function(tree, node), node.lineno, what
            if isinstance(node, ast.With):
                for item in node.items:
                    call = item.context_expr
                    if isinstance(call, ast.Call) and "suppress" in ast.unparse(call.func):
                        yield (rel, _enclosing_function(tree, node), node.lineno, ast.unparse(call))


def test_no_stage_swallows_a_failure_without_saying_so():
    """The regression guard for the whole policy.

    Both cluster-visible failures this repo has had were silent by construction:
    the diagnose stage fed every statistic an empty matrix and reported success,
    and the integrate stage would have written a comparison table containing only
    the methods that happened to import. A new `except ... : continue` is how that
    comes back, so a handler that neither re-raises nor calls `refuse` has to be
    justified here by name.
    """
    unexpected = [
        (f, fn, line, what) for f, fn, line, what in _handlers() if (f, fn) not in ALLOWED_SWALLOWS
    ]
    assert not unexpected, "\n".join(
        f"{f}:{line} `{what}` in {fn}() swallows a failure. Call "
        f"vgtfm.degraded.refuse(), re-raise, or justify it in ALLOWED_SWALLOWS."
        for f, fn, line, what in unexpected
    )


def test_the_allowlist_has_no_entries_for_code_that_is_gone():
    """A stale exemption is a hole nobody remembers opening."""
    live = {(f, fn) for f, fn, _line, _what in _handlers()}
    assert not (set(ALLOWED_SWALLOWS) - live), (
        f"remove from ALLOWED_SWALLOWS: {sorted(set(ALLOWED_SWALLOWS) - live)}"
    )


def test_the_pipeline_modules_import_the_policy_rather_than_reinventing_it():
    """Every stage that can come up short routes through the one module."""
    stages = [
        "diagnostics/report.py",
        "diagnostics/integration.py",
        "diagnostics/scib.py",
        "figures/build.py",
        "evaluate/run_eval.py",
        "biosignal/run_biosignal.py",
        "embed/build.py",
    ]
    for rel in stages:
        text = (ROOT / "vgtfm" / rel).read_text()
        assert "from ..degraded import" in text, f"vgtfm/{rel}"


# ── a table builder that comes back empty ────────────────────────────
#
# The inputs are present and non-empty, so `_source` is satisfied; it is the
# builder that produces nothing. Left alone these are the quietest failures in the
# stage — the artefact is simply absent from the list at the end.


def figures_run(tmp_path, **files):
    """Run the figures stage over a run directory containing only *files*."""
    from vgtfm.config import Config
    from vgtfm.figures import build

    cfg = Config()
    cfg.paths.artifact_root = str(tmp_path)
    for rel, frame in files.items():
        path = tmp_path / cfg.run_name / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    build.run(cfg)


def test_an_annotation_table_that_comes_back_empty_stops_the_stage(tmp_path):
    """`eval` recorded this protocol and level, but with no global-scope rows, so
    the table cannot be built from them. The combo came out of results.csv itself —
    a mismatch, not a stage that has not run."""
    results = pd.DataFrame(
        [
            {
                "model": "ae",
                "seed": 42,
                "protocol": "heldout_donor",
                "level": "cross_donor",
                "scope": "per_tissue",
                "fold": "pooled",
                "macro_f1": 0.4,
                "f1_score": 0.4,
            },
        ]
    )
    with pytest.raises(Incomplete, match="annotation table for heldout_donor"):
        figures_run(tmp_path, **{"eval/results.csv": results})


def test_a_patch_shuffle_table_that_comes_back_empty_stops_the_stage(tmp_path):
    """Table 3. `ablate` wrote per-class scores that carry no `condition`, so there
    is nothing to compare shuffled against unshuffled."""
    per_class = pd.DataFrame(
        [
            {
                "model": "ae",
                "protocol": "heldout_donor",
                "level": "cross_donor",
                "scope": "global",
                "fold": "pooled",
                "class": "TUM",
                "f1_score": 0.4,
            },
        ]
    )
    with pytest.raises(Incomplete, match="patch-shuffle table"):
        figures_run(tmp_path, **{"ablation/per_class.csv": per_class})
