"""Refusing to produce less than was asked for.

Every artefact this pipeline can fail to produce is routed through here, and the
default is to stop: a ``scib_panel.csv`` that was never written and one written
from a broken install look the same to the ``figures`` stage, which would simply
omit the figure.

Two situations are distinguished, because only one of them is a defect:

``refuse``    The stage was asked for something and cannot deliver it — a package
              that will not import, a method that raised, an input that should be
              on disk and is not. This ends the run. :class:`Incomplete` subclasses
              ``SystemExit``, so ``run.py`` records it in the manifest as an error
              and the batch job exits non-zero.

``pending``   The artefact belongs to a stage that has not run yet in this output
              directory. ``figures`` is runnable mid-graph — the ``all`` job builds
              what it can before ``ablate`` and ``biosignal`` land — so "the
              ablation directory does not exist" is a schedule, not a fault. "The
              ablation directory exists but is empty" is a fault, and goes to
              ``refuse``.

There is no third case for skipping on error. Where not doing something is a
legitimate *choice* it is spelled as a config field — ``diagnostics.run_scib`` — so
the decision is recorded in ``config.resolved.json`` beside the results.
"""

from __future__ import annotations

from typing import NoReturn


class Incomplete(SystemExit):
    """A stage could not produce something it was asked for."""


def refuse(what: str, why: str, *, hint: str | None = None) -> NoReturn:
    """Stop the run, naming the artefact, the cause, and the way out."""
    lines = [f"cannot produce {what}: {why}"]
    if hint:
        lines.append(f"  -> {hint}")
    raise Incomplete("\n".join(lines))


def pending(what: str, why: str) -> str:
    """Note an artefact whose upstream stage has not run yet. Prints and returns it."""
    msg = f"{what}: {why}"
    print(f"  pending  {msg}")
    return msg


def report_pending(items: list[str], *, stage: str) -> None:
    """Repeat the pending list at the end of a stage, where it cannot scroll away."""
    if not items:
        return
    print(f"\n  {len(items)} artefact(s) the {stage} stage did not build yet:")
    for it in items:
        print(f"    - {it}")
    print("  Re-run this stage once the stages above have finished.")
