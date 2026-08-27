"""One long-format table holding every score the pipeline produced.

Each stage writes its own artefacts in its own shape — ``eval`` and ``ablate`` in a
long format keyed by (model, seed, protocol, level, fold, scope, class), ``integrate``
in a wide one keyed by method with the scope and level folded into the column names,
``diagnose`` in neither. Tables were then assembled by reading whichever file was
nearest, which is how a number can appear twice in a manuscript with two values: the
reader has no way to tell that two tables sourced the same quantity from two stages
that computed it differently.

This module is the single place those artefacts become comparable. :func:`collect`
normalises every stage into one frame with one row per scored cell::

    run_name substrate stage model condition seed protocol level fold scope class
    metric value ci_lo ci_hi p_two_sided n_donors n_boot source

and :func:`conflicts` then asks the question that matters: *does any cell that two
stages both claim carry two different values?* A stage recomputing a quantity another
stage already produced is not by itself a bug — but the two disagreeing is, and until
this frame existed nothing in the repository could notice.

The correspondences that make that check possible are domain knowledge, not
inference, so they are written down in :data:`ALIASES`.
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

#: Columns every collected row carries, in order. A stage that does not have a
#: concept (``condition`` outside ``ablate``, ``fold`` outside the probe stages)
#: leaves it empty rather than absent, so the frame concatenates without holes.
SCHEMA = (
    "run_name",
    "substrate",
    "stage",
    "model",
    "condition",
    "seed",
    "protocol",
    "level",
    "fold",
    "scope",
    "class",
    "metric",
    "value",
    "ci_lo",
    "ci_hi",
    "p_two_sided",
    "n_donors",
    "n_boot",
    "source",
)

#: ``(stage, model)`` pairs that are the same quantity computed in two places, and
#: must therefore agree wherever both define a cell.
#:
#: ``integrate``'s uncorrected row *is* the ``pca`` baseline: since
#: :func:`vgtfm.diagnostics.integration._reference_embedding` the stage corrects the
#: embedding ``train`` fitted and ``eval`` scores, so the two are the same
#: representation under the same probe at the same folds. They once were not — the
#: stage fitted its own PCA over every spot in the cohort, the evaluated donors'
#: included, and the resulting disagreement with Table 1 went unnoticed because
#: nothing compared them.
ALIASES: dict[tuple[str, str], tuple[str, str]] = {
    ("integrate", "none"): ("eval", "pca"),
    # `diagnose` scores the scIB panel on the same `pca` embedding, through the same
    # `scib.benchmark` call with the same subsample seed, so its row and
    # `integrate`'s uncorrected row are one computation written down twice. They
    # differed while `integrate` corrected a PCA of its own — which is how the batch
    # columns of the integration table came to describe a different matrix from the
    # one Table 2's panel describes.
    ("diagnose", "pca"): ("eval", "pca"),
    # Deliberately absent: `ablate`'s `none` condition and `eval`'s `ae` row. The
    # code says they are one fit — same model, same `fit_rows`, same seed, and
    # `patch_transform="none"` touches neither the features nor the rng — but the fit
    # is not bit-reproducible across machines, and the two stages are separate jobs.
    # On the paper runs they land on different GPU architectures under bf16 and
    # early stopping magnifies the drift into a different epoch count (seed 44: 75
    # epochs in `train`, 199 in `ablate`), for a macro-F1 gap of up to 0.050. They
    # are two draws of one nominal fit, not one quantity written down twice, and the
    # ablation's own claim needs its reference fitted beside its treatments rather
    # than loaded from another node.
}

#: Metrics in ``eval``/``ablate``'s wide metric columns worth carrying across.
_POINT_METRICS = ("accuracy", "precision", "recall", "f1_score")

#: ``integrate``'s batch-mixing columns are cohort-wide by construction — the scIB
#: panel scores one stratified sample of the whole annotated cohort — so they are
#: recorded at this scope rather than at a tissue's.
COHORT_SCOPE = "cohort"


def _read(path: Path) -> pd.DataFrame:
    """One artefact, or an empty frame if the stage has not written it."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df if not df.empty else pd.DataFrame()


def _frame(rows: list[dict]) -> pd.DataFrame:
    """A frame carrying exactly :data:`SCHEMA`, empty columns included."""
    if not rows:
        return pd.DataFrame(columns=list(SCHEMA))
    df = pd.DataFrame(rows)
    for col in SCHEMA:
        if col not in df.columns:
            df[col] = np.nan
    return df[list(SCHEMA)]


def _base(df: pd.DataFrame, run_name: str, substrate: str, stage: str, source: str) -> dict:
    return {"run_name": run_name, "substrate": substrate, "stage": stage, "source": source}


def _melt_probe(
    df: pd.DataFrame, *, stage: str, source: str, run_name: str, substrate: str
) -> list[dict]:
    """``eval``/``ablate``'s point-metric tables, one row per metric per cell."""
    rows = []
    for r in df.to_dict("records"):
        common = {
            **_base(df, run_name, substrate, stage, source),
            "model": r.get("model", ""),
            "condition": r.get("condition", ""),
            "seed": r.get("seed"),
            "protocol": r.get("protocol", ""),
            "level": r.get("level", ""),
            "fold": r.get("fold", ""),
            "scope": r.get("scope", ""),
            "class": r.get("class", "macro"),
            "n_donors": r.get("n_donors"),
        }
        for metric in _POINT_METRICS:
            if metric in r and r[metric] == r[metric]:
                rows.append({**common, "metric": metric, "value": r[metric]})
    return rows


def _melt_bootstrap(
    df: pd.DataFrame, *, stage: str, source: str, run_name: str, substrate: str
) -> list[dict]:
    """The interval tables. ``value`` is the point estimate the CI brackets.

    The pooled-over-seeds interval is preferred where the stage wrote one, matching
    :func:`vgtfm.figures.tables._pooled_or_per_seed`: a per-seed interval describes
    one seed, and the tables quote the pooled form wherever it exists.
    """
    rows = []
    has_pooled = "ci_lo_pooled" in df.columns
    for r in df.to_dict("records"):
        lo, hi = r.get("ci_lo"), r.get("ci_hi")
        if has_pooled and r.get("ci_lo_pooled") == r.get("ci_lo_pooled"):
            lo, hi = r["ci_lo_pooled"], r["ci_hi_pooled"]
        rows.append(
            {
                **_base(df, run_name, substrate, stage, source),
                "model": r.get("model", ""),
                "condition": r.get("condition", ""),
                "seed": r.get("seed"),
                "protocol": r.get("protocol", ""),
                "level": r.get("level", ""),
                "fold": r.get("fold", ""),
                "scope": r.get("scope", ""),
                "class": r.get("class", "macro"),
                "metric": "f1_score",
                "value": r.get("value"),
                "ci_lo": lo,
                "ci_hi": hi,
                "n_donors": r.get("n_donors"),
                "n_boot": r.get("n_boot"),
            }
        )
    return rows


def _melt_deltas(
    df: pd.DataFrame,
    *,
    stage: str,
    source: str,
    run_name: str,
    substrate: str,
    protocol: str = "",
    seed=None,
) -> list[dict]:
    """Paired deltas, recorded under ``metric='delta_f1'``.

    The model a delta belongs to is whichever column the stage varies: ``eval``
    varies the model against a fixed reference, ``ablate`` varies the condition.
    """
    rows = []
    has_pooled = "ci_lo_pooled" in df.columns
    for r in df.to_dict("records"):
        lo, hi = r.get("ci_lo"), r.get("ci_hi")
        p = r.get("p_two_sided")
        if has_pooled and r.get("ci_lo_pooled") == r.get("ci_lo_pooled"):
            lo, hi = r["ci_lo_pooled"], r["ci_hi_pooled"]
            p = r.get("p_two_sided_pooled", p)
        rows.append(
            {
                **_base(df, run_name, substrate, stage, source),
                "model": r.get("model", r.get("method", "")),
                "condition": r.get("condition", ""),
                "seed": r.get("seed", seed) if r.get("seed") == r.get("seed") else seed,
                "protocol": r.get("protocol") or protocol,
                "level": r.get("level", ""),
                "fold": "pooled",
                "scope": r.get("scope", ""),
                "class": r.get("class", "macro"),
                "metric": "delta_f1",
                "value": r.get("delta", r.get("delta_f1")),
                "ci_lo": lo,
                "ci_hi": hi,
                "p_two_sided": p,
                "n_donors": r.get("n_donors"),
                "n_boot": r.get("n_boot"),
            }
        )
    return rows


def _melt_integration(
    df: pd.DataFrame, *, source: str, run_name: str, substrate: str, seed, protocol: str
) -> list[dict]:
    """``integrate``'s wide table, unfolded into the shared schema.

    Its column names carry two different things. ``batch/<m>`` and ``bio/<m>`` are
    scIB panel metrics, cohort-wide (see :data:`COHORT_SCOPE`). ``f1/<scope>/<level>``
    is the probe, with the scope and the fold level folded into the name and the
    interval hung off it as ``_lo``/``_hi``/``_n_donors`` — which is exactly the
    shape that made this table impossible to compare with ``eval``'s.
    """
    rows = []
    for r in df.to_dict("records"):
        # The row's own seed where the stage recorded one; the resolved config only
        # as a fallback for a table written before it did. The two must not disagree
        # silently: a seed read from the wrong place keys every cell apart from its
        # counterpart in `eval`, and the check then passes by comparing nothing.
        own = r.get("seed")
        common = {
            **_base(df, run_name, substrate, "integrate", source),
            "model": r.get("method", ""),
            "seed": own if own == own and own is not None else seed,
        }
        for col, v in r.items():
            if not isinstance(col, str) or v != v:
                continue
            parts = col.split("/")
            if len(parts) == 2 and parts[0] in ("batch", "bio"):
                rows.append(
                    {
                        **common,
                        "scope": COHORT_SCOPE,
                        "class": "macro",
                        "metric": parts[1],
                        "value": v,
                    }
                )
            # A point estimate is a column with an interval hanging off it, the same
            # rule `integration._print_probe` uses. Matching on the shape alone would
            # read `f1/kidney/cross_donor_lo` as a fold level named `cross_donor_lo`.
            elif len(parts) == 3 and parts[0] == "f1" and f"{col}_lo" in r:
                _, scope, level = parts
                # The seed-pooled bracket where the stage wrote one, as
                # :func:`_melt_bootstrap` prefers it for ``eval``: a per-seed
                # interval describes one fit, and these rows now come one per fit.
                lo, hi = r.get(f"{col}_lo"), r.get(f"{col}_hi")
                if (
                    r.get(f"{col}_lo_pooled") == r.get(f"{col}_lo_pooled")
                    and f"{col}_lo_pooled" in r
                ):
                    lo, hi = r[f"{col}_lo_pooled"], r[f"{col}_hi_pooled"]
                rows.append(
                    {
                        **common,
                        "protocol": protocol,
                        "level": level,
                        "fold": "pooled",
                        "scope": scope,
                        "class": "macro",
                        "metric": "f1_score",
                        "value": v,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "n_donors": r.get(f"{col}_n_donors"),
                    }
                )
    return rows


def _melt_effective_rank(
    df: pd.DataFrame, *, run_name: str, substrate: str, source: str
) -> list[dict]:
    return [
        {
            **_base(df, run_name, substrate, "diagnose", source),
            "model": r.get("column", ""),
            "condition": r.get("variant", ""),
            "scope": COHORT_SCOPE,
            "class": "macro",
            "metric": "eff_rank",
            "value": r.get("eff_rank"),
        }
        for r in df.to_dict("records")
    ]


#: ``diagnose`` names a scored representation ``"<model> (seed <n>)"``. The seed
#: belongs in its own column, or the model can never be matched against the same
#: model scored by another stage.
_SEEDED = re.compile(r"^(?P<model>.*?)\s*\(seed\s+(?P<seed>\d+)\)$")


def _melt_scib_panel(df: pd.DataFrame, *, run_name: str, substrate: str, source: str) -> list[dict]:
    rows = []
    for r in df.to_dict("records"):
        name = str(r.get("representation", ""))
        m = _SEEDED.match(name)
        model, seed = (m["model"], int(m["seed"])) if m else (name, np.nan)
        common = {
            **_base(df, run_name, substrate, "diagnose", source),
            "model": model,
            "seed": seed,
            "scope": COHORT_SCOPE,
            "class": "macro",
        }
        for col, v in r.items():
            if isinstance(col, str) and "/" in col and v == v:
                rows.append({**common, "metric": col.split("/", 1)[1], "value": v})
    return rows


def collect_run(run_dir: Path) -> pd.DataFrame:
    """Every score one run directory holds, in the shared schema.

    Missing stages are skipped rather than refused: a run that has not been
    integrated yet is a partial run, not a broken one, and the caller decides
    whether the stages it needs are present.
    """
    run_dir = Path(run_dir)
    cfg_path = run_dir / "config.resolved.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    substrate = cfg.get("data", {}).get("substrate", "")
    run_name = cfg.get("run_name", run_dir.name)
    seed = cfg.get("diagnostics", {}).get("seed")
    # `integrate` scores through `proto.heldout_donor` directly, so its rows belong
    # to that protocol even though its CSV never names one.
    protocol = "heldout_donor"

    rows: list[dict] = []
    for stage, stem in (("eval", "eval"), ("ablate", "ablation")):
        kw = {"stage": stage, "run_name": run_name, "substrate": substrate}
        rows += _melt_probe(
            _read(run_dir / stem / "results.csv"), source=f"{stem}/results.csv", **kw
        )
        rows += _melt_probe(
            _read(run_dir / stem / "per_class.csv"), source=f"{stem}/per_class.csv", **kw
        )
        rows += _melt_bootstrap(
            _read(run_dir / stem / "bootstrap.csv"), source=f"{stem}/bootstrap.csv", **kw
        )
        deltas = "deltas.csv" if stage == "eval" else "paired_deltas.csv"
        rows += _melt_deltas(_read(run_dir / stem / deltas), source=f"{stem}/{deltas}", **kw)

    rows += _melt_integration(
        _read(run_dir / "integration" / "integration.csv"),
        source="integration/integration.csv",
        run_name=run_name,
        substrate=substrate,
        seed=seed,
        protocol=protocol,
    )
    rows += _melt_deltas(
        _read(run_dir / "integration" / "integration_deltas.csv"),
        stage="integrate",
        source="integration/integration_deltas.csv",
        run_name=run_name,
        substrate=substrate,
        protocol=protocol,
        seed=seed,
    )
    rows += _melt_effective_rank(
        _read(run_dir / "diagnostics" / "effective_rank.csv"),
        run_name=run_name,
        substrate=substrate,
        source="diagnostics/effective_rank.csv",
    )
    rows += _melt_scib_panel(
        _read(run_dir / "diagnostics" / "scib_panel.csv"),
        run_name=run_name,
        substrate=substrate,
        source="diagnostics/scib_panel.csv",
    )

    # No blanket backfill of `protocol` here: `integrate`'s batch-mixing rows are a
    # scIB panel, not a probe, and stamping a protocol on them is what stopped them
    # matching the identical panel `diagnose` writes.
    return _frame(rows)


def collect(artifact_root: Path, run_names) -> pd.DataFrame:
    """:func:`collect_run` over several run directories, concatenated."""
    frames = [collect_run(Path(artifact_root) / name) for name in run_names]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else _frame([])


#: Columns identifying one scored cell, once the stage and model have been resolved
#: through :data:`ALIASES`. ``seed`` is included: two stages that scored different
#: seeds are not in disagreement, they are describing different fits.
_CELL = ("run_name", "protocol", "level", "fold", "scope", "class", "metric", "seed")

#: The full grouping key: a cell, plus the identity it is compared under.
_KEYS = (*_CELL, "cmp_stage", "cmp_model", "condition")


def _resolve(df: pd.DataFrame) -> pd.DataFrame:
    """Add the ``(stage, model)`` each row is compared *as*, per :data:`ALIASES`."""
    out = df.copy()
    key = list(zip(out["stage"].astype(str), out["model"].astype(str)))
    resolved = [ALIASES.get(k, k) for k in key]
    out["cmp_stage"] = [s for s, _m in resolved]
    out["cmp_model"] = [m for _s, m in resolved]
    return out


def _comparable(df: pd.DataFrame) -> pd.DataFrame:
    """:func:`_resolve`, with the key columns normalised so they can be grouped.

    An absent field becomes the empty string rather than NaN, which groups; the
    scored value has to be present, since a cell nobody scored is not a claim.
    """
    res = _resolve(df)
    res = res[res["value"].notna()].copy()
    for col in _KEYS:
        res[col] = res[col].fillna("").astype(str)
    return res


def _cell_keys(frame: pd.DataFrame) -> set[tuple]:
    return set(map(tuple, frame[list(_KEYS)].to_numpy()))


def conflicts(df: pd.DataFrame, *, tol: float = 5e-4) -> pd.DataFrame:
    """Cells that two artefacts both claim and disagree on, beyond *tol*.

    Only cells an alias makes comparable are compared *across* stages; two stages
    independently scoring unrelated things share no cell and cannot conflict. Within
    a stage the comparison is free: ``eval`` writes the macro-F1 of one prediction
    vector into ``results.csv`` through :func:`~vgtfm.evaluate.probes.classification_metrics`
    and into ``bootstrap.csv`` as the point estimate of
    :func:`~vgtfm.evaluate.bootstrap.donor_bootstrap`, and those are two code paths
    that must return the same number. Grouping by *source file* rather than by stage
    is what puts them in scope: on the three paper runs it raises the number of
    cross-checked cells from 54 to 10,515.

    *tol* is a hair under the third decimal, which is where the manuscript's
    integration tables are printed: a disagreement smaller than that cannot change a
    printed number, and floating-point paths that differ in their last bits should
    not fail a run.
    """
    if df.empty:
        return _frame([]).assign(n_sources=[], spread=[])
    res = _comparable(df)

    out = []
    for key, g in res.groupby(list(_KEYS), dropna=False):
        if g["source"].nunique() < 2:
            continue
        spread = float(g["value"].max() - g["value"].min())
        if spread <= tol:
            continue
        row = dict(zip(_KEYS, key))
        out.append(
            {
                **row,
                "n_sources": int(g["source"].nunique()),
                "spread": spread,
                "values": "; ".join(f"{s}={v:.6f}" for s, v in zip(g["source"], g["value"])),
            }
        )
    return pd.DataFrame(out)


def cross_checks(df: pd.DataFrame) -> pd.DataFrame:
    """How many cells were actually compared, and between which files.

    :func:`conflicts` returning nothing has two readings: every quantity computed
    twice agreed, or nothing was compared at all. They look identical in a log and
    only one of them is reassuring — a seed, a protocol or a level named differently
    by two stages makes their rows land in different cells, and the check then passes
    by never running. This is the count that tells the two apart, so print it beside
    the verdict rather than the verdict alone.
    """
    if df.empty:
        return pd.DataFrame(columns=["run_name", "metric", "n_cells", "sources"])
    res = _comparable(df)
    rows = []
    for key, g in res.groupby(list(_KEYS), dropna=False):
        if g["source"].nunique() < 2:
            continue
        cell = dict(zip(_KEYS, key))
        rows.append(
            {
                "run_name": cell["run_name"],
                "metric": cell["metric"],
                "sources": " x ".join(sorted(set(g["source"]))),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["run_name", "metric", "n_cells", "sources"])
    out = pd.DataFrame(rows)
    return (
        out.groupby(["run_name", "metric", "sources"])
        .size()
        .reset_index(name="n_cells")
        .sort_values(["run_name", "n_cells"], ascending=[True, False])[
            ["run_name", "metric", "n_cells", "sources"]
        ]
    )


def unexercised(df: pd.DataFrame) -> list[str]:
    """Pairs that should have compared something and compared nothing.

    Two ``(stage, model)`` groups meet only by resolving to the same identity
    through :data:`ALIASES`, so every pair considered here is one the repository
    asserts is the same representation. A pair is only *expected* to fire where both
    sides report the same metric: ``diagnose``'s panel and ``eval``'s probe are both
    the ``pca`` baseline and share no metric at all, so their alias lying idle in a
    run without ``integrate`` is correct rather than broken. Where the metrics do
    overlap and no cell is nonetheless shared, the two sides are keyed apart -- by a
    seed, a protocol or a level one of them names differently -- and the check has
    silently stopped checking.
    """
    if df.empty:
        return []
    res = _comparable(df)
    out = []
    for _identity, g in res.groupby(["cmp_stage", "cmp_model"], dropna=False):
        sides = dict(tuple(g.groupby(["stage", "model"], dropna=False)))
        for a, b in itertools.combinations(sorted(sides), 2):
            ga, gb = sides[a], sides[b]
            shared = sorted(set(ga["metric"]) & set(gb["metric"]))
            if not shared or _cell_keys(ga) & _cell_keys(gb):
                continue
            out.append(
                f"{a[0]}/{a[1]} and {b[0]}/{b[1]} both report "
                f"{', '.join(shared[:4])} but never on the same cell — they are "
                f"keyed apart by seed, protocol or level, so nothing was checked"
            )
    return out


def write(df: pd.DataFrame, out_dir: Path, conflict_df: pd.DataFrame | None = None):
    """``scores.csv``, and ``conflicts.csv`` beside it when there are any."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = out_dir / "scores.csv"
    df.to_csv(scores, index=False)
    written = [scores]
    if conflict_df is not None and not conflict_df.empty:
        path = out_dir / "conflicts.csv"
        conflict_df.to_csv(path, index=False)
        written.append(path)
    return written


def run(cfg) -> None:
    """``results`` stage — join this run's stages and refuse if any two disagree.

    Within one run directory, because that is where the duplication lives: the
    aliases in :data:`ALIASES` pair two stages of the *same* run scoring the same
    representation. Cross-run assembly is the manuscript's job and belongs to
    ``scripts/paper_tables.py``, which reads the same frame.
    """
    from .degraded import refuse

    df = collect_run(cfg.out_dir)
    if df.empty:
        refuse(
            "a results table",
            "no stage has written a score in this run",
            hint="run `python run.py eval` first",
        )
    conf = conflicts(df)
    out = cfg.sub("results")
    for path in write(df, out, conf):
        print(f"  wrote {path}")

    print(
        f"\n  {len(df):,} scored cell(s) from "
        f"{df['stage'].nunique()} stage(s): "
        + ", ".join(f"{s}={n:,}" for s, n in df.groupby("stage").size().items())
    )

    if not conf.empty:
        print(f"\n  {len(conf)} cell(s) claimed twice with different values:")
        print(conf[["level", "scope", "metric", "spread", "values"]].to_string(index=False))
        refuse(
            "a consistent results table",
            f"{len(conf)} cell(s) are computed twice with different values",
            hint="two stages disagree about one quantity; conflicts.csv names "
            "the files. Fix the stage that recomputes rather than reuses.",
        )

    checks = cross_checks(df)
    total = int(checks["n_cells"].sum()) if not checks.empty else 0
    print(f"\n  {total:,} cell(s) computed more than once, all in agreement:")
    if total:
        print(checks[["metric", "n_cells", "sources"]].to_string(index=False))

    idle = unexercised(df)
    if idle:
        for line in idle:
            print(f"  {line}")
        refuse(
            "a meaningful consistency check",
            f"{len(idle)} alias(es) compared nothing",
            hint="the two stages disagree about how a cell is named, so the "
            "check passes by never running. Reconcile the seed, protocol "
            "or level they write, or drop the alias from results.ALIASES.",
        )
