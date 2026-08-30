"""``figures`` stage — regenerate every manuscript figure and table from artefacts.

Reads only what previous stages wrote, and writes PDF (vector, for LaTeX), PNG (for
review) and ``.tex`` fragments side by side.

This stage is meant to run before all its inputs exist: the ``all`` job builds what
it can, then a second pass runs once ``ablate`` and ``biosignal`` have landed. "The
input is not there" is therefore ambiguous, and the two cases are separated by
whether the producing stage left a directory behind:

* no ``ablation/`` directory, or an empty one -> ``ablate`` has left nothing here.
  Pending, listed again at the end of the stage, not an error. (An empty directory
  means it died on its first line, and that job failed on its own.)
* ``ablation/`` holds other outputs but not this one, or holds one with no rows ->
  ``ablate`` ran and came up short. That is a defect and stops the run.

A figure is never dropped because a library would not import; every plotting
dependency is pinned in requirements.txt and checked by ``slurm/preflight.sh``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..biosignal import ridge as ridge_mod
from ..data.tables import load_registry
from ..degraded import pending, refuse, report_pending
from ..config import headline_seed
from ..evaluate.reporting import TISSUE_BALANCED
from ..plotting import (
    MODEL_COLORS,
    RASTER_MIN_POINTS,
    model_color,
    save,
    scatter_categorical_2d,
    scatter_labels_2d,
    set_pub_style,
)
from . import tables as tbl


def _integration_scopes(integration: pd.DataFrame) -> list[str]:
    """Scopes the ``integrate`` stage scored, read off its probe columns.

    The columns are ``f1/<scope>/<level>`` plus interval suffixes. A run made before
    the stage scored per tissue carries ``f1/<level>`` instead and yields the pooled
    scope alone, so an old run directory still builds its one table.
    """
    scopes = {
        c.split("/")[1] for c in integration.columns if c.startswith("f1/") and c.count("/") >= 2
    }
    if not scopes:
        return ["global"] if any(c.startswith("f1/") for c in integration.columns) else []
    # Pooled first, then organs alphabetically: the order the tables are announced in.
    return sorted(scopes, key=lambda s: (s != "global", s))


def _source(
    directory, filename: str, *, what: str, stage: str, notes: list, required: bool = True
) -> pd.DataFrame:
    """Read one upstream artefact, distinguishing "not yet" from "broken".

    An empty frame comes back in exactly one case: *stage* has left nothing here,
    either because it has not run or because it died before writing its first output
    — and in that second case its own job has already failed. Anything else — the
    stage wrote other files but not this one, or wrote one with no rows — ends the
    run rather than dropping the figure.

    ``required=False`` marks the artefacts a stage writes only under some
    configurations (``eval`` skips ``deltas.csv`` with no reference model, and
    ``bootstrap.csv`` when ``eval.bootstrap_n`` is 0). Those are still announced.
    """
    if not directory.exists() or not any(directory.iterdir()):
        # An empty directory carries no information: `cfg.sub()` creates it before
        # the stage does any work, so one is left behind by a stage that died on its
        # first line — and that job has already failed. A directory with *some*
        # output but not this one is the case worth stopping for.
        notes.append(
            pending(what, f"the `{stage}` stage has not produced anything in this run directory")
        )
        return pd.DataFrame()
    path = directory / filename
    if not path.exists():
        if not required:
            notes.append(pending(what, f"`{stage}` ran but wrote no {filename}"))
            return pd.DataFrame()
        refuse(what, f"`{stage}` ran but did not write {path}")
    df = pd.read_csv(path)
    if df.empty:
        refuse(what, f"{path} has no rows")
    return df


def run(cfg) -> None:
    set_pub_style()
    out = cfg.sub("figures")
    data_dir = cfg.out_dir / "data"
    eval_dir = cfg.out_dir / "eval"
    diag_dir = cfg.out_dir / "diagnostics"
    abl_dir = cfg.out_dir / "ablation"
    int_dir = cfg.out_dir / "integration"
    # Same rule the biosignal stage writes under, so a scoped run's figures find
    # the scope that produced them rather than a stale sibling.
    bio_dir = cfg.out_dir / "biosignal" / ridge_mod.scope_name(cfg.biosignal.tissues)

    notes: list[str] = []
    results = _source(
        eval_dir, "results.csv", what="the annotation table and figure", stage="eval", notes=notes
    )
    bootstrap = _source(
        eval_dir,
        "bootstrap.csv",
        what="bootstrap intervals",
        stage="eval",
        notes=notes,
        required=False,
    )
    deltas = _source(
        eval_dir,
        "deltas.csv",
        what="the paired delta tables",
        stage="eval",
        notes=notes,
        required=False,
    )
    er = _source(
        diag_dir,
        "effective_rank.csv",
        what="the effective-rank table",
        stage="diagnose",
        notes=notes,
    )
    var = _source(
        diag_dir,
        "variance_decomposition.csv",
        what="the variance figure",
        stage="diagnose",
        notes=notes,
    )
    abl_class = _source(
        abl_dir, "per_class.csv", what="the patch-shuffle table", stage="ablate", notes=notes
    )
    abl_boot = _source(
        abl_dir,
        "bootstrap.csv",
        what="the patch-shuffle intervals",
        stage="ablate",
        notes=notes,
        required=False,
    )
    abl_delta = _source(
        abl_dir,
        "paired_deltas.csv",
        what="the patch-shuffle delta figure",
        stage="ablate",
        notes=notes,
        required=False,
    )
    per_gene = _source(
        bio_dir, "per_gene_r2.csv", what="the per-gene R^2 figure", stage="biosignal", notes=notes
    )
    integration = _source(
        int_dir, "integration.csv", what="the integration table", stage="integrate", notes=notes
    )
    # Absent from a run made with `eval.bootstrap_n = 0`, and from any run made
    # before the probe carried intervals. The table then prints `--` in its delta
    # columns rather than going missing.
    int_deltas = _source(
        int_dir,
        "integration_deltas.csv",
        what="the integration delta intervals",
        stage="integrate",
        notes=notes,
        required=False,
    )
    cohort = _source(data_dir, "cohort.csv", what="the dataset table", stage="data", notes=notes)
    fold_map = _fold_map(data_dir, notes)

    made = []

    if not cohort.empty:
        # The dataset appendix, built from the cohort table the `data` stage wrote.
        tbl.dataset_table(
            cohort, load_registry(cfg.paths.datasets_json), out, fit_split=cfg.train.fit_split
        )
        made.append("table_datasets")
        if not tbl.fold_hierarchy_table(cohort, fold_map, out).empty:
            made.append("table_fold_hierarchy")

    if not results.empty:
        made.extend(_annotation_tables(results, bootstrap, out))

    if not er.empty:
        tbl.effective_rank_table(er, out)
        made.append("table_effective_rank")
        _fig_effective_rank(er, out)
        made.append("fig_effective_rank")

    if not var.empty:
        _fig_variance(var, out)
        made.append("fig_variance")

    if not deltas.empty:
        for protocol, level in (
            deltas[["protocol", "level"]]
            .drop_duplicates()
            .sort_values(["protocol", "level"])
            .itertuples(index=False)
        ):
            stem = f"table_deltas_{protocol}_{level}"
            if tbl.delta_table(deltas, out, protocol=protocol, level=level, stem=stem).empty:
                refuse(
                    f"the paired delta table for {protocol}/{level}",
                    "deltas.csv lists that protocol and level but holds no macro-class rows for it",
                )
            made.append(stem)

    if not abl_class.empty:
        s = tbl.shuffle_table(abl_class, out, bootstrap=abl_boot)
        if s.empty:
            # `ablate` ran and wrote per-class scores, so an empty shuffle table is
            # a mismatch between them, not a missing stage.
            refuse(
                "the patch-shuffle table",
                "ablation/per_class.csv has rows but none of them are pooled "
                "per-class scores with a `condition`",
            )
        made.append("table_patch_shuffle")
        _fig_shuffle(s, out)
        made.append("fig_patch_shuffle")
    if not abl_delta.empty:
        _fig_shuffle_deltas(abl_delta, out)
        made.append("fig_patch_shuffle_deltas")

    if not per_gene.empty:
        levels = _r2_levels(per_gene, cfg.biosignal.per_gene_r2_levels)
        _fig_r2_scatter(per_gene, out, levels)
        made.append("fig_per_gene_r2")
        if "r2_pca_control" in per_gene:
            _fig_r2_vs_control(per_gene, out, levels)
            made.append("fig_per_gene_r2_vs_control")
        for gene_set, level, table in _setmean_sources(bio_dir, cfg, notes):
            stem = f"fig_setmean_{gene_set}_{level}"
            _fig_setmean(table, out, stem, fdr=cfg.biosignal.fdr)
            made.append(stem)

    if not integration.empty:
        # One table per scope the stage scored: the pooled cohort view first, then
        # one per organ, named the way `eval` names its per-organ tables. The scopes
        # are read off the columns rather than hardcoded, so a cohort with a fourth
        # tissue picks it up without an edit here.
        for scope in _integration_scopes(integration):
            tbl.integration_table(integration, int_deltas, out, scope=scope)
            made.append(tbl.integration_stem(scope))

    _fig_umaps(cfg, out, made, notes)

    if not made:
        refuse(
            "any figure or table",
            "none of the upstream stages has left output in this run directory",
            hint=f"run `python run.py all` first; artefacts are read from {cfg.out_dir}",
        )

    print(f"\n  wrote {len(made)} artefact(s) to {out}:")
    for name in made:
        print(f"    {name}")
    report_pending(notes, stage="figures")


# ── figures ──────────────────────────────────────────────────────────


def _fold_map(data_dir, notes) -> dict:
    """The fold structure the ``data`` stage recorded, or ``{}`` if it has not run.

    Not routed through :func:`_source`: ``folds.json`` is a mapping of level to fold
    list, not a frame, and the hierarchy table degrades to the cohort's own nesting
    without it rather than failing.
    """
    path = data_dir / "folds.json"
    if not path.exists():
        notes.append(
            pending(
                "the fold hierarchy table's held-out column", "`data` ran but wrote no folds.json"
            )
        )
        return {}
    return json.loads(path.read_text())


def _annotation_tables(results, bootstrap, out) -> list[str]:
    """Table 1 and its figure, once per ``(protocol, level)``, plus per-organ tables.

    Every combination present in ``results.csv`` gets a table, so the strict protocol
    and the legacy one are always shown side by side.

    A level that spans more than one cohort also gets one table per organ. Its
    ``global`` row is a macro over ``(tissue, class)`` cells — the right number for
    comparing representations over the whole annotated cohort, and the wrong one to
    read a single organ off, since a cell that only kidney can contain is in the
    average. The per-organ tables are what a claim about one cohort should quote.
    """
    made: list[str] = []
    combos = results[["protocol", "level"]].drop_duplicates().sort_values(["protocol", "level"])
    for protocol, level in combos.itertuples(index=False):
        stem = f"table_annotation_{protocol}_{level}"
        t1 = tbl.annotation_table(
            results, bootstrap, out, protocol=protocol, level=level, stem=stem
        )
        if t1.empty:
            # The combo came out of results.csv itself, so an empty table means the
            # rows are there without the global scope the table reads.
            refuse(
                f"the annotation table for {protocol}/{level}",
                "eval recorded that protocol and level but no global-scope rows for it",
            )
        made.append(stem)
        _fig_annotation(t1, results, out, protocol, level)
        made.append(f"fig_annotation_{protocol}_{level}")

        view = results[(results.protocol == protocol) & (results.level == level)]
        organs = sorted(set(view.scope) - {"global", TISSUE_BALANCED})
        if len(organs) < 2:
            continue  # the global table already is that organ's

        # The headline table for a multi-organ level: one row per representation,
        # each organ weighted equally, with a two-level interval. The `global` table
        # above stays as the pooled-vocabulary view it has always been.
        if TISSUE_BALANCED in set(view.scope):
            bal_stem = f"{stem}_organ_balanced"
            if not tbl.annotation_table(
                results,
                bootstrap,
                out,
                protocol=protocol,
                level=level,
                scope=TISSUE_BALANCED,
                stem=bal_stem,
            ).empty:
                made.append(bal_stem)

        # The spread behind that one number, as a single wide table rather than one
        # file per organ — too large for the main text, and the first thing a
        # reviewer asks for.
        organ_stem = f"table_annotation_by_organ_{protocol}_{level}"
        if not tbl.per_organ_table(
            results, bootstrap, out, protocol=protocol, level=level, stem=organ_stem
        ).empty:
            made.append(organ_stem)

        for organ in organs:
            per_organ_stem = f"{stem}_{organ}"
            if not tbl.annotation_table(
                results,
                bootstrap,
                out,
                protocol=protocol,
                level=level,
                scope=organ,
                stem=per_organ_stem,
            ).empty:
                made.append(per_organ_stem)
    return made


def _fig_annotation(table: pd.DataFrame, results: pd.DataFrame, out, protocol: str, level: str):
    """Headline figure: macro-F1 per model with per-fold dots and reference lines."""
    import matplotlib.pyplot as plt

    models = [m for m in table["model"] if m not in ("pca_oracle", "majority")]
    if not models:
        # Nothing but reference lines: a bar chart of the baselines is not a figure.
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(models))
    heights = [float(table.loc[table.model == m, "macro_f1"].iloc[0]) for m in models]
    ax.bar(x, heights, width=0.6, color=[model_color(m, i) for i, m in enumerate(models)])

    per_fold = results[
        (results.protocol == protocol)
        & (results.level == level)
        & (results.scope == "global")
        & (results.fold != "pooled")
    ]
    for i, m in enumerate(models):
        vals = per_fold.loc[per_fold.model == m, "f1_score"].to_numpy()
        if len(vals):
            jitter = (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.28
            ax.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                s=9,
                color="black",
                alpha=0.55,
                zorder=3,
                linewidths=0,
            )

    for name, style, label in (
        ("pca_oracle", "--", "PCA oracle (H&E)"),
        ("majority", ":", "majority baseline"),
    ):
        row = table.loc[table.model == name, "macro_f1"]
        if len(row):
            y = float(row.iloc[0])
            ax.axhline(y, ls=style, color=MODEL_COLORS.get(name, "#555"), lw=1.2)
            ax.text(
                len(models) - 0.45,
                y,
                f" {label} ({y:.3f})",
                va="bottom",
                ha="right",
                fontsize=7,
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.0),
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            tbl.model_label(m)
            .replace("\\&", "&")
            .replace("\\_", "_")
            .replace(r"$\rightarrow$", "→")
            for m in models
        ],
        fontsize=8,
    )
    ax.set_ylabel("annotation macro-F1")
    ax.set_title(f"{level.replace('_', '-')} annotation prediction ({protocol})")
    ax.set_ylim(0, max(1.0, max(heights) * 1.35))
    save(fig, out / f"fig_annotation_{protocol}_{level}")


def _fig_effective_rank(er: pd.DataFrame, out):
    import matplotlib.pyplot as plt

    obs = er[er.variant == "observed"]
    ctrl = er[er.variant != "observed"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    labels = [c.replace("_features", "") for c in obs["column"]]
    x = np.arange(len(labels))
    ax.bar(x, obs["eff_rank"].to_numpy(dtype=float), width=0.55, color="#1f77b4", label="observed")
    for i, (_, row) in enumerate(obs.iterrows()):
        ax.text(
            i,
            row["eff_rank"],
            f" {row['eff_rank']:.0f}/{int(row['dim'])}\n {row['pct_of_dim']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    for j, variant in enumerate(sorted(set(ctrl["variant"]))):
        sub = ctrl[ctrl.variant == variant].set_index("column").reindex(obs["column"])
        ax.scatter(
            x,
            sub["eff_rank"].to_numpy(dtype=float),
            marker="_",
            s=260,
            linewidths=2,
            label=variant,
            zorder=3,
            color=["#d62728", "#7f7f7f", "#2ca02c"][j % 3],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylabel("effective rank (log scale)")
    ax.set_title("Frozen embeddings use a small fraction of their dimensions")
    ax.legend(frameon=False, fontsize=8)
    save(fig, out / "fig_effective_rank")


def _fig_variance(var: pd.DataFrame, out):
    import matplotlib.pyplot as plt

    v = var[var.grouping == "sample_id"]
    if v.empty:
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    labels = [c.replace("_features", "") for c in v["column"]]
    x = np.arange(len(labels))
    between = v["between_fraction"].to_numpy(dtype=float)
    within = v["within_fraction"].to_numpy(dtype=float)
    ax.bar(x, between, width=0.5, color="#d62728", label="between slides")
    ax.bar(x, within, width=0.5, bottom=between, color="#c9c9c9", label="within slides")
    for i, val in enumerate(v["between_fraction"]):
        ax.text(i, val / 2, f"{val:.0%}", ha="center", va="center", color="white", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of total variance")
    ax.set_title("Slide identity accounts for much of the embedding variance")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, out / "fig_variance")


def _fig_shuffle(df: pd.DataFrame, out):
    import matplotlib.pyplot as plt

    conditions = [
        c for c in ("none", "within-sample", "global", "gaussian") if c in set(df["condition"])
    ]
    keys = df[["tissue", "class"]].drop_duplicates().sort_values(["tissue", "class"])
    if not conditions or keys.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6.4, 0.55 * len(keys) * len(conditions)), 3.8))
    width = 0.8 / max(len(conditions), 1)
    colors = {
        "none": "#2ca02c",
        "within-sample": "#ff7f0e",
        "global": "#d62728",
        "gaussian": "#7f7f7f",
    }
    for i, cond in enumerate(conditions):
        vals = [
            float(df[(df.tissue == t) & (df["class"] == c) & (df.condition == cond)]["f1"].mean())
            for t, c in keys.itertuples(index=False)
        ]
        ax.bar(
            np.arange(len(keys)) + i * width,
            vals,
            width=width,
            color=colors.get(cond, None),
            label={"none": "matched"}.get(cond, cond),
        )
    ax.set_xticks(np.arange(len(keys)) + 0.4 - width / 2)
    ax.set_xticklabels([f"{t}\n{c}" for t, c in keys.itertuples(index=False)], fontsize=7)
    ax.set_ylabel("per-class F1")
    ax.set_title("Breaking the gene/morphology correspondence during training")
    ax.legend(frameon=False, fontsize=8)
    save(fig, out / "fig_patch_shuffle")


def _fig_shuffle_deltas(delta: pd.DataFrame, out):
    """Matched-minus-shuffled macro-F1 with donor-level intervals."""
    import matplotlib.pyplot as plt

    d = delta[(delta["class"] == "macro") & (delta.scope != "global")]
    if d.empty:
        return
    d = (
        d.groupby(["scope", "condition"])
        .agg(delta=("delta", "mean"), ci_lo=("ci_lo", "mean"), ci_hi=("ci_hi", "mean"))
        .reset_index()
    )
    conditions = sorted(set(d["condition"]))
    scopes = sorted(set(d["scope"]))
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    width = 0.8 / max(len(conditions), 1)
    for i, cond in enumerate(conditions):
        sub = d[d.condition == cond].set_index("scope").reindex(scopes)
        # Arrays, and one (2, n) array for yerr rather than a list of two: with a
        # single scope matplotlib unwraps either a Series or a list element as a
        # scalar, which is deprecated and slated to become a TypeError.
        delta = sub["delta"].to_numpy(dtype=float)
        lo = sub["ci_lo"].to_numpy(dtype=float)
        hi = sub["ci_hi"].to_numpy(dtype=float)
        x = np.arange(len(scopes)) + i * width
        ax.bar(x, delta, width=width, label=cond)
        ax.errorbar(
            x,
            delta,
            yerr=np.vstack([delta - lo, hi - delta]),
            fmt="none",
            ecolor="black",
            elinewidth=0.9,
            capsize=2,
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(len(scopes)) + 0.4 - width / 2)
    ax.set_xticklabels(scopes)
    ax.set_ylabel("matched $-$ shuffled macro-F1")
    ax.set_title(
        "Paired effect of the gene/morphology correspondence\n(donor-level bootstrap, 95% CI)"
    )
    ax.legend(frameon=False, fontsize=8)
    save(fig, out / "fig_patch_shuffle_deltas")


def _r2_levels(per_gene: pd.DataFrame, wanted) -> list[str]:
    """The fold levels a per-gene scatter draws, in the order the config names them.

    A level the stage did not score is dropped rather than refused: only TuPro carries
    replicate and region ids, so a lung or kidney scope legitimately has fewer levels
    than the manuscript's skin figure.
    """
    scored = list(dict.fromkeys(per_gene["level"]))
    return [lv for lv in wanted if lv in scored] if wanted else scored


def _fig_r2_scatter(per_gene: pd.DataFrame, out, levels: list[str]):
    """Frozen versus refined per-gene R^2, one panel per fold level.

    The capacity control is deliberately not on these axes. It belongs on
    :func:`_fig_r2_vs_control`, where both series are at the refined width and the
    diagonal is the claim; drawn here it shares an axis with the frozen embedding and
    shows only that the two move together.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(levels), figsize=(4.2 * len(levels), 4.0), squeeze=False)
    for ax, level in zip(axes[0], levels):
        sub = per_gene[per_gene.level == level].dropna(subset=["r2_frozen", "r2_refined"])
        # One marker per gene at ~16k genes: a vector scatter is slow to typeset and
        # megabytes wide, for points no reader can resolve individually. Axes, ticks
        # and labels stay vector.
        ax.scatter(
            sub["r2_frozen"],
            sub["r2_refined"],
            s=3,
            alpha=0.22,
            color="#1f77b4",
            linewidths=0,
            rasterized=len(sub) >= RASTER_MIN_POINTS,
        )
        lim = [
            min(sub["r2_frozen"].min(), sub["r2_refined"].min()),
            max(sub["r2_frozen"].max(), sub["r2_refined"].max()),
        ]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.axhline(0, color="#999", lw=0.6)
        ax.axvline(0, color="#999", lw=0.6)
        ax.set_title(
            f"{level.replace('_', '-')}\n"
            f"refined $\\Delta R^2$ = {float(sub['delta_r2'].mean()):+.3f}",
            fontsize=9,
        )
        ax.set_xlabel("$R^2$ frozen")
        ax.set_ylabel("$R^2$ refined")
    save(fig, out / "fig_per_gene_r2")


def _fig_r2_vs_control(per_gene: pd.DataFrame, out, levels: list[str]):
    """The refinement against its own capacity control, head to head.

    One axis each, so the diagonal is the whole claim: a point below it is a gene
    that ``PCA_k(frozen)`` predicts better than the vision-guided refinement does.
    The scatter above puts both against the frozen embedding, which shows that
    they move together but not which of them is ahead.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(levels), figsize=(4.2 * len(levels), 4.0), squeeze=False)
    for ax, level in zip(axes[0], levels):
        sub = per_gene[per_gene.level == level].dropna(subset=["r2_refined", "r2_pca_control"])
        if sub.empty:
            continue
        ax.scatter(
            sub["r2_pca_control"],
            sub["r2_refined"],
            s=3,
            alpha=0.25,
            color="#1f77b4",
            linewidths=0,
            rasterized=len(sub) >= RASTER_MIN_POINTS,
        )
        lim = [
            min(sub["r2_pca_control"].min(), sub["r2_refined"].min()),
            max(sub["r2_pca_control"].max(), sub["r2_refined"].max()),
        ]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.axhline(0, color="#999", lw=0.6)
        ax.axvline(0, color="#999", lw=0.6)
        wins = 100.0 * float((sub["r2_refined"] > sub["r2_pca_control"]).mean())
        gap = float((sub["r2_refined"] - sub["r2_pca_control"]).mean())
        ax.set_title(
            f"{level.replace('_', '-')}\n"
            f"refined ahead on {wins:.0f}% of genes, "
            f"mean gap {gap:+.3f}",
            fontsize=9,
        )
        ax.set_xlabel("$R^2$ PCA control (no vision)")
        ax.set_ylabel("$R^2$ refined")
    save(fig, out / "fig_per_gene_r2_vs_control")


#: The contrast the dot plot is drawn for. ``guided_vs_frozen`` is the difference the
#: paper reports; the other two live in the same CSV and are read from it directly.
SETMEAN_CONTRAST = "guided_vs_frozen"

#: Tokens ``str.title()`` gets wrong in MSigDB's set names. Enumerated rather than
#: inferred: "Tnfa Signaling Via Nfkb" on a paper figure is worse than no title-casing
#: at all, and the collection is fifty fixed names, not an open vocabulary.
_ACRONYMS = frozenset(
    """TNFA NFKB IL6 IL2 JAK STAT3 STAT5 MYC E2F G2M P53 UV DNA KRAS PI3K AKT MTOR
    MTORC1 TGF ROS EMT WNT V1 V2 DN UP""".split()
)


def _setmean_sources(bio_dir, cfg, notes: list) -> list[tuple[str, str, pd.DataFrame]]:
    """Every ``setmean_<gene_set>_<level>.csv`` this run's biosignal scope holds.

    Absent until `biosignal` has run, which is a "not yet" rather than a defect. A
    file that exists but carries no row for :data:`SETMEAN_CONTRAST` is the defect
    case and stops the run.
    """
    if not bio_dir.exists():
        return []
    found = []
    for gene_set in cfg.biosignal.gene_sets:
        paths = sorted(bio_dir.glob(f"setmean_{gene_set}_*.csv"))
        if not paths:
            notes.append(
                pending(
                    f"the ranked {gene_set} dot plot",
                    f"`biosignal` has not written setmean_{gene_set}_<level>.csv for this scope",
                )
            )
            continue
        for path in paths:
            level = path.stem[len(f"setmean_{gene_set}_") :]
            df = pd.read_csv(path)
            sub = df[df["contrast"].astype(str) == SETMEAN_CONTRAST]
            if sub.empty:
                refuse(
                    f"the ranked {gene_set} dot plot for '{level}'",
                    f"{path} carries no '{SETMEAN_CONTRAST}' rows, only "
                    f"{sorted(df['contrast'].astype(str).unique())}",
                )
            found.append(
                (
                    gene_set,
                    level,
                    sub.sort_values("mean_delta", ascending=False, kind="stable").reset_index(
                        drop=True
                    ),
                )
            )
    return found


def _setmean_label(source: str, *, star: bool) -> str:
    words = str(source).replace("HALLMARK_", "").split("_")
    text = " ".join(w.upper() if w.upper() in _ACRONYMS else w.title() for w in words)
    return f"{text} *" if star else text


def _setmean_counts(sub: pd.DataFrame, fdr: float) -> dict:
    """What the figure's caption has to state, so the two cannot disagree."""
    bg = float(sub["background_mean"].iloc[0])
    out = {
        "n_sets": int(len(sub)),
        "background_mean": bg,
        "n_below": int((sub["mean_delta"] < bg).sum()),
    }
    for key in ("background", "matched"):
        col = f"padj_vs_{key}"
        out[f"n_sig_vs_{key}"] = int((sub[col] < fdr).sum()) if col in sub else 0
    return out


def _fig_setmean(sub: pd.DataFrame, out, stem: str, *, fdr: float) -> dict:
    """Every set of one collection ranked by its mean delta-R^2, against two references.

    A normalised enrichment score cannot say how far a pathway moved, so a table of
    fifty of them cannot answer the question §4.3 leaves: did any pathway move
    differently from the rest of the transcriptome?

    * the dashed line is the transcriptome-wide mean delta-R^2. The bars around each
      point are a bootstrap over member genes — a spread, not a test — so crossing the
      line is not itself a verdict;
    * the open grey marker is ``expected_matched``, what a random set with the same
      baseline predictivity profile did. A set sitting on its own open marker moved
      because it is made of well-predicted genes, not because of what it encodes;
    * the star is the only inferential mark: ``padj_vs_matched`` below *fdr*.

    No title — the caption carries the run, the level and the counts, which are
    returned and printed rather than drawn so the figure and the sentence about it
    come from the same numbers.
    """
    import matplotlib.pyplot as plt

    counts = _setmean_counts(sub, fdr)
    bg = counts["background_mean"]
    y = range(len(sub))

    fig, ax = plt.subplots(figsize=(5.6, 0.135 * len(sub) + 1.7))
    ax.axvline(0.0, color="#333333", lw=0.7, zorder=1)
    ax.axvline(
        bg, color="#d62728", lw=1.1, ls="--", zorder=2, label=f"transcriptome-wide mean ({bg:+.3f})"
    )

    matched = "expected_matched" in sub and sub["expected_matched"].notna().any()
    if matched:
        ax.scatter(
            sub["expected_matched"],
            y,
            s=26,
            facecolors="none",
            edgecolors="#7f7f7f",
            linewidths=0.8,
            zorder=3,
            label="baseline-matched expectation",
        )

    # One call for the bars: fifty per-row errorbars are fifty legend-eligible artists
    # and a slower typeset, for an identical picture.
    ax.errorbar(
        sub["mean_delta"],
        y,
        xerr=[sub["mean_delta"] - sub["ci_lo"], sub["ci_hi"] - sub["mean_delta"]],
        fmt="o",
        ms=3.4,
        lw=0.9,
        color="#1f77b4",
        ecolor="#1f77b4",
        capsize=1.6,
        zorder=4,
        label="mean $\\Delta R^2$ (95% bootstrap CI)",
    )

    survives = (
        sub["padj_vs_matched"] < fdr
        if "padj_vs_matched" in sub
        else pd.Series(False, index=sub.index)
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(
        [_setmean_label(s, star=bool(k)) for s, k in zip(sub["source"], survives.fillna(False))],
        fontsize=6,
    )
    ax.set_ylim(-0.6, len(sub) - 0.4)
    ax.set_xlabel("mean $\\Delta R^2$ over member genes")
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", lw=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    # Above the axes, not inside them: at fifty rows every interior corner is occupied,
    # and a legend over the bottom rows hides the sets nearest the reference line.
    ax.legend(
        frameon=False,
        fontsize=6.5,
        handletextpad=0.5,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.005),
        ncol=2 if matched else 1,
        columnspacing=1.2,
    )
    fig.tight_layout()
    save(fig, out / stem)

    print(
        f"    {stem}: {counts['n_sets']} sets, background mean {bg:+.4f}, "
        f"{counts['n_below']} below it; FDR<{fdr}: "
        f"{counts['n_sig_vs_background']} vs that mean, "
        f"{counts['n_sig_vs_matched']} vs a baseline-matched null"
    )
    return counts


def _fig_umaps(cfg, out, made: list[str], notes: list[str]) -> None:
    """UMAP of the gene-only PCA versus the refined embedding.

    Coloured by annotation, tissue and slide. The slide panel is the informative
    one: spots clustering by slide rather than by annotation means the embedding is
    organised by batch.
    """
    from ..data import tables as tables_mod
    from ..models.train import embedding_path

    try:
        import umap  # noqa: F401
    except ImportError as e:
        refuse(
            "the UMAP figure",
            f"umap-learn is not importable ({e})",
            hint="it is pinned in requirements.txt; `bash slurm/preflight.sh` "
            "checks for it before anything is submitted",
        )

    # One fit, not the three the tables average — see `config.headline_seed` for
    # when that is allowed. `diagnostics.seed` below is the other thing: the
    # algorithm seed, here UMAP's own `random_state`.
    seed = headline_seed(cfg)
    available = {name: embedding_path(cfg, name, seed) for name in cfg.models.names}
    missing = sorted(k for k, v in available.items() if not v.exists())
    available = {k: v for k, v in available.items() if v.exists()}
    if not available:
        notes.append(
            pending(
                "the UMAP figure", f"the `train` stage has written no embedding for seed {seed}"
            )
        )
        return
    if missing:
        # Some but not all: train ran, so an absent embedding is not a schedule.
        refuse(
            "the UMAP figure",
            f"train wrote embeddings for {sorted(available)} but not for {missing} at seed {seed}",
            hint="rerun `python run.py train`, or drop those from models.names",
        )

    # Metadata only: the UMAP needs labels and slide ids, not the frozen features.
    meta = tables_mod.load_meta(cfg)
    rng = np.random.default_rng(cfg.diagnostics.seed)
    n = min(cfg.diagnostics.umap_max_spots, len(meta))
    rows = np.sort(rng.choice(len(meta), size=n, replace=False))
    annotation = meta["annotation"].to_numpy()[rows]
    tissue = meta["tissue"].to_numpy()[rows]
    sample_id = meta["sample_id"].to_numpy()[rows]

    import matplotlib.pyplot as plt
    import umap as umap_lib

    names = list(available)
    fig, axes = plt.subplots(3, len(names), figsize=(4.6 * len(names), 13), squeeze=False)
    for col, name in enumerate(names):
        Z = np.load(available[name])[rows]
        XY = umap_lib.UMAP(n_neighbors=15, random_state=cfg.diagnostics.seed).fit_transform(Z)
        scatter_labels_2d(
            axes[0, col],
            XY,
            annotation,
            size=3,
            title=f"{name} — annotation",
            legend=(col == len(names) - 1),
        )
        scatter_categorical_2d(
            axes[1, col],
            XY,
            tissue,
            size=3,
            title=f"{name} — tissue",
            legend=(col == len(names) - 1),
        )
        scatter_categorical_2d(
            axes[2, col], XY, sample_id, size=3, title=f"{name} — slide", legend=False
        )
    fig.suptitle(f"Embedding structure ({cfg.data.substrate}, seed {seed})", fontsize=13)
    save(fig, out / "fig_umap")
    made.append("fig_umap")
