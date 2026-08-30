"""Manuscript tables that span more than one run.

``run.py figures`` writes one set of tables per run directory, because a run is one
gene-FM substrate and nothing in the pipeline reads a sibling run. The paper puts
all three backbones in a single table, so this script is that last join: it reads
the per-run artefacts the ``figures`` stage already wrote and emits complete LaTeX
``table`` environments in the manuscript's own house style, ready to paste over the
existing ones.

    python scripts/paper_tables.py --artifacts /path/to/artifacts/vgtfm --out paper_tables

Nothing here recomputes a score from an embedding. Every number is read from
``eval/``, ``ablation/`` or ``diagnostics/`` of the runs named on the command line.
The one derived quantity is the USZ cohort row of the patch-shuffle table, which
pools kidney and lung; see :func:`pool_scopes` for why that is exact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vgtfm.figures.tables import (  # noqa: E402
    INTEGRATION_METRICS,
    MODEL_LABELS,
    integration_over_seeds,
    tex_escape,
    tissue_label,
)

#: Substrate key -> the name the paper prints. Bracketed citations are added in the
#: manuscript; keeping bibkeys here would duplicate the .bib.
BACKBONE_LABELS = {
    "cancerfoundation": "CancerFoundation",
    "scgpt": "scGPT",
    "geneformer": "Geneformer V2",
}

#: Paper order for the backbone blocks, as in the manuscript's Table 1.
BACKBONE_ORDER = ["cancerfoundation", "scgpt", "geneformer"]

#: Rows that read no gene-FM features at all — the count baseline, the two
#: morphology oracles and the majority rule — so the three runs reproduce them bit
#: for bit. They get one shared block instead of three identical rows.
SHARED_MODELS = ["hvg_pca", "pca_oracle", "pca_oracle_matched", "majority"]

#: Rows that do depend on the substrate, one column each in the main table.
GUIDED_MODELS = ["pca", "ae", "cdann"]

#: Row labels. ``MODEL_LABELS`` writes "best model" lowercase for the per-run
#: tables' sentence-style rows; the paper's tables capitalise every row label.
ROW_LABELS = {**MODEL_LABELS, "cdann": "Best model"}

#: Two-line column heads for those, so the table stays narrow.
GUIDED_HEADS = {
    "pca": ("PCA", "(gene-only)"),
    "ae": ("AE", r"(gene$\rightarrow$H\&E)"),
    "cdann": ("Best model", ""),
}

#: Ablation condition -> column head, in the order the paper reads them.
CONDITION_HEADS = {
    "none": "Matched",
    "within-sample": "Within-slide shuffle",
    "global": "Global shuffle",
    "gaussian": "Gaussian",
}

#: Registry class keys -> the names the paper prints. The USZ cohorts annotate in
#: abbreviations; the manuscript spells them out.
CLASS_LABELS = {
    "NOR": "Normal",
    "TLS": "Tertiary lymphoid structures",
    "INFL": "Inflammation",
    "Normal lymphoid tissue": "Normal lymphoid",
}

#: Cohort blocks for the patch-shuffle table: display name -> tissue scopes.
#: One scope each, so every cell carries the donor bootstrap ``eval`` and ``ablate``
#: already computed for that organ — kidney over its 2 donors, lung over its 4.
COHORTS = {
    "TuPro, skin melanoma": ["skin"],
    "USZ, kidney carcinoma": ["kidney"],
    "USZ, lung carcinoma": ["lung"],
}

#: ``--pool-usz`` instead reports the two USZ organs as the single cohort the
#: earlier manuscript printed. Exact for the point estimate and intervalless; see
#: :func:`pool_scopes`.
COHORTS_POOLED_USZ = {
    "TuPro, skin melanoma": ["skin"],
    "USZ, kidney and lung carcinoma": ["kidney", "lung"],
}

#: The classes the manuscript's Table 3 carries, per cohort. What it leaves out is
#: the smallest and lowest-scoring class of each; ``--all-classes`` puts them back.
PAPER_CLASSES = {
    "TuPro, skin melanoma": ["Tumor", "Normal lymphoid tissue", "Stroma"],
    "USZ, kidney carcinoma": ["Tumor", "TLS", "NOR"],
    "USZ, lung carcinoma": ["Tumor", "TLS", "NOR"],
    "USZ, kidney and lung carcinoma": ["Tumor", "TLS", "NOR"],
}


def _class_label(name) -> str:
    return tex_escape(CLASS_LABELS.get(str(name), str(name)))


def _num(value, digits: int) -> str:
    return "--" if value is None or value != value else f"{value:.{digits}f}"


def _cell(value, lo=None, hi=None, *, digits: int = 2) -> str:
    """A point estimate with its interval set smaller beside it, house style.

    ``\\tiny`` is applied unbraced, as the manuscript writes it: the cell ends at
    the ``&``, so the size change cannot leak into the next column.
    """
    s = _num(value, digits)
    if s == "--" or lo is None or hi is None or lo != lo or hi != hi:
        return s
    return rf"{s} \tiny [{lo:.{digits}f},\,{hi:.{digits}f}]"


def _signed(value, lo=None, hi=None, *, digits: int = 2) -> str:
    """:func:`_cell` for a difference, with the sign kept."""
    if value is None or value != value:
        return "--"
    s = f"{value:+.{digits}f}"
    if lo is None or hi is None or lo != lo or hi != hi:
        return s
    return rf"{s} \tiny [{lo:+.{digits}f},\,{hi:+.{digits}f}]"


def _p_cell(value, digits: int = 3) -> str:
    """A two-sided bootstrap p-value.

    Printed as ``<0.001`` rather than ``0.000`` at the floor: the statistic counts
    resamples, so zero means "none of them crossed", which is a bound and not a
    probability of zero.
    """
    if value is None or value != value:
        return "--"
    floor = 10.0**-digits
    if value < floor:
        return rf"$<${floor:.{digits}f}"
    return f"{value:.{digits}f}"


def table_env(
    *,
    caption: str,
    label: str,
    colspec: str,
    header: list[str],
    body: list[str],
    note: str | None = None,
    placement: str = "!htbp",
    tabcolsep: str = "4pt",
) -> str:
    """One complete ``table`` environment in the manuscript's house style."""
    lines = [
        rf"\begin{{table}}[{placement}]",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\centering",
        r"\small",
        rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        *header,
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    if note:
        lines.append(rf"\caption*{{{note}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


class Run:
    """One run directory, addressed by the substrate its config records."""

    def __init__(self, path: Path):
        self.path = path
        cfg = json.loads((path / "config.resolved.json").read_text())
        self.substrate = cfg["data"]["substrate"]
        self.run_name = cfg["run_name"]
        self.label = BACKBONE_LABELS.get(self.substrate, self.substrate)

    def csv(self, *parts: str) -> pd.DataFrame:
        path = self.path.joinpath(*parts)
        if not path.exists():
            raise SystemExit(f"{self.run_name}: missing {path} — has `figures` run?")
        return pd.read_csv(path)

    def provenance(self) -> str:
        """The newest manifest that ran every stage, for the header comment."""
        manifests = sorted(self.path.glob("logs/manifest-all-*.json"))
        if not manifests:
            return f"{self.run_name}: no all-stage manifest"
        m = json.loads(manifests[-1].read_text())
        return (
            f"{self.run_name} ({self.substrate}): {m['started']} "
            f"{m['status']}, source {m['source_fingerprint']}"
        )


def _ci_columns(frame: pd.DataFrame) -> tuple[str, str]:
    """Prefer the seed-pooled interval, as the per-run tables do."""
    if "ci_lo_pooled" in frame.columns and frame["ci_lo_pooled"].notna().any():
        return "ci_lo_pooled", "ci_hi_pooled"
    return "ci_lo", "ci_hi"


def pool_scopes(per_class: pd.DataFrame, scopes: list[str], keys: list[str]) -> pd.DataFrame:
    """Micro-average per-class scores over several tissue scopes, exactly.

    ``per_class`` carries precision, recall and support, which determine the
    confusion counts outright: ``TP = recall * support``, ``FN = support - TP`` and
    ``FP = TP / precision - TP``. Summing those over the scopes and recomputing F1
    is therefore identical to scoring the concatenated prediction vectors — the
    predictions themselves are untouched, each spot still scored inside the fold
    that held its donor out. Averaging the per-scope F1 values instead would not
    be, which is why this goes through the counts.

    What it cannot do is produce the interval: a donor bootstrap needs the
    per-donor predictions, and ``ablate`` writes only fold-pooled rows. Those cells
    come back empty rather than filled with a number that is not one.
    """
    d = per_class[per_class.scope.isin(scopes)].copy()
    if d.empty:
        return d
    d["tp"] = d.recall * d.support
    d["fn"] = d.support - d.tp
    d["fp"] = np.where(d.precision > 0, d.tp / d.precision.replace(0, np.nan) - d.tp, 0.0)

    # Per seed first, then average over seeds: the same order the per-run tables
    # pool in, so a pooled row and a per-organ row are the same kind of statistic.
    per_seed = d.groupby(keys + ["seed"])[["tp", "fp", "fn", "support"]].sum()
    tp, fp, fn = per_seed.tp, per_seed.fp, per_seed.fn

    def ratio(num, den):
        # A class the probe never predicts has TP = FP = 0, and one absent from the
        # scope has TP = FN = 0; both are scored 0, as sklearn scores them.
        return np.divide(num, den, out=np.zeros(len(den), float), where=den > 0)

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    per_seed["f1_score"] = ratio(2 * precision * recall, precision + recall)
    out = (
        per_seed.reset_index()
        .groupby(keys)
        .agg(f1_score=("f1_score", "mean"), support=("support", "max"))
        .reset_index()
    )
    return out


def main_skin_table(
    runs: list[Run], *, protocol: str, level: str, digits: int, label: str
) -> tuple[str, int]:
    """Main table — cross-donor macro-F1 on skin, all three backbones.

    Skin is the only cohort with enough donors (7) for the bootstrap to resolve
    anything, which is why it carries the main text alone; the per-organ spread
    lives in :func:`appendix_all_tissues_table`.
    """
    stem = f"table_annotation_{protocol}_{level}_skin"
    frames = {r.substrate: r.csv("figures", f"{stem}.csv").set_index("model") for r in runs}

    # The substrate-independent rows must agree across runs or the shared block is a
    # lie. Checked rather than assumed: a mismatch means the runs saw different spot
    # tables, which would invalidate every comparison below.
    for model in SHARED_MODELS:
        values = {s: f.loc[model, "macro_f1"] for s, f in frames.items() if model in f.index}
        if values and max(values.values()) - min(values.values()) > 5e-4:
            raise SystemExit(
                f"{model} differs across runs ({values}) but reads no gene "
                "features; the runs are not on the same spot table"
            )

    any_frame = next(iter(frames.values()))
    n_donors = int(any_frame["n_donors"].max())
    span = len(GUIDED_MODELS)

    header = [
        "& " + " & ".join(GUIDED_HEADS[m][0] for m in GUIDED_MODELS) + r" \\",
        ("& " + " & ".join(GUIDED_HEADS[m][1] for m in GUIDED_MODELS)).rstrip() + r" \\",
    ]
    body = []
    for model in SHARED_MODELS:
        if model not in any_frame.index:
            continue
        r = any_frame.loc[model]
        cell = _cell(r.macro_f1, r.donor_ci_lo, r.donor_ci_hi, digits=digits)
        body.append(
            f"{ROW_LABELS.get(model, tex_escape(model))} & "
            rf"\multicolumn{{{span}}}{{c}}{{{cell}}} \\"
        )
    body.append(r"\midrule")
    body.append(
        rf"\multicolumn{{{span + 1}}}{{l}}{{\emph{{Transcriptomic "
        r"foundation model + Midnight H\&E}}\\"
    )
    for substrate in BACKBONE_ORDER:
        if substrate not in frames:
            continue
        f = frames[substrate]
        cells = [
            _cell(
                f.loc[m, "macro_f1"],
                f.loc[m, "donor_ci_lo"],
                f.loc[m, "donor_ci_hi"],
                digits=digits,
            )
            if m in f.index
            else "--"
            for m in GUIDED_MODELS
        ]
        body.append(
            rf"\quad {tex_escape(BACKBONE_LABELS.get(substrate, substrate))} & "
            + " & ".join(cells)
            + r" \\"
        )

    caption = (
        r"Cross-donor zero-shot $k$-NN annotation prediction macro-F1 "
        r"($k{=}5$) on TuPro skin melanoma "
        rf"(${n_donors}$ donors). \emph{{Best model}} is the strongest "
        r"guided variant per backbone."
    )
    note = (
        r"Brackets give the $95\%$ CI from a donor bootstrap. "
        r"The count baseline, the morphology oracles and the majority rule "
        r"read no gene-FM features, so they are identical across backbones "
        r"and are reported once."
    )
    tex = table_env(
        caption=caption, label=label, colspec="l" + "c" * span, header=header, body=body, note=note
    )
    return tex, n_donors


def appendix_all_tissues_table(
    runs: list[Run], *, protocol: str, level: str, digits: int, label: str
) -> str:
    """Appendix table — the same comparison, one column per organ.

    The organ-balanced column weights each organ equally. The ``global`` scope the
    per-run tables also carry is a macro over (tissue, class) cells and is not
    reproduced here, since it cannot be read as any one cohort's score.
    """
    stem = f"table_annotation_by_organ_{protocol}_{level}"
    frames = {r.substrate: r.csv("figures", f"{stem}.csv").set_index("model") for r in runs}
    any_frame = next(iter(frames.values()))

    organs = [
        c[: -len("_macro_f1")]
        for c in any_frame.columns
        if c.endswith("_macro_f1") and not c.startswith("tissue_balanced")
    ]
    organs = sorted(organs, key=lambda o: int(any_frame[f"{o}_n_donors"].max()))
    heads = [rf"{tissue_label(o)} (${int(any_frame[f'{o}_n_donors'].max())}$)" for o in organs] + [
        "Organ-balanced"
    ]
    ncols = len(heads) + 1

    def cells(frame, model):
        out = [
            _cell(
                frame.loc[model, f"{o}_macro_f1"],
                frame.loc[model, f"{o}_ci_lo"],
                frame.loc[model, f"{o}_ci_hi"],
                digits=digits,
            )
            for o in organs
        ]
        out.append(
            _cell(
                frame.loc[model, "tissue_balanced_macro_f1"],
                frame.loc[model, "tissue_balanced_ci_lo"],
                frame.loc[model, "tissue_balanced_ci_hi"],
                digits=digits,
            )
        )
        return out

    header = ["Model & " + " & ".join(heads) + r" \\"]
    body = [
        rf"\multicolumn{{{ncols}}}{{l}}{{\emph{{Substrate-independent "
        r"baselines}}\\"
    ]
    for model in SHARED_MODELS:
        if model in any_frame.index:
            body.append(
                rf"\quad {ROW_LABELS.get(model, tex_escape(model))} & "
                + " & ".join(cells(any_frame, model))
                + r" \\"
            )
    for substrate in BACKBONE_ORDER:
        if substrate not in frames:
            continue
        f = frames[substrate]
        body.append(r"\midrule")
        body.append(
            rf"\multicolumn{{{ncols}}}{{l}}{{\emph{{"
            rf"{tex_escape(BACKBONE_LABELS.get(substrate, substrate))}"
            r"} + Midnight H\&E}\\"
        )
        for model in GUIDED_MODELS:
            if model in f.index:
                body.append(
                    rf"\quad {ROW_LABELS.get(model, tex_escape(model))} & "
                    + " & ".join(cells(f, model))
                    + r" \\"
                )

    caption = (
        r"Cross-donor zero-shot $k$-NN annotation prediction macro-F1 "
        r"($k{=}5$) per organ, for every backbone. Column heads give the "
        r"number of donors the fold structure and the bootstrap both rest "
        r"on."
    )
    note = (
        r"Brackets give the $95\%$ CI from a donor bootstrap. "
        r"\emph{Organ-balanced} weights each organ equally rather than by "
        r"spot count. Kidney is two donors, so its interval spans only three "
        r"distinct resamples and should not be read as evidence."
    )
    return table_env(
        caption=caption,
        label=label,
        colspec="l" + "c" * len(heads),
        header=header,
        body=body,
        note=note,
    )


def effective_rank_table(runs: list[Run], *, digits: int, controls: bool, label: str) -> str:
    """Effective rank of every frozen representation, one row per backbone.

    The patch row is the same Midnight cache in all three runs, so it is emitted
    once — and checked, for the same reason the shared baselines above are.
    """
    rows, patch = [], {}
    for r in runs:
        er = r.csv("diagnostics", "effective_rank.csv")
        er = er[er.variant == "observed"]
        rows.append(
            (
                BACKBONE_LABELS.get(r.substrate, r.substrate),
                er[er.column == "gene_features"].iloc[0],
            )
        )
        patch[r.run_name] = er[er.column == "patch_features"].iloc[0]

    ranks = {k: float(v.eff_rank) for k, v in patch.items()}
    if max(ranks.values()) - min(ranks.values()) > 5e-3:
        raise SystemExit(
            f"the Midnight patch rank differs across runs ({ranks}); "
            "the runs are not on the same patch cache"
        )

    order = {s: i for i, s in enumerate(BACKBONE_ORDER)}
    inv = {v: k for k, v in BACKBONE_LABELS.items()}
    rows.sort(key=lambda t: order.get(inv.get(t[0], t[0]), 99))

    def line(name, r):
        return (
            f"{tex_escape(name)} & ${int(r.dim)}$ & ${r.eff_rank:.{digits}f}$ & "
            f"${r.pct_of_dim:.{digits}f}$ \\\\"
        )

    body = [line(r"Midnight (H\&E)", next(iter(patch.values())))]
    body += [line(name, r) for name, r in rows]

    if controls:
        body.append(r"\midrule")
        body.append(r"\multicolumn{4}{l}{\emph{controls}}\\")
        for r in runs:
            er = r.csv("diagnostics", "effective_rank.csv")
            name = BACKBONE_LABELS.get(r.substrate, r.substrate)
            for c in er[(er.column == "gene_features") & (er.variant != "observed")].itertuples():
                body.append(r"\quad " + line(f"{name} ({c.variant})", c))
        er = runs[0].csv("diagnostics", "effective_rank.csv")
        for c in er[(er.column == "patch_features") & (er.variant != "observed")].itertuples():
            body.append(r"\quad " + line(f"Midnight ({c.variant})", c))

    patch_pct = float(next(iter(patch.values())).pct_of_dim)
    gene_pct = [float(r.pct_of_dim) for _name, r in rows]
    caption = (
        r"Effective rank of the frozen embeddings (train split). "
        r"Read the last column rather than the middle one: an absolute rank "
        r"is not comparable across representations of different width. "
        rf"Midnight uses ${patch_pct:.1f}\%$ of its $3072$ dimensions, "
        rf"against ${min(gene_pct):.1f}$--${max(gene_pct):.1f}\%$ for the "
        r"gene-side embeddings on theirs, so the H\&E side is not the "
        r"better-filled space its larger rank suggests."
    )
    return table_env(
        caption=caption,
        label=label,
        colspec="lrrr",
        header=[r"Representation & Dim & Eff.\ rank & \% of dim \\"],
        body=body,
    )


def _organ_scopes(wide, level: str) -> list[tuple[str, int]]:
    """``(organ, n_donors)`` for every tissue scope the probe scored at *level*.

    Read off the ``f1/<scope>/<level>`` columns rather than hardcoded, so a cohort
    with a fourth tissue picks it up without an edit here. ``global`` is dropped:
    this table exists precisely because the pooled macro hides the organs.

    Ordered by donor count ascending, the order
    :func:`integration_by_organ_table` also uses, so the two appendix tables' columns
    line up when they are read side by side.
    """
    found: dict[str, int] = {}
    for col in wide.columns:
        parts = str(col).split("/")
        if len(parts) != 3 or parts[0] != "f1" or parts[2] != level:
            continue
        if parts[1] == "global":
            continue
        n = pd.to_numeric(wide.get(f"{col}_n_donors"), errors="coerce").max()
        found[parts[1]] = int(n) if n == n else 0
    return sorted(found.items(), key=lambda kv: kv[1])


#: Rows of the batch-correction table, in the order Appendix D.2 names them, as
#: ``(label, model, integrate method)``. Each correction appears twice: the
#: corrected embedding probed directly, and the same embedding after
#: histology-guided learning. A label beginning with ``+`` is the guided member of
#: the pair and is indented under it.
#:
#: A row with no model is a method that yields no corrected gene matrix an encoder
#: could be fitted on, so it has no guided arm and is read from ``integrate``:
#: BBKNN corrects a neighbour graph, and scVI is fitted on the raw counts.
CORRECTION_ROWS = (
    ("Uncorrected", "pca", None),
    ("+ guidance", "ae", None),
    ("Harmony", "harmony_pca", None),
    ("+ guidance", "harmony_ae", None),
    ("BBKNN", None, "bbknn"),
    ("ComBat", "combat_pca", None),
    ("+ guidance", "combat_ae", None),
    ("scVI", None, "scvi"),
)


def _scib_over_seeds(run: Run) -> pd.DataFrame:
    """``scib_panel.csv`` keyed by model, averaged over the seeds it scored.

    The panel names a row ``"<model> (seed 42)"``. Averaging here keeps these the
    same statistic :func:`integration_over_seeds` produces for the rows read from
    ``integrate``, so every row of the table is a method rather than a fit.
    """
    panel = run.csv("diagnostics", "scib_panel.csv")
    panel = panel.assign(
        model=panel["representation"].str.replace(r" \(seed \d+\)$", "", regex=True)
    )
    return panel.groupby("model")[[m for m, _h in INTEGRATION_METRICS]].mean()


def correction_table(runs: list[Run], *, protocol: str, level: str, digits: int, label: str) -> str:
    """Appendix D.2 — batch correction of the gene embedding under guidance.

    Each correction is fitted once over the whole cohort with the slide as the batch
    covariate, and its output becomes the new frozen input to the guided encoder.
    Reporting each correction twice — probed directly, then after guidance — makes the
    effect of guidance the step between two adjacent rows, so it is not confounded
    with the correction that produced their shared input.

    Every column group describes the row's own representation: how well it mixes the
    slides, whether the label structure survived, and whether a held-out donor can
    still be annotated from it. The uncorrected pair is Table 1's PCA and AE rows.
    """
    order = {s: i for i, s in enumerate(BACKBONE_ORDER)}
    blocks = sorted(runs, key=lambda r: order.get(r.substrate, 99))
    stem = f"table_annotation_by_organ_{protocol}_{level}.csv"

    loaded, missing = {}, []
    for run in blocks:
        wide = integration_over_seeds(run.csv("integration", "integration.csv")).set_index("method")
        organ = run.csv("figures", stem).set_index("model")
        loaded[run.run_name] = (_scib_over_seeds(run), organ, wide, _organ_scopes(wide, level))
        missing += [
            f"{run.run_name}: {m}" for _l, m, _x in CORRECTION_ROWS if m and m not in organ.index
        ]
    if missing:
        raise SystemExit(
            "the guided batch-correction arms were not scored in these runs: "
            + ", ".join(missing)
            + "\nadd them to models.names and rerun train + eval; see vgtfm/models/corrected.py"
        )

    signatures = {tuple(o) for _s, _o, _w, o in loaded.values()}
    if len(signatures) > 1:
        raise SystemExit(
            f"the {level} probe spans different organs or donor counts across "
            f"runs ({sorted(signatures)}); the runs are not on one cohort"
        )
    organs = next(iter(loaded.values()))[3]
    ncols = 1 + len(INTEGRATION_METRICS) + len(organs)

    body: list[str] = []
    for i, run in enumerate(blocks):
        scib, organ, wide, _o = loaded[run.run_name]
        if i:
            body.append(r"\midrule")
        body.append(rf"\multicolumn{{{ncols}}}{{l}}{{\emph{{{tex_escape(run.label)}}}}}\\")

        for lbl, model, method in CORRECTION_ROWS:
            if model is None:
                if method not in wide.index:
                    continue
                row = wide.loc[method]
                cells = [_num(row.get(m), digits) for m, _h in INTEGRATION_METRICS]
                cells += [
                    _cell(
                        row.get(f"f1/{scope}/{level}"),
                        row.get(f"f1/{scope}/{level}_lo"),
                        row.get(f"f1/{scope}/{level}_hi"),
                        digits=digits,
                    )
                    for scope, _n in organs
                ]
            else:
                cells = [
                    _num(scib.loc[model, m] if model in scib.index else np.nan, digits)
                    for m, _h in INTEGRATION_METRICS
                ]
                cells += [
                    _cell(
                        organ.loc[model, f"{scope}_macro_f1"],
                        organ.loc[model, f"{scope}_ci_lo"],
                        organ.loc[model, f"{scope}_ci_hi"],
                        digits=digits,
                    )
                    for scope, _n in organs
                ]
            indent = r"\quad\quad" if lbl.startswith("+") else r"\quad"
            body.append(f"{indent} {tex_escape(lbl)} & " + " & ".join(cells) + r" \\")

    heads = [h for _key, h in INTEGRATION_METRICS] + [
        rf"{tissue_label(scope)} (${n}$)" for scope, n in organs
    ]
    caption = (
        r"Batch correction applied to the gene expression embeddings, which are "
        r"then used in the histology-guided representation learning as new frozen "
        r"gene expression embeddings. Each correction is fitted once and globally, "
        r"including the evaluation data, with the slide as the batch covariate. "
        r"iLISI and kBET measure batch mixing, cLISI preservation of the biological "
        rf"label structure, and each organ column is the {tex_escape(level.replace('_', '-'))} "
        r"macro-F1 on that tissue alone. Each correction is reported twice, before "
        r"and after guidance, so the step between two adjacent rows is the effect "
        r"of supervision on a shared input; the uncorrected pair is the setting "
        r"described in Section~3."
    )
    note = (
        r"Brackets give the $95\%$ CI from a donor bootstrap within each tissue, and "
        r"donor counts are in the column heads; the two-donor organs admit only "
        r"three distinct resamples, so read their brackets as granular rather than "
        r"tight. Every row is averaged over the same three model seeds as "
        r"Table~\ref{tab:main}. iLISI, kBET and cLISI have no per-organ version, "
        r"since the scIB panel scores one stratified sample of the whole annotated "
        r"cohort batched by slide, but they are measured on each row's own "
        r"representation, so a guided row reports the mixing of the guided "
        r"embedding rather than of the corrected one above it. BBKNN corrects the "
        r"neighbour graph rather than the embedding, so it yields nothing an "
        r"encoder could be fitted on and no F1 is reported for it; scVI produces an "
        r"embedding from the raw counts but was not carried through guidance, so "
        r"both are shown unguided."
    )
    return table_env(
        caption=caption,
        label=label,
        colspec="l" + "r" * (ncols - 1),
        header=["Method & " + " & ".join(heads) + r" \\"],
        body=body,
        note=note,
    )


def _interval(frame, key) -> tuple:
    """``(lo, hi)`` for one cell of a bootstrap frame, or ``(None, None)``."""
    if frame is None or key not in frame.index:
        return None, None
    row = frame.loc[key]
    return float(row.lo), float(row.hi)


def _cohort_intervals(run: Run, scoped, scope: str) -> tuple:
    """The donor bootstraps ``ablate`` and ``eval`` already wrote for one organ.

    Only for a single-scope cohort. Pooling two scopes is exact for the point
    estimate and impossible for the interval — a donor bootstrap needs the per-donor
    predictions, and neither stage writes those — so a pooled cohort's cells stay
    empty rather than carrying a number that is not one.
    """
    ab = scoped(run.csv("ablation", "bootstrap.csv"))
    ab = ab[ab.scope == scope]
    lo, hi = _ci_columns(ab)
    abl_ci = ab.groupby(["class", "condition"]).agg(lo=(lo, "mean"), hi=(hi, "mean"))

    pb = scoped(run.csv("eval", "bootstrap.csv"), model="pca")
    pb = pb[pb.scope == scope]
    lo, hi = _ci_columns(pb)
    return abl_ci, pb.groupby("class").agg(lo=(lo, "mean"), hi=(hi, "mean"))


def shuffle_table(
    run: Run,
    *,
    protocol: str,
    level: str,
    digits: int,
    gaussian: bool,
    all_classes: bool,
    pool_usz: bool,
    label: str,
) -> tuple[str, list[str]]:
    """Per-class F1 under matched vs shuffled morphology targets.

    The gene-only PCA column comes from ``eval/`` at the same protocol and level:
    it is the representation the shuffled conditions are supposed to fall back
    towards, and without it the three ablation columns have no reference.
    """
    per_class = run.csv("ablation", "per_class.csv")
    eval_class = run.csv("eval", "per_class.csv")

    def scoped(df, **extra):
        m = (
            (df.protocol == protocol)
            & (df.level == level)
            & (df.scope != "global")
            & (df["class"] != "macro")
        )
        for k, v in extra.items():
            m &= df[k] == v
        return df[m]

    abl_raw = scoped(per_class, fold="pooled")
    if abl_raw.empty:
        raise SystemExit(f"ablation has no pooled rows for {protocol}/{level}")
    pca_raw = scoped(eval_class, fold="pooled", model="pca")

    cohorts = COHORTS_POOLED_USZ if pool_usz else COHORTS
    counts = abl_raw.groupby("scope")[["n_donors", "n_slides"]].max()

    # Any scope the cohort map does not mention would silently vanish from the
    # table, so name it instead of dropping it.
    mapped = {s for scopes in cohorts.values() for s in scopes}
    unmapped = sorted(set(abl_raw.scope) - mapped)
    if unmapped:
        raise SystemExit(
            f"scopes {unmapped} are scored but not in COHORTS; add "
            "them there or they would be dropped from the table"
        )

    conditions = [c for c in CONDITION_HEADS if c in set(abl_raw.condition)]
    if not gaussian:
        conditions = [c for c in conditions if c != "gaussian"]
    ncols = len(conditions) + 2

    header = [
        r"& PCA & " + rf"\multicolumn{{{len(conditions)}}}{{c}}"
        r"{AE (gene$\rightarrow$H\&E)} \\",
        rf"\cmidrule(lr){{3-{ncols}}}",
        r"Class (support) & (gene-only) & "
        + " & ".join(CONDITION_HEADS[c] for c in conditions)
        + r" \\",
    ]

    body, omitted = [], []
    for cohort, scopes in cohorts.items():
        if not set(scopes) & set(abl_raw.scope):
            continue
        abl = pool_scopes(abl_raw, scopes, ["class", "condition"])
        pca = pool_scopes(pca_raw, scopes, ["class"]).set_index("class")

        abl_ci = pca_ci = None
        if len(scopes) == 1:
            abl_ci, pca_ci = _cohort_intervals(run, scoped, scopes[0])

        keep = PAPER_CLASSES.get(cohort) if not all_classes else None
        order = keep or list(
            abl.groupby("class")["support"].max().sort_values(ascending=False).index
        )
        omitted += [f"{cohort}: {c}" for c in sorted(set(abl["class"]) - set(order))]

        n_slides = int(counts.loc[scopes, "n_slides"].sum())
        n_donors = int(counts.loc[scopes, "n_donors"].sum())
        if body:
            body.append(r"\midrule")
        body.append(
            rf"\multicolumn{{{ncols}}}{{l}}{{\emph{{{tex_escape(cohort)}}} "
            rf"(${n_slides}$ slides, ${n_donors}$ donors)}}\\"
        )

        abl = abl.set_index(["class", "condition"])
        for klass in order:
            if (klass, conditions[0]) not in abl.index:
                continue
            support = int(abl.loc[klass, "support"].max())
            f1 = pca.loc[klass, "f1_score"] if klass in pca.index else None
            cells = [_cell(f1, *_interval(pca_ci, klass), digits=digits)]
            for c in conditions:
                cells.append(
                    _cell(
                        abl.loc[(klass, c), "f1_score"],
                        *_interval(abl_ci, (klass, c)),
                        digits=digits,
                    )
                )
            body.append(
                rf"\quad {_class_label(klass)} ({support:,}) & " + " & ".join(cells) + r" \\"
            )

    caption = (
        r"Per-class cross-donor $k$-NN annotation F1 ($k{=}5$) on "
        rf"{tex_escape(BACKBONE_LABELS.get(run.substrate, run.substrate))} "
        r"embeddings, comparing gene-only PCA with matched and shuffled "
        r"morphology guidance."
    )
    note = (
        r"\emph{Matched} uses true gene and patch pairs. "
        r"\emph{Within-slide shuffle} permutes patches within each slide. "
        r"\emph{Global shuffle} permutes patches across all spots. "
        r"Brackets give the $95\%$ CI from a donor bootstrap."
    )
    if pool_usz:
        note += (
            r" The USZ rows pool kidney and lung over the confusion counts, "
            r"which is exact for the point estimate; a donor bootstrap needs "
            r"the per-donor predictions and is not available for them."
        )
    tex = table_env(
        caption=caption,
        label=label,
        colspec="l" + "c" * (len(conditions) + 1),
        header=header,
        body=body,
        note=note,
    )
    return tex, omitted


def _check_consistency(artifacts: Path, run_names: list[str]) -> None:
    """Refuse to build a manuscript table out of numbers the stages disagree on.

    Every table below reads a per-stage artefact directly, which is fast and keeps
    each table close to the stage that produced it. What it cannot do on its own is
    notice that two stages computed one quantity twice and got two answers -- the
    failure that put a transductive PCA behind the integration table's uncorrected
    row while Table 1 reported an inductive one, with nothing in the repository
    positioned to compare them.

    :mod:`vgtfm.results` joins the stages into one frame and answers exactly that
    question, so it is asked here, once, before any LaTeX is written.

    Two ways for that answer to be "no". The tables are refused when a cell carries
    two values, and equally when an alias was never exercised at all: a check that
    compared nothing prints the same clean line as a check that compared everything,
    and only one of them says the manuscript is safe.
    """
    from vgtfm import results

    df = results.collect(artifacts, run_names)
    if df.empty:
        raise SystemExit(f"no scores found under {artifacts} for runs {run_names}")
    conf = results.conflicts(df)
    if not conf.empty:
        print(
            conf[["run_name", "level", "scope", "metric", "spread", "values"]].to_string(
                index=False
            ),
            file=sys.stderr,
        )
        raise SystemExit(
            f"{len(conf)} cell(s) are computed by two stages with different values; "
            f"the tables would disagree with each other. Rerun the stage that is "
            f"stale (`python run.py results` reports this per run), or fix the "
            f"stage that recomputes a quantity instead of reusing it."
        )
    idle = results.unexercised(df)
    if idle:
        for line in idle:
            print(line, file=sys.stderr)
        raise SystemExit(
            f"{len(idle)} of the correspondences in results.ALIASES compared "
            f"nothing, so 'no stage disagrees' is not a statement about these "
            f"numbers. Reconcile how the two stages key a cell, or drop the alias."
        )
    checked = results.cross_checks(df)
    n_checked = int(checked["n_cells"].sum()) if not checked.empty else 0
    print(
        f"consistency: {len(df):,} scored cell(s) across {len(run_names)} run(s); "
        f"{n_checked:,} computed more than once and all in agreement"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--artifacts",
        type=Path,
        help="directory holding the per-substrate run directories; "
        "defaults to paths.artifact_root from the site profile",
    )
    ap.add_argument(
        "--env",
        help="environments.yaml profile the artifact root is read from when --artifacts is omitted",
    )
    ap.add_argument(
        "--runs",
        nargs="*",
        default=["cancerfoundation", "scgpt", "default"],
        help="run directory names, in any order",
    )
    ap.add_argument(
        "--shuffle-run", default="default", help="run the patch-shuffle ablation is read from"
    )
    ap.add_argument("--out", type=Path, default=Path("paper_tables"))
    ap.add_argument("--protocol", default="heldout_donor")
    ap.add_argument("--level", default="cross_donor")
    ap.add_argument("--digits", type=int, default=2)
    ap.add_argument("--rank-digits", type=int, default=1)
    ap.add_argument(
        "--integration-digits",
        type=int,
        default=3,
        help="the batch-mixing metrics separate methods in the third "
        "decimal, so they are printed wider than the F1 tables",
    )
    ap.add_argument(
        "--controls", action="store_true", help="add the control block to the effective-rank table"
    )
    ap.add_argument(
        "--gaussian",
        action="store_true",
        help="keep the Gaussian column in the patch-shuffle table",
    )
    ap.add_argument(
        "--all-classes", action="store_true", help="keep every class in the patch-shuffle table"
    )
    ap.add_argument(
        "--pool-usz",
        action="store_true",
        help="report kidney and lung as one USZ cohort, as the "
        "earlier manuscript did; exact point estimates, but "
        "those rows lose their donor bootstrap",
    )
    args = ap.parse_args(argv)
    if args.artifacts is None:
        from vgtfm.config import load_config

        args.artifacts = Path(load_config(None, env=args.env).paths.artifact_root)
        print(f"artifact root from the site profile: {args.artifacts}")

    runs = [Run(args.artifacts / name) for name in args.runs]
    _check_consistency(args.artifacts, args.runs)
    shuffle_run = next(r for r in runs if r.run_name == args.shuffle_run)
    args.out.mkdir(parents=True, exist_ok=True)

    main_tex, n_donors = main_skin_table(
        runs, protocol=args.protocol, level=args.level, digits=args.digits, label="tab:main"
    )
    shuffle, omitted = shuffle_table(
        shuffle_run,
        protocol=args.protocol,
        level=args.level,
        digits=args.digits,
        gaussian=args.gaussian,
        all_classes=args.all_classes,
        pool_usz=args.pool_usz,
        label="tab:patchshuffle",
    )
    outputs = {
        "table_main_skin.tex": main_tex,
        "table_appendix_all_tissues.tex": appendix_all_tissues_table(
            runs,
            protocol=args.protocol,
            level=args.level,
            digits=args.digits,
            label="tab:appendixorgans",
        ),
        "table_effective_rank.tex": effective_rank_table(
            runs, digits=args.rank_digits, controls=args.controls, label="tab:effrank"
        ),
        "table_patch_shuffle.tex": shuffle,
        "table_batch_correction.tex": correction_table(
            runs,
            protocol=args.protocol,
            level=args.level,
            digits=args.integration_digits,
            label="tab:integration",
        ),
    }

    provenance = "\n".join(f"%   {r.provenance()}" for r in runs)
    for name, tex in outputs.items():
        header = (
            f"% Generated by scripts/paper_tables.py from "
            f"{args.protocol}/{args.level}.\n{provenance}\n"
        )
        (args.out / name).write_text(header + tex)
        print(f"wrote {args.out / name}")
    print(f"\nskin cohort: n={n_donors} donors")
    if omitted:
        print("classes left out of table_patch_shuffle.tex (--all-classes keeps them):")
        for o in omitted:
            print(f"  {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
