"""Manuscript tables, built from the run's own outputs.

Nothing here embeds a number: every value is read from ``eval/``, ``ablation/``,
``diagnostics/`` or ``integration/``, so a table cannot drift from the run that
produced it.

Each builder returns a tidy ``DataFrame`` and writes both a CSV (for inspection)
and a LaTeX fragment (for the paper) from that same frame.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from ..data.tables import patient_key
from ..evaluate.reporting import TISSUE_BALANCED

#: Display names, in the order the paper lists them.
MODEL_LABELS = {
    "hvg_pca": "HVG + PCA (counts)",
    "pca": "PCA (gene-only)",
    "ae": r"AE (gene$\rightarrow$H\&E)",
    "cdann": "best model",
    "pca_oracle": r"PCA oracle (H\&E)",
    "pca_oracle_matched": r"PCA oracle (H\&E, matched width)",
    "majority": "Majority baseline",
    "dual_decoder": "Dual decoder (gene + H\\&E)",
    "gene_ae": "Gene-only autoencoder",
    "infonce": "Contrastive (InfoNCE)",
    "jepa": "Joint-embedding predictive",
}

#: Organ names for the paper. The registry stores lowercase keys that mix organ with
#: adjective ("ovarian") and hyphenate plurals ("lymph-nodes"); the table wants neither.
TISSUE_LABELS = {
    "skin": "Skin",
    "lung": "Lung",
    "kidney": "Kidney",
    "bladder": "Bladder",
    "ovarian": "Ovary",
    "brain": "Brain",
    "mesothelium": "Mesothelium",
    "lymph-nodes": "Lymph node",
}


def tissue_label(tissue) -> str:
    """Display name for a registry tissue key, degrading gracefully for new ones."""
    key = str(tissue)
    return TISSUE_LABELS.get(key, key.replace("-", " ").replace("_", " ").capitalize())


def tex_escape(text) -> str:
    """Escape the LaTeX-special characters that appear in our identifiers.

    Config-derived names (``column_shuffled``, ``pct_of_dim``) reach the tables
    verbatim, and a bare underscore is a compile error. Narrow on purpose: only
    ``_ % #`` are escaped, so the hand-written LaTeX in :data:`MODEL_LABELS` passes
    through untouched.
    """
    s = str(text)
    for char in ("_", "%", "#"):
        s = s.replace(char, "\\" + char)
    return s


def _fmt(value: float, lo=None, hi=None, digits: int = 3) -> str:
    if value != value:
        return "--"
    s = f"{value:.{digits}f}"
    if lo is not None and hi is not None and lo == lo and hi == hi:
        s += f" \\tiny{{[{lo:.{digits}f}, {hi:.{digits}f}]}}"
    return s


def _fmt_delta(value: float, lo=None, hi=None, digits: int = 3) -> str:
    """A difference, signed. ``+0.027`` and ``0.027`` read the same at a glance in a
    column of absolute scores, and only one of them says which way the method moved.
    """
    if value != value:
        return "--"
    s = f"{value:+.{digits}f}"
    if lo is not None and hi is not None and lo == lo and hi == hi:
        s += f" \\tiny{{[{lo:+.{digits}f}, {hi:+.{digits}f}]}}"
    return s


def _pooled_or_per_seed(frame: pd.DataFrame, *names: str) -> tuple[str, ...]:
    """Prefer the seed-pooled statistic; fall back to the per-seed one.

    ``eval`` and ``ablate`` write ``ci_lo_pooled`` / ``ci_hi_pooled`` / ``p_pooled``
    alongside the per-seed columns: the percentile of every seed's replicate draws
    taken together, which covers training variability as well as donor sampling.
    They are constant within a group, so the aggregation below can keep averaging
    and still return them unchanged. A run made before those columns existed simply
    falls back, at the cost of an interval that describes one seed.
    """
    out = []
    for name in names:
        pooled = f"{name}_pooled"
        use = pooled if pooled in frame.columns and frame[pooled].notna().any() else name
        out.append(use)
    return tuple(out)


def _donors(value) -> str:
    """Donor count for the ``n`` column, or ``--`` when the run did not record one.

    Every interval in these tables is a donor-level bootstrap, so its resolution is
    set by this number and nothing else: at n donors there are only C(2n-1, n)
    distinct resamples, which is 3 at n=2 and 1,716 at n=7. Printing it beside the
    bracket is what stops the two from being read as the same kind of evidence.
    """
    if value is None or pd.isna(value):
        return "--"
    return f"{int(value)}"


def _write(df: pd.DataFrame, out_dir, stem: str, latex: str | None = None) -> None:
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    if latex is not None:
        (out_dir / f"{stem}.tex").write_text(latex)


def dataset_table(
    cohort: pd.DataFrame,
    registry,
    out_dir,
    *,
    fit_split: str = "train",
    stem: str = "table_datasets",
):
    """Dataset appendix — the cohorts, grouped by organ.

    Every number comes from ``data/cohort.csv``, which the ``data`` stage writes
    from the spot table the run loaded. Names, accessions and citation keys come
    from the registry.

    A cohort is *Training* when any of its slides sits in ``fit_split``; the
    annotated cohorts (TuPro, USZ) live entirely in ``test`` and are never seen
    while fitting.
    """
    if cohort is None or len(cohort) == 0:
        return pd.DataFrame()

    specs = getattr(registry, "datasets", registry) or {}

    # Two registry entries can share a display name (USZ is split kidney/lung so the
    # two tissues get their own rows); only those need the organ spelled out again.
    names = [(specs.get(d, {}) or {}).get("name") or d for d in cohort["dataset_id"].unique()]
    repeated = {n for n, c in Counter(names).items() if c > 1}

    rows = []
    for did, g in cohort.groupby("dataset_id", sort=True):
        spec = specs.get(did, {}) or {}
        cit = spec.get("citation") or {}
        tissue = str(g["tissue"].iloc[0])
        name = spec.get("name") or did
        donors = {patient_key(s, spec.get("donor_pattern")) for s in g["sample_id"]}
        rows.append(
            {
                "organ": tissue_label(tissue),
                "split": "Training" if (g["split"] == fit_split).any() else "Held-out eval",
                "dataset": name,
                "qualifier": tissue_label(tissue).lower() if name in repeated else "",
                "dataset_id": did,
                "accession": cit.get("accession") or "",
                "bibkey": cit.get("bibkey") or "",
                "donors": len(donors),
                "samples": int(len(g)),
                "spots": int(g["n_spots"].sum()),
            }
        )

    df = pd.DataFrame(rows)
    # Organs by size, training cohorts before the held-out ones inside each organ.
    df["_organ"] = df["organ"].map(df.groupby("organ")["spots"].sum())
    df["_split"] = (df["split"] != "Training").astype(int)
    df = (
        df.sort_values(
            ["_organ", "organ", "_split", "dataset"], ascending=[False, True, True, True]
        )
        .drop(columns=["_organ", "_split"])
        .reset_index(drop=True)
    )

    lines = [
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        r"Organ & Split & Dataset & Donors & Samples & Spots \\",
        r"\midrule",
    ]
    previous = None
    for r in df.itertuples():
        if previous is not None and r.organ != previous:
            lines.append(r"\cmidrule(lr){1-6}")
        organ = "" if r.organ == previous else tex_escape(r.organ)
        previous = r.organ
        # One parenthesis for both: "TLS Visium USZ (lung, Zenodo 14620362)".
        detail = [d for d in (r.qualifier, r.accession) if d]
        label = tex_escape(r.dataset)
        if detail:
            label += " (" + ", ".join(tex_escape(d) for d in detail) + ")"
        if r.bibkey:
            label += r" \citep{" + r.bibkey + "}"
        lines.append(f"{organ} & {r.split} & {label} & {r.donors} & {r.samples} & {r.spots:,} \\\\")
    lines += [
        r"\midrule",
        (
            r"\textbf{Total} & & & " + f"\\textbf{{{int(df.donors.sum())}}} & "
            f"\\textbf{{{int(df.samples.sum())}}} & "
            f"\\textbf{{{int(df.spots.sum()):,}}} \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    _write(df, out_dir, stem, "\n".join(lines))
    return df


def annotation_table(
    results: pd.DataFrame,
    bootstrap: pd.DataFrame | None,
    out_dir,
    *,
    protocol: str = "heldout_donor",
    level: str = "cross_donor",
    scope: str = "global",
    stem: str | None = None,
):
    """Table 1 — annotation-prediction macro-F1 per model.

    Reports the mean over seeds, the min-max range over folds, and, when a
    bootstrap is available, the donor-level interval — which is far wider.

    ``macro_f1`` and ``fold_min``/``fold_max`` are **different statistics**, not a
    summary and its spread: the first is one macro average over every fold's
    predictions concatenated, the second is the range of per-fold macro averages.
    A pooled value can therefore sit outside its own fold range, and legitimately
    does for the majority baseline at ``cross_replicate`` — 16 of the 18 folds
    predict only ``Tumor`` and 2 predict only ``Normal lymphoid tissue``, so each
    fold alone has one non-zero cell of skin's four (macro <= 0.24) while their
    union has two (macro 0.32). ``fold_mean`` is reported next to the range so the
    two quantities can be told apart instead of the gap reading as an error.
    """
    view = results[
        (results.protocol == protocol) & (results.level == level) & (results.scope == scope)
    ]
    if view.empty:
        return pd.DataFrame()

    pooled = view[view.fold == "pooled"]
    per_fold = view[view.fold != "pooled"]

    rows = []
    for model, g in pooled.groupby("model"):
        folds = per_fold[per_fold.model == model]
        scores = folds["f1_score"]
        row = {
            "model": model,
            "label": MODEL_LABELS.get(model, model),
            "macro_f1": float(g["f1_score"].mean()),
            "sd_over_seeds": float(g["f1_score"].std(ddof=0)) if len(g) > 1 else np.nan,
            "n_seeds": int(len(g)),
            # The mean of the per-fold scores, which is what `fold_min`/`fold_max`
            # bracket. `macro_f1` above is the pooled statistic and need not lie
            # between them; see the docstring.
            "fold_mean": float(scores.mean()) if len(scores) else np.nan,
            "fold_min": float(scores.min()) if len(scores) else np.nan,
            "fold_max": float(scores.max()) if len(scores) else np.nan,
            "n_folds": int(folds["fold"].nunique()),
            "n_donors": int(g["n_donors"].max()) if "n_donors" in g else np.nan,
        }
        if bootstrap is not None and not bootstrap.empty:
            b = bootstrap[
                (bootstrap.protocol == protocol)
                & (bootstrap.level == level)
                & (bootstrap.scope == scope)
                & (bootstrap.model == model)
                & (bootstrap["class"] == "macro")
            ]
            if not b.empty:
                lo_col, hi_col = _pooled_or_per_seed(b, "ci_lo", "ci_hi")
                row["donor_ci_lo"] = float(b[lo_col].mean())
                row["donor_ci_hi"] = float(b[hi_col].mean())
                row["ci_is_pooled"] = lo_col.endswith("_pooled")
        rows.append(row)

    # Increasing access to information: counts, then the gene FM, then the models
    # trained against morphology, then the two oracles that read it outright.
    order = [
        m
        for m in ("hvg_pca", "pca", "ae", "cdann", "pca_oracle_matched", "pca_oracle", "majority")
        if m in {r["model"] for r in rows}
    ]
    order += sorted({r["model"] for r in rows} - set(order))
    df = pd.DataFrame(rows).set_index("model").reindex(order).reset_index()

    # The organ-balanced scope is an average *over* the tissue scopes, so it has no
    # per-fold rows to bracket — a fold spans one organ. Drop the range column there
    # rather than print a table whose third column is "--" all the way down; the
    # per-organ table is where that spread belongs.
    has_range = bool(df["fold_min"].notna().any())
    ci_label = (
        "organ- and donor-level 95\\% CI" if scope == TISSUE_BALANCED else "donor-level 95\\% CI"
    )
    header = ["Model", "macro-F1"] + (["fold range"] if has_range else []) + [ci_label, "$n$"]
    lines = [
        rf"\begin{{tabular}}{{l{'c' * (len(header) - 2)}r}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for r in df.itertuples():
        lo = getattr(r, "donor_ci_lo", np.nan)
        hi = getattr(r, "donor_ci_hi", np.nan)
        ci = "--" if lo != lo else f"[{lo:.3f}, {hi:.3f}]"
        cells = [r.label, f"{r.macro_f1:.3f}"]
        if has_range:
            cells.append(
                "--" if r.fold_min != r.fold_min else f"[{r.fold_min:.3f}, {r.fold_max:.3f}]"
            )
        cells += [ci, _donors(getattr(r, "n_donors", np.nan))]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(df, out_dir, stem or f"table_annotation_{level}", "\n".join(lines))
    return df


def effective_rank_table(er: pd.DataFrame, out_dir):
    """Table 2 — effective rank of the frozen embeddings, with controls.

    Read the ``% of dim`` column: comparing raw effective ranks across
    representations of different width says little, since 78.7 out of 3072 is a
    *smaller* fraction than 29.3 out of 1152.
    """
    if er.empty:
        return er
    df = er.copy()
    df["representation"] = df["column"].str.replace("_features", "", regex=False)
    df = df[
        [
            "representation",
            "variant",
            "dim",
            "eff_rank",
            "eff_rank_std",
            "pct_of_dim",
            "mean_pairwise_cosine",
        ]
    ]

    obs = df[df.variant == "observed"]
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Representation & Dim & Eff.\ rank & \% of dim \\",
        r"\midrule",
    ]
    for r in obs.itertuples():
        lines.append(
            f"{tex_escape(r.representation)} & {int(r.dim)} & {r.eff_rank:.1f} & "
            f"{r.pct_of_dim:.1f} \\\\"
        )
    if (df.variant != "observed").any():
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{4}{l}{\emph{controls}} \\")
        for r in df[df.variant != "observed"].itertuples():
            lines.append(
                f"{tex_escape(r.representation)} ({tex_escape(r.variant)}) & "
                f"{int(r.dim)} & {r.eff_rank:.1f} & "
                f"{r.pct_of_dim:.1f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(df, out_dir, "table_effective_rank", "\n".join(lines))
    return df


#: Display names for the correction methods, in the order the appendix lists them.
#: Anything the `integrate` stage produces that is not named here still reaches the
#: table, under its own key — a new method should appear rather than vanish.
INTEGRATION_LABELS = {
    "none": "Uncorrected",
    "harmony": "Harmony",
    "bbknn": "BBKNN",
    "combat": "ComBat",
    "scvi": "scVI",
}

#: Column headers for the integration table, as ``(csv name, LaTeX header)``. iLISI
#: and cLISI are both "higher is better" but measure opposite things: iLISI rises as
#: batches mix, cLISI falls as labels stop being separable. They are printed side by
#: side because the second is what the first costs.
INTEGRATION_METRICS = (
    ("batch/ilisi", r"iLISI $\uparrow$"),
    ("batch/kbet", r"kBET $\uparrow$"),
    ("bio/clisi", r"cLISI $\uparrow$"),
)


def integration_over_seeds(wide: pd.DataFrame) -> pd.DataFrame:
    """``integration.csv`` reduced to one row per method: the mean over its fits.

    ``integrate`` scores every seed ``eval`` scores — one row per (method, seed) —
    so that its uncorrected row is that seed's PCA row exactly and the three of them
    average to the number Table 1 prints. Every table below reports a method, not a
    fit, so the seeds are collapsed here rather than in four places.

    The point estimate is the mean, matching how the annotation tables pool over
    seeds. The bracket is the *pooled* one where ``integrate`` wrote it — the
    percentile of the seeds' replicate draws taken together, which covers training
    variability as well as donor sampling — and falls back to the mean of the
    per-seed bounds for a run made before those columns existed, exactly as
    :func:`_pooled_or_per_seed` does for ``eval``.

    A run with a single seed passes through unchanged, and a run written before the
    stage recorded a seed at all has nothing to collapse.
    """
    if wide is None or wide.empty or "seed" not in wide.columns:
        return wide
    numeric = [c for c in wide.columns if c != "seed" and pd.api.types.is_numeric_dtype(wide[c])]
    # `note` is the same sentence on every seed of a method; `first` keeps it rather
    # than dropping the one column that says why a row's probe cells are empty.
    text = [c for c in wide.columns if c not in numeric and c not in ("seed", "method")]
    agg = {c: "mean" for c in numeric} | {c: "first" for c in text}
    out = wide.groupby("method", sort=False).agg(agg).reset_index()
    out["n_seeds"] = wide.groupby("method", sort=False).size().to_numpy()

    # The pooled columns are constant within a method, so the mean above returned
    # them unchanged; promote them over the per-seed bounds they supersede.
    for col in list(out.columns):
        if not col.endswith("_lo_pooled"):
            continue
        base = col[: -len("_lo_pooled")]
        hi = f"{base}_hi_pooled"
        if f"{base}_lo" in out.columns and hi in out.columns:
            take = out[col].notna()
            out.loc[take, f"{base}_lo"] = out.loc[take, col]
            out.loc[take, f"{base}_hi"] = out.loc[take, hi]
    return out


def integration_stem(scope: str) -> str:
    """CSV/LaTeX stem for one scope's integration table.

    ``global`` keeps the bare stem it has always had; a tissue scope is suffixed, the
    same way ``eval`` names its per-organ tables.
    """
    return "table_integration" + ("" if scope == "global" else f"_{scope}")


def integration_table(
    integration: pd.DataFrame,
    deltas: pd.DataFrame | None,
    out_dir,
    *,
    level: str = "cross_donor",
    scope: str = "global",
):
    """Appendix — batch correction scored on batch mixing *and* downstream F1.

    Both halves in one row is the whole point of the stage. Every method raises
    iLISI; the probe column is what says whether the mixing bought anything, and a
    composite score would let one be traded for the other and report the trade as an
    improvement — which is why :mod:`vgtfm.diagnostics.scib` never computes one. Read
    the **delta** column rather than the difference of the two F1 columns, for the
    reason :mod:`vgtfm.diagnostics.integration` gives; the same module says why
    BBKNN's probe cells are empty.

    *scope* selects the cohort the probe columns are read at: ``global`` is the
    pooled macro over ``(tissue, class)`` cells from every organ, and a tissue name
    restricts to that organ, as the paper's other appendix tables do. The batch
    metrics are cohort-wide either way -- the scIB panel scores one stratified
    sample of the whole annotated cohort with the slide as the batch, so there is no
    per-organ version of them to select. A tissue-scoped table therefore pairs a
    cohort-wide mixing score with an organ's downstream score, which the caption has
    to say.
    """
    if integration is None or integration.empty:
        return pd.DataFrame()

    # One row per method, not per (method, seed): `integrate` scores every fit `eval`
    # scores, and this table reports methods.
    integration = integration_over_seeds(integration)

    d = pd.DataFrame() if deltas is None or deltas.empty else deltas
    if not d.empty:
        # A pre-scope run wrote no `scope` column; its deltas are the pooled ones.
        if "scope" in d.columns:
            d = d[d["scope"] == scope]
        elif scope != "global":
            d = d.iloc[:0]
        d = d[d["level"] == level]
        # Same collapse on the delta side, and the pooled p-value and bracket where
        # the stage wrote them — a per-seed delta describes one fit.
        if "seed" in d.columns and not d.empty:
            lo, hi = _pooled_or_per_seed(d, "ci_lo", "ci_hi")
            (p_col,) = _pooled_or_per_seed(d, "p_two_sided")
            d = (
                d.assign(ci_lo=d[lo], ci_hi=d[hi], p_two_sided=d[p_col])
                .groupby("method", sort=False)
                .agg(
                    delta_f1=("delta_f1", "mean"),
                    ci_lo=("ci_lo", "mean"),
                    ci_hi=("ci_hi", "mean"),
                    p_two_sided=("p_two_sided", "mean"),
                    n_donors=("n_donors", "max"),
                )
            )
        else:
            d = d.set_index("method")

    key = f"f1/{scope}/{level}"
    f1, lo, hi, nd = key, f"{key}_lo", f"{key}_hi", f"{key}_n_donors"
    rows = []
    for r in integration.to_dict("records"):
        method = str(r["method"])
        row = {
            "method": method,
            "label": INTEGRATION_LABELS.get(method, method),
            "note": r.get("note"),
        }
        for col, _header in INTEGRATION_METRICS:
            row[col] = r.get(col, np.nan)
        row["f1"] = r.get(f1, np.nan)
        row["f1_lo"], row["f1_hi"] = r.get(lo, np.nan), r.get(hi, np.nan)
        row["n_donors"] = r.get(nd, np.nan)
        dr = d.loc[method] if method in d.index else None
        row["delta_f1"] = np.nan if dr is None else dr["delta_f1"]
        row["delta_lo"] = np.nan if dr is None else dr["ci_lo"]
        row["delta_hi"] = np.nan if dr is None else dr["ci_hi"]
        row["p_two_sided"] = np.nan if dr is None else dr["p_two_sided"]
        rows.append(row)
    df = pd.DataFrame(rows)

    headers = " & ".join(h for _c, h in INTEGRATION_METRICS)
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        rf"Method & {headers} & Macro-F1 & $\Delta$ F1 & $p$ \\",
        r"\midrule",
    ]
    for r in df.to_dict("records"):
        cells = [_fmt(r[c]) for c, _h in INTEGRATION_METRICS]
        cells.append(_fmt(r["f1"], r["f1_lo"], r["f1_hi"]))
        cells.append(_fmt_delta(r["delta_f1"], r["delta_lo"], r["delta_hi"]))
        p_val = r["p_two_sided"]
        cells.append("--" if p_val != p_val else f"{p_val:.3f}")
        lines.append(f"{tex_escape(r['label'])} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    # Both notes are LaTeX comments rather than table rows: they belong in the
    # caption, and the builder cannot know how the caption is worded.
    donors = df["n_donors"].dropna()
    if not donors.empty:
        # Every interval here is a donor-level bootstrap, so the donor count sets its
        # resolution and nothing else does. It is one number for the whole table —
        # the folds do not change with the correction — so it goes here rather than
        # into a column that would repeat it on every row.
        lines.append(
            f"% intervals: donor-level bootstrap over "
            f"{_donors(donors.iloc[0])} donors at the {level} level, "
            f"{scope} scope; the delta column is paired on a shared donor "
            f"resample."
        )
    if scope != "global":
        lines.append(
            f"% the probe columns are {scope} only; iLISI, kBET and cLISI "
            f"are cohort-wide -- the scIB panel scores one stratified "
            f"sample of the whole annotated cohort, batched by slide."
        )
    graph_only = df[df["f1"].isna() & df["note"].notna()]
    if not graph_only.empty:
        # The empty cells carry the argument, and a reader who meets them without
        # the reason reads them as a failed run.
        names = ", ".join(str(n) for n in graph_only["label"])
        lines.append(
            f"% {names}: corrects the neighbour graph, not the embedding "
            f"— no corrected matrix for the probe to read, and no "
            f"deployable gene representation at the end of it."
        )
    _write(df, out_dir, integration_stem(scope), "\n".join(lines))
    return df


def shuffle_table(
    per_class: pd.DataFrame,
    out_dir,
    *,
    bootstrap: pd.DataFrame | None = None,
    protocol: str = "pooled_loso",
    level: str = "all",
):
    """Table 3 — per-class F1 under matched vs shuffled morphology targets.

    Rows are (tissue, class); columns are the ablation conditions, each carrying a
    donor-level 95% interval when ``ablate`` produced one.

    Those intervals overlap heavily between conditions: at 7 melanoma and 6 USZ
    patients an absolute per-class F1 is barely resolvable. The resolvable
    comparison is the paired one — same donors resampled for both conditions, so
    the donor effect cancels — in ``ablation/paired_deltas.csv`` and
    ``fig_patch_shuffle_deltas``. Read the two together.
    """
    if per_class.empty or "condition" not in per_class.columns:
        return pd.DataFrame()
    view = per_class[
        (per_class.protocol == protocol)
        & (per_class.level == level)
        & (per_class.fold == "pooled")
        & (per_class.scope != "global")
    ]
    if view.empty:
        return pd.DataFrame()

    df = (
        view.groupby(["scope", "class", "condition"])
        .agg(
            f1=("f1_score", "mean"),
            f1_sd_over_seeds=("f1_score", lambda s: float(s.std(ddof=0))),
            support=("support", "max"),
            n_seeds=("f1_score", "size"),
        )
        .reset_index()
        .rename(columns={"scope": "tissue"})
    )

    # Average the interval over seeds, matching how the point estimate is pooled.
    ci = {}
    donors: dict[str, float] = {}
    if bootstrap is not None and not bootstrap.empty:
        b = bootstrap[
            (bootstrap.protocol == protocol)
            & (bootstrap.level == level)
            & (bootstrap.scope != "global")
            & (bootstrap["class"] != "macro")
        ]
        if not b.empty:
            lo_col, hi_col = _pooled_or_per_seed(b, "ci_lo", "ci_hi")
            cols = {"ci_lo": (lo_col, "mean"), "ci_hi": (hi_col, "mean")}
            if "n_donors" in b.columns:
                cols["n_donors"] = ("n_donors", "max")
            agg = (
                b.groupby(["scope", "class", "condition"])
                .agg(**cols)
                .reset_index()
                .rename(columns={"scope": "tissue"})
            )
            df = df.merge(agg, on=["tissue", "class", "condition"], how="left")
            ci = {
                (r["tissue"], r["class"], r["condition"]): (r["ci_lo"], r["ci_hi"])
                for _, r in df.iterrows()
            }
            # A property of the tissue, not of the class or the condition: USZ
            # kidney is two patients whatever row you read it on.
            if "n_donors" in df.columns:
                donors = df.groupby("tissue")["n_donors"].max().to_dict()

    wide = df.pivot_table(
        index=["tissue", "class", "support"], columns="condition", values="f1"
    ).reset_index()
    conditions = [c for c in ("none", "within-sample", "global", "gaussian") if c in wide.columns]
    headers = {
        "none": "matched",
        "within-sample": "within-slide shuffle",
        "global": "global shuffle",
        "gaussian": "Gaussian",
    }
    lines = [
        r"\begin{tabular}{ll" + "c" * len(conditions) + "}",
        r"\toprule",
        "Tissue & Class (support) & "
        + " & ".join(headers.get(c, tex_escape(c)) for c in conditions)
        + r" \\",
        r"\midrule",
    ]
    for _, row in wide.iterrows():
        cells = " & ".join(
            _fmt(row[c], *ci.get((row["tissue"], row["class"], c), (None, None)))
            for c in conditions
        )
        n = _donors(donors.get(row["tissue"], float("nan")))
        tissue = tex_escape(row["tissue"])
        if n != "--":
            tissue += f" \\tiny{{({n} pt.)}}"
        lines.append(
            f"{tissue} & {tex_escape(row['class'])} ({int(row['support']):,}) & {cells} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(df, out_dir, "table_patch_shuffle", "\n".join(lines))
    return df


def delta_table(
    deltas: pd.DataFrame,
    out_dir,
    *,
    protocol: str = "heldout_donor",
    level: str = "cross_donor",
    stem: str | None = None,
):
    """Paired model-minus-PCA deltas with donor-level intervals.

    The comparison this cohort size supports: a delta whose interval excludes zero
    is a real difference, whereas two absolute scores with overlapping intervals
    are not evidence of one.
    """
    if deltas.empty:
        return deltas
    view = deltas[
        (deltas.protocol == protocol) & (deltas.level == level) & (deltas["class"] == "macro")
    ]
    if view.empty:
        return pd.DataFrame()
    lo_col, hi_col, p_col = _pooled_or_per_seed(view, "ci_lo", "ci_hi", "p_two_sided")
    agg = {
        "delta": ("delta", "mean"),
        "ci_lo": (lo_col, "mean"),
        "ci_hi": (hi_col, "mean"),
        "p": (p_col, "mean"),
    }
    # Constant across seeds — the folds do not change — so max is just "the value".
    if "n_donors" in view.columns:
        agg["n_donors"] = ("n_donors", "max")
    df = view.groupby(["model", "scope"]).agg(**agg).reset_index()

    lines = [
        r"\begin{tabular}{llccr}",
        r"\toprule",
        r"Model & Scope & $\Delta$ macro-F1 vs PCA & 95\% CI & $n$ \\",
        r"\midrule",
    ]
    for r in df.itertuples():
        lines.append(
            f"{MODEL_LABELS.get(r.model, tex_escape(r.model))} & {tex_escape(r.scope)} "
            f"& {r.delta:+.3f} & [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] & "
            f"{_donors(getattr(r, 'n_donors', np.nan))} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(df, out_dir, stem or f"table_deltas_{level}", "\n".join(lines))
    return df


#: Short names for the fold levels, for the "held out in" column. The full names
#: are long enough that the column would dominate the table.
_LEVEL_ABBREV = {
    "cross_replicate": "rep",
    "cross_region": "reg",
    "cross_donor": "donor",
    "cross_slide": "slide",
}


def _region_label(region, replicate) -> str:
    """Display name for a region, or a dash where the cohort has no region level.

    ``-1`` is the sentinel :mod:`vgtfm.data.tables` writes for a slide whose id
    encodes no region or replicate — USZ, MOSAIC, De Zuani. It means "this cohort
    captured one block per patient", not "unknown", so it earns a dash rather than
    an invented ``Region 1``.
    """
    if int(region) < 0:
        return "--" if int(replicate) < 0 else f"Replicate {int(replicate)}"
    return f"Region {int(region)}"


def fold_hierarchy_table(
    cohort: pd.DataFrame, fold_map: dict, out_dir, *, stem: str = "table_fold_hierarchy"
):
    """The evaluation cohorts' donor/region/replicate hierarchy, and the folds it generates.

    One row per annotated slide, grouped organ -> patient -> region, because that
    nesting *is* the fold structure: ``cross_replicate`` holds out one row of a
    region block, ``cross_region`` one region block of a patient, ``cross_donor`` one
    patient of an organ. The ``held out in`` column says which levels actually
    evaluate that slide, which is the part a reader cannot infer from the ids alone —
    USZ carries no region or replicate, so its slides are held out at the patient
    level only.

    Unannotated slides are excluded: they train the representation and are never
    evaluated, so they define no fold.
    """
    if cohort is None or cohort.empty:
        return pd.DataFrame()
    df = cohort[cohort.get("n_annotated", 0) > 0].copy()
    if df.empty:
        return pd.DataFrame()

    # Which levels evaluate each slide, in the order the levels were generated.
    evaluated: dict[str, list[str]] = {}
    for level, specs in (fold_map or {}).items():
        for spec in specs:
            for sid in spec.get("eval_sample_ids", ()):
                seen = evaluated.setdefault(str(sid), [])
                if level not in seen:
                    seen.append(level)
    order = list(fold_map or {})
    df["eval_levels"] = [
        "|".join(
            sorted(evaluated.get(str(s), []), key=lambda lv: order.index(lv) if lv in order else 99)
        )
        for s in df["sample_id"]
    ]
    df["n_eval_folds"] = [len(evaluated.get(str(s), [])) for s in df["sample_id"]]
    df["region_label"] = [_region_label(r, k) for r, k in zip(df["region"], df["replicate"])]

    df = df.sort_values(["tissue", "donor", "region", "replicate", "sample_id"])
    df = df[
        [
            "tissue",
            "dataset_id",
            "donor",
            "region_label",
            "replicate",
            "sample_id",
            "n_annotated",
            "n_classes",
            "eval_levels",
            "n_eval_folds",
        ]
    ].reset_index(drop=True)

    lines = [
        "% requires \\usepackage{booktabs,multirow}",
        r"\begin{tabular}{lllll}",
        r"\toprule",
        r"\textbf{Organ} & \textbf{Donor} & \textbf{Region} & "
        r"\textbf{Sample ID} & \textbf{Held out in} \\",
        r"\midrule",
    ]
    # Spans are precomputed so \multirow knows its height before the rows are
    # emitted; the label is written on a group's first row and blanked after.
    tissue_span = df.groupby("tissue", sort=False)["sample_id"].transform("size")
    donor_span = df.groupby(["tissue", "donor"], sort=False)["sample_id"].transform("size")
    region_span = df.groupby(["tissue", "donor", "region_label"], sort=False)[
        "sample_id"
    ].transform("size")
    prev_tissue = prev_donor = prev_region = None
    for i, r in enumerate(df.itertuples()):
        if prev_tissue is not None:
            if r.tissue != prev_tissue:
                lines.append(r"\midrule")
            elif r.donor != prev_donor:
                lines.append(r"\cmidrule(lr){2-5}")
            elif r.region_label != prev_region:
                lines.append(r"\cmidrule(lr){3-5}")
        cells = []
        for value, changed, span in (
            (tissue_label(r.tissue), r.tissue != prev_tissue, tissue_span[i]),
            (r.donor, (r.tissue, r.donor) != (prev_tissue, prev_donor), donor_span[i]),
            (
                r.region_label,
                (r.tissue, r.donor, r.region_label) != (prev_tissue, prev_donor, prev_region),
                region_span[i],
            ),
        ):
            if not changed:
                cells.append("")
            elif span > 1:
                cells.append(f"\\multirow{{{int(span)}}}{{*}}{{{tex_escape(value)}}}")
            else:
                cells.append(tex_escape(value))
        held = ", ".join(_LEVEL_ABBREV.get(lv, lv) for lv in r.eval_levels.split("|") if lv) or "--"
        lines.append(" & ".join([*cells, tex_escape(r.sample_id), held]) + r" \\")
        prev_tissue, prev_donor, prev_region = r.tissue, r.donor, r.region_label
    lines += [r"\bottomrule", r"\end{tabular}"]

    _write(df, out_dir, stem, "\n".join(lines))
    return df


def per_organ_table(
    results: pd.DataFrame,
    bootstrap: pd.DataFrame | None,
    out_dir,
    *,
    protocol: str = "heldout_donor",
    level: str = "cross_donor",
    stem: str | None = None,
):
    """Macro-F1 per representation and organ, with each organ's own donor-level CI.

    The companion to the headline table, which reports one organ-balanced number.
    This is where the spread behind that number is readable: an organ's macro runs
    over its own class vocabulary, and its interval resamples only its own donors,
    so the ``n`` column is the resolution of the bracket beside it — at two donors
    there are three distinct resamples and the interval is nearly a point.

    Wide rather than long: organs across, representations down, which is the shape
    a reader compares in and keeps the table to one page.
    """
    view = results[
        (results.protocol == protocol) & (results.level == level) & (results.fold == "pooled")
    ]
    organs = sorted(set(view.scope) - {"global", TISSUE_BALANCED})
    if not organs or view.empty:
        return pd.DataFrame()

    def ci_for(model: str, scope: str) -> tuple[float, float]:
        if bootstrap is None or bootstrap.empty:
            return (np.nan, np.nan)
        b = bootstrap[
            (bootstrap.protocol == protocol)
            & (bootstrap.level == level)
            & (bootstrap.scope == scope)
            & (bootstrap.model == model)
            & (bootstrap["class"] == "macro")
        ]
        if b.empty:
            return (np.nan, np.nan)
        lo_col, hi_col = _pooled_or_per_seed(b, "ci_lo", "ci_hi")
        return float(b[lo_col].mean()), float(b[hi_col].mean())

    rows = []
    for model, g in view.groupby("model"):
        row = {"model": model, "label": MODEL_LABELS.get(model, model)}
        for organ in organs:
            s = g[g.scope == organ]["f1_score"]
            lo, hi = ci_for(model, organ)
            row[f"{organ}_macro_f1"] = float(s.mean()) if len(s) else np.nan
            row[f"{organ}_ci_lo"], row[f"{organ}_ci_hi"] = lo, hi
            n = g[g.scope == organ]["n_donors"]
            row[f"{organ}_n_donors"] = int(n.max()) if len(n) else np.nan
        tb = g[g.scope == TISSUE_BALANCED]["f1_score"]
        if len(tb):
            row["tissue_balanced_macro_f1"] = float(tb.mean())
            lo, hi = ci_for(model, TISSUE_BALANCED)
            row["tissue_balanced_ci_lo"], row["tissue_balanced_ci_hi"] = lo, hi
        rows.append(row)

    order = [m for m in MODEL_LABELS if m in {r["model"] for r in rows}]
    order += sorted({r["model"] for r in rows} - set(order))
    df = pd.DataFrame(rows).set_index("model").reindex(order).reset_index()

    has_tb = "tissue_balanced_macro_f1" in df.columns
    cols = "l" + "c" * (len(organs) + (1 if has_tb else 0))
    donors = {o: df[f"{o}_n_donors"].max() for o in organs}
    head = " & ".join(
        [""]
        + [f"{tissue_label(o)} ($n$={_donors(donors[o])})" for o in organs]
        + (["Organ-balanced"] if has_tb else [])
    )
    lines = [rf"\begin{{tabular}}{{{cols}}}", r"\toprule", head + r" \\", r"\midrule"]
    for r in df.itertuples(index=False):
        cells = []
        for key in [*organs] + (["tissue_balanced"] if has_tb else []):
            f1 = getattr(r, f"{key}_macro_f1", np.nan)
            lo = getattr(r, f"{key}_ci_lo", np.nan)
            hi = getattr(r, f"{key}_ci_hi", np.nan)
            if f1 != f1:
                cells.append("--")
            elif lo != lo:
                cells.append(f"{f1:.3f}")
            else:
                cells.append(rf"{f1:.3f} \tiny{{[{lo:.3f}, {hi:.3f}]}}")
        lines.append(" & ".join([r.label] + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(df, out_dir, stem or f"table_annotation_by_organ_{protocol}_{level}", "\n".join(lines))
    return df
