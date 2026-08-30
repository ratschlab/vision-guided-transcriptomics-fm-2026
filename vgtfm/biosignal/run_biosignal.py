"""``biosignal`` stage — what does refinement do to per-gene predictivity?

For each fold: fit a ridge from the *frozen* gene embedding to expression on the
training spots, do the same from the *refined* embedding, and score both on the
held-out spots. Pooling the held-out predictions over a level's folds gives one
R^2 per gene per representation, and the difference between them says whether the
refinement preserved that gene's signal or erased it.

Read the quartile table rather than the mean delta-R^2, which averages two opposite
effects together: a refinement that simply shrinks predictions toward zero lifts
genes the frozen embedding predicted badly and pushes down the ones it predicted
well, monotonically across quartiles of the frozen R^2. :func:`enrichment_tail` then
reads the same ranking twice — by position (GSEA) and by effect size per gene set,
against a null matched on that same frozen R^2 — because only the second can tell a
coherent biological programme from the shrinkage the quartile table tabulates.

``PCA_k(frozen)`` is scored alongside as a capacity control, since the refined
embedding is narrower than the frozen one and part of any drop is dimensionality.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import headline_seed
from ..data import folds as folds_mod
from ..data import tables
from ..degraded import refuse
from ..evaluate.probes import standardize
from ..models.train import embedding_path
from . import ridge as ridge_mod
from .expression import ExpressionIndex


#: Fewest spots a fold must resolve in the raw ``.h5ad`` files, on each side, to be
#: worth scoring. Not every spot in the cohort table is reachable there — ids drift
#: between the tokenizer output and the source cohorts — and a fold that resolves a
#: handful yields an R^2 that is noise with a number attached.
MIN_RESOLVED_TRAIN, MIN_RESOLVED_TEST = 50, 20

#: ``name -> (minuend, subtrahend)`` over the ``r2_*`` columns, naming the differences
#: the enrichment can be ranked on.
#:
#: ``guided_vs_frozen`` is the difference the stage reports as ``delta_r2``, and on its
#: own it cannot attribute anything to morphology: the guided embedding is also the
#: narrower one. ``capacity_vs_frozen`` is that same width change carried out by PCA,
#: with no morphology anywhere in it, so a pathway that scores alike in both was never
#: evidence about guidance. ``guided_vs_capacity`` holds the width fixed and is the
#: contrast that isolates what the morphology supervision did.
GSEA_CONTRASTS = {
    "guided_vs_frozen": ("r2_refined", "r2_frozen"),
    "capacity_vs_frozen": ("r2_pca_control", "r2_frozen"),
    "guided_vs_capacity": ("r2_refined", "r2_pca_control"),
}


def _fold_indices(table, spec):
    train = table.rows_for_samples(spec.train_sample_ids)
    test = table.rows_for_samples(spec.eval_sample_ids)
    return train, test


def _subsample(rows: np.ndarray, cap: int, seed: int) -> np.ndarray:
    if cap <= 0 or len(rows) <= cap:
        return rows
    return np.sort(np.random.default_rng(seed).choice(rows, size=cap, replace=False))


def _scope_to_tissues(fold_map: dict, table, tissues) -> dict:
    """Keep only the folds whose evaluated slides all lie in *tissues*.

    A fold that spans two organs cannot be attributed to either, so it is dropped
    rather than assigned. With no *tissues* the map is returned untouched.
    """
    if not tissues:
        return fold_map
    keep = set(tissues)
    tissue_of = dict(zip(table.sample_id.astype(str), table.col("tissue").astype(str)))
    scoped, dropped = {}, 0
    for level, specs in fold_map.items():
        kept = [
            s
            for s in specs
            if {tissue_of.get(str(x), "unknown") for x in s.eval_sample_ids} <= keep
        ]
        dropped += len(specs) - len(kept)
        scoped[level] = kept
    print(f"  scope: {', '.join(sorted(keep))} ({dropped} fold(s) from other organs dropped)")
    if not any(scoped.values()):
        refuse(
            f"per-gene R^2 for organs {sorted(keep)}",
            "no fold in any level is confined to them",
            hint="biosignal.tissues names organs of the annotated cohort; "
            "leave it empty to score every organ together",
        )
    return scoped


def _representations(cfg, table, seed: int) -> tuple[dict, str]:
    """The embeddings scored against each other, and the refined one's model name.

    Three of them where the capacity control is on. The refined embedding is
    narrower than the frozen one, so part of any drop in per-gene R^2 is
    dimensionality rather than lost information; ``PCA_k(frozen)`` at the refined
    width is what separates the two.
    """
    refined_name = "ae" if "ae" in cfg.models.names else cfg.models.names[-1]
    path = embedding_path(cfg, refined_name, seed)
    if not path.exists():
        raise SystemExit(f"missing {path}. Run `python run.py train` first.")
    Z_refined = np.load(path)
    reps = {"frozen": table.gene, "refined": Z_refined}
    print(
        f"  frozen {table.gene.shape[1]}d vs refined '{refined_name}' "
        f"{Z_refined.shape[1]}d (seed {seed})"
    )

    if cfg.biosignal.include_pca_control:
        from sklearn.decomposition import PCA

        fit_rows = np.where(table.col("split") == cfg.train.fit_split)[0]
        if not len(fit_rows):
            fit_rows = np.arange(table.n)
        pca = PCA(n_components=min(Z_refined.shape[1], table.gene.shape[1]), random_state=seed)
        pca.fit(table.gene[fit_rows])
        reps["pca_control"] = pca.transform(table.gene).astype(np.float32)
        print(f"  capacity control: PCA-{reps['pca_control'].shape[1]} of the frozen embedding")
    return reps, refined_name


def _build_index(cfg, table, fold_map: dict) -> ExpressionIndex:
    """The gene vocabulary and the raw-count reader the folds are scored through.

    Scoped to the folds, the vocabulary would follow the scope — and a skin run and
    a lung run would then be scored over different gene lists and could not be
    compared gene by gene. ``shared_vocabulary`` widens it to every annotated spot so
    the organ runs line up; the ``.h5ad`` reads still cover only what the folds touch.
    """
    b = cfg.biosignal
    used = sorted(
        {
            s
            for specs in fold_map.values()
            for spec in specs
            for s in (*spec.train_sample_ids, *spec.eval_sample_ids)
        }
    )
    rows = table.rows_for_samples(used)
    if len(rows) == 0:
        raise SystemExit("no fold slides present in the loaded cohort")

    vocab_rows = rows
    if b.shared_vocabulary:
        labeled = np.where(table.labeled)[0]
        vocab_rows = labeled if len(labeled) else rows

    index = ExpressionIndex(cfg)
    index.build_vocabulary(
        table.col("dataset_id")[vocab_rows],
        table.sample_id[vocab_rows],
        table.col("spot_id")[vocab_rows],
        min_expressed=b.min_expressed,
        seed=cfg.folds.seed,
    )
    print(
        f"  vocabulary: {len(index.genes):,} genes from "
        f"{'every annotated spot' if b.shared_vocabulary else 'this scope'} "
        f"({len(vocab_rows):,} spots, min_expressed={b.min_expressed})"
    )
    return index


def _fold_targets(index: ExpressionIndex, table, tr, te, name: str):
    """Standardised expression for one fold, or ``None`` if too little resolved.

    Targets are standardised on *training* statistics, so R^2 is in units of
    training-fold SD and comparable across folds. A spot the ``.h5ad`` files cannot
    resolve is dropped from both sides rather than imputed.
    """
    Y_tr, ok_tr = index.matrix(
        table.col("dataset_id")[tr], table.sample_id[tr], table.col("spot_id")[tr]
    )
    Y_te, ok_te = index.matrix(
        table.col("dataset_id")[te], table.sample_id[te], table.col("spot_id")[te]
    )
    if ok_tr.sum() < MIN_RESOLVED_TRAIN or ok_te.sum() < MIN_RESOLVED_TEST:
        print(
            f"    {name}: only {int(ok_tr.sum())}/{int(ok_te.sum())} spots "
            f"resolved in the .h5ad files — skipped"
        )
        return None
    Y_tr, Y_te = Y_tr[ok_tr], Y_te[ok_te]
    mu = Y_tr.mean(axis=0, keepdims=True)
    sd = Y_tr.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (Y_tr - mu) / sd, (Y_te - mu) / sd, tr[ok_tr], te[ok_te]


def _fit_fold(
    reps: dict,
    tr_ok,
    te_ok,
    Y_tr_s,
    *,
    alpha: float,
    grid,
    name: str,
    alphas: dict,
    saturated: dict,
) -> dict:
    """One ridge per representation on one fold, returning held-out predictions."""
    preds = {}
    for rep, Z in reps.items():
        Xtr, Xte = standardize(Z[tr_ok], Z[te_ok])
        if grid is None:
            preds[rep] = ridge_mod.predict_fold(Xtr, Y_tr_s, Xte, alpha)
            continue
        preds[rep], chosen = ridge_mod.predict_fold_gcv(Xtr, Y_tr_s, Xte, grid)
        alphas.setdefault(rep, []).append(float(np.median(chosen)))
        lo_frac, hi_frac = ridge_mod.grid_saturation(chosen, grid)
        if lo_frac > 0.05:
            print(
                f"    {name}: WARNING {100 * lo_frac:.0f}% of genes chose the "
                f"smallest penalty offered for '{rep}' — widen "
                f"biosignal.ridge_alpha_grid downwards"
            )
        saturated.setdefault(rep, []).append(hi_frac)
    return preds


def _level_summary(df: pd.DataFrame, acc, n_folds: int, alphas: dict, saturated: dict) -> dict:
    valid = df.dropna(subset=["delta_r2"])
    summary = {
        "n_folds": n_folds,
        "n_spots_pooled": int(acc.n),
        "n_genes": int(len(valid)),
        "mean_delta_r2": float(valid["delta_r2"].mean()),
        "pct_genes_improved": float(100.0 * (valid["delta_r2"] > 0).mean()),
        "mean_r2_frozen": float(valid["r2_frozen"].mean()),
        "mean_r2_refined": float(valid["r2_refined"].mean()),
    }
    if "delta_r2_pca_control" in valid:
        summary["mean_delta_r2_pca_control"] = float(valid["delta_r2_pca_control"].mean())
    if alphas:
        # The fitted penalty is part of the result, not a hidden knob: a
        # frozen-vs-refined gap is only about information if both sides were
        # regularised for the design they were fitted on.
        summary["median_ridge_alpha"] = {rep: float(np.median(v)) for rep, v in alphas.items()}
        summary["frac_genes_at_max_alpha"] = {
            rep: float(np.mean(v)) for rep, v in saturated.items()
        }
    print(
        f"    pooled over {n_folds} folds, {acc.n:,} spots, {len(valid):,} genes: "
        f"mean dR2 = {summary['mean_delta_r2']:+.4f}, "
        f"{100 - summary['pct_genes_improved']:.1f}% of genes suppressed"
    )
    return summary


def _score_level(
    cfg, out, level: str, specs: list, *, table, reps: dict, index: ExpressionIndex, grid
) -> tuple[pd.DataFrame, dict] | None:
    """Per-gene R^2 for one fold level, pooled over its folds.

    ``None`` when the level has no folds — a property of the cohort, not a failure.
    Folds that individually resolve too few spots are skipped and reported; a level
    where *every* fold is skipped refuses, because that is a broken path to the raw
    counts rather than a cohort that cannot support the level.
    """
    b = cfg.biosignal
    if not specs:
        # The fold builder already printed "<level>: 0 fold(s)" and why.
        print(f"\n  -- {level}: no folds for this cohort --")
        return None
    print(f"\n  -- {level}: {len(specs)} fold(s) --")

    acc = ridge_mod.R2Accumulator(len(index.genes), tuple(reps))
    alphas: dict[str, list[float]] = {}
    saturated: dict[str, list[float]] = {}
    widest = max(Z.shape[1] for Z in reps.values())
    n_folds = 0
    for spec in specs:
        tr, te = _fold_indices(table, spec)
        if len(tr) == 0 or len(te) == 0:
            print(
                f"    {spec.name}: {len(tr)} train / {len(te)} test spots "
                f"present in the cohort — skipped"
            )
            continue
        tr = _subsample(tr, b.max_train_spots, cfg.folds.seed)

        resolved = _fold_targets(index, table, tr, te, spec.name)
        if resolved is None:
            continue
        Y_tr_s, Y_te_s, tr_ok, te_ok = resolved

        if len(tr_ok) < 5 * widest:
            # A ridge with alpha=1 on a design matrix that is nearly square is
            # essentially unregularised; held-out R^2 then goes wildly negative and
            # the frozen-vs-refined comparison becomes meaningless.
            print(
                f"    {spec.name}: WARNING only {len(tr_ok):,} training spots for "
                f"a {widest}-dimensional embedding — R^2 will be unstable"
            )

        preds = _fit_fold(
            reps,
            tr_ok,
            te_ok,
            Y_tr_s,
            alpha=b.ridge_alpha,
            grid=grid,
            name=spec.name,
            alphas=alphas,
            saturated=saturated,
        )
        acc.update(Y_te_s, preds)
        n_folds += 1
        picked = (
            ""
            if grid is None
            else "  alpha " + " ".join(f"{rep}={alphas[rep][-1]:.3g}" for rep in reps)
        )
        print(f"    {spec.name}: train {len(tr_ok):,} / test {len(te_ok):,}{picked}")

    if n_folds == 0:
        refuse(
            f"per-gene R^2 for the '{level}' level",
            f"all {len(specs)} of its folds were skipped (see above)",
            hint="the .h5ad files under paths.raw_h5ad must cover the cohort's "
            "spot ids — `bash slurm/preflight.sh` checks this",
        )

    r2 = acc.r2()
    df = pd.DataFrame({"gene": index.genes, **{f"r2_{k}": v for k, v in r2.items()}})
    df["delta_r2"] = df["r2_refined"] - df["r2_frozen"]
    if "r2_pca_control" in df:
        df["delta_r2_pca_control"] = df["r2_pca_control"] - df["r2_frozen"]
    df.insert(0, "level", level)

    quart = ridge_mod.quartile_table(df["r2_frozen"].to_numpy(), df["delta_r2"].to_numpy())
    if quart.empty:
        refuse(
            f"the frozen-R^2 quartile table for '{level}'",
            "no gene had both a frozen and a refined R^2, or every frozen R^2 was identical",
            hint="check that the .h5ad expression matrices loaded",
        )
    quart.insert(0, "level", level)
    quart.to_csv(out / f"quartiles_{level}.csv", index=False)

    summary = _level_summary(df, acc, n_folds, alphas, saturated)
    print(quart.to_string(index=False))
    enrichment_tail(cfg, out, level, df)
    return df, summary


def run(cfg) -> None:
    b = cfg.biosignal
    out = cfg.sub("biosignal", ridge_mod.scope_name(b.tissues))
    table = tables.load(cfg)
    fold_map = _scope_to_tissues(folds_mod.generate_for(cfg, table), table, b.tissues)

    seed = headline_seed(cfg)
    reps, refined_name = _representations(cfg, table, seed)
    index = _build_index(cfg, table, fold_map)

    grid = ridge_mod.alpha_grid(b.ridge_alpha_grid) if b.ridge_alpha_search else None
    if grid is None:
        print(
            f"  ridge penalty fixed at alpha={b.ridge_alpha} — the frozen-vs-refined "
            f"gap will carry a width term of roughly (d_frozen - d_refined)/n_train"
        )
    else:
        print(
            f"  ridge penalty fitted per gene by GCV over {len(grid)} values in "
            f"[{grid[0]:.3g}, {grid[-1]:.3g}]"
        )

    per_gene_frames, summary = [], {}
    for level, specs in fold_map.items():
        scored = _score_level(
            cfg, out, level, specs, table=table, reps=reps, index=index, grid=grid
        )
        if scored is None:
            continue
        df, summary[level] = scored
        per_gene_frames.append(df)

    if not per_gene_frames:
        refuse(
            "per-gene R^2 at any fold level",
            "no level of the fold hierarchy yielded a usable fold",
            hint="check folds.levels against the slides in this cohort",
        )
    pd.concat(per_gene_frames, ignore_index=True).to_csv(out / "per_gene_r2.csv", index=False)
    (out / "summary.json").write_text(
        json.dumps(
            {
                "substrate": cfg.data.substrate,
                "refined_model": refined_name,
                "seed": seed,
                "ridge_alpha": b.ridge_alpha,
                "levels": summary,
            },
            indent=2,
        )
    )
    index.drop_cache()
    print(
        f"\n  wrote {out}/per_gene_r2.csv, quartiles_*.csv, summary.json, "
        f"gsea_*.csv and setmean_*.csv with their provenance"
    )


def _gsea_rankings(cfg, df: pd.DataFrame) -> dict[tuple[str, bool], pd.Series]:
    """Every ``(contrast, detrended)`` ranking this run's columns support.

    A contrast naming a column the run did not produce is dropped rather than
    refused — with ``include_pca_control`` off there is no capacity control to
    difference against, which is a choice the config is allowed to make.
    """
    b = cfg.biosignal
    unknown = [c for c in b.gsea_contrasts if c not in GSEA_CONTRASTS]
    if unknown:
        refuse(
            f"GSEA on contrast(s) {unknown}",
            "no such difference of per-gene R^2 is defined",
            hint=f"biosignal.gsea_contrasts names any of {sorted(GSEA_CONTRASTS)}",
        )

    rankings: dict[tuple[str, bool], pd.Series] = {}
    for name in b.gsea_contrasts:
        hi, lo = GSEA_CONTRASTS[name]
        if hi not in df.columns or lo not in df.columns:
            continue
        sub = df.dropna(subset=[hi, lo, "r2_frozen"])
        if sub.empty:
            continue
        stat = pd.Series((sub[hi] - sub[lo]).to_numpy(), index=sub["gene"].astype(str))
        rankings[(name, False)] = stat
        if b.gsea_detrend_bins >= 2:
            rankings[(name, True)] = pd.Series(
                ridge_mod.detrend_on_baseline(
                    sub["r2_frozen"].to_numpy(), stat.to_numpy(), b.gsea_detrend_bins
                ),
                index=stat.index,
            )
    return rankings


def _run_one_gsea(cfg, stat: pd.Series, gene_set: str, label: str):
    from . import enrichment

    try:
        res = enrichment.run_gsea(
            stat,
            cfg.biosignal.resources_dir,
            gene_set,
            label=label,
            seed=cfg.folds.seed,
            times=cfg.biosignal.gsea_permutations,
            min_n=cfg.biosignal.gsea_min_set_size,
        )
    except ImportError as e:
        refuse(
            f"GSEA against '{gene_set}'",
            f"decoupler's GSEA kernel is not importable ({e})",
            hint="install the pinned version (`pip install decoupler==2.1.6`); the "
            "unadjusted p-values and set sizes come from that kernel rather than "
            "from the public wrapper. Or set biosignal.gene_sets=[] to record the "
            "enrichment as deliberately omitted from this run",
        )
    except Exception as e:
        refuse(
            f"GSEA against '{gene_set}'",
            f"{type(e).__name__}: {e}",
            hint=f"gene sets are read from {cfg.biosignal.resources_dir}",
        )
    if res.table.empty:
        refuse(
            f"GSEA against '{gene_set}'",
            "decoupler returned no enrichment rows",
            hint="the ranking's gene identifiers must match the gene set's "
            "— Ensembl ids against HGNC symbols silently overlap in "
            "nothing",
        )
    return res


def _signed_significant(table: pd.DataFrame, fdr: float) -> dict[str, float]:
    """``pathway -> NES`` for the sets this ranking calls significant."""
    if "norm" not in table.columns:
        return {}
    sig = table[table["padj"] < fdr] if "padj" in table.columns else table
    return dict(zip(sig["source"].astype(str), sig["norm"]))


def _report_gsea(cfg, gene_set: str, table: pd.DataFrame, coverage: float) -> None:
    """Print the enrichment as the comparison it is, not as one list of pathways.

    The number worth reading is not how many sets are significant for the published
    ranking but how many of them stay significant once the width change and the
    shrinkage trend are taken out. Printing the raw list alone is what let a
    dimensionality artefact be reported as a biological programme.
    """
    fdr = cfg.biosignal.fdr
    n_sets = table.groupby(["contrast", "detrended"]).size().max()
    print(f"    GSEA {gene_set}: coverage {coverage:.2f}, {n_sets} sets scored")

    by_key = {}
    for (contrast, detrended), t in table.groupby(["contrast", "detrended"], sort=False):
        by_key[(contrast, bool(detrended))] = _signed_significant(t, fdr)
        name = f"{contrast}{' (detrended)' if detrended else ''}"
        sig = by_key[(contrast, bool(detrended))]
        neg = sum(1 for v in sig.values() if v < 0)
        print(f"      {name:<34s} {len(sig):>3d} at FDR<{fdr}  ({neg} down / {len(sig) - neg} up)")

    # How much of the published enrichment a run with no morphology in it reproduces.
    published, control = (
        by_key.get(("guided_vs_frozen", False)),
        by_key.get(("capacity_vs_frozen", False)),
    )
    if published and control is not None:
        shared = [p for p, v in published.items() if np.sign(control.get(p, 0.0)) == np.sign(v)]
        print(
            f"      {len(shared)}/{len(published)} of guided_vs_frozen's sets are "
            f"significant with the same sign for capacity_vs_frozen — that much of "
            f"the enrichment is width, not morphology"
        )

    headline = next(
        (
            k
            for k in (
                ("guided_vs_capacity", False),
                ("guided_vs_frozen", False),
                ("guided_vs_capacity", True),
                ("guided_vs_frozen", True),
            )
            if k in by_key
        ),
        None,
    )
    if headline is None:
        return
    sig = by_key[headline]
    label = f"{headline[0]}{' (detrended)' if headline[1] else ''}"
    if not sig:
        print(f"      nothing survives at FDR<{fdr} for {label}")
        return
    print(f"      top {label}:")
    for p, v in sorted(sig.items(), key=lambda kv: -abs(kv[1]))[:6]:
        print(f"        {p:<38s} NES={v:+.2f}")


def _write_gsea_meta(cfg, out, gene_set: str, level: str, metas: list[dict]) -> None:
    """The method, beside the numbers it produced.

    Which collection and which copy of it, the background, the minimum set size, the
    permutation count behind every p-value and what the adjustment ran over — so a
    caption can be written from an artefact rather than from memory.
    """
    from . import enrichment

    payload = {
        "level": level,
        "scope": ridge_mod.scope_name(cfg.biosignal.tissues),
        "substrate": cfg.data.substrate,
        "fdr": cfg.biosignal.fdr,
        "collection": enrichment.collection_provenance(cfg.biosignal.resources_dir, gene_set),
        "versions": enrichment.scoring_versions(),
        "rankings": metas,
    }
    (out / f"gsea_{gene_set}_{level}_meta.json").write_text(json.dumps(payload, indent=2) + "\n")


def _gsea(cfg, out, level: str, rankings: dict) -> None:
    for gene_set in cfg.biosignal.gene_sets:
        frames, metas, coverage = [], [], float("nan")
        for (contrast, detrended), stat in rankings.items():
            label = f"{level}_{contrast}{'_detrended' if detrended else ''}"
            res = _run_one_gsea(cfg, stat, gene_set, label)
            coverage = res.coverage
            t = res.table.copy()
            t.insert(0, "contrast", contrast)
            t.insert(1, "detrended", detrended)
            frames.append(t)
            metas.append({"contrast": contrast, "detrended": bool(detrended), **res.meta()})
        table = pd.concat(frames, ignore_index=True)
        table.to_csv(out / f"gsea_{gene_set}_{level}.csv", index=False)
        _write_gsea_meta(cfg, out, gene_set, level, metas)
        _report_gsea(cfg, gene_set, table, coverage)


def _run_one_set_mean(cfg, stat: pd.Series, baseline: pd.Series, gene_set: str, label: str):
    from . import enrichment

    b = cfg.biosignal
    try:
        return enrichment.run_set_mean(
            stat,
            b.resources_dir,
            gene_set,
            label=label,
            seed=cfg.folds.seed,
            times=b.gsea_permutations,
            n_boot=b.setmean_bootstrap,
            min_n=b.gsea_min_set_size,
            ci=b.setmean_ci,
            baseline=baseline if b.setmean_baseline_bins >= 2 else None,
            baseline_bins=b.setmean_baseline_bins,
        )
    except Exception as e:
        refuse(
            f"the per-set mean test against '{gene_set}'",
            f"{type(e).__name__}: {e}",
            hint=f"gene sets are read from {b.resources_dir}",
        )


def _report_set_mean(cfg, gene_set: str, table: pd.DataFrame) -> None:
    """Print both tests side by side, because only the matched one is news.

    A set clears the against-background column whenever it is made of well-predicted
    genes; the matched column is the only count that is about the set's biology.
    """
    fdr = cfg.biosignal.fdr
    for contrast, t in table.groupby("contrast", sort=False):
        bg = float(t["background_mean"].iloc[0])
        counts = {k: int((t[f"padj_vs_{k}"] < fdr).sum()) for k in ("background", "matched")}
        print(
            f"      {contrast:<22s} {len(t)} sets, background mean {bg:+.4f}, "
            f"{int((t['mean_delta'] < bg).sum())} below it"
        )
        print(
            f"        FDR<{fdr}: {counts['background']} vs background, "
            f"{counts['matched']} vs a baseline-matched null"
        )


def _set_mean(cfg, out, level: str, rankings: dict, baseline: pd.Series) -> None:
    """Per-set mean delta-R^2 with its permutation tests, beside the GSEA.

    Same rankings, different question: effect size rather than position. The detrended
    rankings are deliberately not scored here — the baseline-matched null applies that
    same correction to the null instead of to the statistic, and doing both would
    subtract the trend twice.
    """
    from . import enrichment

    b = cfg.biosignal
    rankings = {c: stat for (c, detrended), stat in rankings.items() if not detrended}
    if not rankings:
        return

    for gene_set in b.gene_sets:
        frames, metas = [], []
        for contrast, stat in rankings.items():
            label = f"{level}_{contrast}"
            res = _run_one_set_mean(cfg, stat, baseline, gene_set, label)
            t = res.table.copy()
            t.insert(0, "contrast", contrast)
            frames.append(t)
            metas.append({"contrast": contrast, **res.meta()})
        table = pd.concat(frames, ignore_index=True)
        table.to_csv(out / f"setmean_{gene_set}_{level}.csv", index=False)
        (out / f"setmean_{gene_set}_{level}_meta.json").write_text(
            json.dumps(
                {
                    "level": level,
                    "scope": ridge_mod.scope_name(b.tissues),
                    "substrate": cfg.data.substrate,
                    "fdr": b.fdr,
                    "collection": enrichment.collection_provenance(b.resources_dir, gene_set),
                    "versions": enrichment.scoring_versions(),
                    "rankings": metas,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"    set-mean {gene_set}:")
        _report_set_mean(cfg, gene_set, table)


def enrichment_tail(cfg, out, level: str, df: pd.DataFrame) -> None:
    """Everything downstream of one level's per-gene R^2.

    A pure function of the scored columns: no embedding, no ridge, no raw counts. The
    rankings are built once and handed to both readings, so the two tables in a scope
    directory cannot be scored on different numbers.
    """
    b = cfg.biosignal
    rankings = _gsea_rankings(cfg, df)
    if not rankings:
        refuse(
            f"the enrichment for the '{level}' level",
            "none of the configured contrasts could be formed from the scored columns",
            hint="biosignal.gsea_contrasts other than guided_vs_frozen need include_pca_control on",
        )
    skipped = set(b.gsea_contrasts) - {c for c, _ in rankings}
    if skipped:
        print(
            f"    enrichment: {', '.join(sorted(skipped))} not scored (no capacity "
            f"control in this run) — it cannot separate the width change from "
            f"information"
        )

    # The frozen R^2 the matched null bins on. Not a ranking: it is the *baseline* each
    # contrast's change is measured from, and the same one for all of them.
    baseline = pd.Series(df["r2_frozen"].to_numpy(), index=df["gene"].astype(str)).dropna()
    _gsea(cfg, out, level, rankings)
    _set_mean(cfg, out, level, rankings, baseline)
