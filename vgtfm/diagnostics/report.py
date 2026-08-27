"""``diagnose`` stage — properties of the representations themselves.

Four diagnostics, run on the frozen features (and, where they exist, on the learned
embeddings), each answering a different question about the same data:

``effective_rank``  How many directions does the embedding actually use? With
                    controls, so the number means something.
``variance``        How much of the variance is just which slide a spot came from?
``cca``             How much of the cross-modal coupling survives a within-slide
                    permutation, and how much survives a held-out patient?
``scib``            The standard integration panel, batch and bio reported apart.

None of these establishes causation on its own; the interventional evidence is the
patch-shuffle ablation in the ``ablate`` stage.
"""

from __future__ import annotations

import json

import numpy as np

from ..data import tables
from ..degraded import pending, refuse
from . import model_seeds
from . import cca as cca_mod
from . import effective_rank as er_mod
from . import variance as var_mod


def _learned_embeddings(cfg, n_spots: int) -> dict[str, np.ndarray]:
    """Every fit ``train`` produced, at every seed :func:`model_seeds` names.

    All of them, not one: ``eval`` scores three replicates of each model and reports
    their mean, so a panel scored on a single fit is not the same experiment and its
    row cannot be read beside Table 1's. ``integrate`` scores the same three, and
    :mod:`vgtfm.results` checks its uncorrected row against this panel seed by seed.

    A row-count mismatch is never tolerated: it means the embedding on disk was
    written against a different cohort than the one loaded, so scoring it would pair
    each spot with another spot's vector and every scIB number downstream would be
    wrong while looking plausible.
    """
    from ..models.train import embedding_path, load_embedding

    out: dict[str, np.ndarray] = {}
    for seed in model_seeds(cfg):
        for name in cfg.models.names:
            p = embedding_path(cfg, name, seed)
            if not p.exists():
                pending(f"scIB panel for '{name}' (seed {seed})", f"{p} does not exist")
                continue
            Z = load_embedding(cfg, name, seed)
            if len(Z) != n_spots:
                refuse(
                    f"the scIB panel for '{name}' (seed {seed})",
                    f"{p} has {len(Z)} rows, the loaded cohort has {n_spots}",
                    hint="the data filters changed since training — rerun `train`",
                )
            out[f"{name} (seed {seed})"] = Z
    return out


def _scib_panel(cfg, out, table) -> None:
    """The scIB panel, or a refusal that names the field which omits it on purpose.

    ``scib-metrics`` is the dependency a cluster environment is most likely to be
    missing. Dropping the panel on an ImportError would leave a run that looks
    complete and is missing a figure, so omitting it has to be stated in the config.
    """
    if not cfg.diagnostics.run_scib:
        print("    omitted: diagnostics.run_scib=false")
        return
    try:
        from . import scib as scib_mod

        learned = _learned_embeddings(cfg, table.n)
        panel = scib_mod.run(cfg, table, learned)
    except ImportError as e:
        refuse(
            "the scIB integration panel",
            f"scib-metrics is not importable ({e})",
            hint="install it (`pip install scib-metrics==0.5.9`), or set "
            "diagnostics.run_scib=false to record the panel as "
            "deliberately omitted from this run",
        )
    if panel.empty:
        refuse("the scIB integration panel", "the benchmark scored no representation")
    panel.to_csv(out / "scib_panel.csv", index=False)


def run(cfg) -> None:
    out = cfg.sub("diagnostics")
    table = tables.load(cfg)

    print("\n  [1/4] effective rank")
    er = er_mod.run(cfg, table)
    er.to_csv(out / "effective_rank.csv", index=False)

    print("\n  [2/4] variance decomposition")
    var = var_mod.run(cfg, table)
    var.to_csv(out / "variance_decomposition.csv", index=False)

    print("\n  [3/4] cross-modal ceiling (CCA)")
    ceiling = cca_mod.run(cfg, table)
    (out / "cca_ceiling.json").write_text(json.dumps(ceiling, indent=2, default=str))

    print("\n  [4/4] scIB integration panel")
    _scib_panel(cfg, out, table)

    _write_summary(cfg, out, er, var, ceiling)
    _plot(cfg, out, er, var, ceiling)
    print(f"\n  wrote {out}/")


def _write_summary(cfg, out, er, var, ceiling) -> None:
    """A short, quotable digest of the diagnostics."""
    obs = er[er.variant == "observed"]
    summary = {
        "substrate": cfg.data.substrate,
        "split": cfg.diagnostics.split,
        "effective_rank": {
            r.column: {
                "dim": int(r.dim),
                "eff_rank": round(float(r.eff_rank), 2),
                "pct_of_dim": round(float(r.pct_of_dim), 2),
                "mean_pairwise_cosine": round(float(r.mean_pairwise_cosine), 4),
            }
            for r in obs.itertuples()
        },
        "between_slide_variance_fraction": {
            r.column: round(float(r.between_fraction), 4)
            for r in var[var.grouping == "sample_id"].itertuples()
        },
        # Keyed `patient`, and derived from the registry's donor_pattern rather than
        # the fold key, which cannot see a multi-slide patient on this split.
        "between_patient_variance_fraction": {
            r.column: round(float(r.between_fraction), 4)
            for r in var[var.grouping == "patient"].itertuples()
        },
        "n_groups": {
            r.grouping: int(r.n_groups) for r in var.drop_duplicates("grouping").itertuples()
        },
        "cca": {
            "rho1_in_sample": round(ceiling["spectrum"]["rho1_observed"], 4),
            "rho1_within_slide_null_q95": round(ceiling["spectrum"]["rho1_null_within_q95"], 4),
            "top_k_variance_beyond_slide_identity": round(
                ceiling["spectrum"]["spot_level_fraction_mean_null"], 4
            ),
            "rho1_held_out_slide": round(ceiling["loso"].get("rho1_test_mean", float("nan")), 4),
            "rho1_held_out_paired_donor": round(
                ceiling["loso"].get("rho1_test_paired_mean", float("nan")), 4
            ),
            "rho1_held_out_unpaired_donor": round(
                ceiling["loso"].get("rho1_test_unpaired_mean", float("nan")), 4
            ),
            "n_held_out_paired": ceiling["loso"].get("n_paired"),
            "n_held_out_unpaired": ceiling["loso"].get("n_unpaired"),
            # Present only when the gap is NaN, saying which of the two reasons it
            # is: no paired slide, or no unpaired one.
            **(
                {"patient_leakage_gap_note": ceiling["loso"]["patient_leakage_gap_note"]}
                if ceiling["loso"].get("patient_leakage_gap_note")
                else {}
            ),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n  summary:")
    print(json.dumps(summary, indent=2))


def _plot(cfg, out, er, var, ceiling) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..plotting import save, set_pub_style

    set_pub_style()

    # -- effective rank, observed vs controls -----------------------
    fig, ax = plt.subplots(figsize=(7, 3.6))
    columns = list(dict.fromkeys(er.column))
    variants = list(dict.fromkeys(er.variant))
    width = 0.8 / max(len(variants), 1)
    for i, variant in enumerate(variants):
        sub = er[er.variant == variant].set_index("column").reindex(columns)
        x = np.arange(len(columns)) + i * width
        ax.bar(x, sub["pct_of_dim"].to_numpy(), width=width, label=variant)
    ax.set_xticks(np.arange(len(columns)) + 0.4 - width / 2)
    ax.set_xticklabels([c.replace("_features", "") for c in columns])
    ax.set_ylabel("effective rank (% of dimension)")
    ax.set_title(f"Spectral collapse — {cfg.data.substrate}")
    ax.legend(frameon=False, fontsize=8)
    save(fig, out / "effective_rank")

    # -- between-slide variance -------------------------------------
    v = var[var.grouping.isin(["sample_id", "patient", "tissue"])]
    if not v.empty:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        groupings = list(dict.fromkeys(v.grouping))
        width = 0.8 / max(len(groupings), 1)
        for i, g in enumerate(groupings):
            sub = v[v.grouping == g].set_index("column").reindex(columns)
            x = np.arange(len(columns)) + i * width
            ax.bar(x, sub["between_fraction"].to_numpy(), width=width, label=g)
        ax.set_xticks(np.arange(len(columns)) + 0.4 - width / 2)
        ax.set_xticklabels([c.replace("_features", "") for c in columns])
        ax.set_ylabel("between-group variance fraction")
        ax.set_ylim(0, 1)
        ax.set_title("Share of embedding variance explained by group identity")
        ax.legend(frameon=False, fontsize=8)
        save(fig, out / "variance_decomposition")

    # -- CCA spectrum against its nulls -----------------------------
    spec = ceiling["spectrum"]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    k = np.arange(1, spec["n_components_reported"] + 1)
    ax.plot(k, spec["rho_observed"], "o-", label="observed", color="#1f77b4")
    ax.plot(k, spec["rho_null_within_q95"], "s--", label="within-slide null (q95)", color="#d62728")
    ax.plot(k, spec["rho_null_global_q95"], "^:", label="global null (q95)", color="#7f7f7f")
    ax.set_xlabel("canonical component")
    ax.set_ylabel(r"canonical correlation $\rho$")
    ax.set_ylim(0, 1.02)
    ax.set_title("Cross-modal coupling is mostly slide identity")
    ax.legend(frameon=False, fontsize=8)
    save(fig, out / "cca_spectrum")
