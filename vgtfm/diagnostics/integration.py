"""``integrate`` stage — do standard batch-correction methods rescue the embedding?

Each corrected embedding is scored on both axes at once:

* **batch mixing**, via the scIB batch panel (iLISI in particular);
* **downstream usefulness**, via the same held-out-donor annotation probe the rest
  of the pipeline uses, scored at every tissue scope ``eval`` reports and carrying
  the same donor-level bootstrap interval as Table 1.

Reporting only the first is how over-correction gets mistaken for success: an
embedding that discards everything mixes batches perfectly. The pairing is the
point.

The probe column is reported twice: as an absolute macro-F1 with its interval, and
as a **paired** delta against the uncorrected embedding on a shared donor resample
(``integration_deltas.csv``). At this cohort size the absolute intervals are far
wider than the differences between the methods, so only the paired form can tell a
cost apart from a null result. Read the delta; the absolute number is there to say
whether the task is supported at all.

Every method that corrects an embedding corrects the *same* one: the ``pca``
baseline that ``train`` fitted and ``eval`` scores, loaded rather than recomputed.
That is what makes the ``none`` row reproduce Table 1's PCA row exactly, and what
makes ``harmony``, ``combat`` and ``bbknn``'s deltas a property of the correction
rather than of the basis under it. See :func:`_reference_embedding` for why
recomputing a PCA here is not the same matrix.

And *every* fit of it, once per seed in :func:`~vgtfm.diagnostics.model_seeds` --
the same replicate seeds ``train``, ``eval`` and ``ablate`` use. A stage that scored
one fit while the annotation tables averaged three would not be the same experiment:
its uncorrected row would sit a few thousandths from Table 1's PCA row for no reason
a reader could see, and its deltas would be single draws printed beside pooled ones.
``diagnostics.seed`` still seeds the subsamples *inside* a diagnostic -- which spots
the scIB panel scores, which correction a stochastic method converges to -- and stays
fixed across fits, so the spread across seeds is the spread across fits and not
across subsamples.

``scvi`` is the exception and deliberately so: it is fitted on raw counts, not on
this embedding, which is the whole reason it is in the table -- it is the only method
here that defines a function a new slide could be pushed through. Its delta is
therefore "scVI's latent against the PCA baseline", not "the PCA baseline corrected",
and a caption comparing it with the other three should say so.

Methods:

``harmony``    iterative linear correction of the embedding (Korsunsky et al.)
``combat``     linear location/scale adjustment per batch
``bbknn``      corrects the neighbour graph rather than the embedding. Scored on the
               metrics that read only a graph — iLISI, kBET, graph connectivity,
               cLISI — and on nothing else, because there is no corrected matrix to
               feed a kNN probe or the metrics that need coordinates. The gap in its
               row is the finding: whatever its batch mixing, it cannot yield a
               deployable gene representation
``scvi``       a batch-conditioned generative model fit on raw counts. Unlike the
               others it is inductive — it defines a function of counts that can be
               applied to a new slide, so it is the only one that could be deployed
               rather than merely reported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import folds as folds_mod
from ..data import tables
from ..degraded import refuse
from ..evaluate import protocol as proto
from ..evaluate import reporting as rep
from ..labels import cohort_classes
from . import model_seeds


def _as_cells_by_features(Z: np.ndarray, n_cells: int, name: str) -> np.ndarray:
    """Return ``Z`` oriented as (cells, features), transposing if needed.

    These libraries disagree on orientation and have changed it between releases,
    and a transposed matrix produces a plausible-looking embedding of the wrong
    thing.
    """
    Z = np.asarray(Z, dtype=np.float32)
    if Z.shape[0] == n_cells:
        return Z
    if Z.shape[1] == n_cells:
        return np.ascontiguousarray(Z.T)
    raise ValueError(f"{name} returned {Z.shape}; neither axis matches {n_cells} cells")


def correct_harmony(X: np.ndarray, batch: np.ndarray, *, seed: int = 42) -> np.ndarray:
    """Harmony's iterative linear correction, returned as a corrected embedding."""
    import harmonypy

    meta = pd.DataFrame({"batch": batch.astype(str)})
    ho = harmonypy.run_harmony(X, meta, ["batch"], max_iter_harmony=20, random_state=seed)
    return _as_cells_by_features(ho.Z_corr, len(X), "harmony")


def correct_combat(X: np.ndarray, batch: np.ndarray) -> np.ndarray:
    """ComBat's per-batch location/scale adjustment, on the PC scores."""
    import anndata as ad
    import scanpy as sc

    a = ad.AnnData(X=np.asarray(X, dtype=np.float32))
    a.obs["batch"] = pd.Categorical(batch.astype(str))
    sc.pp.combat(a, key="batch")
    return _as_cells_by_features(a.X, len(X), "combat")


#: Methods that correct the neighbour graph instead of the embedding. They are scored
#: on the graph half of the scIB panel and skip the kNN probe. See
#: :func:`correct_bbknn`.
GRAPH_ONLY = ("bbknn",)

#: Neighbours BBKNN takes from *each* batch. Its own default. Named here because the
#: caller has to guarantee every batch it passes carries at least this many spots.
BBKNN_NEIGHBORS_WITHIN_BATCH = 3


@dataclass(frozen=True)
class NeighborGraph:
    """A corrected kNN graph: ``(n_spots, k)`` neighbour indices and distances.

    Rows run near to far with the spot itself in column 0, which is the layout
    scib-metrics' graph metrics expect (``NeighborsResults``) and the layout its
    ``Benchmarker`` will not accept from outside — see :func:`..scib.benchmark_graph`.
    """

    indices: np.ndarray
    distances: np.ndarray

    @property
    def k(self) -> int:
        return int(self.indices.shape[1])


def _dense_knn(dist, n_spots: int) -> NeighborGraph:
    """Turn BBKNN's sparse distance matrix into sorted dense neighbour rows.

    BBKNN sorts its neighbours before building the matrix, but sparse round-tripping
    does not promise to preserve that order, and the self edge is written as an
    explicit zero that ``tocsr()`` is entitled to drop. Both are re-established here
    rather than assumed: an unsorted row would silently turn "the 90 nearest
    neighbours" into "90 arbitrary neighbours" for every metric downstream.
    """
    dist = dist.tocsr()
    if dist.shape[0] != n_spots:
        raise ValueError(f"bbknn returned {dist.shape[0]} rows for {n_spots} spots")
    indptr, cols, vals = dist.indptr, dist.indices, dist.data

    rows = []
    for r in range(n_spots):
        lo, hi = indptr[r], indptr[r + 1]
        c, d = cols[lo:hi], vals[lo:hi]
        keep = c != r  # self goes back in at column 0
        c, d = c[keep], d[keep]
        order = np.argsort(d, kind="stable")
        rows.append((np.concatenate(([r], c[order])), np.concatenate(([0.0], d[order]))))

    # Width from the rows themselves, not from `neighbors_within_batch * n_batches`:
    # whether the self edge survived the sparse round trip changes every row's length
    # by one, and guessing wrong pads every row instead of none of them.
    width = max(len(i) for i, _ in rows)
    out_i = np.zeros((n_spots, width), dtype=np.int64)
    out_d = np.zeros((n_spots, width), dtype=np.float64)
    short = 0
    for r, (i, d) in enumerate(rows):
        if len(i) < width:
            short += 1
            pad = width - len(i)
            i = np.concatenate((i, np.full(pad, r, dtype=np.int64)))
            d = np.concatenate((d, np.full(pad, np.inf)))
        out_i[r], out_d[r] = i, d
    if short:
        print(
            f"    note: {short:,} of {n_spots:,} spots returned fewer than {width} "
            f"neighbours and were padded with themselves"
        )
    return NeighborGraph(indices=out_i, distances=out_d)


def correct_bbknn(
    X: np.ndarray,
    batch: np.ndarray,
    *,
    neighbors_within_batch: int = BBKNN_NEIGHBORS_WITHIN_BATCH,
) -> NeighborGraph:
    """BBKNN corrects the neighbour *graph*, not the embedding.

    The absence of a corrected matrix is itself the result — a method with no
    corrected matrix cannot produce a deployable gene representation, however well it
    mixes batches. But "it produces no embedding" and "it does not integrate" are
    different claims, and only the graph can settle the second one, so the graph
    comes back to be scored rather than being thrown away.
    """
    import anndata as ad
    import bbknn

    Xf = np.asarray(X, dtype=np.float32)
    a = ad.AnnData(X=Xf)
    a.obs["batch"] = pd.Categorical(batch.astype(str))
    a.obsm["X_pca"] = a.X
    bbknn.bbknn(
        a,
        batch_key="batch",
        use_rep="X_pca",
        neighbors_within_batch=neighbors_within_batch,
        # bbknn defaults to `n_pcs=50` and truncates silently. Every other method
        # here corrects all `models.pca_components` PCs, so the default would score
        # BBKNN-on-50-PCs against everything else on 128 and read the difference as
        # a property of the method.
        n_pcs=Xf.shape[1],
        # Not bbknn's default backend. `computation="annoy"` killed this stage with
        # SIGILL on every node of two batches; the core dump puts the faulting
        # instruction in annoylib.so, called from CPython rather than from the loader,
        # so it is annoy's index build and not its import. annoy publishes an sdist
        # only and its setup.py compiles with `-march=native`, so the extension is
        # built for whichever machine ran `pip install` — a login node here — and
        # faults on the compute nodes. cKDTree is exact, deterministic and pure scipy;
        # pynndescent would also work but drags in numba and refuses batches with
        # fewer than 11 cells.
        computation="cKDTree",
    )
    return _dense_knn(a.obsp["distances"], len(Xf))


class NativeCrash(RuntimeError):
    """A correction died in a way Python cannot raise — a signal, not an exception."""


#: Seconds to wait for a SIGKILLed child to actually be reaped before giving up on
#: it. Only an uninterruptible syscall can take this long, and no wait length fixes
#: that — the point is to stay bounded, not to succeed.
_REAP_GRACE_S = 60.0


def _isolated(fn, *, name: str, timeout_s: float = 0.0):
    """Run *fn* in a forked child and bring its array back, or report how it died.

    Some corrections fault in compiled code (bbknn has been seen to die with
    ``SIGILL``), taking every method already scored down with them. ``except
    Exception`` cannot see that — the process is simply gone — so the only way to
    survive it is to not be the process it happens in.

    The child writes its result to a temporary ``.npy`` (or a marker file for the
    ``None`` a graph-only method returns); the parent turns a non-zero exit into
    :class:`NativeCrash`. Fork, not spawn, so the child inherits the already
    materialised matrices rather than pickling them — which is also why methods that
    have touched the GPU must not be routed here, and by default are not.

    **A hang is a failure mode too, and the more likely one.** The same fork that
    isolates a fault happens after scib-metrics has started JAX's thread pool and
    after OpenMP has started its own; forking a process with live threads is
    undefined, and the child can just as easily deadlock as fault. Observed: bbknn's
    child never exits, ``join()`` has no deadline, and SLURM kills the stage at its
    wall limit with Harmony already scored and nothing written. *timeout_s* bounds the
    wait and
    turns it into the ordinary recorded row a crash produces; the child is killed so
    it cannot outlive the stage.
    """
    import multiprocessing
    import tempfile

    ctx = multiprocessing.get_context("fork")
    with tempfile.TemporaryDirectory(prefix=f"vgtfm-{name}-") as tmp:
        result = Path(tmp) / "result.npy"
        marker = Path(tmp) / "none"

        def _child():
            Z = fn()
            if Z is None:
                marker.write_text("graph-only")
            else:
                np.save(result, np.asarray(Z, dtype=np.float32))

        proc = ctx.Process(target=_child)
        proc.start()
        proc.join(timeout_s if timeout_s and timeout_s > 0 else None)

        if proc.is_alive():
            proc.kill()  # SIGKILL: a deadlocked child ignores TERM
            # Bounded, because SIGKILL is not instantaneous for a process parked in
            # uninterruptible I/O: a plain join() here would be a second unbounded
            # wait behind the one the timeout just escaped. A child that outlives its
            # own SIGKILL is left to the job's teardown.
            proc.join(_REAP_GRACE_S)
            raise NativeCrash(
                f"timed out after {timeout_s:.0f}s and was killed"
                if not proc.is_alive()
                else f"timed out after {timeout_s:.0f}s and survived SIGKILL for "
                f"{_REAP_GRACE_S:.0f}s (stuck in uninterruptible I/O); abandoned"
            )
        if proc.exitcode != 0:
            how = (
                f"killed by signal {-proc.exitcode}"
                if proc.exitcode < 0
                else f"exited with status {proc.exitcode}"
            )
            raise NativeCrash(how)
        if marker.exists():
            return None
        if not result.exists():
            raise NativeCrash("the child exited cleanly but produced no result")
        return np.load(result)


def _correct_or_crash(fn, *, name: str, isolate: bool, timeout_s: float = 0.0):
    """Run one correction; return ``(Z, crash)`` with exactly one of them set.

    This is the only place a failure becomes a value rather than an immediate
    refusal. :func:`run` refuses on the collected crashes once every method has had
    its turn, so the stage still fails — after the comparison has been carried as
    far as it can go, rather than discarding every method already scored.

    Only :class:`NativeCrash` is caught. An ImportError or an ordinary exception
    propagates to the caller's handlers and refuses on the spot.
    """
    try:
        return (_isolated(fn, name=name, timeout_s=timeout_s) if isolate else fn()), None
    except NativeCrash as e:
        return None, str(e)


def _probe_predictions(cfg, table, Z, fold_map) -> dict[str, dict]:
    """Held-out-donor predictions at each fold level, pooled over that level's folds.

    Pooled through :func:`..evaluate.reporting.pool`, the same helper ``eval`` and
    ``ablate`` use, so the donors and tissues travel with the predictions and a
    donor-level interval can be built from them.
    """
    out = {}
    for level, specs in fold_map.items():
        preds = [
            p for p in (proto.heldout_donor(Z, table.meta, spec, cfg=cfg) for spec in specs) if p
        ]
        pooled = rep.pool(preds)
        if pooled:
            out[level] = pooled
    return out


def _stored_predictions(cfg, fold_map, X: np.ndarray, seed: int) -> dict[str, dict]:
    """``eval``'s own predictions for the ``pca`` baseline, one per fold level.

    The uncorrected row is not recomputed here. It *is* the ``pca`` row of Table 1 --
    same embedding as :func:`_reference_embedding` loads, same probe, same folds --
    and re-running the probe to obtain a number another stage has already written is
    how the two come to disagree without anyone noticing. Reading the cached vector
    makes the row identical by construction rather than by coincidence, and it is
    what lets :mod:`vgtfm.results` check the two stages against each other at all.

    *X* is checked against it rather than assumed. Reusing a cached score while the
    corrections operate on a *different* matrix would be worse than recomputing:
    every row of this table would look consistent and the deltas would compare a
    corrected embedding against a reference that is not its uncorrected form. The
    fingerprint ``eval`` stamped into the file is what rules that out -- one stage
    re-run without the other is otherwise invisible here.

    Which levels to expect comes from ``eval``'s own index rather than from
    *fold_map*. A level whose folds all come up empty is scored by neither stage --
    :func:`_probe_predictions` drops it silently, and a hard failure here on a level
    that simply has no prediction to reuse would refuse a run that is not broken.
    A level ``eval`` *did* score and whose file is gone is still a refusal.
    """
    from ..evaluate.run_eval import REFERENCE, prediction_path
    from ..provenance import array_fingerprint

    protocol = "heldout_donor"
    index_path = cfg.sub("eval", "predictions") / "index.json"
    if not index_path.exists():
        refuse(
            "the uncorrected reference",
            f"{index_path} does not exist",
            hint=f"run `python run.py eval` first — `integrate` reports the "
            f"'{REFERENCE}' baseline's own predictions rather than "
            f"recomputing them",
        )
    scored = {
        e["level"]
        for e in json.loads(index_path.read_text())
        if e.get("model") == REFERENCE and e.get("protocol") == protocol and e.get("seed") == seed
    }
    if not scored & set(fold_map):
        refuse(
            "the uncorrected reference",
            f"`eval` scored no '{REFERENCE}' prediction under '{protocol}' at "
            f"seed {seed} for any of {sorted(fold_map)}",
            hint=f"it recorded {sorted(scored) or 'nothing'} — check that "
            f"'{protocol}' is in eval.protocols and that seed {seed} "
            f"is one `eval` ran with ({list(cfg.seeds)})",
        )

    want = array_fingerprint(X)
    out: dict[str, dict] = {}
    for level in fold_map:
        if level not in scored:
            print(
                f"    {level}: `eval` scored no '{REFERENCE}' prediction there — "
                f"skipped, as the probe below skips it too"
            )
            continue
        path = prediction_path(cfg, REFERENCE, seed, protocol, level)
        if not path.exists():
            refuse(
                f"the uncorrected reference at {level}",
                f"{index_path.name} lists it but {path} does not exist",
                hint="the prediction cache is incomplete — rerun `python run.py eval`",
            )
        with np.load(path, allow_pickle=False) as z:
            refit = "refit_per_fold" in z and bool(z["refit_per_fold"])
            got = str(z["embedding_fingerprint"]) if "embedding_fingerprint" in z else ""
            if refit:
                refuse(
                    f"the uncorrected reference at {level}",
                    "`eval` ran with train.refit_per_fold, so its predictions "
                    "come from a representation refitted inside each fold, not "
                    "from the frozen embedding these corrections are applied to",
                    hint="rerun `eval` with train.refit_per_fold=false, or drop "
                    "the integrate stage for this run",
                )
            if not got:
                refuse(
                    f"the uncorrected reference at {level}",
                    f"{path} carries no embedding fingerprint",
                    hint="it predates the check; rerun `python run.py eval` so "
                    "the predictions record which matrix produced them",
                )
            if got != want:
                refuse(
                    f"the uncorrected reference at {level}",
                    f"{path} was produced from embedding {got}, but the "
                    f"embedding loaded here is {want}",
                    hint="`train` and `eval` are out of step — rerun `eval` so "
                    "its predictions describe the current embedding",
                )
            out[level] = {k: z[k] for k in ("y_true", "y_pred", "donor", "sample_id", "tissue")}
    print(
        f"    reusing `eval`'s {REFERENCE} predictions at {', '.join(out)} "
        f"(embedding {want}) — this row is Table 1's, not a recomputation"
    )
    return out


def _probe_scores(
    cfg, preds: dict, classes, *, method: str, seed: int, reference: dict | None = None
) -> tuple[dict, list[dict], dict, dict]:
    """Macro-F1 per (fold level, tissue scope), with a donor interval and a paired delta.

    Both statistics come from :mod:`..evaluate.reporting`, which is what ``eval`` and
    ``ablate`` score through. Scoring this stage's probe by hand is what once made its
    F1 column incomparable with Table 1 -- a vocabulary taken from ``y_true`` alone
    spans no column for a class the probe predicted but the held-out slides do not
    carry, so those rows were dropped instead of counted as errors. Routing through
    the shared helpers means that question is settled in one place.

    Every scope is scored, not just the pooled one. The pooled ``global`` macro runs
    over ``(tissue, class)`` cells from all three organs while its bootstrap resamples
    all donors regardless of organ, so a resample that happens to draw no kidney donor
    still carries kidney's cells in the denominator and scores them zero. That is a
    fine cohort-level summary and a poor per-organ one, and the paper's other tables
    report organs separately for the same reason.

    Both statistics per cell -- absolute interval and paired delta -- for the reason
    the module docstring gives.
    """
    cols: dict = {}
    deltas: list[dict] = []
    # Replicate draws per (method, level, scope), so `_write_outputs` can publish the
    # percentile of the seeds' union rather than one seed's bracket. Same convention
    # as `eval` and `ablate`; see `..evaluate.bootstrap.pool_replicates`.
    draws: dict[tuple, list] = {}
    abs_draws: dict[tuple, list] = {}
    for level, pred in preds.items():
        vocab = rep.score_vocabulary(pred)
        base = {"level": level}
        brows, bdraws = rep.bootstrap_rows(pred, classes, base, cfg, vocab, return_draws=True)
        for row in brows:
            if row["class"] != "macro":
                continue
            abs_draws[(method, level, row["scope"])] = bdraws.get((row["scope"], "macro"))
            key = f"f1/{row['scope']}/{level}"
            cols[key] = row["value"]
            cols[f"{key}_lo"] = row["ci_lo"]
            cols[f"{key}_hi"] = row["ci_hi"]
            # n_donors travels with every interval in this repository: at 2 donors
            # there are 3 distinct resamples and at 7 there are 1,716, and the
            # bracket alone does not say which of those produced it.
            cols[f"{key}_n_donors"] = row["n_donors"]

        ref = (reference or {}).get(level)
        if ref is None:
            continue
        # A paired delta is only defined if both vectors describe the same rows. They
        # do -- the evaluated spots follow from the fold and the annotation mask, never
        # from the embedding -- so a mismatch here is a bug in the pooling, not a
        # condition to work around.
        if not np.array_equal(pred["y_true"], ref["y_true"]):
            refuse(
                f"the paired delta for '{method}' at {level}",
                "its predictions and the uncorrected reference's describe different rows",
            )
        # (corrected, uncorrected), so a positive delta means the correction helped.
        rows, ddraws = rep.paired_delta_rows(
            pred, pred["y_pred"], ref["y_pred"], base, classes, cfg, vocab, return_draws=True
        )
        for row in rows:
            if row["class"] != "macro":
                continue
            deltas.append(
                {
                    "method": method,
                    "seed": seed,
                    "level": level,
                    "scope": row["scope"],
                    "delta_f1": row["delta"],
                    "ci_lo": row["ci_lo"],
                    "ci_hi": row["ci_hi"],
                    "p_two_sided": row.get("p_two_sided", np.nan),
                    "n_donors": row["n_donors"],
                    "n_boot": int(cfg.eval.bootstrap_n),
                }
            )
            draws[(method, level, row["scope"])] = ddraws.get((row["scope"], "macro"))
    return cols, deltas, draws, abs_draws


def _print_probe(cols: dict, deltas: list[dict]) -> None:
    """One block per fold level, one line per tissue scope inside it."""
    by_cell = {(d["level"], d["scope"]): d for d in deltas}
    # The point-estimate keys are the ones with an interval hanging off them, rather
    # than whatever does not end in a known suffix -- a scope is free to be named
    # `..._hi` without silently losing its line.
    cells = [k for k in cols if f"{k}_lo" in cols]
    levels: dict[str, list[tuple[str, str]]] = {}
    for key in cells:
        _, scope, level = key.split("/", 2)
        levels.setdefault(level, []).append((scope, key))
    for level, entries in levels.items():
        print(f"      probe {level}:")
        for scope, key in sorted(entries, key=lambda e: e[0] != "global"):
            lo, hi = cols[f"{key}_lo"], cols[f"{key}_hi"]
            line = f"        {scope:<8s} {cols[key]:.4f}"
            if lo == lo and hi == hi:
                line += f" [{lo:.4f}, {hi:.4f}] (n={cols[f'{key}_n_donors']})"
            d = by_cell.get((level, scope))
            if d is not None and d["ci_lo"] == d["ci_lo"]:
                line += (
                    f"   delta={d['delta_f1']:+.4f} "
                    f"[{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}] "
                    f"p={d['p_two_sided']:.3f}"
                )
            elif d is not None:
                line += f"   delta={d['delta_f1']:+.4f}"
            print(line)


def _graph_plan(
    scored: np.ndarray,
    batch: np.ndarray,
    *,
    min_neighbors: int,
) -> tuple[np.ndarray, int]:
    """Spots to fit a graph-only correction on, and the per-batch neighbour count.

    Two constraints meet here. The panel scores iLISI and cLISI at k=90
    (``scib.GRAPH_METRIC_K``), and a graph narrower than that cannot be compared with
    one the Benchmarker built for itself — so BBKNN's width, ``neighbors_within_batch
    * n_batches``, has to clear it. And BBKNN refuses outright if any batch holds
    fewer spots than ``neighbors_within_batch``.

    Its default of 3 is not enough here: of 120 slides only ~24 carry annotated spots,
    so the default yields 72. Raising the per-batch count is the conservative
    direction — more cross-batch neighbours can only help BBKNN's batch mixing, and
    the comparison is being run to see whether that mixing is worth anything.
    Dropping an undersized batch lowers the batch count and raises the requirement
    again, so this settles rather than computes.
    """
    rows = np.asarray(scored)
    for _ in range(8):
        b = np.asarray(batch)[rows]
        uniq, counts = np.unique(b, return_counts=True)
        if not len(uniq):
            break
        per_batch = max(BBKNN_NEIGHBORS_WITHIN_BATCH, int(np.ceil(min_neighbors / len(uniq))))
        small = uniq[counts < per_batch]
        if not len(small):
            print(
                f"  graph-only methods: {len(rows):,} scored spot(s) over "
                f"{len(uniq)} batch(es), {per_batch} neighbour(s) per batch "
                f"-> {per_batch * len(uniq)} neighbours per spot "
                f"(panel scores at k={min_neighbors})"
            )
            return rows, per_batch
        keep = ~np.isin(b, small)
        print(
            f"  graph-only methods: {len(small)} batch(es) hold fewer than "
            f"{per_batch} scored spots and cannot supply a per-batch neighbour set "
            f"— {int((~keep).sum())} spot(s) held out of the graph comparison"
        )
        rows = rows[keep]
    refuse(
        "a graph wide enough to score",
        f"no set of batches among the scored spots can supply {min_neighbors} neighbours per spot",
        hint="raise diagnostics.scib_max_spots so more spots per slide are "
        "scored, or drop the graph-only methods from "
        "diagnostics.integration_methods",
    )


def _reference_embedding(cfg, table, seed: int) -> np.ndarray:
    """The matrix the corrections are applied to: the ``pca`` baseline itself.

    Loaded from ``train``, not recomputed here. A fresh PCA of the gene features
    would be the same width and look interchangeable, and is not the same matrix:
    ``train`` fits the baseline on ``train.fit_split`` alone -- slides that carry no
    annotation and none of the evaluated donors -- while a PCA fitted in this stage
    sees every spot in the cohort, the held-out donors' included. Both are defensible
    bases, but only one of them is the row Table 1 reports, and a ``none`` row that
    does not reproduce Table 1's PCA row reads as a discrepancy in the results rather
    than as the difference of protocol it actually is.

    Reusing it also means every method in this table starts from one matrix, so a
    delta is a property of the correction and not of the basis underneath it.
    """
    from ..models.train import embedding_path, load_embedding

    path = embedding_path(cfg, "pca", seed)
    if not path.exists():
        refuse(
            f"the uncorrected reference embedding at seed {seed}",
            f"{path} does not exist",
            hint=f"run `python run.py train` first, or narrow "
            f"diagnostics.model_seeds to the seeds it was run with "
            f"({list(cfg.seeds)})",
        )
    # Uncast, through the loader `eval` reads with. The fingerprint in
    # :func:`_stored_predictions` hashes the dtype along with the bytes, so an
    # `astype` here would turn "the matrix `eval` scored" into a different matrix and
    # refuse a run that is in fact consistent. See `train.load_embedding`.
    Z = load_embedding(cfg, "pca", seed)
    # The embedding is addressed by row against `table`, so a length mismatch would
    # silently pair every spot with another spot's features.
    if len(Z) != len(table.gene):
        refuse(
            f"the uncorrected reference embedding at seed {seed}",
            f"{path} has {len(Z):,} rows but the cohort has {len(table.gene):,}",
            hint="the data filters changed since training — rerun `train`",
        )
    print(
        f"  uncorrected reference: {path} {Z.shape} — the `pca` baseline fitted "
        f"on split '{cfg.train.fit_split}', the matrix `eval` scores"
    )
    return Z


def _build_methods(cfg, *, X, batch, table, graph_rows, graph_per_batch, seed: int) -> dict:
    """Name -> a thunk returning that method's correction, ``none`` first.

    Insertion order matters downstream: :func:`run` scores in this order and takes
    the uncorrected embedding as the reference every paired delta is measured
    against, so it has to be scored before anything needs it.
    """
    methods = {"none": lambda: X}
    for name in cfg.diagnostics.integration_methods:
        if name == "harmony":
            # The method's own stochasticity follows the run seed, as
            # `ablate` does with the patch permutation: the spread across seeds
            # then covers the correction's variability as well as the basis's.
            methods[name] = lambda: correct_harmony(X, batch, seed=seed)
        elif name == "combat":
            methods[name] = lambda: correct_combat(X, batch)
        elif name == "bbknn":
            methods[name] = lambda: correct_bbknn(
                X[graph_rows], batch[graph_rows], neighbors_within_batch=graph_per_batch
            )
        elif name == "scvi":
            methods[name] = lambda: _scvi_embedding(cfg, table, seed)
        else:
            refuse(
                f"integration method '{name}'",
                "no such method",
                hint="diagnostics.integration_methods accepts harmony, combat, bbknn, scvi",
            )
    return methods


def _reuses_diagnoses_panel(cfg) -> bool:
    """Whether the uncorrected row reads ``diagnose``'s panel or is scored here.

    ``diagnostics.run_scib=false`` is the recorded way to skip that panel. There is
    then nothing to read, and nothing for :mod:`vgtfm.results` to find a second
    value of either, so the row falls back to being scored in this stage as every
    row was before. Refusing would make a supported opt-out unrunnable.
    """
    return bool(cfg.diagnostics.run_scib)


def _stored_panel(cfg, X: np.ndarray, seed: int) -> dict:
    """``diagnose``'s scIB panel for the ``pca`` baseline, as this row's columns.

    The uncorrected row's panel is not recomputed here, for the reason
    :func:`_stored_predictions` does not recompute its probe: ``diagnose`` already
    scores this exact embedding, through the same :func:`scib.benchmark` call at the
    same ``diagnostics.seed``, and :mod:`vgtfm.results` checks the two stages against
    each other. Two stages computing one quantity is how they come to disagree.

    kBET is the metric that actually did. It is the only one of the panel whose path
    reaches ``scipy.sparse.linalg.eigsh`` -- through
    ``scib_metrics.utils.diffusion_nn``, which calls it with no ``v0`` -- so its
    value is not reproducible across processes the way iLISI, cLISI, graph
    connectivity, BRAS, the k-means scores and the silhouettes are; those agreed to
    1e-9 between the two stages while kBET moved by up to 0.023. Reading the panel
    makes the row identical by construction rather than by coincidence, and leaves
    exactly one number in the manuscript per computation.

    *X* is checked against the fingerprint the panel recorded rather than assumed,
    as :func:`_stored_predictions` checks its own. Reusing a panel scored on a
    *different* matrix would be worse than recomputing: the batch columns would
    describe one embedding and the probe columns beside them another, and nothing
    downstream could tell.
    """
    import pandas as pd

    from ..evaluate.run_eval import REFERENCE
    from ..provenance import array_fingerprint

    path = cfg.sub("diagnostics") / "scib_panel.csv"
    if not path.exists():
        refuse(
            f"the uncorrected panel at seed {seed}",
            f"{path} does not exist",
            hint=f"run `python run.py diagnose` first — `integrate` reports the "
            f"'{REFERENCE}' baseline's own panel rather than recomputing it, "
            f"so that one computation reaches the manuscript per number",
        )

    panel = pd.read_csv(path)
    want_row = f"{REFERENCE} (seed {seed})"
    match = panel[panel["representation"].astype(str) == want_row]
    if match.empty:
        refuse(
            f"the uncorrected panel at seed {seed}",
            f"{path} has no '{want_row}' row",
            hint=f"it scored {sorted(panel['representation'].astype(str)) or 'nothing'} "
            f"— check that seed {seed} is one `diagnose` ran with "
            f"({list(cfg.seeds)}) and that '{REFERENCE}' is in models.names",
        )
    row = match.iloc[0]

    # A panel written before the fingerprint column existed cannot be checked, and
    # an unchecked reuse is the failure mode this function exists to prevent.
    if "embedding" not in panel.columns or row["embedding"] != row["embedding"]:
        refuse(
            f"the uncorrected panel at seed {seed}",
            f"{path} records no embedding fingerprint for '{want_row}'",
            hint="it was written by an older `diagnose` — rerun `python run.py "
            "diagnose` so the panel can be tied to the matrix it scored",
        )
    got, expected = str(row["embedding"]), array_fingerprint(X)
    if got != expected:
        refuse(
            f"the uncorrected panel at seed {seed}",
            f"`diagnose` scored '{want_row}' on embedding {got}, but this stage loaded {expected}",
            hint="one stage has been re-run without the other — rerun `python "
            "run.py diagnose` against the current `train` output",
        )

    cols = {
        c: float(row[c])
        for c in panel.columns
        if isinstance(c, str) and c.split("/", 1)[0] in ("batch", "bio") and row[c] == row[c]
    }
    print(
        f"    reusing `diagnose`'s {REFERENCE} panel (embedding {got}) — "
        f"this row's batch columns are that panel's, not a recomputation"
    )
    return cols


def _scib_panel(fn, *, what: str, hint: str = "") -> dict:
    """Run one scIB panel, turning either way it can fail into a refusal.

    A missing scib-metrics and a panel that raised are different problems with
    different fixes, and neither may be swallowed: this stage exists to report batch
    mixing and downstream F1 together, so a row with the second half and not the
    first is not a partial result but a misleading one.
    """
    try:
        return fn()
    except ImportError as e:
        refuse(
            what,
            f"scib-metrics is not importable ({e})",
            hint=("install it (`pip install scib-metrics==0.5.9`)" + (f"; {hint}" if hint else "")),
        )
    except Exception as e:
        refuse(what, f"{type(e).__name__}: {e}")


def _panel_columns(metrics: dict) -> dict:
    """A scIB panel as ``batch/<name>`` and ``bio/<name>`` columns."""
    from . import scib as scib_mod

    batch, bio, _other = scib_mod.split_panels(metrics)
    return {f"batch/{name}": v for name, v in batch.items()} | {
        f"bio/{name}": v for name, v in bio.items()
    }


def _print_panel(cols: dict, *namespaces: str) -> None:
    for ns in namespaces:
        scored = {
            k.split("/", 1)[1]: v
            for k, v in cols.items()
            if k.startswith(f"{ns}/") and v is not None
        }
        if scored:
            print(f"      {ns + ':':<8s}" + "  ".join(f"{k}={v:.3f}" for k, v in scored.items()))


def _attach_pooled_probe(rows: list[dict], draws: dict, cfg) -> None:
    """Seed-pooled brackets on the wide probe columns, in place.

    :func:`..evaluate.reporting.attach_pooled` matches on flat ``level``/``scope``
    fields and these rows fold both into the column name, so the merge is written out
    here. The columns are constant within a method, deliberately: a
    ``groupby("method").mean()`` over the seeds returns them unchanged, which is what
    lets the table layer collapse the seeds with no special case -- the same trick
    ``eval``'s ``ci_lo_pooled`` plays.
    """
    cols = rep.pooled_columns(draws, cfg, with_p=False)
    for row in rows:
        for (method, level, scope), c in cols.items():
            key = f"f1/{scope}/{level}"
            if row.get("method") != method or key not in row:
                continue
            row[f"{key}_lo_pooled"] = c["ci_lo_pooled"]
            row[f"{key}_hi_pooled"] = c["ci_hi_pooled"]
            row[f"{key}_n_seeds"] = c["n_seeds_pooled"]


def _write_outputs(
    cfg,
    out,
    rows: list[dict],
    delta_rows: list[dict],
    delta_draws: dict,
    probe_draws: dict,
    methods: dict,
    seeds: tuple,
) -> None:
    """``integration.csv``, the tidy deltas beside it, and the run summary.

    One row per (method, seed): this stage scores every fit ``eval`` scores, so its
    uncorrected row is that seed's PCA row exactly and the three of them average to
    the number Table 1 prints. The deltas carry the seed-pooled interval beside the
    per-seed one, through the same :func:`..evaluate.reporting.attach_pooled` that
    ``eval`` and ``ablate`` use -- so a batch-correction delta and an annotation
    delta are the same statistic, and the reader is not comparing a single draw
    against a pooled one.
    """
    rep.attach_pooled(
        delta_rows, delta_draws, cfg, with_p=True, group_fields=("method", "level", "scope")
    )
    _attach_pooled_probe(rows, probe_draws, cfg)
    df = pd.DataFrame(rows)
    df.to_csv(out / "integration.csv", index=False)
    # Tidy rather than folded into the wide table, following `ablation/paired_deltas`:
    # a delta carries its own interval, p-value and donor count, and those do not
    # belong in a frame keyed by method alone.
    ddf = pd.DataFrame(delta_rows)
    if not ddf.empty:
        ddf.to_csv(out / "integration_deltas.csv", index=False)
    (out / "summary.json").write_text(
        json.dumps(
            {
                "substrate": cfg.data.substrate,
                "seeds": list(seeds),
                "methods": list(methods),
                "rows": rows,
                "deltas": delta_rows,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\n  wrote {out}/integration.csv{', integration_deltas.csv' if not ddf.empty else ''}")
    if not df.empty:
        print(df.to_string(index=False))
    if not ddf.empty:
        print("\n  paired deltas vs. the uncorrected embedding (same donors resampled for both):")
        print(ddf.to_string(index=False))


def _score_seed(
    cfg,
    *,
    table,
    fold_map,
    classes,
    methods,
    X,
    seed,
    isolate,
    graph_rows,
    scib_mod,
    rows,
    crashed,
    delta_rows,
    delta_draws,
    probe_draws,
) -> None:
    """Every method, on one fit of the ``pca`` baseline. Appends to *rows* in place.

    One call per seed in :func:`~vgtfm.diagnostics.model_seeds`. The scIB panel and
    the graph plan are seeded from ``diagnostics.seed`` throughout, not from *seed*:
    which spots are scored must not move between fits, or the spread across fits
    would carry the spread across subsamples with it.
    """
    print(f"\n  ===== seed {seed} =====")
    # The uncorrected predictions for *this* fit, that every later method's paired
    # delta is measured against — see :func:`_build_methods` for why "none" is
    # guaranteed to come first.
    reference: dict | None = None
    for name, fn in methods.items():
        print(f"\n  -- {name} (seed {seed}) --")
        # This stage's output is a *comparison*, and a table missing the methods
        # that failed is a table of the methods that worked, which reads the same
        # and is not. A failure is therefore still fatal — but only at the end: a
        # method that faults in compiled code is isolated, recorded as a failed row,
        # and the remaining methods still run.
        try:
            Z, crash = _correct_or_crash(
                fn, name=name, isolate=name in isolate, timeout_s=cfg.diagnostics.isolate_timeout_s
            )
        except ImportError as e:
            refuse(
                f"the '{name}' integration",
                f"its package is not importable ({e})",
                hint=f"install it, or drop '{name}' from diagnostics.integration_methods",
            )
        except Exception as e:
            refuse(f"the '{name}' integration at seed {seed}", f"{type(e).__name__}: {e}")

        if crash is not None:
            print(f"    CRASHED: {crash} — recorded; the remaining methods continue")
            rows.append({"method": name, "seed": seed, "note": f"crashed: {crash}"})
            crashed.append(f"{name} at seed {seed} ({crash})")
            continue

        if isinstance(Z, NeighborGraph):
            cols = _panel_columns(
                _scib_panel(
                    lambda: scib_mod.benchmark_graph(
                        Z, table.sample_id[graph_rows], table.annotation[graph_rows]
                    ),
                    what=f"the graph panel for '{name}'",
                )
            )
            _print_panel(cols, "batch", "bio")
            print(
                "    corrects the graph, not the embedding — the probe and the "
                "metrics that need coordinates are skipped, which is the result"
            )
            rows.append(
                {
                    "method": name,
                    "seed": seed,
                    "note": f"graph-only; no embedding to probe; corrected on "
                    f"the {len(graph_rows):,} scored spots",
                    **cols,
                }
            )
            continue

        if Z is None:
            rows.append({"method": name, "seed": seed, "note": "graph-only; no embedding to probe"})
            print("    produces a corrected graph, not an embedding — probe skipped")
            continue

        # The uncorrected row reads `diagnose`'s panel; only the corrections are
        # scored here. `diagnose` never sees a corrected embedding, so those four
        # have no other source and this is the stage that computes them.
        #
        # See :func:`_reuses_diagnoses_panel` for the one config that skips it.
        cols = (
            _stored_panel(cfg, X, seed)
            if name == "none" and _reuses_diagnoses_panel(cfg)
            else _panel_columns(
                _scib_panel(
                    lambda: scib_mod.benchmark(
                        Z,
                        table.sample_id,
                        table.annotation,
                        max_spots=cfg.diagnostics.scib_max_spots,
                        seed=cfg.diagnostics.seed,
                    ),
                    what=f"the batch panel for '{name}'",
                    hint="this stage reports batch mixing and downstream F1 together, so "
                    "it cannot run without the batch half",
                )
            )
        )
        _print_panel(cols, "batch")

        preds = (
            _stored_predictions(cfg, fold_map, X, seed)
            if name == "none"
            else _probe_predictions(cfg, table, Z, fold_map)
        )
        scores, deltas, draws, abs_draws = _probe_scores(
            cfg, preds, classes, method=name, seed=seed, reference=reference
        )
        if name == "none":
            reference = preds
        delta_rows.extend(deltas)
        for store, got in ((delta_draws, draws), (probe_draws, abs_draws)):
            for key, d in got.items():
                if d is not None:
                    store.setdefault(key, []).append(d)
        if scores:
            _print_probe(scores, deltas)
        rows.append({"method": name, "seed": seed, "dim": int(Z.shape[1]), **cols, **scores})


def run(cfg) -> None:
    # Before the table load and the PCA, not after: this is a config error, and it
    # should cost a second rather than an hour.
    isolate = set(cfg.diagnostics.isolate_methods)
    if isolate & set(GRAPH_ONLY):
        refuse(
            "an isolated graph-only correction",
            f"{sorted(isolate & set(GRAPH_ONLY))} return a neighbour graph, and "
            f"`_isolated` can only carry an array or None back from its child",
            hint="drop them from diagnostics.isolate_methods",
        )

    from . import scib as scib_mod

    out = cfg.sub("integration")
    table = tables.load(cfg)
    fold_map = folds_mod.generate_for(cfg, table)

    # The whole cohort's annotation vocabulary, exactly as `eval` derives it. The
    # confusion matrix spans this rather than the classes a level's held-out slides
    # happen to carry, so a prediction outside the averaged subset still counts as a
    # false negative for its true class. See :func:`_probe_scores`.
    classes = cohort_classes(table)

    batch = table.sample_id.astype(str)

    # A graph-only method is fitted on exactly the spots the scIB panel will score,
    # because a neighbour graph cannot be subsampled after the fact. Every other
    # method is corrected on all spots and subsampled afterwards, so BBKNN's
    # correction sees fewer spots than Harmony's — an asymmetry that is unavoidable
    # if both are to be scored on the same rows, and that the row's note records.
    #
    # Drawn once, from `diagnostics.seed`, outside the seed loop: the spots depend on
    # the annotations and not on any fit, and redrawing them per seed would confound
    # the spread across fits with the spread across subsamples.
    scored = scib_mod.scoring_rows(
        table.sample_id,
        table.annotation,
        max_spots=cfg.diagnostics.scib_max_spots,
        seed=cfg.diagnostics.seed,
    )
    graph_rows, graph_per_batch = _graph_plan(
        scored, batch, min_neighbors=max(scib_mod.GRAPH_METRIC_K.values())
    )

    seeds = model_seeds(cfg)
    print(
        f"  scoring {len(seeds)} fit(s): seeds {list(seeds)} — the same replicates "
        f"`eval` scores, so a row here averages to Table 1's"
    )

    rows: list[dict] = []
    crashed: list[str] = []
    delta_rows: list[dict] = []
    delta_draws: dict[tuple, list] = {}
    probe_draws: dict[tuple, list] = {}
    methods: dict = {}
    for seed in seeds:
        X = _reference_embedding(cfg, table, seed)
        methods = _build_methods(
            cfg,
            X=X,
            batch=batch,
            table=table,
            graph_rows=graph_rows,
            graph_per_batch=graph_per_batch,
            seed=seed,
        )
        _score_seed(
            cfg,
            table=table,
            fold_map=fold_map,
            classes=classes,
            methods=methods,
            X=X,
            seed=seed,
            isolate=isolate,
            graph_rows=graph_rows,
            scib_mod=scib_mod,
            rows=rows,
            crashed=crashed,
            delta_rows=delta_rows,
            delta_draws=delta_draws,
            probe_draws=probe_draws,
        )

    _write_outputs(cfg, out, rows, delta_rows, delta_draws, probe_draws, methods, seeds)

    if crashed:
        # Everything that could be scored has been, and integration.csv records both
        # the scores and the crash. Fail now, so a dependent job does not treat a
        # partial comparison as the comparison.
        refuse(
            "a complete integration comparison",
            f"{len(crashed)} method(s) died in native code: {', '.join(crashed)}",
            hint="the partial table is on disk. A SIGILL here is usually a "
            "compiled dependency built for a different instruction set than "
            "the node provides — rebuild the environment from "
            "requirements.txt, or drop the method from "
            "diagnostics.integration_methods",
        )


def _scvi_embedding(cfg, table, seed: int) -> np.ndarray:
    """Train scVI on raw counts with slide as the batch covariate.

    The one method here that is a function of counts rather than a transformation of
    an existing embedding, so its correction transfers to a slide it was not fit on.
    """
    import gc

    import anndata as ad
    import scvi
    import torch

    from ..biosignal.expression import ExpressionIndex

    # The run seed drives the fit; `diagnostics.seed` still picks the gene
    # vocabulary, which must not move between fits or the three would be trained on
    # three different gene sets.
    scvi.settings.seed = int(seed)
    index = ExpressionIndex(cfg)
    ds, sid, spot = (table.col("dataset_id"), table.sample_id, table.col("spot_id"))
    # `stream` on *both* sweeps: this is the caller that reads all 120 slides once
    # and then stops needing them, which is the case both flags document. Without
    # them the index holds every slide's dense count matrix -- tens of GB, at the
    # full gene panel rather than the vocabulary -- and `matrix()` in particular
    # holds it at the moment it has also finished materialising `Y`.
    #
    # Streaming only the vocabulary pass, as this did before, moves that peak rather
    # than removing it: `matrix()` reloads every slide regardless of what the
    # vocabulary pass evicted, so the `drop_cache()` between them frees a cache that
    # is about to be rebuilt. The stage died at seed 44 both at 96G and, having
    # streamed only the vocabulary pass, at 64G -- the last thing it printed being
    # this function's vocabulary line, because the kill lands inside `matrix()`
    # before training starts.
    index.build_vocabulary(
        ds,
        sid,
        spot,
        min_expressed=cfg.biosignal.min_expressed,
        seed=cfg.diagnostics.seed,
        stream=True,
    )
    index.drop_cache()
    # `normalize=False`, so the counts reach scVI as counts. Its likelihood is a
    # negative binomial over integer counts and it models each spot's library size
    # itself; that size factor is how it separates sequencing depth from biology.
    #
    # Asking for the normalised matrix and inverting it -- what this did before --
    # does not give counts back. `matrix` divides each spot by its library size over
    # the full gene panel before the log1p, so `expm1` recovers CP10k rather than
    # counts, and rounding CP10k to integers rescales every spot to a common ~10k
    # depth: it multiplies up the counts of a shallow spot and rounds a deep one's
    # small counts to zero. The depth variation scVI exists to absorb was being
    # flattened out of its input before it ever saw it.
    Y, found = index.matrix(ds, sid, spot, normalize=False, stream=True)
    index.drop_cache()
    if found.sum() < 100:
        raise RuntimeError("too few spots resolved in the raw .h5ad files for scVI")

    # `Y` itself when every spot resolved, rather than a mask-indexed copy of it:
    # boolean indexing always copies, and this array is ~14G at cohort scale, so the
    # copy doubles the peak for as long as both are alive. `del` then either frees
    # `Y` or just drops the second name, and AnnData wraps the buffer rather than
    # copying it again.
    counts = Y if found.all() else Y[found]
    del Y
    a = ad.AnnData(X=counts)
    a.obs["batch"] = pd.Categorical(table.sample_id[found].astype(str))
    scvi.model.SCVI.setup_anndata(a, batch_key="batch")
    model = scvi.model.SCVI(a, n_latent=cfg.models.pca_components)
    model.train(max_epochs=cfg.models.ae.num_epochs // 4 or 1, accelerator="auto")

    Z = np.zeros((table.n, cfg.models.pca_components), dtype=np.float32)
    Z[found] = model.get_latent_representation().astype(np.float32)

    # Everything above is per-seed and none of it survives usefully, but two of the
    # references to it are held on the model *class* rather than by any local name:
    # scvi-tools' `setup_anndata` records each AnnDataManager in
    # `_setup_adata_manager_store`, and constructing the model records it again in
    # `_per_instance_manager_store`. Neither is ever evicted, and each manager holds
    # its AnnData, which holds the ~14G count matrix. Returning from this function
    # therefore frees nothing: seed 42's matrix is still resident when seed 43 asks
    # for its own, and seed 43's when seed 44 does. That is the accumulation that
    # made the stage die one seed later at 96G than at 64G -- the per-seed peak was
    # never the whole story, the floor under it was rising by 14G a fit.
    #
    # `gc.collect()` because refcounting alone will not do it in time: the manager
    # and the AnnData reference each other, as do the lightning module and its
    # trainer, so they are cycle-collected rather than freed at the last `del`, and
    # the next seed allocates its matrix immediately. `empty_cache` is the same
    # argument on the GPU side, where three fits' allocations otherwise stack up.
    scvi.model.SCVI._setup_adata_manager_store.clear()
    scvi.model.SCVI._per_instance_manager_store.clear()
    del model, a, counts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return Z
