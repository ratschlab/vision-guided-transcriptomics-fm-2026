"""Refit the AE as the ``train`` stage does and save its weights.

    python scripts/export_ae_checkpoint.py --config configs/default.yaml --env <profile>

Takes ``--config``, ``--env`` and ``--set`` like ``run.py``. Writes
``model.safetensors``, ``config.json`` and ``history.json`` to
``<artifact_root>/checkpoints/<run_name>_ae_seed-<seed>/``, and exits 1 if the refit does
not match the run's stored history and embedding.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vgtfm import perf  # noqa: E402
from vgtfm.config import headline_seed, load_config, to_dict  # noqa: E402
from vgtfm.data import tables  # noqa: E402
from vgtfm.models.base import Inputs  # noqa: E402
from vgtfm.models.nn import layer_sizes  # noqa: E402
from vgtfm.models.train import fit_and_embed, fit_rows  # noqa: E402
from vgtfm.provenance import array_fingerprint, source_fingerprint  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

#: History fields a refit on the same data must reproduce exactly.
EXACT_FIELDS = ("n_params", "n_fit_spots", "n_train", "n_val", "gene_dim", "patch_dim")

#: Minimum mean per-spot cosine between refit and stored embedding.
MIN_COSINE = 0.99


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def check_history(history: dict, run_dir: Path, seed: int) -> list[str]:
    """Bookkeeping that must match the reference run exactly; losses are shown."""
    path = run_dir / "train" / "history" / f"ae__seed-{seed}.json"
    if not path.exists():
        return [f"no {path}"]
    ref = json.loads(path.read_text())
    for k in ("epochs_run", "best_val_loss"):
        print(f"  {k:<14} run {ref.get(k)}  here {history.get(k)}")
    return [
        f"{k}: run {ref.get(k)}, here {history.get(k)}"
        for k in EXACT_FIELDS
        if ref.get(k) != history.get(k)
    ]


def check_embedding(Z: np.ndarray, run_dir: Path, seed: int) -> tuple[dict, list[str]]:
    """Per-spot agreement between the refit and the reference run's embedding."""
    path = run_dir / "embeddings" / "ae" / f"seed-{seed}.npy"
    if not path.exists():
        return {}, [f"no {path}"]
    ref = np.load(path)
    if ref.shape != Z.shape:
        return {}, [f"embedding shape: run {ref.shape}, here {Z.shape}"]
    cos = np.sum(ref * Z, axis=1) / np.maximum(
        np.linalg.norm(ref, axis=1) * np.linalg.norm(Z, axis=1), 1e-12
    )
    stats = {
        "reference_fingerprint": array_fingerprint(ref),
        "identical": bool(np.array_equal(ref, Z)),
        "max_abs_diff": float(np.max(np.abs(ref - Z))),
        "cosine_mean": float(cos.mean()),
        "cosine_p01": float(np.quantile(cos, 0.01)),
        "cosine_min": float(cos.min()),
    }
    for k, v in stats.items():
        print(f"  {k:<22} {v}")
    if stats["cosine_mean"] < MIN_COSINE:
        return stats, [f"mean per-spot cosine {stats['cosine_mean']:.4f} < {MIN_COSINE}"]
    return stats, []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None, help="YAML config path")
    ap.add_argument("--env", default=None, help="site profile from environments.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="dotted config overrides, as run.py")
    ap.add_argument("--seed", type=int, default=None, help="default: the run's headline seed")
    args = ap.parse_args(argv)

    overrides = dict(tuple(p.strip() for p in s.split("=", 1)) for s in args.set)
    cfg = load_config(args.config, overrides, env=args.env)
    seed = headline_seed(cfg) if args.seed is None else args.seed
    run_dir = cfg.out_dir
    out = Path(cfg.paths.artifact_root) / "checkpoints" / f"{cfg.run_name}_ae_seed-{seed}"
    print(f"  reference run  {run_dir}\n  output         {out}")
    print(f"  src={source_fingerprint()}  git={_git('rev-parse', '--short', 'HEAD') or '?'}")

    table = tables.load(cfg)
    rows = fit_rows(cfg, table)
    model, Z, history = fit_and_embed(cfg, Inputs.from_table(table), rows, "ae", seed)

    print("\n  -- verification: training history --")
    issues = check_history(history, run_dir, seed)
    print("\n  -- verification: embedding --")
    emb_stats, emb_issues = check_embedding(Z, run_dir, seed)
    issues += emb_issues

    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.net.state_dict().items()}
    save_file(state, out / "model.safetensors")
    (out / "history.json").write_text(json.dumps(history, indent=2))

    p = cfg.models.ae
    d_gene, d_patch = int(history["gene_dim"]), int(history["patch_dim"])
    config = {
        "architecture": {
            "class": "vgtfm.models.ae._Autoencoder",
            "d_gene": d_gene,
            "d_patch": d_patch,
            "d_latent": p.latent_dim,
            "enc_layers": p.enc_layers,
            "dec_layers": p.dec_layers,
            "dropout": p.dropout,
            "encoder_sizes": layer_sizes(d_gene, p.latent_dim, p.enc_layers),
            "decoder_sizes": layer_sizes(p.latent_dim, d_patch, p.dec_layers),
            "embedding": "encoder_norm(encoder(x))",
        },
        "training": {
            "seed": seed,
            "n_fit_spots": history["n_fit_spots"],
            "n_fit_slides": int(table.meta.iloc[rows].sample_id.nunique()),
            "epochs_run": history["epochs_run"],
            "best_val_loss": history["best_val_loss"],
        },
        "provenance": {
            "command": " ".join(sys.argv),
            "src_fingerprint": source_fingerprint(),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain", "--", "vgtfm", "run.py", "configs")),
            "device": perf.describe(cfg),
            "torch": __import__("torch").__version__,
            "python": platform.python_version(),
            "embedding_fingerprint": array_fingerprint(Z),
        },
        "verification": {"against": str(run_dir), "issues": issues, "embedding": emb_stats},
        "config": to_dict(cfg),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))
    print(f"\n  wrote {out}/{{model.safetensors,config.json,history.json}}")

    if issues:
        print(f"\n  the refit does NOT reproduce {run_dir}:")
        for d in issues:
            print(f"    - {d}")
        return 1
    print(f"\n  the refit reproduces {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
