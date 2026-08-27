#!/usr/bin/env python
"""vgtfm pipeline CLI.

    python run.py data       [--config configs/default.yaml] [--env cluster]
                             [--set key.sub=val ...]
    python run.py train
    python run.py eval
    python run.py diagnose
    python run.py integrate
    python run.py ablate
    python run.py biosignal
    python run.py figures
    python run.py all

Every stage reads one YAML config (falling back to the dataclass defaults in
``vgtfm/config.py``) plus a site profile from ``environments.yaml`` (``--env``,
which supplies every machine-specific path) and writes under
``<artifact_root>/<run_name>/``. The resolved config is dumped next to the outputs,
stdout/stderr are teed to a per-stage logfile, and a
``logs/manifest-<stage>-<timestamp>.json`` records host, device, per-stage wall time
and status — including when a stage fails. Both names carry the stage because
stages are submitted in parallel and the timestamp alone is not unique.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import platform
import socket
import sys
import time
from pathlib import Path

# Make ``vgtfm`` importable when invoked as ``python run.py`` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vgtfm.config import ENVIRONMENTS_FILE, load_config, to_dict  # noqa: E402
from vgtfm.provenance import source_fingerprint  # noqa: E402

#: Stages run by ``all``, in dependency order. ``embed`` (heavy, cluster-only) and
#: ``ablate`` / ``biosignal`` (expensive, and not needed for the headline table)
#: are opt-in.
_ALL_STAGES = ["data", "train", "eval", "diagnose", "figures"]

_STAGES = [
    "data",
    "embed",
    "train",
    "eval",
    "diagnose",
    "integrate",
    "results",
    "ablate",
    "biosignal",
    "figures",
    "all",
]


def _parse_sets(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--set expects key=value, got '{it}'")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


class _Tee:
    """Duplicate a stream to several sinks (console + per-run logfile)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            with contextlib.suppress(Exception):
                s.flush()

    def isatty(self):
        return getattr(self.streams[0], "isatty", lambda: False)()

    def close(self):
        """No-op: the sinks outlive the tee, and `main` closes the logfile itself.

        Required because libraries that install logging handlers on the replaced
        stream — absl, which scib-metrics pulls in through JAX — call ``close()``
        on it at interpreter shutdown, and would otherwise put an ``AttributeError``
        traceback on stderr after the run's own status line.
        """


def _stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _device_info(cfg) -> str:
    try:
        from vgtfm import perf

        perf.configure(cfg)
        return perf.describe(cfg)
    except Exception as e:  # pragma: no cover
        return f"(torch backend unavailable: {type(e).__name__})"


def _run_stage(stage: str, cfg) -> None:
    if stage == "data":
        from vgtfm.data import build

        build.run(cfg)
    elif stage == "embed":
        from vgtfm.embed import build as embed_build

        embed_build.run(cfg)
    elif stage == "train":
        from vgtfm.models import train

        train.run(cfg)
    elif stage == "eval":
        from vgtfm.evaluate import run_eval

        run_eval.run(cfg)
    elif stage == "diagnose":
        from vgtfm.diagnostics import report

        report.run(cfg)
    elif stage == "integrate":
        from vgtfm.diagnostics import integration

        integration.run(cfg)
    elif stage == "ablate":
        from vgtfm.ablations import run_ablation

        run_ablation.run(cfg)
    elif stage == "biosignal":
        from vgtfm.biosignal import run_biosignal

        run_biosignal.run(cfg)
    elif stage == "results":
        from vgtfm import results

        results.run(cfg)
    elif stage == "figures":
        from vgtfm.figures import build as fig_build

        fig_build.run(cfg)
    else:  # pragma: no cover
        raise SystemExit(f"unknown stage {stage}")


def _banner(cfg, *, started: str, device: str, logpath) -> None:
    """The five lines every run opens with, and the manifest records alongside.

    ``src`` is a content hash of the shipped sources; ``bash slurm/preflight.sh``
    prints the same value, so two checkouts can be compared in one line rather than
    by reading line numbers out of a traceback.
    """
    print(
        f"== run '{cfg.run_name}'  [{started}]  host={socket.gethostname()}  "
        f"py={platform.python_version()}  src={source_fingerprint()}"
    )
    print(f"   env={cfg.env or '(none)'}  out={cfg.out_dir}")
    print(f"   data={cfg.data_path()}")
    print(
        f"   device={device}  amp={cfg.perf.amp}({cfg.perf.amp_dtype})  compile={cfg.perf.compile}"
    )
    print(
        f"   substrate={cfg.data.substrate}  models={list(cfg.models.names)}  "
        f"seeds={list(cfg.seeds)}"
    )
    if logpath is not None:
        print(f"   log={logpath}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="vgtfm - vision-guided transcriptomics FM pipeline")
    ap.add_argument("stage", choices=_STAGES)
    ap.add_argument("--config", default=None, help="YAML config path")
    ap.add_argument(
        "--env",
        default=None,
        help="site profile from environments.yaml (default: $VGTFM_ENV, "
        "else the file's `default:`). Supplies machine-specific paths.",
    )
    ap.add_argument(
        "--set",
        nargs="*",
        default=[],
        help="dotted config overrides, e.g. models.pca_components=50",
    )
    ap.add_argument("--no-log", action="store_true", help="do not tee output to a per-run logfile")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, _parse_sets(args.set), env=args.env)
    try:
        cfg.sub()
    except OSError as e:
        # Almost always a site profile pointing somewhere this machine cannot see.
        raise SystemExit(
            f"cannot create the output directory {cfg.out_dir}: {e}\n"
            f"  site profile '{cfg.env or '(none)'}' from {ENVIRONMENTS_FILE}\n"
            f"  fix artifact_root there, or pass "
            f"--set paths.artifact_root=/a/writable/path"
        ) from None
    (cfg.out_dir / "config.resolved.json").write_text(
        json.dumps(to_dict(cfg), indent=2, default=str)
    )

    stages = _ALL_STAGES if args.stage == "all" else [args.stage]
    started = _stamp()
    logfh = None
    orig_out, orig_err = sys.stdout, sys.stderr
    logpath = None
    if not args.no_log:
        logpath = cfg.sub("logs") / f"{args.stage}-{started}.log"
        logfh = open(logpath, "w", buffering=1)
        sys.stdout = _Tee(orig_out, logfh)
        sys.stderr = _Tee(orig_err, logfh)

    device = _device_info(cfg)
    _banner(cfg, started=started, device=device, logpath=logpath)

    manifest = {
        "run_name": cfg.run_name,
        "stage_arg": args.stage,
        "started": started,
        "env": cfg.env,
        "source_fingerprint": source_fingerprint(),
        "host": socket.gethostname(),
        "device": device,
        "substrate": cfg.data.substrate,
        "seeds": list(cfg.seeds),
        "config": str(cfg.out_dir / "config.resolved.json"),
        "stages": [],
    }
    t0 = time.time()
    status = "ok"
    try:
        for stage in stages:
            print(f"\n===== stage: {stage} =====")
            st = time.time()
            try:
                _run_stage(stage, cfg)
                dt = time.time() - st
                manifest["stages"].append({"stage": stage, "seconds": round(dt, 2), "status": "ok"})
                print(f"----- stage {stage} done in {dt:.1f}s -----")
            # BaseException, not Exception: stages abort with SystemExit for
            # actionable configuration errors, and a run that stopped early must
            # never be recorded in the manifest as a success.
            except BaseException as e:
                dt = time.time() - st
                manifest["stages"].append(
                    {
                        "stage": stage,
                        "seconds": round(dt, 2),
                        "status": "error",
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                status = "error"
                raise
    finally:
        manifest["total_seconds"] = round(time.time() - t0, 2)
        manifest["status"] = status
        # The stage belongs in the filename, not only inside the file: stages are
        # submitted in parallel and `_stamp()` has one-second resolution, so
        # `manifest-<ts>.json` alone lets two jobs started in the same second
        # overwrite each other's provenance record.
        (cfg.sub("logs") / f"manifest-{args.stage}-{started}.json").write_text(
            json.dumps(manifest, indent=2)
        )
        print(
            f"\n== {status} in {manifest['total_seconds']:.1f}s "
            f"({len(manifest['stages'])} stage(s))"
        )
        if logfh is not None:
            sys.stdout, sys.stderr = orig_out, orig_err
            logfh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
