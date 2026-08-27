"""Torch performance helpers — device, TF32/bf16 backends, AMP, compile.

Driven entirely by ``cfg.perf``. Every helper accepts a possibly-``None`` config and
degrades to safe CPU/eager defaults, so enabling these changes speed, never results:
``auto`` AMP selects bf16 (full fp32 exponent range, no GradScaler) only where the
GPU supports it, and never silently falls back to lossy fp16.
"""

from __future__ import annotations

import contextlib

import torch


def _perf(cfg):
    return getattr(cfg, "perf", None)


def configure(cfg=None) -> None:
    """Set process-wide backend flags. Idempotent; call once per stage."""
    p = _perf(cfg)
    tf32 = getattr(p, "tf32", True)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cudnn.benchmark = bool(getattr(p, "cudnn_benchmark", True))
    with contextlib.suppress(Exception):
        torch.set_float32_matmul_precision(getattr(p, "matmul_precision", "high"))
    n = int(getattr(p, "num_threads", 0) or 0)
    if n > 0:
        torch.set_num_threads(n)


def device(cfg=None) -> torch.device:
    """CUDA when it is there and the config has not asked for CPU."""
    if getattr(_perf(cfg), "device", "auto") == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe(cfg=None) -> str:
    """The device line every run banner carries, for the manifest to record."""
    dev = device(cfg)
    if dev.type == "cuda":
        return (
            f"cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name(0)}, "
            f"bf16={'yes' if torch.cuda.is_bf16_supported() else 'no'})"
        )
    return f"cpu (threads={torch.get_num_threads()})"


def batch_size(cfg, base: int) -> int:
    """Allow the config to override batch size on CUDA only (VRAM-driven)."""
    gb = int(getattr(_perf(cfg), "gpu_batch_size", 0) or 0)
    return gb if gb > 0 and device(cfg).type == "cuda" else base


def _amp_dtype(cfg, dev):
    p = _perf(cfg)
    if not bool(getattr(p, "amp", True)) or dev.type != "cuda":
        return None
    want = getattr(p, "amp_dtype", "auto")
    if want == "fp16":
        return torch.float16
    if want == "bf16":
        return torch.bfloat16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else None


def amp(cfg, dev):
    """Return ``(autocast_factory, scaler)``. Both are no-ops on CPU."""
    dtype = _amp_dtype(cfg, dev)
    use_scaler = dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, TypeError):  # pragma: no cover - older torch
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    def autocast():
        if dtype is None:
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=dtype)

    return autocast, scaler


def free_cuda() -> None:
    """Best-effort VRAM release between folds/models."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        with contextlib.suppress(Exception):
            torch.cuda.ipc_collect()
