"""Torch building blocks and the training loop every neural model shares."""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .. import perf


def layer_sizes(input_size: int, target_size: int, num_layers: int) -> list[int]:
    """Geometrically interpolated widths from ``input_size`` to ``target_size``.

    ``num_layers`` counts *linear layers*, so the list has ``num_layers + 1``
    entries: ``layer_sizes(1152, 128, 2)`` gives 1152 -> 512 -> 128.
    """
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    if num_layers == 1:
        return [input_size, target_size]

    log_in, log_out = math.log2(input_size), math.log2(target_size)
    step = (log_out - log_in) / num_layers
    sizes = [int(2 ** round(log_in + step * i)) for i in range(num_layers + 1)]
    sizes[0], sizes[-1] = input_size, target_size
    return sizes


def mlp(sizes: list[int], *, dropout: float = 0.1, norm: bool = True) -> nn.Sequential:
    """Linear stack with BatchNorm + GELU + Dropout between hidden layers.

    The final linear layer is left bare: encoder outputs get their own LayerNorm
    (part of the deployed embedding) and decoder outputs are compared directly
    against raw morphology features.
    """
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            if norm:
                layers.append(nn.BatchNorm1d(b))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class TensorStore:
    """Holds training matrices on GPU when they fit, otherwise on CPU.

    The full cohort is ~3.3 GB of features, which does not fit on a small card.
    Resident storage is used when there is room and per-batch transfers otherwise,
    with identical numerics either way.
    """

    def __init__(self, device: torch.device, *arrays: np.ndarray, reserve_frac: float = 0.6):
        self.device = device
        self.resident = False
        nbytes = sum(int(a.size) * 4 for a in arrays)
        if device.type == "cuda":
            try:
                free, _total = torch.cuda.mem_get_info()
                self.resident = nbytes < reserve_frac * free
            except Exception:  # pragma: no cover
                self.resident = False
        self.tensors = []
        for a in arrays:
            t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
            if self.resident:
                t = t.to(device)
            elif device.type == "cuda":
                t = t.pin_memory()  # async host->device copies per batch
            self.tensors.append(t)
        where = "gpu" if self.resident else ("cpu (pinned)" if device.type == "cuda" else "cpu")
        print(f"  [data] {nbytes / 1e9:.2f} GB of features held on {where}")

    def batch(self, idx: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.resident:
            return tuple(t[idx] for t in self.tensors)
        cpu_idx = idx.cpu()
        return tuple(t[cpu_idx].to(self.device, non_blocking=True) for t in self.tensors)

    def slice(self, start: int, stop: int) -> tuple[torch.Tensor, ...]:
        if self.resident:
            return tuple(t[start:stop] for t in self.tensors)
        return tuple(t[start:stop].to(self.device, non_blocking=True) for t in self.tensors)

    def __len__(self) -> int:
        return int(self.tensors[0].shape[0])


def effective_rank(emb: np.ndarray) -> float:
    """``exp(H(p))`` with ``p = sigma^2 / sum(sigma^2)`` over centred singular values.

    Roy & Vetterli's effective rank. Note the exponential: it is on the scale of a
    dimension count and therefore comparable to the width.
    """
    if emb.shape[0] < 2:
        return 0.0
    X = np.asarray(emb, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    s2 = s**2
    total = s2.sum()
    if total <= 0:
        return 0.0
    p = np.clip(s2 / total, 1e-20, None)
    return float(np.exp(-(p * np.log(p)).sum()))


# ── shared training machinery ────────────────────────────────────────
#
# The autoencoder, the three head variants and the JEPA differ only in their loss.
# Optimiser, schedule, validation split, early stopping and history live here once,
# so a change to the schedule cannot apply to some models and not others.

#: Batches smaller than this are skipped: BatchNorm needs >1 row, InfoNCE needs a
#: negative, and a one-row variance term is undefined.
MIN_BATCH = 2


def seed_torch(cfg, seed: int) -> None:
    """Configure backends and seed torch + numpy for one fit."""
    perf.configure(cfg)
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_val_split(n: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Random ``(train_idx, val_idx)`` for early stopping.

    Drawn across the training slides, so it measures fit quality rather than
    cross-donor transfer.
    """
    order = np.random.default_rng(seed).permutation(n)
    n_val = max(MIN_BATCH, int(val_frac * n))
    return order[n_val:], order[:n_val]


def train_loop(
    net: nn.Module,
    train_store: TensorStore,
    val_store: TensorStore,
    *,
    loss_fn,
    params,
    cfg,
    device,
    seed: int,
    label: str,
) -> dict:
    """AdamW + cosine annealing + early stopping on the validation split.

    ``loss_fn(net, gene_batch, patch_batch) -> Tensor`` is the only part that
    differs between models. ``params`` supplies the schedule (``lr``,
    ``weight_decay``, ``batch_size``, ``num_epochs``, ``patience``).

    On return *net* holds the best-validation weights, not the last ones.
    """
    batch_size = perf.batch_size(cfg, params.batch_size)
    optimizer = optim.AdamW(
        net.parameters(),
        lr=params.lr,
        weight_decay=params.weight_decay,
        fused=(device.type == "cuda"),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params.num_epochs, eta_min=params.lr * 0.01
    )
    autocast, scaler = perf.amp(cfg, device)

    n_train, n_val = len(train_store), len(val_store)
    shuffle_gen = torch.Generator(device="cpu").manual_seed(seed)
    best_val, best_state, patience = float("inf"), None, 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    t0 = time.time()

    for epoch in range(params.num_epochs):
        net.train()
        batch_order = torch.randperm(n_train, generator=shuffle_gen)
        running = 0.0
        for start in range(0, n_train, batch_size):
            idx = batch_order[start : start + batch_size]
            if idx.numel() < MIN_BATCH:
                continue
            g_b, p_b = train_store.batch(idx)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                loss = loss_fn(net, g_b, p_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach()) * g_b.shape[0]
        train_losses.append(running / max(n_train, 1))

        net.eval()
        running = 0.0
        with torch.no_grad(), autocast():
            for start in range(0, n_val, batch_size):
                g_b, p_b = val_store.slice(start, min(start + batch_size, n_val))
                if g_b.shape[0] < MIN_BATCH:
                    continue
                running += float(loss_fn(net, g_b, p_b)) * g_b.shape[0]
        val_losses.append(running / max(n_val, 1))

        if val_losses[-1] < best_val:
            best_val = val_losses[-1]
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            patience = 0
        else:
            patience += 1

        scheduler.step()
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"    epoch {epoch + 1:3d}/{params.num_epochs}  "
                f"train {train_losses[-1]:.6f}  val {val_losses[-1]:.6f}  "
                f"lr {optimizer.param_groups[0]['lr']:.2e}"
                f"{'  *best*' if patience == 0 else ''}",
                flush=True,
            )
        if patience >= params.patience:
            print(
                f"    early stop at epoch {epoch + 1} "
                f"(no val improvement for {params.patience} epochs)"
            )
            break

    if best_state is not None:
        net.load_state_dict(best_state)

    seconds = round(time.time() - t0, 2)
    print(
        f"  [{label}] done in {seconds:.1f}s ({len(train_losses)} epochs, best val {best_val:.6f})"
    )
    return {
        "n_train": int(n_train),
        "n_val": int(n_val),
        "epochs_run": len(train_losses),
        "best_val_loss": float(best_val),
        "final_train_loss": float(train_losses[-1]) if train_losses else None,
        "train_losses": [float(x) for x in train_losses],
        "val_losses": [float(x) for x in val_losses],
        "train_seconds": seconds,
    }


def batched_embed(encode, gene: np.ndarray, *, batch_size: int, device) -> np.ndarray:
    """Apply ``encode`` (a torch callable) to *gene* in blocks; return float32."""
    X = torch.from_numpy(np.ascontiguousarray(gene, dtype=np.float32))
    out = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            block = X[start : start + batch_size].to(device)
            out.append(encode(block).float().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)
