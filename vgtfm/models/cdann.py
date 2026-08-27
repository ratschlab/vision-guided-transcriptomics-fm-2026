"""Conditional domain-adversarial network.

Aligns the two modalities in a shared latent space with a symmetric InfoNCE loss,
while a gradient-reversed discriminator predicts which slide a spot came from. The
discriminator is *conditional*: it sees the tissue alongside the latent, so it is
asked to separate slides within a tissue rather than to separate tissues.

Both projectors end in LayerNorm. The deployed embedding is the gene projector
alone; the patch projector only provides the alignment target and is discarded.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .. import perf
from .base import Inputs, Model
from .nn import MIN_BATCH, TensorStore, batched_embed, seed_torch


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class _GRL(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_ = 0.0

    def set_lambda(self, value: float) -> None:
        self.lambda_ = float(value)

    def forward(self, x):
        return _GradientReversal.apply(x, self.lambda_)


def _projector(d_in: int, d_hidden: int, d_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.LayerNorm(d_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_hidden, d_out),
        nn.LayerNorm(d_out),
    )


def _discriminator(
    d_in: int, d_hidden: int, d_out: int, n_layers: int, dropout: float
) -> nn.Sequential:
    dims = [d_in] + [d_hidden] * (n_layers - 1) + [d_out]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class _CDANN(nn.Module):
    def __init__(
        self,
        d_gene: int,
        d_patch: int,
        d_latent: int,
        n_tissues: int,
        n_slides: int,
        hidden: int,
        disc_hidden: int,
        disc_layers: int,
        dropout: float,
    ):
        super().__init__()
        # CLIP's learned temperature, initialised at 1/0.07.
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self.gene_projector = _projector(d_gene, hidden, d_latent, dropout)
        self.patch_projector = _projector(d_patch, hidden, d_latent, dropout)
        self.grl = _GRL()
        self.discriminator = _discriminator(
            d_latent + n_tissues, disc_hidden, n_slides, disc_layers, dropout
        )

    def embed(self, gene: torch.Tensor) -> torch.Tensor:
        return self.gene_projector(gene)

    def discriminate(self, z: torch.Tensor, tissue_onehot: torch.Tensor) -> torch.Tensor:
        return self.discriminator(torch.cat([self.grl(z), tissue_onehot], dim=-1))


def info_nce(z_a: torch.Tensor, z_b: torch.Tensor, logit_scale) -> torch.Tensor:
    """Symmetric InfoNCE over L2-normalised embeddings within a batch."""
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    scale = torch.clamp(logit_scale.exp(), max=100.0)
    logits = z_a @ z_b.T * scale
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def masked_batch_ce(
    logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor, label_smoothing: float
) -> torch.Tensor:
    """Cross-entropy over the slides that can occur in the spot's tissue.

    Without the mask the discriminator can score well by learning the tissue rather
    than the slide, which is the confound the conditioning exists to remove. The
    smoothing mass is spread over the valid slides only, for the same reason.
    """
    logits = logits.masked_fill(~valid, -1e9)
    if label_smoothing <= 0:
        return F.cross_entropy(logits, targets)
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
    smooth = (
        -log_probs.masked_fill(~valid, 0.0).sum(dim=-1) / valid.sum(dim=-1).clamp(min=1).float()
    )
    return ((1.0 - label_smoothing) * nll + label_smoothing * smooth).mean()


def grl_lambda(step: int, total_steps: int, max_lambda: float = 0.5) -> float:
    """DANN's schedule: ramp the reversal strength from 0 up to ``max_lambda``.

    Starting at zero lets the projectors learn something before the discriminator
    starts pushing back.
    """
    p = step / max(total_steps, 1)
    return (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0) * max_lambda


class CDANNModel(Model):
    name = "cdann"

    def __init__(self, cfg, seed: int = 42):
        super().__init__(cfg, seed)
        self.p = cfg.models.cdann
        self.device = perf.device(cfg)
        self.net: _CDANN | None = None
        self._tissues: np.ndarray | None = None

    # -- batch metadata ---------------------------------------------
    def _metadata(self, inputs: Inputs):
        """Slide and tissue indices, plus which slides each tissue can contain."""
        tissues = np.unique(inputs.tissue)
        slides = np.unique(inputs.sample_id)
        t_idx = {t: i for i, t in enumerate(tissues)}
        s_idx = {s: i for i, s in enumerate(slides)}

        valid = np.zeros((len(tissues), len(slides)), dtype=bool)
        for s, t in zip(inputs.sample_id, inputs.tissue):
            valid[t_idx[t], s_idx[s]] = True

        tissue_codes = np.array([t_idx[t] for t in inputs.tissue], dtype=np.int64)
        slide_codes = np.array([s_idx[s] for s in inputs.sample_id], dtype=np.int64)
        return tissues, slides, tissue_codes, slide_codes, valid

    # -- training ---------------------------------------------------
    def fit(self, inputs: Inputs) -> dict:
        cfg, p = self.cfg, self.p
        seed_torch(cfg, self.seed)

        tissues, slides, tissue_codes, slide_codes, valid = self._metadata(inputs)
        self._tissues = tissues
        n_t, n_s = len(tissues), len(slides)

        d_g, d_p = inputs.gene.shape[1], inputs.patch.shape[1]
        self.net = _CDANN(
            d_g,
            d_p,
            p.latent_dim,
            n_t,
            n_s,
            p.hidden_dim,
            p.disc_hidden_dim,
            p.disc_layers,
            p.dropout,
        ).to(self.device)
        n_params = sum(t.numel() for t in self.net.parameters())
        print(
            f"  [cdann] {n_params:,} params | latent {p.latent_dim} | "
            f"{n_t} tissue(s), {n_s} slide(s)"
        )

        # The discriminator learns faster than the projectors it is fighting, so it
        # gets its own (much smaller) learning rate.
        opt_proj = optim.AdamW(
            [
                *self.net.gene_projector.parameters(),
                *self.net.patch_projector.parameters(),
                self.net.logit_scale,
            ],
            lr=p.projector_lr,
            weight_decay=p.weight_decay,
            fused=(self.device.type == "cuda"),
        )
        opt_disc = optim.AdamW(
            self.net.discriminator.parameters(),
            lr=p.discriminator_lr,
            weight_decay=p.weight_decay,
            fused=(self.device.type == "cuda"),
        )

        batch_size = perf.batch_size(cfg, p.batch_size)
        store = TensorStore(self.device, inputs.gene, inputs.patch)
        tissue_t = torch.from_numpy(tissue_codes)
        slide_t = torch.from_numpy(slide_codes)
        valid_t = torch.from_numpy(valid).to(self.device)
        eye = torch.eye(n_t, device=self.device)

        n = len(store)
        steps_per_epoch = max(1, math.ceil(n / batch_size))
        total_steps = p.num_epochs * steps_per_epoch
        gen = torch.Generator(device="cpu").manual_seed(self.seed)

        history_align, history_disc = [], []
        t0 = time.time()
        self.net.train()
        for epoch in range(p.num_epochs):
            order = torch.randperm(n, generator=gen)
            align_sum = disc_sum = 0.0
            for step in range(steps_per_epoch):
                idx = order[step * batch_size : (step + 1) * batch_size]
                if idx.numel() < MIN_BATCH:
                    continue  # InfoNCE needs at least one negative
                g_b, p_b = store.batch(idx)
                t_b = tissue_t[idx].to(self.device)
                s_b = slide_t[idx].to(self.device)
                tissue_onehot = eye[t_b]

                self.net.grl.set_lambda(
                    grl_lambda(epoch * steps_per_epoch + step, total_steps, p.max_grl_lambda)
                )

                z_gene = self.net.gene_projector(g_b)
                z_patch = self.net.patch_projector(p_b)
                loss_align = info_nce(z_gene, z_patch, self.net.logit_scale)
                mask = valid_t[t_b]
                loss_disc = 0.5 * (
                    masked_batch_ce(
                        self.net.discriminate(z_gene, tissue_onehot), s_b, mask, p.label_smoothing
                    )
                    + masked_batch_ce(
                        self.net.discriminate(z_patch, tissue_onehot), s_b, mask, p.label_smoothing
                    )
                )
                loss = loss_align + loss_disc

                opt_proj.zero_grad(set_to_none=True)
                opt_disc.zero_grad(set_to_none=True)
                loss.backward()
                opt_proj.step()
                opt_disc.step()

                align_sum += float(loss_align.detach()) * idx.numel()
                disc_sum += float(loss_disc.detach()) * idx.numel()

            history_align.append(align_sum / n)
            history_disc.append(disc_sum / n)
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(
                    f"    epoch {epoch + 1:3d}/{p.num_epochs}  "
                    f"align {history_align[-1]:.4f}  disc {history_disc[-1]:.4f}  "
                    f"grl_lambda {self.net.grl.lambda_:.3f}",
                    flush=True,
                )

        self.history = {
            "n_params": int(n_params),
            "n_fit_spots": int(n),
            "latent_dim": int(p.latent_dim),
            "n_tissues": int(n_t),
            "n_slides": int(n_s),
            "epochs_run": int(p.num_epochs),
            "final_align_loss": float(history_align[-1]) if history_align else None,
            "final_disc_loss": float(history_disc[-1]) if history_disc else None,
            "align_losses": [float(x) for x in history_align],
            "disc_losses": [float(x) for x in history_disc],
            "train_seconds": round(time.time() - t0, 2),
        }
        print(f"  [cdann] done in {self.history['train_seconds']:.1f}s")
        del store
        perf.free_cuda()
        return self.history

    # -- inference --------------------------------------------------
    def embed(self, inputs: Inputs) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("cdann.embed called before fit")
        self.net.eval()
        return batched_embed(
            self.net.embed,
            inputs.gene,
            device=self.device,
            batch_size=perf.batch_size(self.cfg, self.p.batch_size),
        )
