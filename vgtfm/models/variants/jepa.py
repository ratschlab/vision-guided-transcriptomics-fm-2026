"""Joint-embedding predictive variant.

Instead of reconstructing the 3072-dimensional morphology vector, a predictor head
maps the gene latent into the *latent* space of a morphology encoder and the loss
is computed there, which lets the model ignore whatever part of the morphology
signal is unpredictable in principle rather than spending capacity on it.

The failure mode is collapse: nothing in a latent-prediction loss stops both
encoders mapping everything to a constant, which drives the loss to zero while
carrying no information. The variance-preservation term is therefore mandatory.
Watch effective rank alongside the loss — a run that converges at effective rank
one has learned nothing.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import perf
from ..base import Inputs, Model
from ..nn import (
    TensorStore,
    batched_embed,
    layer_sizes,
    mlp,
    seed_torch,
    train_loop,
    train_val_split,
)


class _JEPANet(nn.Module):
    def __init__(self, d_gene: int, d_patch: int, d_latent: int, enc_layers: int, dropout: float):
        super().__init__()
        self.encoder = mlp(layer_sizes(d_gene, d_latent, enc_layers), dropout=dropout)
        self.encoder_norm = nn.LayerNorm(d_latent)
        self.target = mlp(layer_sizes(d_patch, d_latent, enc_layers), dropout=dropout)
        self.predictor = nn.Sequential(
            nn.Linear(d_latent, d_latent),
            nn.BatchNorm1d(d_latent),
            nn.GELU(),
            nn.Linear(d_latent, d_latent),
        )

    def embed(self, gene):
        return self.encoder_norm(self.encoder(gene))

    def forward(self, gene, patch):
        z = self.embed(gene)
        with torch.no_grad():
            t = self.target(patch)
        return z, self.predictor(z), t


def variance_hinge(z: torch.Tensor, target_std: float = 1.0) -> torch.Tensor:
    """VICReg's hinge: penalise any latent dimension whose SD falls below 1.

    This is what keeps the predictive objective from collapsing to a constant.
    """
    std = torch.sqrt(z.var(dim=0) + 1e-6)
    return F.relu(target_std - std).mean()


class JEPAModel(Model):
    name = "jepa"

    def __init__(self, cfg, seed: int = 42):
        super().__init__(cfg, seed)
        self.p = cfg.models.ae
        self.v = cfg.models.variants
        self.device = perf.device(cfg)
        self.net: _JEPANet | None = None

    def fit(self, inputs: Inputs) -> dict:
        p = self.p
        seed_torch(self.cfg, self.seed)

        gene, patch = inputs.gene, inputs.patch
        self.net = _JEPANet(
            gene.shape[1], patch.shape[1], p.latent_dim, p.enc_layers, p.dropout
        ).to(self.device)
        n_params = sum(t.numel() for t in self.net.parameters())
        print(
            f"  [jepa] {n_params:,} params | latent {p.latent_dim} | "
            f"variance weight {self.v.variance_weight}"
        )

        def loss_fn(net, g_b, p_b):
            z, pred, tgt = net(g_b, p_b)
            return F.mse_loss(pred, tgt) + self.v.variance_weight * variance_hinge(z)

        train_idx, val_idx = train_val_split(len(gene), p.val_frac, self.seed)
        train_store = TensorStore(self.device, gene[train_idx], patch[train_idx])
        val_store = TensorStore(self.device, gene[val_idx], patch[val_idx])

        history = train_loop(
            self.net,
            train_store,
            val_store,
            loss_fn=loss_fn,
            params=p,
            cfg=self.cfg,
            device=self.device,
            seed=self.seed,
            label="jepa",
        )

        self.history = {
            "variant": "jepa",
            "n_params": int(n_params),
            "n_fit_spots": int(len(gene)),
            "latent_dim": int(p.latent_dim),
            **history,
        }
        del train_store, val_store
        perf.free_cuda()
        return self.history

    def embed(self, inputs: Inputs) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("jepa.embed called before fit")
        self.net.eval()
        return batched_embed(
            self.net.embed,
            inputs.gene,
            device=self.device,
            batch_size=perf.batch_size(self.cfg, self.p.batch_size),
        )
