"""Objective variants sharing the autoencoder's encoder and training schedule.

Every variant keeps the encoder, optimiser, scheduler, early stopping and
validation split of :class:`~vgtfm.models.ae.AutoencoderModel` and changes only
what the latent is trained to do:

``dual_decoder``  reconstruct the morphology embedding *and* the gene input, so a
                  gene head stops the latent discarding gene-only variance while
                  chasing morphology.
``gene_ae``       reconstruct only the gene input. A morphology-free control: the
                  gap to PCA is what the nonlinearity buys with no vision guidance.
``infonce``       align gene and morphology projections contrastively instead of
                  regressing one onto the other — what BLEEP-style methods do.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import perf
from ...ablations.patch_shuffle import apply_patch_transform
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

VARIANT_KINDS = ("dual_decoder", "gene_ae", "infonce")


class _Encoder(nn.Module):
    """Identical to the autoencoder's encoder, including the output LayerNorm."""

    def __init__(self, d_gene: int, d_latent: int, n_layers: int, dropout: float):
        super().__init__()
        self.body = mlp(layer_sizes(d_gene, d_latent, n_layers), dropout=dropout)
        self.norm = nn.LayerNorm(d_latent)

    def forward(self, x):
        return self.norm(self.body(x))


class _VariantNet(nn.Module):
    def __init__(
        self,
        kind: str,
        d_gene: int,
        d_patch: int,
        d_latent: int,
        enc_layers: int,
        dec_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.kind = kind
        self.encoder = _Encoder(d_gene, d_latent, enc_layers, dropout)
        self.patch_head = None
        self.gene_head = None
        self.patch_projector = None
        if kind == "dual_decoder":
            self.patch_head = mlp(layer_sizes(d_latent, d_patch, dec_layers), dropout=dropout)
            self.gene_head = mlp(layer_sizes(d_latent, d_gene, dec_layers), dropout=dropout)
        elif kind == "gene_ae":
            self.gene_head = mlp(layer_sizes(d_latent, d_gene, dec_layers), dropout=dropout)
        elif kind == "infonce":
            # Projects morphology into the latent space so the two views are
            # comparable; discarded at inference like every other patch-side head.
            self.patch_projector = _Encoder(d_patch, d_latent, enc_layers, dropout)
        else:  # pragma: no cover
            raise ValueError(f"unknown variant '{kind}'. Known: {VARIANT_KINDS}")

    def forward(self, gene, patch):
        z = self.encoder(gene)
        out = {"z": z}
        if self.patch_head is not None:
            out["patch_hat"] = self.patch_head(z)
        if self.gene_head is not None:
            out["gene_hat"] = self.gene_head(z)
        if self.patch_projector is not None:
            out["z_patch"] = self.patch_projector(patch)
        return out


def variant_loss(
    kind: str,
    out: dict,
    gene: torch.Tensor,
    patch: torch.Tensor,
    gene_weight: float,
    temperature: float,
) -> torch.Tensor:
    if kind == "gene_ae":
        return F.mse_loss(out["gene_hat"], gene)
    if kind == "dual_decoder":
        return F.mse_loss(out["patch_hat"], patch) + gene_weight * F.mse_loss(out["gene_hat"], gene)
    if kind == "infonce":
        a = F.normalize(out["z"], dim=-1)
        b = F.normalize(out["z_patch"], dim=-1)
        logits = a @ b.T / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    raise ValueError(kind)  # pragma: no cover


class HeadVariantModel(Model):
    """One trainer for all three objective variants."""

    def __init__(self, kind: str, cfg, seed: int = 42):
        super().__init__(cfg, seed)
        self.name = kind
        self.kind = kind
        self.p = cfg.models.ae  # shares the autoencoder's schedule
        self.v = cfg.models.variants
        self.device = perf.device(cfg)
        self.net: _VariantNet | None = None

    def fit(self, inputs: Inputs) -> dict:
        p = self.p
        seed_torch(self.cfg, self.seed)

        gene = inputs.gene
        patch, _perm = apply_patch_transform(
            p.patch_transform, inputs.patch, inputs.sample_id, np.random.default_rng(p.shuffle_seed)
        )

        d_g, d_p = gene.shape[1], patch.shape[1]
        self.net = _VariantNet(
            self.kind, d_g, d_p, p.latent_dim, p.enc_layers, p.dec_layers, p.dropout
        ).to(self.device)
        n_params = sum(t.numel() for t in self.net.parameters())
        print(f"  [{self.kind}] {n_params:,} params | {d_g} -> {p.latent_dim}")

        def loss_fn(net, g_b, p_b):
            return variant_loss(
                self.kind, net(g_b, p_b), g_b, p_b, self.v.gene_recon_weight, self.v.temperature
            )

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
            label=self.kind,
        )

        self.history = {
            "variant": self.kind,
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
            raise RuntimeError(f"{self.kind}.embed called before fit")
        self.net.eval()
        return batched_embed(
            self.net.encoder,
            inputs.gene,
            device=self.device,
            batch_size=perf.batch_size(self.cfg, self.p.batch_size),
        )


def build_head_variant(name: str, cfg, seed: int = 42) -> HeadVariantModel:
    return HeadVariantModel(name, cfg, seed)
