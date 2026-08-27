"""The gene -> morphology encoder-decoder.

An encoder compresses the frozen gene-FM embedding to a bottleneck; a decoder
expands it back to the frozen Midnight H&E embedding; the loss is MSE against the
real morphology features of the same spot. After training the decoder is discarded
and the deployed representation is ``LayerNorm(Encoder(gene))`` — morphology has
shaped the encoder's weights but is never an input at inference.

``patch_transform`` drives the correspondence ablation: training against permuted
morphology destroys the gene/patch pairing while leaving the marginal distribution
of the target untouched. See :mod:`vgtfm.ablations.patch_shuffle`.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .. import perf
from ..ablations.patch_shuffle import apply_patch_transform
from .base import Inputs, Model
from .nn import (
    TensorStore,
    batched_embed,
    layer_sizes,
    mlp,
    seed_torch,
    train_loop,
    train_val_split,
)


class _Autoencoder(nn.Module):
    def __init__(
        self,
        d_gene: int,
        d_patch: int,
        d_latent: int,
        enc_layers: int,
        dec_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.encoder = mlp(layer_sizes(d_gene, d_latent, enc_layers), dropout=dropout)
        self.decoder = mlp(layer_sizes(d_latent, d_patch, dec_layers), dropout=dropout)
        # Part of the deployed embedding: it fixes the latent scale so downstream
        # kNN distances are comparable across models.
        self.encoder_norm = nn.LayerNorm(d_latent)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_norm(self.encoder(x))

    def forward(self, x: torch.Tensor):
        z = self.embed(x)
        return self.decoder(z), z


def _reconstruction_loss(net, gene, patch):
    return nn.functional.mse_loss(net(gene)[0], patch)


class AutoencoderModel(Model):
    name = "ae"

    def __init__(self, cfg, seed: int = 42):
        super().__init__(cfg, seed)
        self.p = cfg.models.ae
        self.device = perf.device(cfg)
        self.net: _Autoencoder | None = None

    def fit(self, inputs: Inputs) -> dict:
        p = self.p
        seed_torch(self.cfg, self.seed)

        gene = inputs.gene
        patch, perm = apply_patch_transform(
            p.patch_transform,
            inputs.patch,
            inputs.sample_id,
            np.random.default_rng(p.shuffle_seed),
        )
        fixed_frac = float(np.mean(perm == np.arange(len(perm))))

        d_g, d_p = gene.shape[1], patch.shape[1]
        self.net = _Autoencoder(d_g, d_p, p.latent_dim, p.enc_layers, p.dec_layers, p.dropout).to(
            self.device
        )
        n_params = sum(t.numel() for t in self.net.parameters())
        print(
            f"  [ae] {n_params:,} params | {d_g} -> {p.latent_dim} -> {d_p} | "
            f"transform={p.patch_transform} (fixed-point frac {fixed_frac:.3f})"
        )

        train_idx, val_idx = train_val_split(len(gene), p.val_frac, self.seed)
        train_store = TensorStore(self.device, gene[train_idx], patch[train_idx])
        val_store = TensorStore(self.device, gene[val_idx], patch[val_idx])

        history = train_loop(
            self.net,
            train_store,
            val_store,
            loss_fn=_reconstruction_loss,
            params=p,
            cfg=self.cfg,
            device=self.device,
            seed=self.seed,
            label="ae",
        )

        self.history = {
            "n_params": int(n_params),
            "n_fit_spots": int(len(gene)),
            "gene_dim": int(d_g),
            "patch_dim": int(d_p),
            "latent_dim": int(p.latent_dim),
            "patch_transform": p.patch_transform,
            "shuffle_seed": int(p.shuffle_seed),
            "permutation_fixed_fraction": fixed_frac,
            **history,
        }
        del train_store, val_store
        perf.free_cuda()
        return self.history

    def embed(self, inputs: Inputs) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("ae.embed called before fit")
        self.net.eval()
        return batched_embed(
            self.net.embed,
            inputs.gene,
            device=self.device,
            batch_size=perf.batch_size(self.cfg, self.p.batch_size),
        )
