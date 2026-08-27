"""PCA baselines.

``pca``         PCA of the frozen gene-FM embedding. The reference baseline, and a
                strong one: the frozen embeddings have an effective rank well under
                50, so a linear projection captures nearly everything they contain.

``pca_oracle``  PCA of the frozen Midnight H&E embedding. Not a competitor — it
                reads the morphology the gene models are trying to predict — but
                the ceiling a perfect gene->morphology model could reach.

``pca_oracle_matched``
                The oracle at ``models.pca_oracle_matched_components`` instead of
                the shared ``models.pca_components``. All baselines share a nominal
                width, but they are not equally *loaded* at it: the gene PCAs
                saturate around effective rank 17-29 while the oracle's 128-d output
                reaches ~43. Refitting the oracle where those coincide separates its
                lead from its capacity.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from .base import Inputs, Model


class _PCABase(Model):
    #: Which feature block this baseline is fit on and transforms.
    feature: str = "gene"

    def __init__(self, cfg, seed: int = 42):
        super().__init__(cfg, seed)
        self.n_components = int(self._components(cfg))
        self.pca: PCA | None = None

    @staticmethod
    def _components(cfg) -> int:
        """Width of this baseline. Shared by default; overridden by the control."""
        return cfg.models.pca_components

    def _X(self, inputs: Inputs) -> np.ndarray:
        return inputs.gene if self.feature == "gene" else inputs.patch

    def fit(self, inputs: Inputs) -> dict:
        X = self._X(inputs)
        n = min(self.n_components, X.shape[0], X.shape[1])
        if n < self.n_components:
            print(f"  [{self.name}] capping components {self.n_components} -> {n} (data {X.shape})")
        self.pca = PCA(n_components=n, random_state=self.seed)
        self.pca.fit(X)
        evr = float(np.sum(self.pca.explained_variance_ratio_))
        self.history = {
            "n_components": int(n),
            "n_fit_spots": int(X.shape[0]),
            "input_dim": int(X.shape[1]),
            "explained_variance_ratio": evr,
        }
        print(f"  [{self.name}] {n} PCs on {X.shape[0]:,}x{X.shape[1]} -> {evr:.4f} of variance")
        return self.history

    def embed(self, inputs: Inputs) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError(f"{self.name}.embed called before fit")
        return self.pca.transform(self._X(inputs)).astype(np.float32)


class PCAGene(_PCABase):
    name = "pca"
    feature = "gene"


class PCAOracle(_PCABase):
    name = "pca_oracle"
    feature = "patch"
    uses_patch_at_inference = True


class PCAOracleMatched(PCAOracle):
    """The oracle at a width matched on *effective* rather than nominal rank."""

    name = "pca_oracle_matched"

    @staticmethod
    def _components(cfg) -> int:
        return cfg.models.pca_oracle_matched_components
