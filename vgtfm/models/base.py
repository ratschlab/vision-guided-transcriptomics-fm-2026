"""Model interface.

Every model here is a *gene-embedding* model: whatever it is trained on, the
deployed artefact maps gene features alone to a vector, since morphology is a
training signal rather than an input. ``pca_oracle`` is the declared exception (it
reads H&E directly) and marks the morphology ceiling the gene models are measured
against.

The contract is three methods wide::

    fit(inputs, seed)  -> history dict     # trains on one set of spots
    embed(inputs)      -> (N, d) float32   # embeds any set of spots
    state()            -> dict             # small, JSON-serialisable provenance
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Inputs:
    """The arrays a model may look at, for one set of spots.

    ``patch`` is available at fit time for every model and at embed time only for
    ``pca_oracle``; a gene model that reads it in :meth:`Model.embed` would break
    the deployment contract, so models take the whole struct and are individually
    responsible for using only what they should.

    ``dataset_id`` and ``spot_id`` carry no features themselves — they are the key
    that resolves a spot back to its slide's raw ``.h5ad``. Only ``hvg_pca`` uses
    them, because it is the one baseline that starts from counts rather than from
    a foundation model's output; see :mod:`vgtfm.models.hvg_pca`.
    """

    gene: np.ndarray  # (N, Dg) float32
    patch: np.ndarray  # (N, Dp) float32
    sample_id: np.ndarray  # (N,) slide id — the batch variable
    tissue: np.ndarray  # (N,) tissue of the slide
    donor: np.ndarray  # (N,) patient key
    annotation: np.ndarray  # (N,) pathology label; NEVER read at fit time
    dataset_id: np.ndarray  # (N,) cohort key, for the raw .h5ad lookup
    spot_id: np.ndarray  # (N,) Visium barcode, ditto

    @property
    def n(self) -> int:
        return int(self.gene.shape[0])

    def select(self, rows: np.ndarray) -> "Inputs":
        rows = np.asarray(rows)
        return Inputs(
            gene=np.ascontiguousarray(self.gene[rows]),
            patch=np.ascontiguousarray(self.patch[rows]),
            sample_id=self.sample_id[rows],
            tissue=self.tissue[rows],
            donor=self.donor[rows],
            annotation=self.annotation[rows],
            dataset_id=self.dataset_id[rows],
            spot_id=self.spot_id[rows],
        )

    @classmethod
    def from_table(cls, table, rows: np.ndarray | None = None) -> "Inputs":
        full = cls(
            gene=table.gene,
            patch=table.patch,
            sample_id=table.sample_id,
            tissue=table.tissue,
            donor=table.donor,
            annotation=table.annotation,
            dataset_id=table.col("dataset_id"),
            spot_id=table.col("spot_id"),
        )
        return full if rows is None else full.select(rows)


class Model(ABC):
    """Base class for every embedder."""

    #: Registry key; also the directory name embeddings are written to.
    name: str = "model"
    #: Set on models that read H&E at inference (only the oracle baseline).
    uses_patch_at_inference: bool = False

    def __init__(self, cfg, seed: int = 42):
        self.cfg = cfg
        self.seed = seed
        self.history: dict = {}

    @abstractmethod
    def fit(self, inputs: Inputs) -> dict:
        """Train on *inputs*; return a JSON-serialisable history."""

    @abstractmethod
    def embed(self, inputs: Inputs) -> np.ndarray:
        """Return an ``(N, d)`` float32 embedding for *inputs*."""

    def state(self) -> dict:
        return {"name": self.name, "seed": self.seed, **self.history}


# ── registry ─────────────────────────────────────────────────────────


def build(name: str, cfg, seed: int = 42) -> Model:
    """Instantiate a model by registry name.

    Imports are local so a run using only PCA never pays for torch.
    """
    if name == "pca":
        from .pca import PCAGene

        return PCAGene(cfg, seed)
    if name == "pca_oracle":
        from .pca import PCAOracle

        return PCAOracle(cfg, seed)
    if name == "pca_oracle_matched":
        from .pca import PCAOracleMatched

        return PCAOracleMatched(cfg, seed)
    if name == "hvg_pca":
        from .hvg_pca import HVGPCA

        return HVGPCA(cfg, seed)
    if name == "ae":
        from .ae import AutoencoderModel

        return AutoencoderModel(cfg, seed)
    if name == "cdann":
        from .cdann import CDANNModel

        return CDANNModel(cfg, seed)
    from .corrected import parse

    if parse(name) is not None:
        # The correction needs every spot at once, and `build` is handed a name and
        # no data, so `train` resolves a corrected arm before it gets here.
        raise SystemExit(
            f"'{name}' is a batch-corrected arm and is not built through the "
            f"registry; the train stage corrects the gene features and builds the "
            f"inner model. See vgtfm/models/corrected.py."
        )

    from .variants import build_variant

    model = build_variant(name, cfg, seed)
    if model is None:
        raise SystemExit(f"unknown model '{name}'. Known: {sorted(MODEL_NAMES)}")
    return model


MODEL_NAMES = (
    "pca",
    "hvg_pca",
    "pca_oracle",
    "pca_oracle_matched",
    "ae",
    "cdann",
    "dual_decoder",
    "infonce",
    "gene_ae",
    "jepa",
)
