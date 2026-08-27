"""Model contracts and the training machinery they all share.

The load-bearing property is the *deployment contract*: whatever a model trains on,
the artefact that gets scored is a function of gene features alone. A model that
read morphology at inference would score well and mean nothing, so every model is
checked against scrambled morphology.

The fingerprint test pins the numerics of the shared training loop: all six torch
models are trained on a fixed fixture and their embeddings hashed, so a refactor
that changes a result cannot pass silently.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vgtfm.config import load_config  # noqa: E402
from vgtfm.models.base import MODEL_NAMES, Inputs, build  # noqa: E402
from vgtfm.models.nn import (  # noqa: E402
    MIN_BATCH,
    batched_embed,
    effective_rank,
    layer_sizes,
    mlp,
    train_val_split,
)

GENE_MODELS = ("pca", "ae", "cdann", "dual_decoder", "gene_ae", "infonce", "jepa")

#: sha256 of the float32 embedding each model produces on :func:`fixture`, with
#: the config in :func:`_cfg` and seed 0. Update these only when a numerical
#: change is intended, and say why in the commit message.
FINGERPRINTS = {
    "ae": "f885711e1da1fe76",
    "cdann": "b77e442b8767f33f",
    "dual_decoder": "1ed0eaac68953d29",
    "gene_ae": "dd714254c36df354",
    "infonce": "c43409959e972bb8",
    "jepa": "cea36d85ae557a91",
}


def fixture(n=192, d_gene=16, d_patch=12, seed=0) -> Inputs:
    """Two views of a shared 4-d latent, on 4 slides across 2 tissues."""
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((n, 4)).astype(np.float32)
    Wg = rng.standard_normal((4, d_gene)).astype(np.float32)
    Wp = rng.standard_normal((4, d_patch)).astype(np.float32)
    return Inputs(
        gene=(latent @ Wg + 0.1 * rng.standard_normal((n, d_gene))).astype(np.float32),
        patch=(latent @ Wp + 0.1 * rng.standard_normal((n, d_patch))).astype(np.float32),
        sample_id=np.array([f"s{i % 4}" for i in range(n)]),
        tissue=np.array([f"t{i % 2}" for i in range(n)]),
        donor=np.array([f"d{i % 4}" for i in range(n)]),
        annotation=np.array([["A", "B", "C"][i % 3] for i in range(n)]),
        dataset_id=np.array([f"ds{i % 2}" for i in range(n)]),
        spot_id=np.array([f"bc{i}" for i in range(n)]),
    )


def _cfg():
    return load_config(
        None,
        {
            "models.ae.latent_dim": 8,
            "models.ae.num_epochs": 6,
            "models.ae.batch_size": 32,
            "models.ae.enc_layers": 2,
            "models.ae.dec_layers": 2,
            "models.ae.val_frac": 0.1,
            "models.cdann.latent_dim": 8,
            "models.cdann.hidden_dim": 16,
            "models.cdann.disc_hidden_dim": 16,
            "models.cdann.num_epochs": 4,
            "models.cdann.batch_size": 32,
            "models.pca_components": 8,
            "perf.device": "cpu",
            "perf.amp": "false",
        },
    )


def _digest(Z: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(Z, dtype=np.float32).tobytes()).hexdigest()[:16]


# ── the deployment contract ──────────────────────────────────────────


@pytest.mark.parametrize("name", GENE_MODELS)
def test_gene_models_never_read_morphology_at_inference(name):
    """Scrambling H&E after training must not move a single deployed value."""
    inputs = fixture()
    model = build(name, _cfg(), seed=0)
    model.fit(inputs)
    Z = model.embed(inputs)

    rng = np.random.default_rng(99)
    scrambled = Inputs(
        **{**inputs.__dict__, "patch": rng.standard_normal(inputs.patch.shape).astype(np.float32)}
    )
    assert np.array_equal(Z, model.embed(scrambled))
    assert not model.uses_patch_at_inference


def test_the_oracle_is_the_declared_exception():
    """``pca_oracle`` reads H&E on purpose; it is the ceiling, not a competitor."""
    inputs = fixture()
    model = build("pca_oracle", _cfg(), seed=0)
    model.fit(inputs)
    assert model.uses_patch_at_inference

    rng = np.random.default_rng(99)
    scrambled = Inputs(
        **{**inputs.__dict__, "patch": rng.standard_normal(inputs.patch.shape).astype(np.float32)}
    )
    assert not np.array_equal(model.embed(inputs), model.embed(scrambled))


@pytest.mark.parametrize("name", (*GENE_MODELS, "pca_oracle"))
def test_embeddings_are_well_formed_and_deterministic(name):
    inputs = fixture()
    model = build(name, _cfg(), seed=0)
    model.fit(inputs)
    Z = model.embed(inputs)

    assert Z.shape == (inputs.n, 8)
    assert Z.dtype == np.float32
    assert np.isfinite(Z).all()
    assert np.array_equal(Z, model.embed(inputs))  # repeatable
    assert effective_rank(Z) > 1.0  # not collapsed


@pytest.mark.parametrize("name", GENE_MODELS)
def test_fitting_twice_with_the_same_seed_gives_the_same_model(name):
    inputs = fixture()
    a, b = build(name, _cfg(), seed=7), build(name, _cfg(), seed=7)
    a.fit(inputs)
    b.fit(inputs)
    assert np.array_equal(a.embed(inputs), b.embed(inputs))


@pytest.mark.parametrize("name", GENE_MODELS)
def test_embedding_a_subset_matches_embedding_the_whole_cohort(name):
    """Embedding is row-wise: eval slices one saved matrix and must get the same rows."""
    inputs = fixture()
    model = build(name, _cfg(), seed=0)
    model.fit(inputs)
    rows = np.array([0, 5, 17, 100, 191])
    assert np.allclose(model.embed(inputs)[rows], model.embed(inputs.select(rows)), atol=1e-5)


@pytest.mark.parametrize("name", GENE_MODELS + ("pca_oracle",))
def test_embed_before_fit_is_an_error(name):
    with pytest.raises((RuntimeError, AttributeError)):
        build(name, _cfg(), seed=0).embed(fixture())


def test_unknown_model_names_are_rejected_with_the_known_list():
    with pytest.raises(SystemExit, match="unknown model"):
        build("transformer", _cfg(), seed=0)


def test_every_registered_name_can_be_built():
    for name in MODEL_NAMES:
        assert build(name, _cfg(), seed=0).name == name


# ── numerical fingerprints of the shared training loop ───────────────


@pytest.mark.parametrize("name", sorted(FINGERPRINTS))
def test_training_numerics_are_unchanged(name):
    """Pins the shared loop: same fixture, same seed, same bits.

    A failure here means a change to the optimiser, schedule, validation split or
    batching moved the results. That may be intended, but it is never incidental,
    so the recorded digest has to be updated on purpose.
    """
    model = build(name, _cfg(), seed=0)
    model.fit(fixture())
    assert _digest(model.embed(fixture())) == FINGERPRINTS[name]


def test_history_records_what_the_run_manifest_needs():
    model = build("ae", _cfg(), seed=0)
    history = model.fit(fixture())
    for key in (
        "n_params",
        "n_fit_spots",
        "n_train",
        "n_val",
        "epochs_run",
        "best_val_loss",
        "train_losses",
        "val_losses",
        "train_seconds",
        "patch_transform",
        "permutation_fixed_fraction",
    ):
        assert key in history, key
    assert history["epochs_run"] == len(history["train_losses"])
    assert history["n_train"] + history["n_val"] == history["n_fit_spots"]
    assert model.state()["seed"] == 0


def test_early_stopping_keeps_the_best_weights_not_the_last():
    """Training must end on the best validation epoch, whenever that was."""
    cfg = load_config(
        None,
        {
            "models.ae.latent_dim": 4,
            "models.ae.num_epochs": 40,
            "models.ae.batch_size": 32,
            "models.ae.patience": 3,
            "perf.device": "cpu",
            "perf.amp": "false",
        },
    )
    model = build("ae", cfg, seed=0)
    history = model.fit(fixture())
    assert history["best_val_loss"] == pytest.approx(min(history["val_losses"]))
    if history["epochs_run"] < 40:
        # It stopped early, so the last `patience` epochs were all worse.
        assert history["val_losses"][-1] > history["best_val_loss"]


def test_the_ablation_hook_reaches_the_autoencoder():
    """Training against shuffled morphology must produce a different model."""
    matched = load_config(
        None,
        {
            "models.ae.latent_dim": 8,
            "models.ae.num_epochs": 5,
            "models.ae.batch_size": 32,
            "perf.device": "cpu",
            "perf.amp": "false",
        },
    )
    shuffled = load_config(
        None,
        {
            "models.ae.latent_dim": 8,
            "models.ae.num_epochs": 5,
            "models.ae.batch_size": 32,
            "perf.device": "cpu",
            "perf.amp": "false",
            "models.ae.patch_transform": "global",
        },
    )
    inputs = fixture()
    a, b = build("ae", matched, seed=0), build("ae", shuffled, seed=0)
    ha, hb = a.fit(inputs), b.fit(inputs)

    assert ha["permutation_fixed_fraction"] == 1.0
    assert hb["permutation_fixed_fraction"] < 0.1
    assert hb["patch_transform"] == "global"
    assert not np.allclose(a.embed(inputs), b.embed(inputs))


# ── shared building blocks ───────────────────────────────────────────


def test_layer_sizes_interpolate_geometrically():
    assert layer_sizes(1152, 128, 2) == [1152, 512, 128]
    assert layer_sizes(128, 3072, 2) == [128, 512, 3072]
    assert layer_sizes(100, 10, 1) == [100, 10]
    assert layer_sizes(1024, 32, 5)[0] == 1024
    assert layer_sizes(1024, 32, 5)[-1] == 32
    assert len(layer_sizes(1024, 32, 5)) == 6
    with pytest.raises(ValueError):
        layer_sizes(64, 8, 0)


def test_mlp_leaves_the_final_layer_bare():
    """The last layer must be a plain Linear: the encoder norms its own output."""
    net = mlp([16, 8, 4], dropout=0.1)
    kinds = [type(m).__name__ for m in net]
    assert kinds == ["Linear", "BatchNorm1d", "GELU", "Dropout", "Linear"]
    assert isinstance(net[-1], torch.nn.Linear)

    assert [type(m).__name__ for m in mlp([16, 8, 4], dropout=0.0)] == [
        "Linear",
        "BatchNorm1d",
        "GELU",
        "Linear",
    ]
    assert [type(m).__name__ for m in mlp([16, 8, 4], norm=False, dropout=0.0)] == [
        "Linear",
        "GELU",
        "Linear",
    ]
    assert len(mlp([16, 4])) == 1


def test_train_val_split_is_a_partition_and_is_reproducible():
    train, val = train_val_split(1000, 0.1, seed=42)
    assert len(val) == 100 and len(train) == 900
    assert sorted(np.concatenate([train, val]).tolist()) == list(range(1000))
    assert np.array_equal(train, train_val_split(1000, 0.1, seed=42)[0])
    assert not np.array_equal(train, train_val_split(1000, 0.1, seed=43)[0])


def test_train_val_split_always_leaves_a_usable_validation_batch():
    """Below MIN_BATCH rows the validation loss would be undefined, not just noisy."""
    for n in (4, 9, 20):
        _train, val = train_val_split(n, 0.01, seed=0)
        assert len(val) >= MIN_BATCH


def test_batched_embed_is_independent_of_the_batch_size():
    """Blocking is an implementation detail; it must not change a single value."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((77, 6)).astype(np.float32)
    layer = torch.nn.Linear(6, 3)
    layer.eval()
    device = torch.device("cpu")

    whole = batched_embed(layer, X, batch_size=1000, device=device)
    for bs in (1, 7, 64):
        assert np.allclose(whole, batched_embed(layer, X, batch_size=bs, device=device), atol=1e-6)
    assert whole.shape == (77, 3) and whole.dtype == np.float32


def test_layernorm_without_affine_is_exactly_standardised():
    """Guards the intent of the encoder's output norm, independent of learned affine."""
    z = torch.nn.LayerNorm(16, elementwise_affine=False)(torch.randn(64, 16)).numpy()
    assert np.allclose(z.mean(axis=1), 0.0, atol=1e-5)
    assert np.allclose(z.std(axis=1), 1.0, atol=1e-2)


def test_deployed_embedding_has_a_fixed_scale():
    """LayerNorm on the encoder output is what keeps kNN distances comparable."""
    model = build("ae", _cfg(), seed=0)
    inputs = fixture()
    model.fit(inputs)
    norms = np.linalg.norm(model.embed(inputs), axis=1)
    assert norms.std() / norms.mean() < 0.2


# ── the count-level baseline ─────────────────────────────────────────


class _FakeExpression:
    """Stand-in for ExpressionIndex: deterministic counts, no .h5ad on disk.

    Two gene blocks: the first `n_signal` genes track the shared latent, the rest
    are near-constant noise. A correct HVG step must pick the first block.
    """

    def __init__(self, n_spots, n_genes=40, n_signal=6, missing=(), seed=0):
        rng = np.random.default_rng(seed)
        # Abundance is spread over the same range for both groups, so the test
        # separates "variable" from "highly expressed" — which is the entire job
        # of the mean-variance trend the dispersion is measured against.
        mean = 10 ** rng.uniform(1.4, 2.7, size=n_genes)
        counts = rng.poisson(mean, size=(n_spots, n_genes)).astype(np.float64)
        # The signal block gets latent-driven swing far above the Poisson trend.
        latent = rng.standard_normal((n_spots, 2))
        swing = latent @ rng.standard_normal((2, n_signal))
        counts[:, :n_signal] *= np.exp(0.9 * swing)
        self.counts = np.rint(counts).astype(np.float32)
        self.all_genes = np.array([f"g{i:03d}" for i in range(n_genes)])
        self.signal_genes = set(self.all_genes[:n_signal])
        self.missing = set(missing)
        self.genes = None
        self.dropped = 0

    def build_vocabulary(self, dataset_ids, sample_ids, spot_ids, **kw):
        self.genes = self.all_genes
        return self.genes

    def matrix(self, dataset_ids, sample_ids, spot_ids, *, normalize=True):
        idx = np.array([int(str(s)[2:]) for s in np.asarray(spot_ids)])
        col = np.array([np.where(self.all_genes == g)[0][0] for g in self.genes])
        counts = self.counts[idx][:, col]
        if normalize:
            totals = counts.sum(axis=1, keepdims=True)
            totals[totals <= 0] = 1.0
            counts = np.log1p(counts / totals * 1e4)
        Y = counts.astype(np.float32)
        found = ~np.isin(np.asarray(sample_ids).astype(str), list(self.missing))
        Y[~found] = 0.0
        return Y, found

    def drop_cache(self):
        self.dropped += 1


def _hvg_model(inputs, fake, cfg=None, seed=0):
    from vgtfm.models.hvg_pca import HVGPCA

    cfg = cfg or _cfg()
    model = HVGPCA(cfg, seed=seed)
    model._index = fake
    # `fit` asks the cohort table for the slides the embedding will cover; the
    # fake resolves every spot, so the vocabulary sweep has nothing to add.
    model._cohort_meta = None
    return model


@pytest.fixture
def hvg_patched(monkeypatch):
    """Point `hvg_pca`'s cohort lookup at the fixture instead of a real cache."""
    import pandas as pd

    from vgtfm.data import tables as tables_mod

    def fake_load_meta(cfg, substrate=None):
        n = 192
        return pd.DataFrame(
            {
                "dataset_id": ["ds"] * n,
                "sample_id": [f"s{i % 4}" for i in range(n)],
                "spot_id": [f"bc{i}" for i in range(n)],
            }
        )

    monkeypatch.setattr(tables_mod, "load_meta", fake_load_meta)


def test_hvg_pca_selects_the_variable_genes(hvg_patched):
    """The whole point of the HVG step: constant noise genes must not survive."""
    from vgtfm.models.hvg_pca import HVGPCA

    inputs = fixture()
    cfg = _cfg()
    cfg.models.hvg.n_top_genes = 6
    fake = _FakeExpression(inputs.n)
    model = HVGPCA(cfg, seed=0)
    model._index = fake
    model.fit(inputs)

    assert set(model.genes) == fake.signal_genes
    assert model.history["input_dim"] == 6
    assert model.history["n_shared_genes"] == 40


def test_hvg_pca_never_reads_morphology(hvg_patched):
    """Same deployment contract as every other gene model."""
    from vgtfm.models.hvg_pca import HVGPCA

    inputs = fixture()
    model = HVGPCA(_cfg(), seed=0)
    model._index = _FakeExpression(inputs.n)
    model.fit(inputs)
    Z = model.embed(inputs)

    rng = np.random.default_rng(99)
    scrambled = Inputs(
        **{**inputs.__dict__, "patch": rng.standard_normal(inputs.patch.shape).astype(np.float32)}
    )
    assert np.array_equal(Z, model.embed(scrambled))
    assert not model.uses_patch_at_inference


def test_hvg_pca_scaler_and_pca_come_from_the_fit_rows_only(hvg_patched):
    """Fitting either on the evaluated spots would leak the held-out donors."""
    from vgtfm.models.hvg_pca import HVGPCA

    inputs = fixture()
    model = HVGPCA(_cfg(), seed=0)
    model._index = _FakeExpression(inputs.n)
    fit_rows = np.arange(0, 96)
    model.fit(inputs.select(fit_rows))

    # Embedding the fit rows alone, or as part of the whole cohort, must give the
    # same vectors — which is only true if nothing is refitted at embed time.
    Z_all = model.embed(inputs)
    Z_fit = model.embed(inputs.select(fit_rows))
    assert np.allclose(Z_all[fit_rows], Z_fit, atol=1e-5)


def test_hvg_pca_reports_spots_it_could_not_resolve(hvg_patched, capsys):
    """A slide with no .h5ad becomes visible zeros, never a silently dropped row."""
    from vgtfm.models.hvg_pca import HVGPCA

    inputs = fixture()
    model = HVGPCA(_cfg(), seed=0)
    model._index = _FakeExpression(inputs.n, missing={"s3"})
    model.fit(inputs)
    Z = model.embed(inputs)

    assert Z.shape == (inputs.n, 8)
    absent = np.asarray(inputs.sample_id) == "s3"
    assert np.all(Z[absent] == 0)
    assert np.any(Z[~absent] != 0)
    assert "embedded as zeros" in capsys.readouterr().out


def test_hvg_pca_drops_the_slide_cache_between_chunks(hvg_patched):
    """Holding 120 dense count matrices at once is more RAM than the stage has."""
    from vgtfm.models.hvg_pca import HVGPCA

    inputs = fixture()
    cfg = _cfg()
    cfg.models.hvg.slides_per_chunk = 1
    fake = _FakeExpression(inputs.n)
    model = HVGPCA(cfg, seed=0)
    model._index = fake
    model.fit(inputs)
    before = fake.dropped
    model.embed(inputs)
    # Four slides, one per chunk: one eviction each.
    assert fake.dropped - before == 4


def test_seurat_v3_dispersion_ranks_variable_genes_above_constant_ones():
    from vgtfm.models.hvg_pca import seurat_v3_dispersion

    rng = np.random.default_rng(0)
    counts = np.hstack([rng.integers(0, 500, (200, 3)).astype(float), np.full((200, 3), 7.0)])
    d = seurat_v3_dispersion(counts)
    assert d[:3].min() > d[3:].max()
    assert np.all(d[3:] == 0.0)  # zero variance never ranks


# ── the capacity-matched oracle ──────────────────────────────────────


def test_the_matched_oracle_is_narrower_and_still_reads_morphology():
    """It answers "is the oracle's lead just capacity?", so it must differ only
    in width."""
    inputs = fixture()
    cfg = _cfg()
    cfg.models.pca_oracle_matched_components = 4

    wide = build("pca_oracle", cfg, seed=0)
    narrow = build("pca_oracle_matched", cfg, seed=0)
    wide.fit(inputs)
    narrow.fit(inputs)

    assert wide.embed(inputs).shape[1] == 8
    assert narrow.embed(inputs).shape[1] == 4
    assert narrow.uses_patch_at_inference
    # The narrow one is a truncation of the same basis, not a different fit.
    assert np.allclose(np.abs(narrow.embed(inputs)), np.abs(wide.embed(inputs)[:, :4]), atol=1e-4)


def test_hvg_selection_ranks_on_counts_not_on_normalised_values(hvg_patched):
    """seurat_v3 standardises each gene against a mean-variance trend fitted
    across genes. CP10k rescales every spot and therefore moves that trend, so
    ranking on normalised values selects a different gene set than the recipe
    names — silently, and only visibly as a slightly different Table 1 row."""
    from vgtfm.models.hvg_pca import HVGPCA

    inputs = fixture()
    cfg = _cfg()
    cfg.models.hvg.n_top_genes = 6

    seen = []
    fake = _FakeExpression(inputs.n)
    real_matrix = fake.matrix

    def spy(*a, normalize=True, **kw):
        seen.append(normalize)
        return real_matrix(*a, normalize=normalize, **kw)

    fake.matrix = spy
    model = HVGPCA(cfg, seed=0)
    model._index = fake
    model.fit(inputs)

    # The ranking pass asks for counts; the pass that fits the scaler and PCA
    # asks for log-normalised values.
    assert False in seen, "HVG ranking must read raw counts"
    assert True in seen, "the PCA must be fitted on log-normalised values"
    assert seen.index(False) < seen.index(True)
