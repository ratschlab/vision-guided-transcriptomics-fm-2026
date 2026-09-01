"""Config resolution and the spot-table plumbing underneath every stage.

Config errors are the cheapest way to produce a wrong number: a typo'd override
that silently does nothing, or a spot cap that drops a whole slide and therefore
a whole fold. Both fail quietly, so both are asserted here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]

from vgtfm.config import (  # noqa: E402
    Config,
    environment,
    load_config,
    load_environments,
    resolve_env_name,
    to_dict,
)
from vgtfm.data.tables import (  # noqa: E402
    Registry,
    SpotTable,
    infer_tissue,
    load_registry,
    parse_tupro_sample_id,
    select_rows,
)
from vgtfm.models.base import Inputs  # noqa: E402


# ── config ───────────────────────────────────────────────────────────


def test_defaults_load_without_a_file():
    cfg = load_config(None)
    assert isinstance(cfg, Config)
    assert cfg.data.substrate == "geneformer"
    assert cfg.eval.protocols[0] == "heldout_donor"


@pytest.mark.parametrize("path", sorted((REPO / "configs").glob("*.yaml")))
def test_every_shipped_config_resolves(path):
    """A config that no longer parses is a broken reproduction command."""
    cfg = load_config(path)
    assert cfg.run_name
    assert set(cfg.eval.protocols) <= {
        "heldout_donor",
        "loso_donor",
        "pooled_loso",
        "pooled_loso_on_fold",
    }
    assert json.dumps(to_dict(cfg), default=str)  # must be dumpable to the run dir


def test_overrides_are_coerced_to_the_field_type():
    cfg = load_config(
        None,
        {
            "models.pca_components": "50",  # int
            "models.ae.lr": "3e-4",  # float
            "eval.standardize": "false",  # bool
            "seeds": "1,2,3",  # tuple of ints
            "eval.protocols": "pooled_loso",  # tuple of one string
            "run_name": "smoke",  # str
        },
    )
    assert cfg.models.pca_components == 50 and isinstance(cfg.models.pca_components, int)
    assert cfg.models.ae.lr == pytest.approx(3e-4)
    assert cfg.eval.standardize is False
    assert cfg.seeds == (1, 2, 3)
    assert cfg.eval.protocols == ("pooled_loso",)
    assert cfg.run_name == "smoke"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("true", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("0", False),
        ("no", False),
    ],
)
def test_boolean_overrides_accept_the_usual_spellings(value, expected):
    assert load_config(None, {"eval.standardize": value}).eval.standardize is expected


def test_a_misspelled_override_is_an_error_not_a_no_op():
    """Silently ignoring `--set eval.standardise=false` would be the worst outcome."""
    for key in ("eval.standardise", "models.pca_component", "nonsense"):
        with pytest.raises(SystemExit, match="unknown config key"):
            load_config(None, {key: "1"})


def test_an_unknown_yaml_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("data:\n  substrat: geneformer\n")
    with pytest.raises(KeyError, match="substrat"):
        load_config(path)


def test_yaml_lists_become_tuples(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("seeds: [1, 2]\nmodels:\n  names: [pca, ae]\n")
    cfg = load_config(path)
    assert cfg.seeds == (1, 2)
    assert cfg.models.names == ("pca", "ae")


def test_output_directories_are_derived_from_the_run_name(tmp_path):
    cfg = load_config(None, {"run_name": "trial", "paths.artifact_root": str(tmp_path)})
    assert cfg.out_dir == tmp_path / "trial"
    made = cfg.sub("eval", "predictions")
    assert made.is_dir() and made == tmp_path / "trial" / "eval" / "predictions"


def test_the_feature_cache_is_shared_across_runs(tmp_path):
    """The cache is a function of the substrate, not of the run, or every run pays 5 GB."""
    a = load_config(None, {"run_name": "a", "paths.artifact_root": str(tmp_path)})
    b = load_config(None, {"run_name": "b", "paths.artifact_root": str(tmp_path)})
    assert a.cache_dir("geneformer") == b.cache_dir("geneformer")
    assert a.cache_dir("scgpt") != a.cache_dir("geneformer")
    assert a.out_dir != b.out_dir


def test_an_unknown_substrate_names_the_known_ones():
    with pytest.raises(SystemExit, match="Unknown substrate"):
        load_config(None).data_path("scfoundation")


def test_relative_dataset_paths_resolve_under_the_data_root():
    cfg = load_config(None, {"paths.data_root": "/data"})
    assert cfg.data_path("geneformer").is_absolute()
    assert str(cfg.data_path("geneformer")).startswith("/data/")


def test_dict_valued_fields_are_addressable_by_dotted_key():
    """On a cluster the raw cohorts sit under several roots, not one `data_root`."""
    cfg = load_config(
        None,
        {
            "paths.raw_h5ad.10x_TuPro": "/st/10x_TuPro/out_all_genes/data/h5ad",
            "paths.merged_datasets.geneformer": (
                "/tok/none_midnight_geneformer_none_none/merged_dataset"
            ),
        },
    )
    assert cfg.paths.raw_h5ad["10x_TuPro"] == "/st/10x_TuPro/out_all_genes/data/h5ad"
    # Untouched entries survive; an absolute override bypasses the data root.
    assert cfg.paths.raw_h5ad["TLS_VISIUM_USZ_kidney"] == "TLS_VISIUM_USZ/h5ad_preprocessed"
    assert str(cfg.data_path("geneformer")).startswith("/tok/")


def test_a_misspelled_dict_key_names_the_known_ones():
    with pytest.raises(SystemExit, match="unknown config key"):
        load_config(None, {"paths.raw_h5ad.10x_TuPRO": "/x"})


# ── site profiles (environments.yaml) ────────────────────────────────


@pytest.fixture
def envfile(tmp_path):
    """A two-profile environments.yaml, written where the loader can be pointed."""
    f = tmp_path / "environments.yaml"
    f.write_text("""
default: alpha
environments:
  alpha:
    data_root: /alpha/tok
    artifact_root: /alpha/out
  beta:
    data_root: /beta/tok
    raw_data_root: /beta/raw
    raw_h5ad:
      10x_TuPro: /beta/special/tupro
    conda_env: beta-env
    slurm_extra_args: ["--partition=gpu"]
""")
    return f


def test_a_profile_supplies_paths(envfile):
    cfg = load_config(None, env="beta", environments_file=envfile)
    assert cfg.env == "beta"
    assert cfg.paths.data_root == "/beta/tok"
    assert cfg.raw_root() == Path("/beta/raw")


def test_a_profile_is_a_diff_not_a_second_copy_of_the_defaults(envfile):
    """Stating one cohort must not silently drop the others."""
    default = load_config(None).paths.raw_h5ad
    cfg = load_config(None, env="beta", environments_file=envfile)
    assert cfg.raw_h5ad_dir("10x_TuPro") == Path("/beta/special/tupro")
    # untouched cohorts survive, and still resolve under the profile's raw root
    assert set(cfg.paths.raw_h5ad) == set(default)
    assert cfg.raw_h5ad_dir("TLS_VISIUM_USZ_kidney") == Path(
        "/beta/raw/TLS_VISIUM_USZ/h5ad_preprocessed"
    )
    # a key the profile omits keeps the dataclass default
    assert cfg.paths.artifact_root == load_config(None).paths.artifact_root


def test_set_overrides_beat_the_profile(envfile):
    cfg = load_config(None, {"paths.data_root": "/cli"}, env="beta", environments_file=envfile)
    assert cfg.paths.data_root == "/cli"


def test_an_unknown_profile_names_the_known_ones(envfile):
    with pytest.raises(SystemExit, match="unknown environment"):
        load_config(None, env="gamma", environments_file=envfile)


def test_the_default_profile_applies_when_none_is_named(envfile, monkeypatch):
    monkeypatch.delenv("VGTFM_ENV", raising=False)
    assert resolve_env_name(None, envfile) == "alpha"
    monkeypatch.setenv("VGTFM_ENV", "beta")
    assert resolve_env_name(None, envfile) == "beta"
    assert resolve_env_name("alpha", envfile) == "alpha"  # explicit wins


def test_a_relative_profile_path_resolves_against_the_repo(tmp_path):
    f = tmp_path / "e.yaml"
    f.write_text("environments:\n  rel:\n    artifact_root: artifacts\n")
    cfg = load_config(None, env="rel", environments_file=f)
    assert Path(cfg.paths.artifact_root).is_absolute()
    assert Path(cfg.paths.artifact_root) == REPO / "artifacts"


def test_a_missing_environments_file_is_not_an_error(tmp_path):
    assert load_environments(tmp_path / "nope.yaml") == {}
    assert resolve_env_name(None, tmp_path / "nope.yaml") is None
    assert load_config(None, environments_file=tmp_path / "nope.yaml").env == ""


@pytest.mark.parametrize("name", sorted((load_environments().get("environments") or {})))
def test_every_shipped_profile_resolves(name):
    """A profile that no longer applies is a broken cluster run."""
    cfg = load_config("configs/default.yaml", env=name)
    assert cfg.env == name
    assert Path(cfg.paths.data_root).is_absolute()
    assert Path(cfg.paths.artifact_root).is_absolute()
    assert cfg.data_path().is_absolute()
    for k in cfg.paths.raw_h5ad:
        assert cfg.raw_h5ad_dir(k).is_absolute()
    assert isinstance(environment(name).get("slurm_extra_args", []), list)


# ── sample ids, registry, tissue ─────────────────────────────────────


def test_tupro_ids_parse_into_donor_region_replicate():
    p = parse_tupro_sample_id("MACEGEJ-1-2")
    assert (p.donor, p.region, p.replicate, p.raw) == ("MACEGEJ", 1, 2, "MACEGEJ-1-2")
    assert parse_tupro_sample_id("KC1") is None
    assert parse_tupro_sample_id("MW-B-001a-vis") is None


def test_registry_indexes_samples_by_cohort_tissue_and_split(tmp_path):
    path = tmp_path / "datasets.json"
    path.write_text(
        json.dumps(
            {
                "datasets": {
                    "usz_kidney": {
                        "tissue": "kidney",
                        "annotation_col": "annotation",
                        "samples": [{"id": "KC1", "split": "test"}],
                    },
                    "mosaic": {
                        "tissue": "melanoma",
                        "samples": [{"id": "MW-B-001a-vis", "split": "train"}],
                    },
                }
            }
        )
    )
    reg = load_registry(path)
    assert reg.sample_tissue == {"KC1": "kidney", "MW-B-001a-vis": "melanoma"}
    assert reg.sample_dataset["KC1"] == "usz_kidney"
    assert reg.sample_split["MW-B-001a-vis"] == "train"
    assert reg.annotated_cohorts() == ["usz_kidney"]  # mosaic has no labels


def test_tissue_comes_from_the_registry_before_any_heuristic():
    assert infer_tissue("KC1", {"KC1": "kidney"}) == "kidney"
    assert infer_tissue("KC1", {}) == "KC"
    assert infer_tissue("MW-B-001a-vis", {}) == "MW-B"
    assert infer_tissue("weird_name", {}) == "weird_name"


def test_the_shipped_registry_is_well_formed():
    reg = load_registry(REPO / "configs" / "datasets.json")
    assert reg.datasets, "registry is empty"
    for name, spec in reg.datasets.items():
        assert spec.get("tissue"), f"{name} has no tissue"
        ids = [s["id"] for s in spec.get("samples", [])]
        assert len(ids) == len(set(ids)), f"{name} has duplicate sample ids"
    all_ids = list(reg.sample_tissue)
    assert len(all_ids) == len(set(all_ids)), "a sample id appears in two cohorts"
    assert isinstance(reg, Registry)


def test_the_shipped_registry_splits_on_annotation_and_nothing_else():
    """Annotation decides the split, so `train.fit_split` never sees a label.

    Every slide of an unannotated cohort is fitted on; the annotated cohorts are the
    whole of `test`. Nothing is held back into a cohort-level `validation` split —
    the neural models carve their early-stopping rows out of the fit set itself
    (:func:`vgtfm.models.nn.train_val_split`), so a third split would only shrink
    the representation's training set without being used.

    The counts are pinned because the manuscript quotes them: a cohort added here
    changes what the paper must say about the fit.
    """
    reg = load_registry(REPO / "configs" / "datasets.json")
    annotated = set(reg.annotated_cohorts())
    cohort_of = reg.sample_dataset
    by_split: dict[str, list[str]] = {}
    for sid, split in reg.sample_split.items():
        expected = "test" if cohort_of[sid] in annotated else "train"
        assert split == expected, f"{sid} ({cohort_of[sid]}) is '{split}', not '{expected}'"
        by_split.setdefault(split, []).append(sid)

    assert sorted(by_split) == ["test", "train"], "the registry grew a third split"
    assert len(by_split["train"]) == 96
    assert len(by_split["test"]) == 24


def test_every_shipped_donor_pattern_matches_its_own_sample_ids():
    """A typo'd regex would silently fall back to slide ids and inflate donor counts.

    The pattern is reporting-only (it feeds the dataset table), so nothing would
    crash — the appendix would just claim 22 patients where there are 5.
    """
    reg = load_registry(REPO / "configs" / "datasets.json")
    for name, spec in reg.datasets.items():
        pattern = spec.get("donor_pattern")
        if not pattern:
            continue
        for s in (sample["id"] for sample in spec.get("samples", [])):
            assert re.match(pattern, s), f"{name}: {pattern!r} does not match {s!r}"


def test_the_dataset_table_citation_fields_are_present_and_latex_safe():
    """`accession` and `bibkey` are written into the paper verbatim."""
    reg = load_registry(REPO / "configs" / "datasets.json")
    for name, spec in reg.datasets.items():
        cit = spec.get("citation") or {}
        assert "accession" in cit and "bibkey" in cit, f"{name} has no citation fields"
        assert cit["bibkey"], f"{name} has no bibkey"
        for field in ("accession", "bibkey"):
            value = cit[field] or ""
            assert "&" not in value and "%" not in value, f"{name}.{field}"
        assert " " not in cit["bibkey"], f"{name} bibkey is not a single cite key"


# ── spot table ───────────────────────────────────────────────────────


def table(n=12, seed=0):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(
        {
            "sample_id": np.repeat(["A-1-1", "A-1-2", "B-1-1"], n // 3),
            "donor": np.repeat(["A", "A", "B"], n // 3),
            "tissue": np.repeat(["mel", "mel", "mel"], n // 3),
            "split": np.repeat(["test"], n),
            "annotation": ["TUM", "STR", "UNASSIGNED", "TUM"] * (n // 4),
            "dataset_id": np.repeat(["ds_a", "ds_a", "ds_b"], n // 3),
            "spot_id": [f"bc{i}" for i in range(n)],
        }
    )
    return SpotTable(
        gene=rng.standard_normal((n, 5)).astype(np.float32),
        patch=rng.standard_normal((n, 3)).astype(np.float32),
        meta=meta,
        substrate="geneformer",
    )


def test_a_row_count_mismatch_is_caught_at_construction():
    """Features and metadata that disagree would misattribute every prediction."""
    with pytest.raises(ValueError, match="row mismatch"):
        SpotTable(
            gene=np.zeros((5, 2), dtype=np.float32),
            patch=np.zeros((4, 2), dtype=np.float32),
            meta=pd.DataFrame({"sample_id": ["a"] * 5}),
            substrate="x",
        )


def test_subsetting_keeps_features_and_metadata_aligned():
    t = table()
    idx = np.array([0, 4, 11])
    sub = t.subset(idx)
    assert sub.n == 3
    assert np.array_equal(sub.gene, t.gene[idx])
    assert np.array_equal(sub.patch, t.patch[idx])
    assert list(sub.meta.sample_id) == list(t.meta.sample_id.iloc[idx])
    assert list(sub.meta.index) == [0, 1, 2]  # reset, so .iloc/.loc agree


def test_rows_for_samples_selects_exactly_the_named_slides():
    t = table()
    rows = t.rows_for_samples(["A-1-2", "missing"])
    assert set(t.sample_id[rows]) == {"A-1-2"}
    assert len(t.rows_for_samples([])) == 0


def test_feature_blocks_are_addressable_by_either_name():
    t = table()
    assert t.features("gene") is t.gene and t.features("gene_features") is t.gene
    assert t.features("patch") is t.patch and t.features("patch_features") is t.patch
    with pytest.raises(KeyError):
        t.features("expression")


def test_labeled_mask_reaches_the_table():
    t = table()
    assert t.labeled.sum() == 9  # a quarter are UNASSIGNED
    assert "labelled=9" in t.describe()


# ── row selection ────────────────────────────────────────────────────


def _meta(counts: dict[str, int]) -> pd.DataFrame:
    return pd.DataFrame({"sample_id": [s for s, n in counts.items() for _ in range(n)]})


def test_include_and_exclude_filters_compose():
    meta = _meta({"a": 10, "b": 10, "c": 10})
    cfg = load_config(None, {"data.include_samples": "a,b", "data.exclude_samples": "b"})
    assert set(meta.sample_id.iloc[select_rows(cfg, meta)]) == {"a"}


def test_the_spot_cap_keeps_every_slide():
    """A cap that dropped a slide would silently delete a fold from a smoke run."""
    meta = _meta({"big": 5000, "medium": 500, "tiny": 30})
    cfg = load_config(None, {"data.max_spots": 500})
    kept = meta.sample_id.iloc[select_rows(cfg, meta)]

    assert set(kept) == {"big", "medium", "tiny"}
    assert len(kept) == pytest.approx(500, rel=0.1)
    # Sampling is proportional, so the big slide still dominates.
    assert (kept == "big").sum() > (kept == "medium").sum() > (kept == "tiny").sum()


def test_row_selection_is_sorted_and_deterministic():
    meta = _meta({"a": 400, "b": 400})
    cfg = load_config(None, {"data.max_spots": 100})
    first = select_rows(cfg, meta)
    assert np.array_equal(first, np.sort(first))
    assert np.array_equal(first, select_rows(cfg, meta))


def test_no_filters_selects_everything():
    meta = _meta({"a": 7, "b": 3})
    assert len(select_rows(load_config(None), meta)) == 10


# ── model inputs ─────────────────────────────────────────────────────


def test_inputs_from_table_carries_every_axis_a_model_may_condition_on():
    t = table()
    inputs = Inputs.from_table(t)
    assert inputs.n == t.n
    assert np.array_equal(inputs.gene, t.gene)
    assert np.array_equal(inputs.sample_id, t.sample_id)
    assert np.array_equal(inputs.donor, t.donor)
    assert np.array_equal(inputs.annotation, t.annotation)


def test_selecting_inputs_keeps_every_field_row_aligned():
    """Misalignment here would pair a spot's genes with another spot's slide id."""
    t = table()
    rows = np.array([1, 7, 9])
    sel = Inputs.from_table(t, rows)
    whole = Inputs.from_table(t)
    assert sel.n == 3
    for field in ("gene", "patch", "sample_id", "tissue", "donor", "annotation"):
        assert np.array_equal(getattr(sel, field), getattr(whole, field)[rows]), field
    assert np.array_equal(sel.gene, whole.select(rows).gene)


# ── cache schema ─────────────────────────────────────────────────────


def _schema1_cache(tmp_path):
    """A cache directory as written before annotations were canonicalised."""
    import json

    from vgtfm.data.tables import _cache_paths

    paths = _cache_paths(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    meta = pd.DataFrame(
        {
            "sample_id": ["KC1", "KC1", "MACEGEJ-1-1"],
            "spot_id": ["a", "b", "c"],
            "annotation": ["TUM", "UNASSIGNED", "Tumor"],
            "dataset_id": ["usz", "usz", "tupro"],
        }
    )
    meta.to_parquet(paths["meta"], index=False)
    np.save(paths["gene"], np.zeros((3, 2), dtype=np.float32))
    np.save(paths["patch"], np.zeros((3, 2), dtype=np.float32))
    paths["info"].write_text(
        json.dumps({"substrate": "geneformer", "source": "/src", "n_spots": 3})
    )
    return paths


def test_a_schema_1_cache_is_migrated_rather_than_rebuilt(tmp_path):
    """The feature matrices are several GB, so an exact migration beats a rebuild.

    Schema 1 -> 2 is exact without the source, because a schema-1 `annotation`
    column *is* the raw cohort spelling schema 2 stores as `annotation_raw`.
    """
    from vgtfm.data.tables import _CACHE_SCHEMA, _migrate_meta

    paths = _schema1_cache(tmp_path / "cache")
    assert _migrate_meta(paths, 1) is True

    meta = pd.read_parquet(paths["meta"])
    assert meta["annotation"].tolist() == ["Tumor", "UNASSIGNED", "Tumor"]
    assert meta["annotation_raw"].tolist() == ["TUM", "UNASSIGNED", "Tumor"]
    # The features were not touched.
    assert np.load(paths["gene"]).shape == (3, 2)
    # And a backup of the pre-migration frame is left behind.
    assert paths["meta"].with_suffix(".parquet.bak").exists()
    assert _CACHE_SCHEMA == 2


def test_an_unknown_cache_schema_falls_back_to_a_rebuild(tmp_path):
    """Silence here would reuse a frame whose columns mean something else."""
    from vgtfm.data.tables import _migrate_meta

    paths = _schema1_cache(tmp_path / "cache")
    assert _migrate_meta(paths, 99) is False


def test_the_cohort_table_records_which_spellings_were_merged(tmp_path):
    """The merge has to stay auditable in the artefact the appendix reads."""
    from vgtfm.data.build import _class_table

    meta = pd.DataFrame(
        {
            "sample_id": ["KC1", "KC1", "MACEGEJ-1-1", "MACEGEJ-1-1"],
            "spot_id": list("abcd"),
            "tissue": ["kidney", "kidney", "skin", "skin"],
            "annotation": ["Tumor", "TLS", "Tumor", "Stroma"],
            "annotation_raw": ["TUM", "TLS", "Tumor", "Stroma"],
            "dataset_id": ["usz", "usz", "tupro", "tupro"],
        }
    )
    table = SpotTable(
        gene=np.zeros((4, 2), dtype=np.float32),
        patch=np.zeros((4, 2), dtype=np.float32),
        meta=meta,
        substrate="geneformer",
    )
    df = _class_table(table)
    pooled = df[(df.tissue == "ALL") & (df.annotation == "Tumor")].iloc[0]
    assert pooled["raw_labels"] == "TUM|Tumor"
    assert pooled["n_spots"] == 2
    kidney = df[(df.tissue == "kidney") & (df.annotation == "Tumor")].iloc[0]
    assert kidney["raw_labels"] == "TUM"


# ── building the feature cache ───────────────────────────────────────


def _merged_dataset(
    path, n_per_split=(("train", 4), ("test", 3)), gene_dim=3, patch_dim=2, missing=()
):
    """A merged HuggingFace DatasetDict in the shape the embed stage writes."""
    from datasets import Array2D, Dataset, DatasetDict, Features, Value

    features = Features(
        {
            "sample_id": Value("string"),
            "spot_id": Value("string"),
            "array_row": Value("int32"),
            "array_col": Value("int32"),
            "dataset_id": Value("string"),
            "annotation": Value("string"),
            "gene_features": Array2D(shape=(1, gene_dim), dtype="float32"),
            "patch_features": Array2D(shape=(1, patch_dim), dtype="float32"),
        }
    )
    for column in missing:
        del features[column]

    splits = {}
    for split, n in n_per_split:
        rows = []
        for i in range(n):
            row = {
                "sample_id": "MACEGEJ-1-1" if i % 2 else "KC1",
                "spot_id": f"{split}-bc{i}",
                "array_row": i,
                "array_col": 2 * i,
                "dataset_id": "10x_TuPro" if i % 2 else "TLS_VISIUM_USZ_kidney",
                "annotation": ["TUM", "Tumor", "UNASSIGNED"][i % 3],
                "gene_features": [[float(i)] * gene_dim],
                "patch_features": [[float(-i)] * patch_dim],
            }
            rows.append({k: v for k, v in row.items() if k in features})
        splits[split] = Dataset.from_list(rows, features=features)
    DatasetDict(splits).save_to_disk(str(path))
    return path


@pytest.fixture
def cached(tmp_path):
    """A config whose substrate points at a real merged dataset under *tmp_path*."""
    src = _merged_dataset(tmp_path / "merged")
    return load_config(
        None,
        {
            "paths.data_root": str(tmp_path),
            "paths.artifact_root": str(tmp_path / "artifacts"),
            "paths.merged_datasets.geneformer": str(src),
        },
    )


def test_the_cache_materialises_both_feature_matrices_and_the_metadata(cached):
    from vgtfm.data.tables import _cache_paths, build_cache

    paths = _cache_paths(build_cache(cached))

    gene = np.load(paths["gene"])
    patch = np.load(paths["patch"])
    assert gene.shape == (7, 3) and gene.dtype == np.float32
    assert patch.shape == (7, 2)
    meta = pd.read_parquet(paths["meta"])
    assert len(meta) == 7
    assert meta["split"].tolist() == ["train"] * 4 + ["test"] * 3


def test_the_cache_derives_the_columns_the_folds_and_the_bootstrap_need(cached):
    from vgtfm.data.tables import _cache_paths, build_cache

    meta = pd.read_parquet(_cache_paths(build_cache(cached))["meta"])
    tupro = meta[meta.sample_id == "MACEGEJ-1-1"].iloc[0]
    usz = meta[meta.sample_id == "KC1"].iloc[0]

    assert tupro["donor"] == "MACEGEJ" and tupro["region"] == 1
    assert tupro["replicate"] == 1 and tupro["tissue"] == "skin"
    assert usz["donor"] == "KC1", "a slide with no replicate id is its own donor"
    assert usz["region"] == -1 and usz["tissue"] == "kidney"


def test_the_cache_canonicalises_annotations_and_keeps_the_raw_spelling(cached):
    from vgtfm.data.tables import _cache_paths, build_cache

    meta = pd.read_parquet(_cache_paths(build_cache(cached))["meta"])
    merged = meta[meta.annotation_raw == "TUM"]

    assert not merged.empty
    assert set(merged["annotation"]) == {"Tumor"}


def test_a_second_run_reuses_the_cache_rather_than_extracting_again(cached, capsys):
    from vgtfm.data.tables import _cache_paths, build_cache

    build_cache(cached)
    paths = _cache_paths(cached.cache_dir())
    stamped = paths["gene"].stat().st_mtime_ns
    capsys.readouterr()

    build_cache(cached)
    assert "cache hit" in capsys.readouterr().out
    assert paths["gene"].stat().st_mtime_ns == stamped


def test_force_rebuild_re_extracts_even_on_a_hit(cached, capsys):
    from vgtfm.data.tables import build_cache

    build_cache(cached)
    capsys.readouterr()
    build_cache(cached, force=True)
    assert "cache hit" not in capsys.readouterr().out


def test_the_cache_records_the_source_it_was_built_from(cached):
    from vgtfm.data.tables import _CACHE_SCHEMA, _cache_paths, build_cache

    info = json.loads(_cache_paths(build_cache(cached))["info"].read_text())
    assert info["source"] == str(cached.data_path())
    assert info["schema"] == _CACHE_SCHEMA
    assert info["n_spots"] == 7 and info["gene_dim"] == 3 and info["patch_dim"] == 2


def test_a_merged_dataset_missing_a_required_column_is_refused(tmp_path):
    from vgtfm.data.tables import build_cache

    src = _merged_dataset(tmp_path / "merged", missing=("annotation",))
    cfg = load_config(
        None,
        {
            "paths.data_root": str(tmp_path),
            "paths.artifact_root": str(tmp_path / "artifacts"),
            "paths.merged_datasets.geneformer": str(src),
        },
    )
    with pytest.raises(SystemExit, match="annotation"):
        build_cache(cfg)


def test_a_merged_dataset_that_is_not_there_names_the_stage_that_makes_one(tmp_path):
    from vgtfm.data.tables import build_cache

    cfg = load_config(
        None,
        {
            "paths.data_root": str(tmp_path),
            "paths.artifact_root": str(tmp_path / "artifacts"),
            "paths.merged_datasets.geneformer": str(tmp_path / "absent"),
        },
    )
    with pytest.raises(SystemExit, match="embed stage"):
        build_cache(cfg)


def test_requesting_a_split_the_dataset_does_not_have_is_refused(cached):
    from vgtfm.data.tables import build_cache

    cached.data.splits = ("validation",)
    with pytest.raises(SystemExit, match="validation"):
        build_cache(cached)


def test_loading_applies_the_config_filters_to_features_and_metadata_alike(cached):
    from vgtfm.data.tables import load

    cached.data.include_samples = ("KC1",)
    table = load(cached)

    assert set(table.sample_id) == {"KC1"}
    assert table.gene.shape[0] == table.patch.shape[0] == table.n
    assert table.substrate == "geneformer"


def test_loading_metadata_alone_gives_the_same_rows_without_the_features(cached):
    """The figures stage reads labels and slide ids for hundreds of thousands of
    spots; materialising several GB of float32 to do it would be waste."""
    from vgtfm.data.tables import load, load_meta

    cached.data.include_samples = ("MACEGEJ-1-1",)
    assert load_meta(cached)["spot_id"].tolist() == load(cached).meta["spot_id"].tolist()
    assert len(load_meta(cached)) == 3


def test_an_unfiltered_load_returns_every_spot(cached):
    from vgtfm.data.tables import load, load_meta

    assert load(cached).n == 7 and len(load_meta(cached)) == 7


# ── class support floor ──────────────────────────────────────────────


def _floor_meta() -> pd.DataFrame:
    """One organ with a well-supported class and one carried by a single slide."""
    rows = (
        [("lung", "L1", "Tumor")] * 4
        + [("lung", "L2", "Tumor")] * 4
        + [("lung", "L1", "LN")] * 2
        + [("lung", "L1", "UNASSIGNED")] * 2
    )
    return pd.DataFrame(rows, columns=["tissue", "sample_id", "annotation"])


def test_the_class_floor_demotes_a_thin_class_to_unlabelled():
    from vgtfm.data.tables import apply_class_floor
    from vgtfm.labels import BELOW_FLOOR, labeled_mask

    cfg = load_config(None, {"data.min_class_spots": "3", "data.min_class_slides": "2"})
    out = apply_class_floor(cfg, _floor_meta(), verbose=False)

    assert set(out.loc[out.annotation_floored != "", "annotation_floored"]) == {"LN"}
    assert set(out.loc[out.annotation_floored != "", "annotation"]) == {BELOW_FLOOR}
    # 8 Tumor spots survive; LN and UNASSIGNED are both unlabelled now.
    assert int(labeled_mask(out["annotation"].to_numpy()).sum()) == 8


def test_the_class_floor_leaves_a_supported_class_alone():
    from vgtfm.data.tables import apply_class_floor

    cfg = load_config(None, {"data.min_class_spots": "1", "data.min_class_slides": "1"})
    out = apply_class_floor(cfg, _floor_meta(), verbose=False)

    assert set(out["annotation_floored"]) == {""}
    assert set(out.loc[out.annotation != "UNASSIGNED", "annotation"]) == {"Tumor", "LN"}


def test_the_class_floor_column_is_present_even_when_nothing_is_dropped():
    """Consumers test the column, not whether the floor happened to bite."""
    from vgtfm.data.tables import apply_class_floor

    cfg = load_config(None, {"data.min_class_spots": "0", "data.min_class_slides": "0"})
    assert "annotation_floored" in apply_class_floor(cfg, _floor_meta(), verbose=False).columns


def test_the_shipped_default_floor_would_keep_the_usz_minority_classes():
    """Guards the threshold against the real cohort's smallest scored classes.

    `kidney | INFL` at 164 spots on 2 slides is the tightest class the headline
    run keeps; `lung | LN` at 33 spots on 1 slide is the one it drops.
    """
    from vgtfm.labels import class_floor_mask

    cfg = load_config(None, {})
    ann = np.array(["INFL"] * 164 + ["LN"] * 33, dtype=object)
    tis = np.array(["kidney"] * 164 + ["lung"] * 33, dtype=object)
    sid = np.array(["K1"] * 82 + ["K2"] * 82 + ["L1"] * 33, dtype=object)
    _, dropped = class_floor_mask(
        ann, tis, sid, min_spots=cfg.data.min_class_spots, min_slides=cfg.data.min_class_slides
    )
    assert [d["annotation"] for d in dropped] == ["LN"]
