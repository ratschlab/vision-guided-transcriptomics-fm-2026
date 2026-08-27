# Isolated environments for the gene-side foundation models

Geneformer, scGPT and CancerFoundation pin mutually incompatible versions of
torch, transformers and flash-attention, so they cannot share an environment with
each other or with `requirements.txt`.

Rather than hide that behind a plugin layer, each runs as a separate process in its
own environment and writes parquet:

```
<artifact_root>/_embed/<model>/<dataset_id>/<sample_id>.parquet
columns: spot_id, e0 .. e{D-1}
```

`vgtfm.embed.build.merge` joins those with the morphology parquets on `spot_id`.
Nothing downstream of the `embed` stage knows which environment produced a feature
matrix, which is what keeps the analysis code free of their dependencies.

The pins here reflect the versions the published embeddings were produced with. If
a model's upstream has moved on, prefer reproducing the pinned version over
upgrading: the embedding is the input to everything else, and a different checkpoint
is a different experiment.
