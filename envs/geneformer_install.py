"""Install Geneformer and its weights into the environment envs/geneformer.yaml made.

    conda run -n vgtfm-geneformer python envs/geneformer_install.py

Geneformer ships as a HuggingFace model repository rather than a package, and the
install line its README gives does not work here — envs/geneformer.yaml records both
reasons. This does the same job through huggingface_hub, which resolves git-lfs
without git-lfs being installed:

    1. fetch the Python package *and its gene dictionaries*, which are lfs blobs and
       are what a smudge-free git clone silently replaces with pointer files;
    2. pip install that directory, with --no-deps so it cannot undo the pins;
    3. fetch the pretrained weights into the cache, which is where
       ``vgtfm.embed.gene_fm._resolve_hf_cache`` looks when no --model-dir is given.

Weights are the repository-root model: 18 layers, hidden size 1152, 4096-token
context, ~1.3 GB. That 1152 is the width the ``gene_features`` column carries and
what ``configs/default.yaml`` sizes its encoder for, so a different variant here
would need that changed with it.

Re-running is cheap: huggingface_hub skips what the cache already has.
"""

from __future__ import annotations

import subprocess
import sys

#: Everything but the weights. ``geneformer/**`` matters as much as the code: the
#: tokenizer reads its gene median, token and Ensembl-mapping dictionaries from
#: inside the installed package.
PACKAGE = ["geneformer/*", "geneformer/**", "setup.py", "MANIFEST.in", "README.md"]

#: The repository-root model. Named exactly, so the same-named files under each
#: variant's own directory are not pulled with it.
WEIGHTS = ["config.json", "model.safetensors", "generation_config.json", "training_args.bin"]

REPO = "ctheodoris/Geneformer"


def main() -> int:
    from huggingface_hub import snapshot_download

    print(f"== {REPO}: package and gene dictionaries")
    package = snapshot_download(REPO, allow_patterns=PACKAGE)

    print(f"== pip install {package}")
    pip = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", package],
        check=False,
    )
    if pip.returncode != 0:
        return pip.returncode

    print(f"== {REPO}: pretrained weights (~1.3 GB)")
    weights = snapshot_download(REPO, allow_patterns=WEIGHTS)

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(weights)
    print(f"\n  weights  {weights}")
    print(
        f"  {config.hidden_size}-d, {config.num_hidden_layers} layers, "
        f"{config.max_position_embeddings}-token context"
    )
    if config.hidden_size != 1152:
        print(
            f"  WARNING this checkpoint is {config.hidden_size}-d, not the 1152 the "
            f"rest of the pipeline is configured for (configs/default.yaml)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
