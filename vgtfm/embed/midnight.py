"""Midnight H&E patch embeddings.

One 224x224 RGB patch is cut from the whole-slide image around each Visium spot
and encoded with Midnight (kaiko-ai). The embedding is the CLS token concatenated
with the mean of the patch tokens, per the model's recommended usage, which is why
the morphology side is 3072-wide throughout.

Patches are cut at the spot's pixel centre in the resolution the alignment refers
to (see :mod:`vgtfm.embed.alignment`). Spots whose patch would fall outside the
image are padded with white rather than dropped, so the output stays row-aligned
with the expression matrix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PATCH_SIZE = 224
EMBEDDING_DIM = 3072
MODEL_NAME = "kaiko-ai/midnight"


def load_model(model_name: str = MODEL_NAME, device: str = "auto"):
    """Load Midnight and its normalisation transform."""
    import torch
    from torchvision import transforms
    from transformers import AutoModel

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return model, transform, torch.device(device)


def cut_patch(image, cx: float, cy: float, size: int = PATCH_SIZE) -> np.ndarray:
    """Crop a ``size x size`` RGB patch centred on ``(cx, cy)``, padding with white.

    Padding rather than dropping keeps the output row-aligned with the expression
    matrix; an edge spot yields a mostly-white patch, which is what it physically is.
    """
    half = size // 2
    h, w = image.shape[:2]
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    patch = np.full((size, size, 3), 255, dtype=np.uint8)

    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + size, w), min(y0 + size, h)
    if sx1 <= sx0 or sy1 <= sy0:
        return patch
    patch[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = image[sy0:sy1, sx0:sx1, :3]
    return patch


def embed_slide(
    image_path: str | Path,
    coords,
    *,
    model,
    transform,
    device,
    batch_size: int = 32,
    amp: bool = True,
) -> np.ndarray:
    """Embed every spot of one slide. ``coords`` needs ``image_x`` / ``image_y``."""
    import torch
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None  # whole-slide TIFFs exceed the guard
    image = np.asarray(Image.open(image_path).convert("RGB"))

    use_amp = amp and device.type == "cuda" and torch.cuda.is_bf16_supported()
    out = np.empty((len(coords), EMBEDDING_DIM), dtype=np.float32)

    for start in range(0, len(coords), batch_size):
        block = coords.iloc[start : start + batch_size]
        tensors = [transform(cut_patch(image, r.image_x, r.image_y)) for r in block.itertuples()]
        batch = torch.stack(tensors).to(device, non_blocking=True)
        with (
            torch.inference_mode(),
            torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp),
        ):
            hidden = model(batch).last_hidden_state
            # CLS token plus the mean over patch tokens, per Midnight's usage notes.
            features = torch.cat([hidden[:, 0, :], hidden[:, 1:, :].mean(dim=1)], dim=-1)
        out[start : start + len(block)] = features.float().cpu().numpy()

    del image
    return out
