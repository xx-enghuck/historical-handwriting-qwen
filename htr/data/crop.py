"""Conservative, deterministic margin cropping with no intensity normalization."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from htr.config import CropConfig


def crop_line(image: Image.Image, cfg: CropConfig, debug_path: Path | None = None) -> Image.Image:
    """Keep the bounding box of every pixel darker than threshold, plus padding.

    Blank pages fall back to the original. No component pruning, stretching,
    deskewing or background normalization is performed; inspect faint ink manually.
    """
    image = image.convert("RGB")
    if not cfg.enabled:
        return image.copy()
    foreground = np.asarray(image.convert("L")) < cfg.threshold
    ys, xs = np.where(foreground)
    if len(xs):
        p = cfg.padding
        box = (
            max(0, int(xs.min()) - p),
            max(0, int(ys.min()) - p),
            min(image.width, int(xs.max()) + p + 1),
            min(image.height, int(ys.max()) + p + 1),
        )
    else:
        box = (0, 0, image.width, image.height)
    if debug_path:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        preview = image.copy()
        ImageDraw.Draw(preview).rectangle((box[0], box[1], box[2] - 1, box[3] - 1), outline="red")
        preview.save(debug_path)
    return image.crop(box)
