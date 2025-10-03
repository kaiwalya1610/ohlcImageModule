"""I/O utilities for persisting generated images and metadata."""
from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd
from PIL import Image


def save_image(image: Image.Image, path: str) -> None:
    """Persist the PIL image to disk as PNG."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path, format="PNG")


def write_metadata(rows: List[Dict[str, object]], path: str) -> None:
    """Persist metadata rows to a CSV file."""

    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def ensure_outdirs(out_dir: str) -> Dict[str, str]:
    """Create output directories if they do not exist."""

    os.makedirs(out_dir, exist_ok=True)
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    return {"root": out_dir, "images": images_dir}
