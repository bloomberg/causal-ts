# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

#!/usr/bin/env python3
"""Extract a plot from each notebook as a thumbnail for the docs gallery.

Usage:
    python docs/scripts/generate_thumbnails.py

Selection priority (highest to lowest):
  1. Manual override — place your own PNG in docs/_static/img/thumbnails/overrides/
     named {notebook_stem}_thumb.png.  It will be resized but never overwritten.
  2. THUMBNAIL_INDEX — set a per-notebook image index (0-based) below.
     Index counts only code-cell outputs that contain image/png data.
  3. Default — first image output found in the notebook.

Requires Pillow: pip install Pillow
"""

import base64
import json
from io import BytesIO
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
THUMB_DIR = REPO_ROOT / "docs" / "_static" / "img" / "thumbnails"
OVERRIDE_DIR = THUMB_DIR / "overrides"
THUMB_SIZE = (400, 300)

NOTEBOOK_SOURCES = [
    REPO_ROOT / "tutorial.ipynb",
    *sorted((REPO_ROOT / "examples").glob("*.ipynb")),
]

# Per-notebook image index (0 = first image, 1 = second, etc.).
# Only entries listed here override the default (first image).
THUMBNAIL_INDEX: dict[str, int] = {
    # "tutorial": 2,           # use the 3rd image in tutorial.ipynb
    # "grace_high_dim": 1,     # use the 2nd image
    "multi_c_nonstationarity": 4,
    "cedar_benchmark": 1,       # use the scaling line-plot (2nd image) instead of the compare_graphs
}

BG_LIGHT = (248, 248, 253)


def extract_image(nb_path: Path, index: int = 0) -> bytes | None:
    with open(nb_path) as f:
        nb = json.load(f)

    found = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") in ("display_data", "execute_result"):
                img_b64 = output.get("data", {}).get("image/png")
                if img_b64:
                    if found == index:
                        if isinstance(img_b64, list):
                            img_b64 = "".join(img_b64)
                        return base64.b64decode(img_b64)
                    found += 1
    return None


def make_thumbnail(raw: bytes, size: tuple[int, int] = THUMB_SIZE) -> Image.Image:
    img = Image.open(BytesIO(raw))
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, BG_LIGHT)
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    img.thumbnail(size, Image.LANCZOS)

    canvas = Image.new("RGB", size, BG_LIGHT)
    offset = ((size[0] - img.width) // 2, (size[1] - img.height) // 2)
    canvas.paste(img, offset)
    return canvas


def main():
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    OVERRIDE_DIR.mkdir(parents=True, exist_ok=True)

    generated = []
    overrides_used = []
    skipped = []

    for nb_path in NOTEBOOK_SOURCES:
        if not nb_path.exists():
            skipped.append(nb_path.name)
            continue

        stem = nb_path.stem
        out = THUMB_DIR / f"{stem}_thumb.png"

        override = OVERRIDE_DIR / f"{stem}_thumb.png"
        if override.exists():
            thumb = make_thumbnail(override.read_bytes())
            thumb.save(out, optimize=True)
            overrides_used.append(stem)
            continue

        idx = THUMBNAIL_INDEX.get(stem, 0)
        raw = extract_image(nb_path, index=idx)
        if raw is None:
            skipped.append(f"{nb_path.name} (no image at index {idx})")
            continue

        thumb = make_thumbnail(raw)
        thumb.save(out, optimize=True)
        generated.append(f"{out.name} (image #{idx})")

    print(f"Generated {len(generated)} thumbnails in {THUMB_DIR}")
    for name in generated:
        print(f"  {name}")
    if overrides_used:
        print(f"\nUsed {len(overrides_used)} manual overrides from {OVERRIDE_DIR}:")
        for name in overrides_used:
            print(f"  {name}")
    if skipped:
        print(f"\nSkipped {len(skipped)}:")
        for name in skipped:
            print(f"  {name}")


if __name__ == "__main__":
    main()
