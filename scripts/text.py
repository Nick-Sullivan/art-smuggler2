"""
Aim: Given a target image, build multiple semi-transparent layers that,
when stacked together over bright light, reveal the image.

This script: Show the concept with a simple target, a word made entirely of
black blobs, split up across opaque layers like stickers on clear acrylic,
where you see the top-most painted pixel. Each layer looks like random dots,
but the stack spells the word.

Every layer starts empty (fully clear). Then:
1. Lay the word out on a grid. Each letter is a small 5x5 pattern of filled
   cells, with a gap between letters. With a round blob drawn on every filled
   cell, this is the target.
2. Put a round blob on each filled cell, on a randomly chosen layer. Blobs are
   slightly bigger than a cell, so neighbouring blobs join up into strokes
   once the layers are stacked, while the gap keeps letters apart.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from all_things_ones.core import combine_layers_by_transparency
from all_things_ones.files import clear_folder, save_image
from all_things_ones.text import get_letter_patterns

text = "HELLO"
img_size = 2000
num_images = 4
blob_scale = 1.3
seed = 1

output_folder = Path("data/output/text")


def main():
    clear_folder(output_folder)
    cells, cell_size = layout_cells()
    layers = draw_blobs(cells, cell_size)

    save_image(combine_layers_by_transparency(layers), output_folder / "stack.png")
    for i, layer in enumerate(layers):
        save_image(layer, output_folder / f"layer_{i}.png")
    print(f"Saved to {output_folder}")


def layout_cells() -> tuple[list[tuple[int, int]], int]:
    patterns = get_letter_patterns()
    letter_width = 6
    cell_size = img_size // (len(text) * letter_width + 1)
    left = (img_size - (len(text) * letter_width - 1) * cell_size) // 2
    top = (img_size - 5 * cell_size) // 2

    cells = []
    for i, char in enumerate(text.upper()):
        for row, col in patterns.get(char, []):
            x = left + (i * letter_width + col) * cell_size
            y = top + row * cell_size
            cells.append((x, y))

    target_img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(target_img)
    for x, y in cells:
        draw_blob(draw, x, y, cell_size, fill=(0, 0, 0))
    save_image(np.array(target_img) / 255.0, output_folder / "1_target.png")
    print(f"{len(cells)} cells, {cell_size}px each")
    return cells, cell_size


def draw_blobs(cells: list[tuple[int, int]], cell_size: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    imgs = [
        Image.new("RGBA", (img_size, img_size), (255, 255, 255, 0))
        for _ in range(num_images)
    ]
    draws = [ImageDraw.Draw(img) for img in imgs]
    for x, y in cells:
        layer_idx = rng.integers(num_images)
        draw_blob(draws[layer_idx], x, y, cell_size, fill=(0, 0, 0, 255))

    layers = [np.array(img).astype(np.float32) / 255.0 for img in imgs]
    for i, layer in enumerate(layers):
        save_image(layer, output_folder / f"2_blobs_layer_{i}.png")
    return layers


def draw_blob(
    draw: ImageDraw.ImageDraw, x: int, y: int, cell_size: int, fill: tuple
) -> None:
    radius = cell_size * blob_scale / 2
    cx = x + cell_size / 2
    cy = y + cell_size / 2
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=fill)


if __name__ == "__main__":
    main()
