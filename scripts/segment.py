"""
Aim: Given a target image, build multiple semi-transparent layers that,
when stacked together over bright light, reveal the image.

This script: Use opaque layers like stickers on clear acrylic, where
you see the top-most painted pixel. Put the fine detail on the bottom layer
where it is more easily 'hidden'. For the other layers, group the changes
by colour in an attempt to make it easier to hide.

Every layer starts empty (fully clear). Then:
1. Find the fine detail: blur the target more and more until enough pixels
   change noticeably.
2. Paint the fine detail onto the bottom layer, and cut it out of the target.
   What's left is the smoother parts of the image.
3. Reduce what's left to a small number of flat colours, and score each colour
   by how busy it is: how many separate patches it breaks into.
4. Deal whole colours out to the remaining layers, least busy on the top layer
   and busiest lower down. Each layer shows flat colour shapes rather than a
   recognisable part of the picture.
5. Mark where each layer could be filled in. Wherever a higher layer is
   painted, anything on this layer is hidden, so it can be filled freely
   (bright green). Wherever a lower layer is painted, this layer must stay
   clear or it would cover it (left transparent). The bottom layer can be
   filled everywhere it isn't painted; the top layer can't be filled at all.

Notes:
I like how putting the detail on the bottom makes it easier to hide.
Top layer looks like nothing and has no options for making it better.
"""

from pathlib import Path

import numpy as np

from all_things_ones.core import (
    combine_layers_by_transparency,
    find_fillable,
    resize_image,
)
from all_things_ones.files import clear_folder, load_image, save_image
from all_things_ones.segmentation import (
    extract_frequency_layer,
    segment_image_to_colours,
)

target_file = "data/input/target/cats.png"
num_images = 4
img_size = 2000
num_colours = 12
fillable_colour = (0.0, 1.0, 0.0, 1.0)

output_folder = Path("data/output/segment")


def main():
    clear_folder(output_folder)
    target_img = load_image(target_file)[:, :, :3]
    target_img = resize_image(target_img, (img_size, img_size, 3))
    save_image(target_img, output_folder / "target.png")

    detail_layer, target_without_detail = create_detail_layer(target_img)
    colour_layers = create_colour_layers(target_without_detail)
    layers = [detail_layer] + colour_layers

    mark_fillable(layers)

    save_image(combine_layers_by_transparency(layers), output_folder / "stack.png")
    for i, layer in enumerate(layers):
        save_image(layer, output_folder / f"layer_{i}.png")
    print(f"Saved to {output_folder}")


def create_detail_layer(target_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask, blurred_img = extract_frequency_layer(
        target_img, mask_pct_threshold=100.0 / (num_images + 2)
    )
    save_image(blurred_img, output_folder / "1_blurred.png")

    detail_layer = np.zeros((img_size, img_size, 4), dtype=np.float32)
    detail_layer[mask, :3] = target_img[mask]
    detail_layer[mask, 3] = 1.0
    save_image(detail_layer, output_folder / "2_detail.png")

    target_without_detail = np.concatenate(
        [target_img, np.ones((img_size, img_size, 1), dtype=np.float32)], axis=2
    )
    target_without_detail[mask, 3] = 0
    save_image(target_without_detail, output_folder / "2_target_without_detail.png")
    return detail_layer, target_without_detail


def create_colour_layers(target_without_detail: np.ndarray) -> list[np.ndarray]:
    labels, colours, polygon_counts = segment_image_to_colours(
        target_without_detail, num_colours
    )
    flat_colours = np.zeros_like(target_without_detail)
    painted = labels >= 0
    flat_colours[painted, :3] = colours[labels[painted]]
    flat_colours[painted, 3] = 1.0
    save_image(flat_colours, output_folder / "3_colours.png")

    # Fewer polygons = less busy = top layer first
    polygon_counts.sort(key=lambda x: x[1])

    num_layers = num_images - 1
    colours_per_layer, extra_colours = divmod(len(polygon_counts), num_layers)

    layers = []
    colour_idx = 0
    for layer_num in range(num_layers):
        layer = np.zeros((img_size, img_size, 4), dtype=np.float32)
        num_colours_for_layer = colours_per_layer + (
            1 if layer_num < extra_colours else 0
        )
        for kmeans_idx, _, _ in polygon_counts[
            colour_idx : colour_idx + num_colours_for_layer
        ]:
            mask = labels == kmeans_idx
            layer[mask] = target_without_detail[mask]
        colour_idx += num_colours_for_layer
        layers.append(layer)

    # Least busy goes on top, which is last in the stack
    return layers[::-1]


def mark_fillable(layers: list[np.ndarray]) -> None:
    for i, (layer, fillable) in enumerate(zip(layers, find_fillable(layers))):
        marked = layer.copy()
        marked[fillable] = fillable_colour
        save_image(marked, output_folder / f"5_fillable_layer_{i}.png")
        print(f"Layer {i}: {fillable.mean():.1%} can be filled in")


if __name__ == "__main__":
    main()
