"""
Aim: Given a target image, build multiple semi-transparent layers that,
when stacked together over bright light, reveal the image.

This script: Build the layers using coarse-to-fine approach, where each
pass adds more detail to the layers.

Every layer starts blank (white, so it lets all light through). Then the steps
below repeat once per blur level, going from very blurry to sharp:
1. Set a goal for this pass: a blurred copy of the target.
2. Subtract what the stack currently shows from the goal. This shows what
   adjustments need to be made for this pass.
3. Group that difference into blobs: patches of similar colour that sit close
   together.
4. Fix each blob in turn:
   a. Paint the blob onto every layer, plus a faint copy on the opposite side
      of the image. The copy is a decoy, so each layer shows shapes that
      aren't in the target, which makes the real ones harder to pick out.
   b. Adjust the layers until the stack matches the goal in that area. The
      layers take turns, and a different layer sits out for each blob, so no
      single layer ends up carrying the picture.
   c. A layer can't let through more light than clear film. If the stack
      needs to be brighter than that allows, dim the target a little and
      carry on. This is why results come out darker than the original.

Notes:
I like how it creates the natural shapes and blobs.
The final result isn't great, can clearly see the target image as a ghost in each layer.
Final image is very dark.

"""

from pathlib import Path

import numpy as np

from all_things_ones.blob import detect_blobs, tessellate_blob, visualise_blobs
from all_things_ones.core import (
    combine_images,
    darken_image_pct,
    low_pass_filter,
    resize_image,
)
from all_things_ones.files import clear_folder, load_image, save_image

target_file = "data/input/target/cats_chair.png"
img_size = 1000
num_images = 3
sigma_values = [80, 40, 20, 15, 10, 6, 4, 2, 1, 0]

output_folder = Path("data/output/pyramid")


def main():
    clear_folder(output_folder)
    target_img = load_image(target_file)
    target_img = resize_image(target_img, (img_size, img_size, 3))
    save_image(target_img, output_folder / "target.png")
    canvases = [
        np.ones((img_size, img_size, 3), dtype=np.float32) for _ in range(num_images)
    ]
    max_brightness = 1.0
    for pass_num, sigma in enumerate(sigma_values, start=1):
        print(f"--- pass {pass_num}, blur level {sigma} ---")
        folder = output_folder / f"pass_{pass_num:02d}_blur_{sigma:02d}"
        target_img, canvases, max_brightness = apply_sigma(
            target_img, canvases, sigma, max_brightness, folder
        )

    save_image(combine_images(canvases), output_folder / "stack.png")
    for i, canvas in enumerate(canvases):
        save_image(canvas, output_folder / f"layer_{i}.png")
    print(f"Saved to {output_folder}")


def apply_sigma(
    target_img: np.ndarray,
    canvases: list[np.ndarray],
    sigma: int,
    max_brightness: float,
    folder: Path,
) -> tuple[np.ndarray, list[np.ndarray], float]:
    banned_idx = 0
    blurred_img = low_pass_filter(target_img, sigma=sigma)
    save_image(blurred_img, folder / "1_goal.png")

    combined_img = combine_images(canvases)
    inc_target_img = combined_img.copy()
    diff = blurred_img - combined_img
    diff_normalized = (diff - diff.min()) / (diff.max() - diff.min())
    save_image(diff_normalized, folder / "2_goal_minus_stack.png")

    labels, blobs = detect_blobs(diff, exclude_zero_pixels=False)
    num_blobs = len(blobs)
    save_image(visualise_blobs(labels, blobs), folder / "3_blobs.png")

    for i, blob in enumerate(blobs):
        print(f"Processing blob {i + 1}/{num_blobs}")

        inc_target_img[blob.mask] = blurred_img[blob.mask]

        tessellations = tessellate_blob(blob, blurred_img, num_blobs=2)
        for j, (colors, rotated_y, rotated_x) in enumerate(tessellations):
            ratio = 0.6 if j == 0 else 0.1
            for idx in range(len(canvases)):
                canvases[idx][rotated_y, rotated_x] = (
                    canvases[idx][rotated_y, rotated_x] * (1 - ratio) + colors * ratio
                )

        # Apply correction on the other canvases
        canvas_idx = 0
        banned_idx = (banned_idx + 1) % len(canvases)
        for _ in range(100):
            combined = combine_images(canvases)
            img_diff = inc_target_img - combined
            canvas_idx = (canvas_idx + 1) % len(canvases)
            if canvas_idx == banned_idx:
                canvas_idx = (canvas_idx + 1) % len(canvases)

            canvases[canvas_idx] += img_diff * 0.3
            if canvases[canvas_idx].max() > 1.0:
                canvases[canvas_idx] = np.clip(canvases[canvas_idx], 0, 1)
                max_brightness *= 0.9
                inc_target_img = darken_image_pct(inc_target_img, 10)
                blurred_img = darken_image_pct(blurred_img, 10)
                target_img = darken_image_pct(target_img, 10)
                print(
                    f"Darkening target image, max brightness now {max_brightness:.1%}"
                )
                continue

            if np.round(np.abs(img_diff).max(), 3) < 0.01:
                break

    for i, canvas in enumerate(canvases):
        save_image(canvas, folder / f"4_layer_{i}.png")
    save_image(combine_images(canvases), folder / "4_stack.png")
    # The goal may have been dimmed during this pass
    save_image(blurred_img, folder / "1_goal.png")
    return target_img, canvases, max_brightness


if __name__ == "__main__":
    main()
