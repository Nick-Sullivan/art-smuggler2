"""
Aim: Given a target image, build multiple semi-transparent layers that,
when stacked together over bright light, reveal the image.

This script: Start each layer from a colourful AI-generated pattern that only
carries a small, scattered part of the target, then nudge the patterns until
the stack reveals the target.

These steps have been done before running the script:
1. Shatter the target into many small pieces, like a smashed windscreen, and
   deal the pieces out between the layers.
2. Manual: give each layer's pieces to an AI image generator as a reference,
   and ask for a colourful pattern that mimics its shapes. Save them to seed_folder.
   These patterns are the starting point for each layer, and act as its camouflage.

When the script runs:
3. Set the goal: the target squeezed into a narrower brightness range, with
   the patterns brightened. Stacking can only darken, so this leaves the
   layers room to move both lighter and darker.
4. Repeatedly nudge the layers towards the goal:
   a. Subtract what the stack currently shows from the goal.
   b. Shatter that difference with a fresh set of pieces, and give each layer
      only the corrections for its own pieces. This way no single layer ends
      up carrying a faint copy of the whole target.
   c. Stop once the stack is close enough, or it stops improving.

Notes:
Does a good job of hiding the target, but the final stack has 'bits' in it.
The combined image is only visible in real life with a bright white backing in a dark room.
"""

import time
from pathlib import Path

import numpy as np

from all_things_ones.core import (
    adjust_image_brightness,
    brighten_image,
    combine_images,
    resize_image,
)
from all_things_ones.files import clear_folder, load_image, save_image
from all_things_ones.shatter import shatter_image_with_cache

target_file = "data/input/target/cats.png"
seed_folder = Path("data/input/seed/005_london")
seed_suffix = "1"
seed = 1
num_images = 4
num_pieces = 500
img_size = 2000
tolerance = 0.05
min_increment = 0.01
stale_iterations = 11
save_every = 5

output_folder = Path("data/output/shatter")
setup_folder = output_folder / "0_setup"


def main():
    clear_folder(output_folder)
    target_img = load_image(target_file)
    target_img = resize_image(target_img, (img_size, img_size, 3))
    save_image(target_img, output_folder / "target.png")

    shatter_target(target_img)
    # Manual: generate an AI image from each layer's pieces and save as
    # {seed_folder}/shattered_piece_XX_gen_{seed_suffix}.jpg
    output_imgs = refine_with_seeds(target_img)

    save_image(combine_images(output_imgs), output_folder / "stack.png")
    for i, output_img in enumerate(output_imgs):
        save_image(output_img, output_folder / f"layer_{i}.png")
    print(f"Saved to {output_folder}")


def shatter_target(target_img: np.ndarray) -> None:
    pieces = shatter_image_with_cache(
        target_img,
        num_pieces=num_pieces,
        num_images=num_images,
        seed=seed,
        background_color=1,
    )
    for i, piece in enumerate(pieces):
        save_image(piece, setup_folder / f"1_pieces_layer_{i}.png")


def refine_with_seeds(target_img: np.ndarray) -> list[np.ndarray]:
    start_time = time.time()
    goal_img = adjust_image_brightness(target_img, min_=0.2, max_=0.8)
    save_image(goal_img, setup_folder / "3_goal.png")

    output_imgs = []
    for i in range(num_images):
        seed_img = load_image(
            seed_folder / f"shattered_piece_{i:02d}_gen_{seed_suffix}.jpg"
        )
        seed_img = resize_image(seed_img, goal_img.shape)
        save_image(seed_img, setup_folder / f"2_seed_layer_{i}.png")
        seed_img = brighten_image(seed_img, min_brightness=0.4)
        save_image(seed_img, setup_folder / f"3_brightened_seed_layer_{i}.png")
        output_imgs.append(seed_img)

    iteration = 0
    needs_iteration = True
    diff_indicator_history = []
    while needs_iteration:
        iteration += 1
        needs_iteration = False
        combined = combine_images(output_imgs)
        img_diff = goal_img - combined
        folder = None
        if iteration == 1 or iteration % save_every == 0:
            folder = output_folder / f"iteration_{iteration:03d}"
            save_image(combined, folder / "4a_stack.png")
            save_image(0.5 + img_diff, folder / "4a_goal_minus_stack.png")

        diff_indicator = np.round(np.abs(img_diff).mean(), 3)
        diff_indicator_history.append(diff_indicator)
        if len(diff_indicator_history) >= stale_iterations and all(
            x == diff_indicator_history[-1]
            for x in diff_indicator_history[-stale_iterations:]
        ):
            print(f"No improvement in last {stale_iterations} iterations, stopping")
            break
        print(
            f"Iteration {iteration}, diff: {diff_indicator: .3f}, time: {time.time() - start_time:.2f}s"
        )
        start_time = time.time()

        fragments_pos = shatter_image_with_cache(
            img_diff.clip(min=0),
            num_pieces=num_pieces,
            num_images=num_images,
            seed=iteration // 5,
            background_color=0,
        )
        fragments_neg = shatter_image_with_cache(
            img_diff.clip(max=0),
            num_pieces=num_pieces,
            num_images=num_images,
            seed=100 * (iteration // 5),
            background_color=1,
        )

        for i in range(num_images):
            correction = np.zeros_like(output_imgs[i])
            # too dark
            fragment_pos = fragments_pos[i]
            pos_indices = np.where(fragment_pos > tolerance)
            correction[pos_indices] = np.maximum(
                min_increment, np.abs(fragment_pos[pos_indices]) / 2
            )
            # too bright
            fragment_neg = fragments_neg[i]
            neg_indices = np.where(fragment_neg < -tolerance)
            correction[neg_indices] = -np.maximum(
                min_increment, np.abs(fragment_neg[neg_indices]) / 2
            )

            if len(pos_indices[0]) > 0 or len(neg_indices[0]) > 0:
                needs_iteration = True
            output_imgs[i] = np.clip(output_imgs[i] + correction, 0, 1)
            if folder:
                save_image(0.5 + correction, folder / f"4b_correction_layer_{i}.png")
                save_image(output_imgs[i], folder / f"4b_layer_{i}.png")

    return output_imgs


if __name__ == "__main__":
    main()
