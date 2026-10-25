"""
Aim: Given a target image, build multiple semi-transparent layers that,
when stacked together over bright light, reveal the image.

This script: Split the image into blobs and give each blob a light random
colour on every layer. Then slowly adjust the layers until the stack matches
the target. Each adjustment goes to one randomly chosen layer, and the choice
changes every few rounds, so the picture is spread thinly across all layers.

1. Load the target and resize it.
2. Find the busy parts of the image (edges, lace, leaves) by applying blurring.
   Blurring changes these a lot, and barely changes the smooth parts.
3. Group the pixels by into bands, from busiest to smoothest.
4. Smooth the bands into rounded regions.
5. Cut each region into random blobs, each owned by one layer. Busy regions
   get small blobs, smooth regions get big ones.
6. Set the goal: the target with its blacks lifted and its whites
   dimmed. Stacking layers can only make things darker, so the
   layers need this room for their colours.
7. Give every blob on every layer a random light colour.
8. Add decoys: copy the target's fine detail onto each layer, turned, flipped
   and moved, mostly in the smooth areas. The real detail is then not the only
   detail on a layer.
9. Repeat until the stack matches the goal, or for max_iterations rounds:
   a. Compare the stack with the goal.
   b. Split the image into random patches, as in step 5. In each patch, one
      layer moves a small step towards the goal.
   c. Pick new patches every reshuffle_every rounds. Picking often spreads
      the picture thinly, so every layer shows a faint copy. Picking rarely
      puts stronger pieces of the picture on fewer layers.
10. Score how much of the target shows in each layer, from 0 (none) to 1
    (a faint copy). Lower is better. The score is taken at each blur in
    leak_sigmas: a high score after heavy blurring means the target's light
    and dark areas show, a high score with no blur means its detail shows.
"""

from pathlib import Path

import cv2
import numpy as np

from all_things_ones.core import (
    adjust_image_brightness,
    combine_images,
    low_pass_filter,
    resize_image,
)
from all_things_ones.files import clear_folder, load_image, save_image

target_file = "data/input/target/bride2.jpg"
img_width = 1831
img_height = 1000
num_images = 4
num_bands = 4
sigma_values = [1, 2, 4, 6, 10, 15, 20, 40, 80]
blob_sigma = 15
goal_min = 0.2
goal_max = 0.8
off_white_min = 0.9
decoy_sigma = 8
decoy_strength = 1.0
step_size = 0.2
max_iterations = 100
tolerance = 0.00001
reshuffle_every = 5
save_every = 10
leak_sigmas = [0, 8, 32]
seed = 1
palette = [(0.9, 0.3, 0.2), (0.2, 0.7, 0.3), (0.2, 0.4, 0.9), (0.9, 0.8, 0.2)]

output_folder = Path("data/output/blob_levels")


def main():
    clear_folder(output_folder)
    rng = np.random.default_rng(seed)
    target_img = load_image(target_file)[:, :, :3]
    target_img = resize_image(target_img, (img_height, img_width, 3))
    save_image(target_img, output_folder / "1_target.png")

    fineness = measure_fineness(target_img)
    bands = split_into_bands(fineness)
    blobs = smooth_into_blobs(bands)
    blob_ids = split_into_blobs(blobs, rng)
    goal_img = adjust_image_brightness(target_img, min_=goal_min, max_=goal_max)
    save_image(goal_img, output_folder / "6_goal.png")
    layers = tint_blobs(blob_ids, rng)
    layers = add_decoys(layers, goal_img, fineness, rng)
    layers = nudge_towards_goal(layers, blobs, goal_img, rng)

    stack = combine_images(list(layers))
    print(
        f"Max difference between stack and goal: {np.abs(stack - goal_img).max():.4f}"
    )
    save_image(stack, output_folder / "stack.png")
    for i, layer in enumerate(layers):
        save_image(layer, output_folder / f"layer_{i}.png")
    measure_leak(layers, goal_img)
    print(f"Saved to {output_folder}")


def measure_fineness(target_img: np.ndarray) -> np.ndarray:
    fineness = np.zeros(target_img.shape[:2], dtype=np.float32)
    for sigma in sigma_values:
        change = np.abs(low_pass_filter(target_img, sigma=sigma) - target_img)
        fineness += change.max(axis=2) / sigma

    ranked = fineness.flatten().argsort().argsort().reshape(fineness.shape)
    save_image(ranked / ranked.max(), output_folder / "2_fineness.png")
    return fineness


def split_into_bands(fineness: np.ndarray) -> np.ndarray:
    ranked = fineness.flatten().argsort().argsort().reshape(fineness.shape)
    smoothness = ranked / ranked.size
    bands = np.minimum((smoothness * num_bands).astype(int), num_bands - 1)
    bands = num_bands - 1 - bands
    save_image(colour_map(bands), output_folder / "3_bands.png")
    return bands


def smooth_into_blobs(bands: np.ndarray) -> np.ndarray:
    votes = np.stack(
        [
            low_pass_filter((bands == i).astype(np.float32), sigma=blob_sigma)
            for i in range(num_bands)
        ],
        axis=2,
    )
    blobs = votes.argmax(axis=2)
    save_image(colour_map(blobs), output_folder / "4_regions.png")
    return blobs


def split_into_blobs(blobs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    owners = random_owners(blobs, rng)
    blob_ids = np.zeros(blobs.shape, dtype=np.int32)
    num_blobs = 0
    for label in np.unique(blobs * num_images + owners):
        count, labels = cv2.connectedComponents(
            (blobs * num_images + owners == label).astype(np.uint8)
        )
        mask = labels > 0
        blob_ids[mask] = labels[mask] - 1 + num_blobs
        num_blobs += count - 1

    print(f"{num_blobs} blobs")
    blob_colours = np.random.default_rng(0).random((num_blobs, 3))
    save_image(blob_colours[blob_ids], output_folder / "5_blobs.png")
    return blob_ids


def tint_blobs(blob_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    num_blobs = blob_ids.max() + 1
    layers = np.stack(
        [
            rng.uniform(off_white_min, 1.0, (num_blobs, 3)).astype(np.float32)[blob_ids]
            for _ in range(num_images)
        ]
    )
    for i, layer in enumerate(layers):
        save_image(layer, output_folder / f"7_tinted_layer_{i}.png")
    return layers


def add_decoys(
    layers: np.ndarray,
    goal_img: np.ndarray,
    fineness: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    log_goal = np.log(goal_img)
    detail = log_goal - low_pass_filter(log_goal, sigma=decoy_sigma)
    ranked = fineness.flatten().argsort().argsort().reshape(fineness.shape)
    smoothness = (1 - ranked / ranked.size)[:, :, None]

    for i, layer in enumerate(layers):
        decoy = np.rot90(detail, rng.integers(1, 4))
        if rng.random() < 0.5:
            decoy = decoy[:, ::-1]
        # Turning a non-square image swaps its sides, so tile it and crop to fit
        reps = (-(-img_height // decoy.shape[0]), -(-img_width // decoy.shape[1]), 1)
        decoy = np.tile(decoy, reps)[:img_height, :img_width]
        decoy = np.roll(
            decoy,
            (rng.integers(img_height), rng.integers(img_width)),
            axis=(0, 1),
        )
        layers[i] = np.clip(layer * np.exp(decoy_strength * decoy * smoothness), 0, 1)
        save_image(layers[i], output_folder / f"8_decoy_layer_{i}.png")
    return layers


def nudge_towards_goal(
    layers: np.ndarray,
    blobs: np.ndarray,
    goal_img: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    rows, cols = np.indices(blobs.shape)
    for iteration in range(1, max_iterations + 1):
        stack = layers.prod(axis=0)
        diff = np.abs(goal_img - stack).mean()
        print(f"Iteration {iteration}, diff: {diff:.5f}")
        if diff < tolerance:
            break

        if (iteration - 1) % reshuffle_every == 0:
            owners = random_owners(blobs, rng)
        correction = np.exp(step_size * (np.log(goal_img) - np.log(stack)))
        layers[owners, rows, cols] = np.clip(
            layers[owners, rows, cols] * correction, 0, 1
        )

        if iteration == 1 or iteration % save_every == 0:
            folder = output_folder / f"iteration_{iteration:03d}"
            save_image(stack, folder / "9a_stack.png")
            save_image(0.5 + goal_img - stack, folder / "9a_goal_minus_stack.png")
            save_image(colour_map(owners), folder / "9b_owners.png")
            for i, layer in enumerate(layers):
                save_image(layer, folder / f"9b_layer_{i}.png")
    return layers


def measure_leak(layers: np.ndarray, goal_img: np.ndarray) -> None:
    log_goal = np.log(goal_img.mean(axis=2))
    for sigma in leak_sigmas:
        blurred_goal = low_pass_filter(log_goal, sigma) if sigma else log_goal
        scores = []
        for layer in layers:
            log_layer = np.log(np.maximum(layer.mean(axis=2), 1e-3))
            if sigma:
                log_layer = low_pass_filter(log_layer, sigma)
            scores.append(np.corrcoef(log_layer.ravel(), blurred_goal.ravel())[0, 1])
        print(
            f"Leak at blur {sigma:2d}: {np.mean(scores):.2f} "
            f"(layers {', '.join(f'{s:.2f}' for s in scores)})"
        )
    print(f"Stack brightness: {layers.prod(axis=0).mean():.2f}")


def random_owners(blobs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    owners = np.zeros(blobs.shape, dtype=np.int32)
    for band in range(num_bands):
        mask = blobs == band
        owners[mask] = smoothed_noise_vote(blob_sigma * (band + 1), rng)[mask]
    return owners


def smoothed_noise_vote(sigma: float, rng: np.random.Generator) -> np.ndarray:
    small = (img_height // 4, img_width // 4)
    votes = np.stack(
        [
            low_pass_filter(rng.random(small, dtype=np.float32), sigma / 4)
            for _ in range(num_images)
        ],
        axis=2,
    )
    votes = cv2.resize(votes, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
    return votes.argmax(axis=2)


def colour_map(indices: np.ndarray) -> np.ndarray:
    return np.array(palette)[indices % len(palette)]


if __name__ == "__main__":
    main()
