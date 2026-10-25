"""
Aim: Given a target image, build multiple semi-transparent layers that,
when stacked together over bright light, reveal the image.

This script: Give every layer light blotches of colour and the fine detail of
a decoy picture. Then slowly adjust the layers until the stack matches the
target. Each part of the image is adjusted by one layer only, so each layer
holds scattered pieces of the target instead of a faint copy of all of it.

1. Load the target, and turn it grey if grayscale_target is set.
2. Set the goal: the target with its blacks lifted and its whites
   dimmed. Stacking layers can only make things darker, so the
   layers need this room for their colours.
3. Give every layer random blotches of light colour.
4. Add decoys: copy the fine detail of a different decoy picture onto each
   layer. If there are fewer decoys than layers, they're reused mirrored. The
   target's detail is then never the only detail on a layer.
5. Share the image out between the layers twice: in small patches for its
   detail (fine_patch_size), and in large patches for its broad light and
   dark areas (coarse_patch_size). Each patch belongs to one layer. Keep the
   small patches smaller than a face, so no layer gets a whole one.
6. Repeat until every pixel of the stack matches the goal, or for
   max_iterations rounds:
   a. Compare the stack with the goal, and work out a small step towards it.
   b. Blur the step to split it into its broad part and its detail.
   c. In each patch, its layer takes its part of the step. If the layer is
      already clear and can't get any lighter, the next layer takes what's
      left.
7. Score how much of the target shows in each layer, from 0 (none) to 1
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

# target_file = "data/input/target/london.png"
target_file = "data/input/target/bride2.jpg"
decoy_files = [
    # "data/input/target/mish.png",
    # "data/input/target/trees.png",
    "data/input/decoy/bride2-decoy.jpg",
    "data/input/decoy/bride2-decoy2.jpg",
    "data/input/decoy/bride2-decoy3.jpg",
]
grayscale_target = False
img_width = 1000
num_images = 3
goal_min = 0.2
goal_max = 0.8
off_white_min = 0.9
fine_patch_size = 5
coarse_patch_size = 60
decoy_strength = 1.0
step_size = 0.2
max_iterations = 100
save_every = 1
leak_sigmas = [0, 8, 32]
seed = 2
palette = [(0.9, 0.3, 0.2), (0.2, 0.7, 0.3), (0.2, 0.4, 0.9), (0.9, 0.8, 0.2)]

output_folder = Path("data/output/blob_levels3")

# Anything finer than a quarter of a large patch counts as detail
detail_sigma = coarse_patch_size / 4


def main():
    clear_folder(output_folder)
    rng = np.random.default_rng(seed)
    target_img = load_image(target_file)[:, :, :3]
    img_height = round(img_width * target_img.shape[0] / target_img.shape[1])
    target_img = resize_image(target_img, (img_height, img_width, 3))
    if grayscale_target:
        grey = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        target_img = np.repeat(grey[:, :, None], 3, axis=2)
    save_image(target_img, output_folder / "1_target.png")

    goal_img = adjust_image_brightness(target_img, min_=goal_min, max_=goal_max)
    save_image(goal_img, output_folder / "2_goal.png")
    layers = tint_layers(goal_img.shape[:2], rng)
    layers = add_decoys(layers)
    fine_owners, coarse_owners = share_out(goal_img.shape[:2], rng)
    layers = nudge_towards_goal(layers, fine_owners, coarse_owners, goal_img)

    stack = combine_images(list(layers))
    print(f"Stack: {describe_error(stack, goal_img)}")
    save_image(stack, output_folder / "stack.png")
    for i, layer in enumerate(layers):
        save_image(layer, output_folder / f"layer_{i}.png")
    measure_leak(layers, goal_img)
    print(f"Saved to {output_folder}")


def tint_layers(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    layers = []
    for i in range(num_images):
        blob_ids = split_into_blobs(random_owners(detail_sigma, shape, rng))
        colours = rng.uniform(off_white_min, 1.0, (blob_ids.max() + 1, 3))
        layers.append(colours.astype(np.float32)[blob_ids])
        save_image(layers[i], output_folder / f"3_tinted_layer_{i}.png")
    return np.stack(layers)


def split_into_blobs(owners: np.ndarray) -> np.ndarray:
    blob_ids = np.zeros(owners.shape, dtype=np.int32)
    num_blobs = 0
    for owner in range(num_images):
        count, labels = cv2.connectedComponents((owners == owner).astype(np.uint8))
        mask = labels > 0
        blob_ids[mask] = labels[mask] - 1 + num_blobs
        num_blobs += count - 1
    return blob_ids


def add_decoys(layers: np.ndarray) -> np.ndarray:
    height, width = layers.shape[1:3]
    for i, layer in enumerate(layers):
        decoy_img = load_image(decoy_files[i % len(decoy_files)])[:, :, :3]
        decoy_img = resize_image(decoy_img, (height, width, 3))
        decoy_img = adjust_image_brightness(decoy_img, min_=goal_min, max_=goal_max)
        # Mirror a decoy each time it's reused
        if (i // len(decoy_files)) % 2:
            decoy_img = decoy_img[:, ::-1]
        log_decoy = np.log(decoy_img)
        detail = log_decoy - low_pass_filter(log_decoy, sigma=detail_sigma)
        layers[i] = np.clip(layer * np.exp(decoy_strength * detail), 0, 1)
        save_image(layers[i], output_folder / f"4_decoy_layer_{i}.png")
    return layers


def share_out(
    shape: tuple[int, int], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    fine_owners = random_owners(fine_patch_size, shape, rng)
    coarse_owners = random_owners(coarse_patch_size, shape, rng)
    save_image(colour_map(fine_owners), output_folder / "5_fine_owners.png")
    save_image(colour_map(coarse_owners), output_folder / "5_coarse_owners.png")
    return fine_owners, coarse_owners


def nudge_towards_goal(
    layers: np.ndarray,
    fine_owners: np.ndarray,
    coarse_owners: np.ndarray,
    goal_img: np.ndarray,
) -> np.ndarray:
    for iteration in range(1, max_iterations + 1):
        stack = layers.prod(axis=0)
        print(f"Iteration {iteration}: {describe_error(stack, goal_img)}")
        if (pixel_error(stack, goal_img) < 1).all():
            print("All pixels exact, stopping")
            break

        step = step_size * (np.log(goal_img) - np.log(stack))
        coarse = low_pass_filter(step, sigma=detail_sigma)
        apply_step(layers, coarse_owners, coarse)
        apply_step(layers, fine_owners, step - coarse)

        if iteration == 1 or iteration % save_every == 0:
            folder = output_folder / f"iteration_{iteration:03d}"
            save_image(stack, folder / "6a_stack.png")
            save_image(0.5 + goal_img - stack, folder / "6a_goal_minus_stack.png")
            save_image(0.5 + coarse, folder / "6b_coarse_step.png")
            for i, layer in enumerate(layers):
                save_image(layer, folder / f"6c_layer_{i}.png")
    return layers


def pixel_error(stack: np.ndarray, goal_img: np.ndarray) -> np.ndarray:
    # Each pixel's error out of 255, as in the saved images. Under 1 is exact.
    return (255 * np.abs(goal_img - stack)).max(axis=2)


def describe_error(stack: np.ndarray, goal_img: np.ndarray) -> str:
    error = pixel_error(stack, goal_img)
    # Round down, so 100% only shows once every pixel is exact
    exact = int(100 * (error < 1).mean())
    return (
        f"mean error {error.mean():.2f}/255, max error {error.max():.2f}/255, {exact}%"
    )


def apply_step(layers: np.ndarray, owners: np.ndarray, step: np.ndarray) -> None:
    rows, cols = np.indices(owners.shape)
    for offset in range(num_images):
        idx = (owners + offset) % num_images
        before = np.maximum(layers[idx, rows, cols], 1e-6)
        after = np.clip(before * np.exp(step), 0, 1)
        layers[idx, rows, cols] = after
        step = step - (np.log(np.maximum(after, 1e-6)) - np.log(before))


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


def random_owners(
    sigma: float, shape: tuple[int, int], rng: np.random.Generator
) -> np.ndarray:
    height, width = shape
    votes = np.stack(
        [
            low_pass_filter(
                rng.random((height // 4, width // 4), dtype=np.float32), sigma / 4
            )
            for _ in range(num_images)
        ],
        axis=2,
    )
    votes = cv2.resize(votes, (width, height), interpolation=cv2.INTER_LINEAR)
    return votes.argmax(axis=2)


def colour_map(indices: np.ndarray) -> np.ndarray:
    return np.array(palette)[indices % len(palette)]


if __name__ == "__main__":
    main()
