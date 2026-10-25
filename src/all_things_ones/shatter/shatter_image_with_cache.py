import numpy as np

from .create_shatter_pattern_with_cache import create_shatter_pattern_with_cache
from .shatter_image import shatter_image


def shatter_image_with_cache(
    target: np.ndarray,
    num_pieces: int,
    num_images: int,
    seed: int,
    background_color: float,
) -> list[np.ndarray]:
    height, width = target.shape[:2]
    pattern = create_shatter_pattern_with_cache(width, height, num_pieces, seed)
    return shatter_image(
        target=target,
        num_pieces=num_pieces,
        num_images=num_images,
        seed=seed,
        background_color=background_color,
        pattern=pattern,
    )
