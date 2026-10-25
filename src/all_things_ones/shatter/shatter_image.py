import numpy as np

from .apply_shatter_pattern import apply_shatter_pattern
from .construct_image_from_pieces import construct_image_from_pieces
from .create_shatter_pattern import create_shatter_pattern
from .model import ShatterPattern


def shatter_image(
    target: np.ndarray,
    num_pieces: int = 15,
    num_images: int = 2,
    seed: int | None = None,
    background_color: float = 0.6,
    pattern: ShatterPattern = None,
) -> list[np.ndarray]:
    """
    Split image into irregular pieces like a shattered windshield, then distribute across multiple images.
    """
    height, width, channels = target.shape

    # Use provided pattern or create a new one
    if pattern is None:
        pattern = create_shatter_pattern(width, height, num_pieces, seed)

    pieces = apply_shatter_pattern(target, pattern)

    # Distribute pieces across images
    distributed_images = [[] for _ in range(num_images)]

    # Strategy: distribute pieces as evenly as possible
    for i, piece in enumerate(pieces):
        target_image = i % num_images
        distributed_images[target_image].append(piece)

    # Create composite images
    composite_images = []
    for pieces_list in distributed_images:
        if not pieces_list:
            continue

        composite = construct_image_from_pieces(
            pieces_list, (height, width, channels), background_color
        )
        composite_images.append(composite)

    return composite_images
