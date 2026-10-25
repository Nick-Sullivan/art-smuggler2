import numpy as np

from .model import ShatterPiece


def construct_image_from_pieces(
    pieces: list[ShatterPiece],
    target_shape: tuple[int, int, int],
    background_color: float = 0.6,
) -> np.ndarray:
    """
    Reconstruct a full image from shatter pieces.
    """
    height, width, channels = target_shape
    result = np.full((height, width, channels), background_color, dtype=np.float32)

    for piece in pieces:
        min_row = piece.bbox.min_row
        min_col = piece.bbox.min_col
        max_row = piece.bbox.max_row
        max_col = piece.bbox.max_col

        # Extract alpha mask and expand to match channels
        alpha_mask = piece.data[:, :, -1:] > 0  # Keep as 3D
        piece_rgb = piece.data[:, :, :channels]

        # Get target slice
        target_slice = result[min_row:max_row, min_col:max_col]

        # Use np.where for vectorized conditional assignment
        result[min_row:max_row, min_col:max_col] = np.where(
            alpha_mask, piece_rgb, target_slice
        )

    return result
