import numpy as np
from scipy import ndimage

from .model import BoundingBox, ShatterPattern, ShatterPiece


def apply_shatter_pattern(
    target: np.ndarray,
    pattern: ShatterPattern,
) -> list[ShatterPiece]:
    """
    Apply a pre-computed shatter pattern to an image, returning only the cropped pieces.
    """
    height, width, channels = target.shape

    if pattern.width != width or pattern.height != height:
        raise ValueError(
            f"Pattern dimensions ({pattern.width}x{pattern.height}) don't match target ({width}x{height})"
        )

    # Find unique pieces and their bounding boxes in one pass
    unique_pieces = np.unique(pattern.region_map)

    # Use scipy to find bounding boxes efficiently
    slice_objects = ndimage.find_objects(
        pattern.region_map, max_label=unique_pieces.max()
    )

    all_pieces = []

    for piece_idx in unique_pieces:
        if piece_idx == 0:  # Skip background
            continue

        slice_obj = slice_objects[piece_idx - 1]  # find_objects is 1-indexed
        if slice_obj is None:
            continue

        row_slice, col_slice = slice_obj
        min_row, max_row = row_slice.start, row_slice.stop
        min_col, max_col = col_slice.start, col_slice.stop

        # Extract regions
        region_mask = pattern.region_map[row_slice, col_slice] == piece_idx
        piece_target = target[row_slice, col_slice]

        # Create piece data efficiently
        piece_height, piece_width = region_mask.shape
        piece_data = np.zeros(
            (piece_height, piece_width, channels + 1), dtype=target.dtype
        )
        piece_data[:, :, :channels] = piece_target
        piece_data[:, :, channels] = region_mask.astype(target.dtype)

        all_pieces.append(
            ShatterPiece(
                data=piece_data,
                bbox=BoundingBox(min_row, min_col, max_row, max_col),
                piece_id=piece_idx,
            )
        )

    return all_pieces
