import numpy as np
from scipy.ndimage import rotate

from .model import Blob


def tessellate_blob(
    blob: Blob, img: np.ndarray, num_blobs: int
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Returns list of (colors, rotated_y, rotated_x) tuples for efficient mapping.
    """
    angle_step = 360 / num_blobs

    tessellations = []

    # Create a masked image with only the blob
    blob_img = np.zeros_like(img)
    blob_img[blob.mask] = img[blob.mask]

    for j in range(num_blobs):
        angle = j * angle_step

        # Rotate the entire blob image
        rotated_img = rotate(blob_img, angle, axes=(0, 1), reshape=False, order=0)

        # Find non-zero pixels in the rotated image
        rotated_mask = np.any(rotated_img != 0, axis=2)
        rotated_y, rotated_x = np.where(rotated_mask)
        colors = rotated_img[rotated_y, rotated_x]

        tessellations.append((colors, rotated_y, rotated_x))

    return tessellations
