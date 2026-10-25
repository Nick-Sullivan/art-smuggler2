import cv2
import numpy as np

from .model import Blob


def visualise_blobs(labels: np.ndarray, blob_info: list[Blob]) -> np.ndarray:
    """
    Create a visualization of the detected blobs with different colors.

    Args:
        labels: Label array from blob detection
        blob_info: Blob information from blob detection

    Returns:
        RGB visualization image
    """
    height, width = labels.shape
    visualization = np.zeros((height, width, 3), dtype=np.float32)

    # Generate distinct colors for each blob
    num_blobs = len(blob_info)

    # Use HSV color space for better color distribution
    colors = []
    for i in range(num_blobs):
        hue = (i * 137.5) % 360  # Golden angle for good distribution
        sat = 0.7 + 0.3 * (i % 3) / 3  # Vary saturation
        val = 0.8 + 0.2 * (i % 2)  # Vary brightness

        # Convert HSV to RGB
        hsv = np.array([[[hue, sat, val]]], dtype=np.float32)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
        colors.append(rgb)

    colors = np.array(colors)

    for blob in blob_info:
        cluster_id = blob.id
        mask = blob.mask
        visualization[mask] = colors[cluster_id % num_blobs]

    return visualization
