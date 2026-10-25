import cv2
import numpy as np


def segment_image_to_colours(
    image: np.ndarray, num_colours: int
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Segment image by color using k-means clustering.
    Returns (labels, colors, polygon_counts) where labels is -1 for transparent pixels.
    """
    height, width = image.shape[:2]

    # Extract non-transparent pixels for k-means
    valid_mask = image[:, :, 3] > 0
    img_for_kmeans = image[:, :, :3]
    pixels = img_for_kmeans[valid_mask]

    print(f"K-means clustering {np.sum(valid_mask)} pixels into {num_colours} colors")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels_flat, colors = cv2.kmeans(
        pixels.astype(np.float32),
        num_colours,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS,
    )

    # Build full label array
    labels = np.full((height, width), -1, dtype=np.int32)
    labels[valid_mask] = labels_flat.flatten()

    # Count polygons (connected components) for each color
    polygon_counts = []
    for i in range(num_colours):
        mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygon_counts.append((i, len(contours), np.sum(labels == i)))
        print(f"  Color {i}: {len(contours)} polygons, {np.sum(labels == i)} pixels")

    print(f"Color quantization complete: {num_colours} colors")

    return labels, colors, polygon_counts
