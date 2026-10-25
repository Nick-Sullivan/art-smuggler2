import cv2
import numpy as np

from .model import Blob


def detect_blobs(
    image: np.ndarray,
    num_clusters: int = 200,
    max_iterations: int = 20,
    min_blob_size: int = 1000,
    exclude_zero_pixels: bool = False,
) -> tuple[np.ndarray, list[Blob]]:
    """
    Detect blobs using K-means clustering on color and spatial features.

    Args:
        image: Input image as numpy array
        num_clusters: Number of blobs/clusters to create
        max_iterations: Maximum iterations for K-means
        min_blob_size: Minimum number of pixels required for a blob
        exclude_zero_pixels: If True, exclude pixels with all channels = 0

    Returns:
        Tuple of labels and blob info
    """
    height, width = image.shape[:2]
    zero_threshold = 0.001

    # Create feature vectors: [R, G, B, X, Y]
    # Normalize spatial coordinates to [0,1] range
    features = []
    valid_positions = [] if exclude_zero_pixels else None

    for y in range(height):
        for x in range(width):
            r, g, b = image[y, x]

            # Skip zero pixels if filtering is enabled
            is_zero = (
                -zero_threshold <= r <= zero_threshold
                and -zero_threshold <= g <= zero_threshold
                and -zero_threshold <= b <= zero_threshold
            )
            if exclude_zero_pixels and is_zero:
                continue

            norm_x = x / width
            norm_y = y / height
            # Weight spatial features less than color features
            features.append([r, g, b, norm_x * 0.5, norm_y * 0.5])

            if exclude_zero_pixels:
                valid_positions.append((y, x))

    if exclude_zero_pixels and not features:
        # Return empty results if no valid pixels found
        return np.zeros((height, width), dtype=int), []

    features = np.array(features, dtype=np.float32)

    # Adjust num_clusters if we have fewer valid pixels than clusters
    actual_num_clusters = (
        min(num_clusters, len(features)) if exclude_zero_pixels else num_clusters
    )

    # Apply K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iterations, 1.0)
    _, labels_flat, centers = cv2.kmeans(
        features, actual_num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # Create labels array
    if exclude_zero_pixels:
        # Initialize with -1 (indicating no cluster)
        labels = np.full((height, width), -1, dtype=int)
        # Fill in the labels for valid positions only
        for i, (y, x) in enumerate(valid_positions):
            labels[y, x] = labels_flat[i]
    else:
        # Reshape labels back to image shape (original behavior)
        labels = labels_flat.reshape((height, width))

    # Calculate initial blob statistics and identify small blobs
    small_blobs = []
    large_blobs = []

    for cluster_id in range(actual_num_clusters):
        mask = labels == cluster_id
        pixels = np.where(mask)
        size = len(pixels[0])

        if size > 0:
            # Calculate statistics
            centroid_y = np.mean(pixels[0])
            centroid_x = np.mean(pixels[1])
            mean_color = centers[cluster_id][:3]

            blob_data = {
                "id": cluster_id,
                "size": size,
                "centroid": (centroid_x, centroid_y),
                "mean_color": mean_color,
                "mask": mask,
                "pixels": pixels,
            }

            if size < min_blob_size:
                small_blobs.append(blob_data)
            else:
                large_blobs.append(blob_data)

    # Merge small blobs with nearest large blobs
    new_labels = labels.copy()

    for small_blob in small_blobs:
        small_centroid = small_blob["centroid"]

        # Find nearest large blob by centroid distance
        min_distance = float("inf")
        nearest_blob_id = None

        for large_blob in large_blobs:
            large_centroid = large_blob["centroid"]
            distance = np.sqrt(
                (small_centroid[0] - large_centroid[0]) ** 2
                + (small_centroid[1] - large_centroid[1]) ** 2
            )

            if distance < min_distance:
                min_distance = distance
                nearest_blob_id = large_blob["id"]

        # If we found a large blob to merge with, reassign pixels
        if nearest_blob_id is not None:
            small_mask = small_blob["mask"]
            new_labels[small_mask] = nearest_blob_id

            # Update the large blob's data
            for large_blob in large_blobs:
                if large_blob["id"] == nearest_blob_id:
                    # Combine masks
                    combined_mask = new_labels == nearest_blob_id
                    combined_pixels = np.where(combined_mask)

                    # Update blob data
                    large_blob["mask"] = combined_mask
                    large_blob["pixels"] = combined_pixels
                    large_blob["size"] = len(combined_pixels[0])
                    large_blob["centroid"] = (
                        np.mean(combined_pixels[1]),
                        np.mean(combined_pixels[0]),
                    )
                    break

    # Create final blob info from large blobs (now including merged small ones)
    blob_info = []
    for blob_data in large_blobs:
        pixels = blob_data["pixels"]

        min_y, max_y = np.min(pixels[0]), np.max(pixels[0])
        min_x, max_x = np.min(pixels[1]), np.max(pixels[1])

        blob = Blob(
            id=blob_data["id"],
            size=blob_data["size"],
            centroid=blob_data["centroid"],
            bbox=(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1),
            mean_color=blob_data["mean_color"],
            mask=blob_data["mask"],
        )
        blob_info.append(blob)

    return new_labels, blob_info
