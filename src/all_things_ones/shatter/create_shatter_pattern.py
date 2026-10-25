import random

import numpy as np
from scipy.spatial import cKDTree

from .model import ShatterPattern


def create_shatter_pattern(
    width: int,
    height: int,
    num_pieces: int = 15,
    seed: int | None = None,
) -> ShatterPattern:
    """
    Create a reusable shatter pattern for images of the given dimensions.
    This can be cached and reused for multiple images.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Generate random points for Voronoi diagram (impact points)
    points = []

    # Add some points near the center (main impact)
    center_x, center_y = width // 2, height // 2
    center_points = max(1, num_pieces // 4)

    for _ in range(center_points):
        x = center_x + np.random.normal(0, width * 0.1)
        y = center_y + np.random.normal(0, height * 0.1)
        points.append([x, y])

    # Add random scattered points
    scatter_points = num_pieces - len(points)

    for _ in range(scatter_points):
        x = np.random.uniform(0, width)
        y = np.random.uniform(0, height)
        points.append([x, y])

    # Only use the first num_pieces points
    points = np.array(points[:num_pieces])

    # Use KDTree for extremely fast nearest neighbor lookup
    tree = cKDTree(points)

    # Process in chunks to avoid memory issues
    chunk_size = 1000000  # Process 1M pixels at a time
    region_map = np.zeros((height, width), dtype=np.int32)

    total_pixels = height * width

    for start_idx in range(0, total_pixels, chunk_size):
        end_idx = min(start_idx + chunk_size, total_pixels)

        # Convert flat indices to 2D coordinates for this chunk
        flat_indices = np.arange(start_idx, end_idx)
        y_coords = flat_indices // width
        x_coords = flat_indices % width

        # Create coordinate pairs for this chunk
        chunk_coords = np.column_stack([x_coords, y_coords])

        # Query KDTree for nearest points
        _, closest_indices = tree.query(chunk_coords)

        # Assign to region map
        region_map[y_coords, x_coords] = closest_indices

    return ShatterPattern(region_map, num_pieces, width, height)
