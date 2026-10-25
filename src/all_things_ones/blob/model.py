from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Blob:
    id: int
    size: int
    centroid: tuple[float, float]  # (x, y)
    bbox: tuple[float, float, float, float]  # (x, y, width, height)
    mean_color: np.ndarray
    mask: np.ndarray
