from pathlib import Path

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert image is not None, f"Failed to load image: {path}"
    return image.astype(np.float32) / 255.0
