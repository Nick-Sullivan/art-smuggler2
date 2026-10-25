from pathlib import Path

import cv2
import numpy as np


def save_image(image: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(path), image)
