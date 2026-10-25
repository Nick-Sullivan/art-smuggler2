import cv2
import numpy as np


def resize_image(image: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    # OpenCV resize expects (width, height)
    height, width, _ = target_shape
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
