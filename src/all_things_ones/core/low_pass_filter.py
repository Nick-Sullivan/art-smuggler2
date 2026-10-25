import cv2
import numpy as np


def low_pass_filter(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply a low pass filter (Gaussian blur) to an image.

    Args:
        image: Input image as numpy array with shape (height, width, channels)
        sigma: Standard deviation for Gaussian kernel. Higher values = more blur

    Returns:
        Filtered image with same shape as input
    """
    kernel_size = max(3, int(2 * np.ceil(2 * sigma) + 1))
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return blurred
