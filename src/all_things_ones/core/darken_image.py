import numpy as np


def darken_image(image: np.ndarray, max_brightness: float = 0.5) -> np.ndarray:
    # Find current min and max values
    current_min = image.min()
    current_max = image.max()

    # Scale image from [current_min, current_max] to [0.0, max_brightness]
    if current_max > current_min:  # Avoid division by zero
        scaled = (image - current_min) / (current_max - current_min)
        scaled = scaled * max_brightness
        return scaled
    else:
        # All pixels same value, set to max_brightness
        return np.full_like(image, max_brightness)


def darken_image_pct(image: np.ndarray, pct_reduction: int) -> np.ndarray:
    reduction_factor = pct_reduction / 100.0
    return image * (1.0 - reduction_factor)
