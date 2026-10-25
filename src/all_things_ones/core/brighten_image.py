import numpy as np


def brighten_image(image: np.ndarray, min_brightness: float = 0.2) -> np.ndarray:
    # Find current min and max values
    current_min = image.min()
    current_max = image.max()

    # Scale image from [current_min, current_max] to [min_brightness, 1.0]
    if current_max > current_min:  # Avoid division by zero
        scaled = (image - current_min) / (current_max - current_min)
        scaled = scaled * (1.0 - min_brightness) + min_brightness
        return scaled
    else:
        # All pixels same value, set to min_brightness
        return np.full_like(image, min_brightness)
