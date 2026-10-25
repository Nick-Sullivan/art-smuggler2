import numpy as np


def adjust_image_brightness(
    image: np.ndarray,
    min_: float = 0.0,
    max_: float = 1.0,
) -> np.ndarray:
    """
    Adjust image brightness to fit within a specified range.

    Args:
        image: Input image as numpy array
        min_brightness: Minimum brightness value (default: 0.0)
        max_brightness: Maximum brightness value (default: 1.0)

    Returns:
        Image scaled to the specified brightness range
    """
    # Find current min and max values
    current_min = image.min()
    current_max = image.max()

    # Scale image from [current_min, current_max] to [min_brightness, max_brightness]
    if current_max > current_min:  # Avoid division by zero
        scaled = (image - current_min) / (current_max - current_min)
        scaled = scaled * (max_ - min_) + min_
        return scaled
    else:
        # All pixels same value, set to average of min and max brightness
        avg_brightness = (min_ + max_) / 2
        return np.full_like(image, avg_brightness)
