import numpy as np


def combine_images(images: list[np.ndarray]) -> np.ndarray:
    """
    Combine images according to light.

    Say you stack:
        Cyan: R=0, G=1, B=1
        Magenta: R=1, G=0, B=1
        Resulting transmission:
        R = 0 × 1 = 0
        G = 1 × 0 = 0
        B = 1 × 1 = 1
        So the final light that passes through is pure blue → RGB: (0, 0, 255)
    """
    if not images:
        raise ValueError("No images to combine")
    return np.prod(images, axis=0)


def combine_layers_by_transparency(layers: list[np.ndarray]) -> np.ndarray:
    """
    Combine RGBA images by transparency (alpha channel).
    Top layer shows unless transparent, then layers below show through.

    Args:
        layers: List of RGBA images (H, W, 4) with alpha channel

    Returns:
        Combined RGB image (H, W, 3)
    """
    if not layers:
        raise ValueError("No layers to combine")

    # Start with first layer
    result = layers[0][:, :, :3].copy()  # RGB only

    # Overlay subsequent layers
    for layer in layers[1:]:
        alpha = layer[:, :, 3:4]  # Keep as (H, W, 1) for broadcasting
        rgb = layer[:, :, :3]

        # Where alpha > 0, use the layer's color, otherwise keep what's below
        result = np.where(alpha > 0, rgb, result)

    return result
