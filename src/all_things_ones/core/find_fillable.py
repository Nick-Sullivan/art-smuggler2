import numpy as np


def find_fillable(layers: list[np.ndarray]) -> list[np.ndarray]:
    """
    For opaque RGBA layers stacked bottom (index 0) to top, find where each layer
    is hidden by a painted pixel on a higher layer, so it can be filled freely.
    """
    height, width = layers[0].shape[:2]
    owner = np.full((height, width), -1)
    for i, layer in enumerate(layers):
        owner[layer[:, :, 3] > 0] = i
    return [owner > i for i in range(len(layers))]
