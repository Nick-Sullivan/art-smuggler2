import numpy as np

from all_things_ones.core import low_pass_filter


def extract_frequency_layer(
    target_img: np.ndarray,
    mask_pct_threshold: float,
    diff_threshold: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Blur the target with increasing sigma until enough pixels change by more than
    diff_threshold. Those pixels are the high frequency detail of the image.

    Returns (mask, filtered_img).
    """
    height, width = target_img.shape[:2]
    sigma = 0
    sigma_increment = 1
    mask_pct = 0
    prev_mask_pct = 0
    while mask_pct < mask_pct_threshold:
        filtered_img = low_pass_filter(target_img, sigma=sigma)
        diff = filtered_img - target_img
        mask = np.max(np.abs(diff), axis=2) >= diff_threshold
        mask_pct = np.sum(mask) / (height * width) * 100
        print(f"sigma {sigma} mask percentage: {mask_pct:.2f}%")
        delta = mask_pct - prev_mask_pct
        if delta < 1:
            sigma_increment += 1
        elif delta > 2:
            sigma_increment = max(1, sigma_increment - 1)
        sigma += sigma_increment
        prev_mask_pct = mask_pct
    return mask, filtered_img
