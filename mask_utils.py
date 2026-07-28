"""
Mask processing utilities: binary refinement, largest blob selection, visualization.
"""

import numpy as np
import cv2
from scipy.ndimage import binary_closing, binary_opening, median_filter
from .io_utils import Visualizer
import matplotlib.pyplot as plt


class MaskProcessor:
    """Handles morphological cleaning and visualization of masks."""

    @staticmethod
    def mask_from_indices(H, W, indices):
        """Create a binary mask (H,W) from a flat list of pixel indices."""
        mask = np.zeros(H * W, dtype=np.uint8)
        if indices.size:
            mask[indices] = 1
        return mask.reshape(H, W)

    @staticmethod
    def refine_and_keep_largest(mask, close_size=7, open_size=5):
        """
        Apply morphological closing/opening and keep only the largest connected component.
        """
        m = mask.astype(bool)
        m = binary_closing(m, np.ones((close_size, close_size)), iterations=2)
        m = binary_opening(m, np.ones((open_size, open_size)), iterations=2)
        m = median_filter(m.astype(np.uint8), size=3)
        nlab, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if nlab <= 1:
            return m
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return (labels == largest).astype(np.uint8)

    @staticmethod
    def visualize_mask(mask, title, tag: str | None = None):
        """Display and save a binary mask as grayscale image."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(mask, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
        Visualizer.save_plot(fig, title, tag=tag)
