"""
3D box measurement utilities for width, length, and height estimation.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from .io_utils import Visualizer


class BoxMeasurer:
    """Extracts box corners, computes 3D dimensions, and visualizes annotations."""

    @staticmethod
    def find_box_corners(top_mask):
        """
        Find 4 polygon corners from the top mask contour.
        """
        mask_u8 = (top_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise RuntimeError("No contours found in top mask.")
        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True).reshape(-1, 2)
        if len(approx) != 4:
            raise RuntimeError(f"Expected 4 corners, got {len(approx)}")

        pts = approx.astype(float)
        idx = np.argsort(pts[:, 1])
        top2, bottom2 = pts[idx[:2]], pts[idx[2:]]
        tl, tr = top2[np.argsort(top2[:, 0])]
        bl, br = bottom2[np.argsort(bottom2[:, 0])]
        return np.array([tl, tr, br, bl])

    @staticmethod
    def compute_dimensions(corners2D, plane, top_plane, pc):
        """
        Compute box width, length, and height from 2D corners and plane parameters.
        """
        H, W, _ = pc.shape
        flat = pc.reshape(-1, 3)
        inds = (corners2D[:, 1].astype(int) * W + corners2D[:, 0].astype(int)).astype(int)
        corners3D = flat[inds]
        width = np.linalg.norm(corners3D[1] - corners3D[0])
        length = np.linalg.norm(corners3D[2] - corners3D[1])
        a, b, c, d = plane
        a2, b2, c2, d2 = top_plane
        height = abs(d2 - d) / np.linalg.norm([a, b, c])
        return width, length, height

    @staticmethod
    def visualize_box(mask, corners2D, dim, top_mask, floor_mask, tag: str | None = None):
        """
        Overlay detected box corners and dimensions on the combined mask.
        """
        width, length, height = dim
        overlay = np.zeros((*mask.shape, 3), dtype=np.uint8)
        overlay[..., 0] = (floor_mask > 0) * 150
        overlay[..., 2] = (top_mask > 0) * 255

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(overlay)
        for i, (x, y) in enumerate(corners2D):
            ax.plot(x, y, "yo")
            ax.text(x + 4, y + 4, f"{i}", color="yellow", weight="bold")
            ax.plot([x, corners2D[(i + 1) % 4, 0]], [y, corners2D[(i + 1) % 4, 1]], "lime", lw=2)
        ax.set_title(f"H={height:.3f}, W={width:.3f}, L={length:.3f}")
        ax.axis("off")
        Visualizer.save_plot(fig, "Box_Dimensions", tag=tag)
