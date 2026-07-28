"""
Geometric and RANSAC-based plane fitting utilities.
"""

import numpy as np


class PlaneFitter:
    """Provides methods for fitting planes to 3D points."""

    @staticmethod
    def preprocess_point_cloud(pc):
        """
        Flatten and filter valid 3D points.

        Returns:
            flat_points, valid_mask, height, width
        """
        H, W, _ = pc.shape
        flat = pc.reshape(-1, 3)
        valid = (~np.isnan(flat).any(axis=1)) & (~np.all(flat == 0, axis=1))
        return flat.astype(np.float32), valid, H, W

    @staticmethod
    def fit_plane(points3):
        """
        Fit a plane from 3 non-collinear 3D points.

        Returns:
            Plane coefficients (a, b, c, d)
        """
        p1, p2, p3 = points3
        v1, v2 = p2 - p1, p3 - p1
        normal = np.cross(v1, v2)
        nrm = np.linalg.norm(normal)
        if nrm == 0 or not np.isfinite(nrm):
            return None
        a, b, c = normal / nrm
        d = -np.dot([a, b, c], p1)
        return a, b, c, d

    @staticmethod
    def distances_to_plane(pts, plane):
        """Compute perpendicular distances from points to a plane."""
        a, b, c, d = plane
        return pts @ np.array([a, b, c]) + d

