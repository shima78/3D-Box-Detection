"""
High-level box detection pipeline integrating all modules.
"""

import numpy as np

from .measurements import BoxMeasurer
from .config import CFG
from .io_utils import DataLoader, Visualizer
from .geometry import PlaneFitter
from .ransac import RANSACPlaneDetector
from .mask_utils import MaskProcessor


class BoxDetectionPipeline:
    """Full processing pipeline from Kinect data to box measurements."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.amp, self.dist, self.pc = DataLoader.read_mat(file_path)
        CFG.set_current_file(file_path)

    def run(self):
        """
        Run All plane-detection variants (RANSAC, MLESAC, Preemptive RANSAC),
        and save every output with the algorithm tag in the filename.
        """
        algorithms = [
            ("ransac", self._make_floor_detector_ransac, self._make_top_detector_ransac),
            ("mlesac", self._make_floor_detector_mlesac, self._make_top_detector_mlesac),
            ("preemptive", self._make_floor_detector_preemptive, self._make_top_detector_preemptive),
        ]

        for algo_tag, floor_factory, top_factory in algorithms:
            print("\n" + "=" * 60)
            print(f"RUNNING: {algo_tag.upper()}")
            print("=" * 60)
            self._run_single(algo_tag, floor_factory(), top_factory())

    # ----------------------------
    # Detector factories (floor)
    # ----------------------------
    def _make_floor_detector_ransac(self):
     
        return RANSACPlaneDetector(
            CFG.RANSAC_ITER,
            CFG.RANSAC_DIST_THRESH,
            CFG.RANSAC_MIN_INLIERS,
            CFG.RANDOM_SEED,
            score_method="ransac",
            preemptive=False,
        )

    def _make_floor_detector_mlesac(self):
        return RANSACPlaneDetector(
            CFG.MLESAC_ITER,
            CFG.MLESAC_DIST_THRESH,
            CFG.MLESAC_MIN_INLIERS,
            CFG.RANDOM_SEED,
            score_method="mlesac",
            gamma=CFG.MLESAC_GAMMA,
            preemptive=False, 
        )

    def _make_floor_detector_preemptive(self):
        return RANSACPlaneDetector(
            1,  # ignored in preemptive mode
            CFG.PREEMPTIVE_DIST_THRESH,
            CFG.PREEMPTIVE_MIN_INLIERS,
            CFG.RANDOM_SEED,
            score_method=CFG.PREEMPTIVE_SCORE_METHOD,  # MUST exist in config
            gamma=CFG.PREEMPTIVE_GAMMA,
            preemptive=True,
            M=CFG.PREEMPTIVE_M,
            B=CFG.PREEMPTIVE_B,
        )

    # ----------------------------
    # Detector factories (top)
    # ----------------------------
    def _make_top_detector_ransac(self):
        return RANSACPlaneDetector(
            CFG.TOP_RANSAC_ITER,
            CFG.TOP_RANSAC_DIST_THRESH,
            CFG.TOP_RANSAC_MIN_INLIERS,
            CFG.RANDOM_SEED,
            score_method="ransac",
            preemptive=False,
        )

    def _make_top_detector_mlesac(self):
        return RANSACPlaneDetector(
            CFG.TOP_MLESAC_ITER,
            CFG.TOP_MLESAC_DIST_THRESH,
            CFG.TOP_MLESAC_MIN_INLIERS,
            CFG.RANDOM_SEED,
            score_method="mlesac",
            gamma=CFG.TOP_MLESAC_GAMMA,
            preemptive=False,
        )

    def _make_top_detector_preemptive(self):
        return RANSACPlaneDetector(
            1,  # ignored in preemptive mode
            CFG.TOP_PREEMPTIVE_DIST_THRESH,
            CFG.TOP_PREEMPTIVE_MIN_INLIERS,
            CFG.RANDOM_SEED,
            score_method=CFG.TOP_PREEMPTIVE_SCORE_METHOD,  # MUST exist in config
            gamma=CFG.TOP_PREEMPTIVE_GAMMA,
            preemptive=True,
            M=CFG.TOP_PREEMPTIVE_M,
            B=CFG.TOP_PREEMPTIVE_B,
        )

    # ----------------------------
    # Single algorithm run
    # ----------------------------
    def _run_single(self, algo_tag: str, floor_det: RANSACPlaneDetector, top_det: RANSACPlaneDetector):
        # Save common visualizations with algorithm tag in filename
        Visualizer.show_amp_and_dist(self.amp, self.dist, tag=algo_tag)
        Visualizer.show_pc(self.pc, tag=algo_tag)

        flat, valid, H, W = PlaneFitter.preprocess_point_cloud(self.pc)
        valid_idx = np.flatnonzero(valid)
        pts_valid = flat[valid_idx]

        # --- Floor plane detection ---
        print("=== Plane Detection (floor) ===")
        floor_plane, floor_inliers = floor_det.detect_plane(pts_valid)
        print(f"[{algo_tag}] Floor plane: {floor_plane}, inliers={floor_inliers.size}")

        floor_full = valid_idx[floor_inliers]
        floor_mask = MaskProcessor.mask_from_indices(H, W, floor_full)
        clean_floor = MaskProcessor.refine_and_keep_largest(
            floor_mask,
            close_size=CFG.CLOSE_SIZE,
            open_size=CFG.OPEN_SIZE
        )
        MaskProcessor.visualize_mask(clean_floor, "Floor Mask", tag=algo_tag)

        # --- Box mask ---
        floor_flat = floor_mask.reshape(-1).astype(bool)
        box_mask_flat = valid & (~floor_flat)
        box_mask = box_mask_flat.reshape(H, W).astype(np.uint8)
        clean_box = MaskProcessor.refine_and_keep_largest(
            box_mask,
            close_size=CFG.CLOSE_SIZE,
            open_size=CFG.OPEN_SIZE
        )
        MaskProcessor.visualize_mask(clean_box, "Box Mask", tag=algo_tag)

        # --- Top plane detection on box points ---
        box_idx = np.flatnonzero(clean_box.reshape(-1))
        box_pts = flat[box_idx]

        print("=== Plane Detection (top) ===")
        top_plane, top_inliers = top_det.detect_plane(box_pts)
        print(f"[{algo_tag}] Top plane: {top_plane}, inliers={top_inliers.size}")

        top_full = box_idx[top_inliers]
        top_mask = MaskProcessor.mask_from_indices(H, W, top_full)
        MaskProcessor.visualize_mask(top_mask, "Top Plane Mask", tag=algo_tag)

        # --- Box measurement ---
        corners2D = BoxMeasurer.find_box_corners(top_mask)
        width, length, height = BoxMeasurer.compute_dimensions(corners2D, floor_plane, top_plane, self.pc)
        BoxMeasurer.visualize_box(clean_box, corners2D, (width, length, height), top_mask, clean_floor, tag=algo_tag)
        print(f"[{algo_tag}] Dimensions — H={height:.3f}, W={width:.3f}, L={length:.3f}")
