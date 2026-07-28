import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import csv
import time
import numpy as np
import matplotlib.pyplot as plt

from boxdetect.config import CFG
from boxdetect.io_utils import DataLoader
from boxdetect.geometry import PlaneFitter
from boxdetect.mask_utils import MaskProcessor
from boxdetect.ransac import RANSACPlaneDetector


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def plane_dists(points: np.ndarray, plane) -> np.ndarray:
    return np.abs(PlaneFitter.distances_to_plane(points, plane))


def save_plane_viz(points: np.ndarray, inliers: np.ndarray, out_path: str, title: str):
    """Simple 3D scatter showing inliers vs outliers."""
    m = np.zeros(points.shape[0], dtype=bool)
    m[inliers] = True
    in_pts = points[m]
    out_pts = points[~m]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(out_pts[:, 0], out_pts[:, 1], out_pts[:, 2], s=1)
    ax.scatter(in_pts[:, 0], in_pts[:, 1], in_pts[:, 2], s=1)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def heatmap(values, M_list, B_list, title, out_path):
    """
    values shape: (len(M_list), len(B_list))
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(values, aspect="auto")
    ax.set_xticks(range(len(B_list)))
    ax.set_xticklabels([str(b) for b in B_list])
    ax.set_yticks(range(len(M_list)))
    ax.set_yticklabels([str(m) for m in M_list])
    ax.set_xlabel("B (batch size)")
    ax.set_ylabel("M (hypotheses)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_preemptive(points: np.ndarray, eps: float, min_inliers: int, M: int, B: int, score_method: str, gamma: float):
    """
    score_method:
      - "ransac": batch-wise outlier counting (lower is better)
      - "mlesac": truncated distance cost (lower is better)
    """
    det = RANSACPlaneDetector(
        1,  # ignored in preemptive mode
        eps,
        min_inliers,
        CFG.RANDOM_SEED,
        score_method=score_method,
        gamma=gamma,
        preemptive=True,
        M=M,
        B=B,
    )

    t0 = time.perf_counter()
    plane, inliers = det.detect_plane(points)
    dt = time.perf_counter() - t0

    if plane is None:
        return None, np.array([], dtype=int), {
            "n_inliers": 0,
            "inlier_ratio": 0.0,
            "mean_inlier_dist": np.nan,
            "median_inlier_dist": np.nan,
            "runtime_s": float(dt),
        }

    d = plane_dists(points, plane)
    in_d = d[inliers] if inliers.size else np.array([], dtype=float)

    metrics = {
        "n_inliers": int(inliers.size),
        "inlier_ratio": float(inliers.size) / float(points.shape[0]),
        "mean_inlier_dist": float(np.mean(in_d)) if in_d.size else np.nan,
        "median_inlier_dist": float(np.median(in_d)) if in_d.size else np.nan,
        "runtime_s": float(dt),
    }
    return plane, inliers, metrics


def build_floor_and_box_points(pc: np.ndarray):
    """
    Same preprocessing idea as pipeline:
      - preprocess point cloud (flat Nx3 + valid mask)
      - floor is estimated on pts_valid
      - box points are valid minus floor mask (after refinement)
    """
    flat, valid, H, W = PlaneFitter.preprocess_point_cloud(pc)
    valid_idx = np.flatnonzero(valid)
    pts_valid = flat[valid_idx]
    return flat, valid, H, W, valid_idx, pts_valid


def write_csv(rows, out_csv):
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to exercise .mat file (e.g., example4kinect.mat)")
    parser.add_argument("--outdir", default="outputs/preemptive_eval", help="Output directory")
    parser.add_argument("--save-masks", action="store_true", help="Save floor/box/top masks for every (M,B)")
    args = parser.parse_args()

    # Load data
    amp, dist, pc = DataLoader.read_mat(args.file)
    CFG.set_current_file(args.file)
    CFG.ensure_dirs()

    base = os.path.splitext(os.path.basename(args.file))[0]
    out_root = ensure_dir(os.path.join(args.outdir, base))

    # Use config lists if you have them, otherwise define here
    # (You asked: use exact config values; so we assume you have these in CFG.)
    M_list = CFG.PREEMPTIVE_M_LIST
    B_list = CFG.PREEMPTIVE_B_LIST

    score_methods = ["ransac", "mlesac"]  # both scenarios, always

    # Prepare point sets
    flat, valid, H, W, valid_idx, pts_valid = build_floor_and_box_points(pc)

    rows = []

    # For time budget demonstration: 3 choices of M (small/medium/large)
    demo_Ms = [M_list[0], M_list[len(M_list)//2], M_list[-1]]
    demo_B = B_list[len(B_list)//2]  # keep B fixed for fair comparison

    for score_method in score_methods:
        score_dir = ensure_dir(os.path.join(out_root, f"preemptive_{score_method}"))

        # Matrices for heatmaps (floor & top)
        floor_inliers_mat = np.zeros((len(M_list), len(B_list)), dtype=float)
        floor_mean_mat = np.zeros((len(M_list), len(B_list)), dtype=float)
        floor_time_mat = np.zeros((len(M_list), len(B_list)), dtype=float)

        top_inliers_mat = np.zeros((len(M_list), len(B_list)), dtype=float)
        top_mean_mat = np.zeros((len(M_list), len(B_list)), dtype=float)
        top_time_mat = np.zeros((len(M_list), len(B_list)), dtype=float)

        for i, M in enumerate(M_list):
            for j, B in enumerate(B_list):
                # ---- FLOOR (preemptive) ----
                floor_plane, floor_inliers, floor_metrics = run_preemptive(
                    points=pts_valid,
                    eps=CFG.PREEMPTIVE_DIST_THRESH,
                    min_inliers=CFG.PREEMPTIVE_MIN_INLIERS,
                    M=M,
                    B=B,
                    score_method=score_method,
                    gamma=CFG.PREEMPTIVE_GAMMA,
                )

                # Build floor mask -> box points
                if floor_plane is None:
                    clean_floor = np.zeros((H, W), dtype=np.uint8)
                    box_pts = np.empty((0, 3), dtype=np.float32)
                    clean_box = np.zeros((H, W), dtype=np.uint8)
                else:
                    floor_full = valid_idx[floor_inliers]
                    floor_mask = MaskProcessor.mask_from_indices(H, W, floor_full)
                    clean_floor = MaskProcessor.refine_and_keep_largest(
                        floor_mask, close_size=CFG.CLOSE_SIZE, open_size=CFG.OPEN_SIZE
                    )

                    floor_flat = clean_floor.reshape(-1).astype(bool)
                    box_mask_flat = valid & (~floor_flat)
                    box_mask = box_mask_flat.reshape(H, W).astype(np.uint8)

                    clean_box = MaskProcessor.refine_and_keep_largest(
                        box_mask, close_size=CFG.CLOSE_SIZE, open_size=CFG.OPEN_SIZE
                    )

                    box_idx = np.flatnonzero(clean_box.reshape(-1))
                    box_pts = flat[box_idx]

                # ---- TOP (preemptive) ----
                if box_pts.shape[0] >= 3:
                    top_plane, top_inliers, top_metrics = run_preemptive(
                        points=box_pts,
                        eps=CFG.TOP_PREEMPTIVE_DIST_THRESH,
                        min_inliers=CFG.TOP_PREEMPTIVE_MIN_INLIERS,
                        M=M,
                        B=B,
                        score_method=score_method,
                        gamma=CFG.TOP_PREEMPTIVE_GAMMA,
                    )
                else:
                    top_plane = None
                    top_inliers = np.array([], dtype=int)
                    top_metrics = {
                        "n_inliers": 0,
                        "inlier_ratio": 0.0,
                        "mean_inlier_dist": np.nan,
                        "median_inlier_dist": np.nan,
                        "runtime_s": np.nan,
                    }

                # Optional: save masks for this (M,B)
                if args.save_masks:
                    MaskProcessor.visualize_mask(clean_floor, f"Floor_preemptive_{score_method}_M{M}_B{B}", tag=f"{score_method}_M{M}_B{B}")
                    MaskProcessor.visualize_mask(clean_box, f"Box_preemptive_{score_method}_M{M}_B{B}", tag=f"{score_method}_M{M}_B{B}")

                    if top_plane is not None and top_inliers.size:
                        # top mask in original image indices
                        box_idx = np.flatnonzero(clean_box.reshape(-1))
                        top_full = box_idx[top_inliers]
                        top_mask = MaskProcessor.mask_from_indices(H, W, top_full)
                        MaskProcessor.visualize_mask(top_mask, f"Top_preemptive_{score_method}_M{M}_B{B}", tag=f"{score_method}_M{M}_B{B}")

                # Fill heatmap matrices
                floor_inliers_mat[i, j] = floor_metrics["n_inliers"]
                floor_mean_mat[i, j] = floor_metrics["mean_inlier_dist"]
                floor_time_mat[i, j] = floor_metrics["runtime_s"]

                top_inliers_mat[i, j] = top_metrics["n_inliers"]
                top_mean_mat[i, j] = top_metrics["mean_inlier_dist"]
                top_time_mat[i, j] = top_metrics["runtime_s"]

                # CSV rows (floor + top)
                rows.append({
                    "file": base,
                    "variant": f"preemptive_{score_method}",
                    "plane": "floor",
                    "M": M,
                    "B": B,
                    "eps": CFG.PREEMPTIVE_DIST_THRESH,
                    "gamma": CFG.PREEMPTIVE_GAMMA if score_method == "mlesac" else 0.0,
                    **floor_metrics
                })
                rows.append({
                    "file": base,
                    "variant": f"preemptive_{score_method}",
                    "plane": "top",
                    "M": M,
                    "B": B,
                    "eps": CFG.TOP_PREEMPTIVE_DIST_THRESH,
                    "gamma": CFG.TOP_PREEMPTIVE_GAMMA if score_method == "mlesac" else 0.0,
                    **top_metrics
                })

        # Save heatmaps
        heatmap(floor_inliers_mat, M_list, B_list,
                "FLOOR: #inliers (higher is better)",
                os.path.join(score_dir, "heatmap_floor_inliers.png"))
        heatmap(floor_mean_mat, M_list, B_list,
                "FLOOR: mean inlier distance (lower is better)",
                os.path.join(score_dir, "heatmap_floor_mean_inlier_dist.png"))
        heatmap(floor_time_mat, M_list, B_list,
                "FLOOR: runtime (s)",
                os.path.join(score_dir, "heatmap_floor_runtime.png"))

        heatmap(top_inliers_mat, M_list, B_list,
                "TOP: #inliers (higher is better)",
                os.path.join(score_dir, "heatmap_top_inliers.png"))
        heatmap(top_mean_mat, M_list, B_list,
                "TOP: mean inlier distance (lower is better)",
                os.path.join(score_dir, "heatmap_top_mean_inlier_dist.png"))
        heatmap(top_time_mat, M_list, B_list,
                "TOP: runtime (s)",
                os.path.join(score_dir, "heatmap_top_runtime.png"))

        # ---- Time budget demonstration: 3 M values, fixed B ----
        for M in demo_Ms:
            # floor
            floor_plane, floor_inliers, _ = run_preemptive(
                points=pts_valid,
                eps=CFG.PREEMPTIVE_DIST_THRESH,
                min_inliers=CFG.PREEMPTIVE_MIN_INLIERS,
                M=M,
                B=demo_B,
                score_method=score_method,
                gamma=CFG.PREEMPTIVE_GAMMA,
            )
            if floor_plane is not None and floor_inliers.size:
                save_plane_viz(
                    pts_valid, floor_inliers,
                    os.path.join(score_dir, f"plane_viz_FLOOR_M{M}_B{demo_B}.png"),
                    f"FLOOR preemptive-{score_method} (M={M}, B={demo_B})"
                )

            # build box points from this floor plane for top viz
            if floor_plane is not None:
                floor_full = valid_idx[floor_inliers]
                floor_mask = MaskProcessor.mask_from_indices(H, W, floor_full)
                clean_floor = MaskProcessor.refine_and_keep_largest(
                    floor_mask, close_size=CFG.CLOSE_SIZE, open_size=CFG.OPEN_SIZE
                )
                floor_flat = clean_floor.reshape(-1).astype(bool)
                box_mask_flat = valid & (~floor_flat)
                box_mask = box_mask_flat.reshape(H, W).astype(np.uint8)
                clean_box = MaskProcessor.refine_and_keep_largest(
                    box_mask, close_size=CFG.CLOSE_SIZE, open_size=CFG.OPEN_SIZE
                )
                box_idx = np.flatnonzero(clean_box.reshape(-1))
                box_pts = flat[box_idx]
            else:
                box_pts = np.empty((0, 3), dtype=np.float32)

            if box_pts.shape[0] >= 3:
                top_plane, top_inliers, _ = run_preemptive(
                    points=box_pts,
                    eps=CFG.TOP_PREEMPTIVE_DIST_THRESH,
                    min_inliers=CFG.TOP_PREEMPTIVE_MIN_INLIERS,
                    M=M,
                    B=demo_B,
                    score_method=score_method,
                    gamma=CFG.TOP_PREEMPTIVE_GAMMA,
                )
                if top_plane is not None and top_inliers.size:
                    save_plane_viz(
                        box_pts, top_inliers,
                        os.path.join(score_dir, f"plane_viz_TOP_M{M}_B{demo_B}.png"),
                        f"TOP preemptive-{score_method} (M={M}, B={demo_B})"
                    )

    # Save CSV summary
    out_csv = os.path.join(out_root, "results.csv")
    write_csv(rows, out_csv)
    print("Saved:", out_csv)
    print("Done. Outputs in:", out_root)


if __name__ == "__main__":
    main()
