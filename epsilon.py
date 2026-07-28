from __future__ import annotations

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

# Project imports (assumes this script is run from repo root)
from boxdetect.geometry import PlaneFitter
from boxdetect.ransac import RANSACPlaneDetector


def _pick_first_key(mat: dict, prefix: str) -> str:
    keys = [k for k in mat.keys() if k.startswith(prefix)]
    if not keys:
        raise KeyError(f"No keys starting with '{prefix}' found in .mat. Available keys: {list(mat.keys())}")
    # deterministic: sort keys
    return sorted(keys)[0]


def read_kinect_mat(file_path: str):
    """
    Robust reader:
    - tries the project's convention (amplitudesX, distancesX, cloudX)
    - falls back to the first matching key if that convention doesn't fit
    """
    mat = scipy.io.loadmat(file_path)
    amp_key = dist_key = cloud_key = None
    try:
        number = file_path[-11]
        amp_key = f"amplitudes{number}"
        dist_key = f"distances{number}"
        cloud_key = f"cloud{number}"
        if amp_key not in mat or dist_key not in mat or cloud_key not in mat:
            amp_key = dist_key = cloud_key = None
    except Exception:
        amp_key = dist_key = cloud_key = None

    if amp_key is None:
        amp_key = _pick_first_key(mat, "amplitudes")
        dist_key = _pick_first_key(mat, "distances")
        cloud_key = _pick_first_key(mat, "cloud")

    amp = mat[amp_key]
    dist = np.float32(mat[dist_key])
    pc = mat[cloud_key]
    return amp, dist, pc


def mlesac_cost(dists: np.ndarray, eps: float, gamma: float) -> float:
    """Eq.(2): sum(d_i if d_i < eps else gamma)"""
    return float(np.sum(np.where(dists < eps, dists, gamma)))


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_fig(fig, out_dir: Path, name: str):
    fig.savefig(out_dir / name, bbox_inches="tight", dpi=300)
    plt.close(fig)


def mask_from_inliers(H: int, W: int, valid_idx: np.ndarray, inlier_idx_validspace: np.ndarray) -> np.ndarray:
    """Return (H,W) uint8 mask for inliers."""
    full_idx = valid_idx[inlier_idx_validspace]
    m = np.zeros(H * W, dtype=np.uint8)
    if full_idx.size:
        m[full_idx] = 1
    return m.reshape(H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to one .mat sample")
    ap.add_argument("--iters", type=int, default=200, help="Iterations for (non-preemptive) RANSAC/MLESAC")
    ap.add_argument("--min-inliers", type=int, default=10, help="Minimum inliers to accept a plane")
    ap.add_argument("--seed", type=int, default=42)

    # epsilon choices: either explicit list OR linspace
    ap.add_argument("--eps", type=float, nargs="*", default=None, help="Explicit epsilon list, e.g. --eps 0.01 0.02 0.05")
    ap.add_argument("--eps-min", type=float, default=0.01)
    ap.add_argument("--eps-max", type=float, default=0.10)
    ap.add_argument("--eps-num", type=int, default=10)

    ap.add_argument("--gamma-mult", type=float, default=2.0, help="gamma = gamma_mult * eps (MLESAC outlier penalty)")
    ap.add_argument("--save-masks", action="store_true", help="Save mask comparisons for a few eps values")
    args = ap.parse_args()

    file_path = args.file
    sample_name = Path(file_path).stem

    out_dir = ensure_dir(Path("outputs") / "epsilon_sweep" / sample_name)

    amp, dist, pc = read_kinect_mat(file_path)
    flat, valid, H, W = PlaneFitter.preprocess_point_cloud(pc)
    valid_idx = np.flatnonzero(valid)
    pts_valid = flat[valid_idx]

    if args.eps is None or len(args.eps) == 0:
        eps_list = np.linspace(args.eps_min, args.eps_max, args.eps_num).tolist()
    else:
        eps_list = list(args.eps)

    results = []

    # Choose eps indices to visualize (min, mid, max)
    vis_indices = set()
    if args.save_masks and len(eps_list) >= 1:
        vis_indices.add(0)
        vis_indices.add(len(eps_list) // 2)
        vis_indices.add(len(eps_list) - 1)

    for i, eps in enumerate(eps_list):
        gamma = args.gamma_mult * eps

        # --- Classic RANSAC ---
        ransac = RANSACPlaneDetector(
            num_iterations=args.iters,
            distance_threshold=eps,
            min_inliers=args.min_inliers,
            seed=args.seed,
            score_method="ransac",
            preemptive=False,
        )
        r_plane, r_inl = ransac.detect_plane(pts_valid)

        # --- MLESAC ---
        mlesac = RANSACPlaneDetector(
            num_iterations=args.iters,
            distance_threshold=eps,
            min_inliers=args.min_inliers,
            seed=args.seed,
            score_method="mlesac",
            gamma=gamma,
            preemptive=False,
        )
        m_plane, m_inl = mlesac.detect_plane(pts_valid)

        # compute diagnostics (using all points for cost; inlier mean distance for fit tightness)
        def plane_stats(plane, inl):
            if plane is None or inl.size == 0:
                return dict(inliers=0, mean_inlier_dist=np.nan, cost=np.nan)
            d_all = np.abs(PlaneFitter.distances_to_plane(pts_valid, plane))
            d_inl = d_all[inl]
            return dict(
                inliers=int(inl.size),
                mean_inlier_dist=float(np.mean(d_inl)),
                cost=mlesac_cost(d_all, eps, gamma),  # compute MLESAC-style cost for both (comparable)
            )

        r_stat = plane_stats(r_plane, r_inl)
        m_stat = plane_stats(m_plane, m_inl)

        results.append(
            dict(
                eps=float(eps),
                gamma=float(gamma),
                ransac_inliers=r_stat["inliers"],
                ransac_mean_inlier_dist=r_stat["mean_inlier_dist"],
                ransac_cost=r_stat["cost"],
                mlesac_inliers=m_stat["inliers"],
                mlesac_mean_inlier_dist=m_stat["mean_inlier_dist"],
                mlesac_cost=m_stat["cost"],
            )
        )

        # Optional: save a side-by-side inlier mask comparison for a few eps values
        if args.save_masks and i in vis_indices:
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            if r_plane is None:
                axs[0].set_title(f"RANSAC eps={eps:.4f}\n(no plane)")
                axs[0].imshow(np.zeros((H, W), dtype=np.uint8), cmap="gray")
            else:
                r_mask = mask_from_inliers(H, W, valid_idx, r_inl)
                axs[0].set_title(f"RANSAC eps={eps:.4f}\n#inliers={r_inl.size}")
                axs[0].imshow(r_mask, cmap="gray")
            axs[0].axis("off")

            if m_plane is None:
                axs[1].set_title(f"MLESAC eps={eps:.4f}\n(no plane)")
                axs[1].imshow(np.zeros((H, W), dtype=np.uint8), cmap="gray")
            else:
                m_mask = mask_from_inliers(H, W, valid_idx, m_inl)
                axs[1].set_title(f"MLESAC eps={eps:.4f}\n#inliers={m_inl.size}")
                axs[1].imshow(m_mask, cmap="gray")
            axs[1].axis("off")

            plt.tight_layout()
            save_fig(fig, out_dir, f"mask_compare_eps_{eps:.4f}.png")

    # --- Summary plots ---
    eps_arr = np.array([r["eps"] for r in results])
    r_inl = np.array([r["ransac_inliers"] for r in results])
    m_inl = np.array([r["mlesac_inliers"] for r in results])
    r_md = np.array([r["ransac_mean_inlier_dist"] for r in results])
    m_md = np.array([r["mlesac_mean_inlier_dist"] for r in results])

    # Inliers vs eps
    fig = plt.figure(figsize=(8, 5))
    plt.plot(eps_arr, r_inl, marker="o", label="RANSAC (#inliers)")
    plt.plot(eps_arr, m_inl, marker="o", label="MLESAC (#inliers)")
    plt.xlabel("epsilon (distance threshold)")
    plt.ylabel("inliers")
    plt.title("Inlier count vs epsilon")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_fig(fig, out_dir, "sweep_inliers.png")



    # Mean inlier distance vs eps (lower is tighter fit)
    fig = plt.figure(figsize=(8, 5))
    plt.plot(eps_arr, r_md, marker="o", label="RANSAC (mean inlier |dist|)")
    plt.plot(eps_arr, m_md, marker="o", label="MLESAC (mean inlier |dist|)")
    plt.xlabel("epsilon (distance threshold)")
    plt.ylabel("mean |distance| of inliers")
    plt.title("Fit tightness vs epsilon")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_fig(fig, out_dir, "sweep_mean_inlier_dist.png")

    # Save CSV-like table
    header = [
        "eps", "gamma",
        "ransac_inliers", "ransac_mean_inlier_dist", "ransac_cost",
        "mlesac_inliers", "mlesac_mean_inlier_dist", "mlesac_cost",
    ]
    lines = [",".join(header)]
    for r in results:
        lines.append(
            ",".join(
                [
                    f"{r['eps']:.6f}",
                    f"{r['gamma']:.6f}",
                    str(r["ransac_inliers"]),
                    f"{r['ransac_mean_inlier_dist']:.6f}" if np.isfinite(r["ransac_mean_inlier_dist"]) else "nan",
                    f"{r['ransac_cost']:.6f}" if np.isfinite(r["ransac_cost"]) else "nan",
                    str(r["mlesac_inliers"]),
                    f"{r['mlesac_mean_inlier_dist']:.6f}" if np.isfinite(r["mlesac_mean_inlier_dist"]) else "nan",
                    f"{r['mlesac_cost']:.6f}" if np.isfinite(r["mlesac_cost"]) else "nan",
                ]
            )
        )
    (out_dir / "results.csv").write_text("\n".join(lines), encoding="utf-8")

    print("\nDone.")
    print(f"Saved outputs to: {out_dir.resolve()}")
    print("Key files:")
    print("  - sweep_inliers.png")
    print("  - sweep_mean_inlier_dist.png")
    if args.save_masks:
        print("  - mask_compare_eps_*.png (for a few eps values)")
    print("  - results.csv")


if __name__ == "__main__":
    main()
