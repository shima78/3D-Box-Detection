"""
I/O and visualization utilities for loading data and saving plots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from datetime import datetime
from .config import CFG


class DataLoader:
    """Handles loading of Kinect .mat files."""

    @staticmethod
    def read_mat(file_path: str):
        """
        Load amplitude, distance, and point cloud arrays from a .mat file.

        Args:
            file_path: Path to the .mat file.

        Returns:
            Tuple of (amplitude, distance, point_cloud) numpy arrays.
        """
        mat = scipy.io.loadmat(file_path)
        number = file_path[-11]
        amp = mat["amplitudes" + number]
        dist = np.float32(mat["distances" + number])
        pc = mat["cloud" + number]
        return amp, dist, pc


class Visualizer:
    """Handles plotting and saving visual outputs."""

    @staticmethod
    def save_plot(fig, title: str, tag: str | None = None):
        """Save a matplotlib figure with timestamped filename."""
        CFG.ensure_dirs()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if tag is None:
            fname = f"{CFG.prefix()}_{title.replace(' ', '_')}_{timestamp}.png"
        else:
            tag_s = str(tag).replace(" ", "_")
            fname = f"{CFG.prefix()}_{tag_s}_{title.replace(' ', '_')}_{timestamp}.png"
        fpath = os.path.join(CFG.OUTPUT_DIR, fname)
        fig.savefig(fpath, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"[Saved] {title} -> {fpath}")

    @staticmethod
    def show_amp_and_dist(amp, dist, tag: str | None = None):
        """Plot amplitude and distance images side-by-side."""
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(amp, cmap="gray")
        axs[0].set_title("Amplitude")
        axs[0].axis("off")
        axs[1].imshow(dist, cmap="gray")
        axs[1].set_title("Distance")
        axs[1].axis("off")
        plt.tight_layout()
        Visualizer.save_plot(fig, "Amplitude_Distance", tag=tag)

    @staticmethod
    def show_pc(pc, sample_rate=CFG.SAMPLE_RATE_PC, tag: str | None = None):
        """Plot a sampled 3D point cloud."""
        H, W, _ = pc.shape
        valid = ~np.isnan(pc).any(axis=2) & ~np.all(pc == 0, axis=2)
        subs = pc[::sample_rate, ::sample_rate, :]
        valid_s = valid[::sample_rate, ::sample_rate]
        x, y, z = subs[..., 0][valid_s], subs[..., 1][valid_s], subs[..., 2][valid_s]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x, y, z, c=z, s=1, cmap="jet")
        ax.set_title("Point Cloud")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.tight_layout()
        Visualizer.save_plot(fig, "PointCloud", tag=tag)
