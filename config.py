"""
Configuration module for the box measurement pipeline.
"""

import os
import numpy as np
from dataclasses import dataclass


@dataclass
class Config:
    """
    Stores global configuration parameters for the pipeline.
    """
    PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR: str = os.path.join(PROJECT_ROOT, "")
    OUTPUT_DIR: str = os.path.join(PROJECT_ROOT, "outputs")

    RANDOM_SEED: int = 42
    RANSAC_ITER: int = 200
    RANSAC_DIST_THRESH: float = 0.1
    RANSAC_MIN_INLIERS: int = 500
    TOP_RANSAC_ITER: int = 200
    TOP_RANSAC_DIST_THRESH: float = 0.02
    TOP_RANSAC_MIN_INLIERS: int = 50
    CLOSE_SIZE: int = 9
    OPEN_SIZE: int = 5
    SAMPLE_RATE_PC: int = 4
    CURRENT_DATA_FILE: str | None = None

    # ================= MLESAC =================
    MLESAC_ITER = 200
    MLESAC_DIST_THRESH = 0.1
    MLESAC_MIN_INLIERS = 500
    MLESAC_GAMMA = 0.2   # must be > DIST_THRESH

    TOP_MLESAC_ITER: int = 200
    TOP_MLESAC_DIST_THRESH: float = 0.02
    TOP_MLESAC_MIN_INLIERS: int = 50
    TOP_MLESAC_GAMMA: float = 0.04

    # ================= Preemptive RANSAC =================
    PREEMPTIVE_M = 1000      # number of initial hypotheses
    PREEMPTIVE_B = 50        # batch size
    PREEMPTIVE_DIST_THRESH = 0.05
    PREEMPTIVE_MIN_INLIERS = 10
    PREEMPTIVE_GAMMA = 0.1
    PREEMPTIVE_SCORE_METHOD: str = "mlesac"
    TOP_PREEMPTIVE_SCORE_METHOD: str = "mlesac"
    TOP_PREEMPTIVE_M: int = 1000
    TOP_PREEMPTIVE_B: int = 50
    TOP_PREEMPTIVE_DIST_THRESH: float = 0.02
    TOP_PREEMPTIVE_MIN_INLIERS: int = 50
    TOP_PREEMPTIVE_GAMMA: float = 0.04

        # sweep lists for evaluation
    PREEMPTIVE_M_LIST = [128, 256, 512, 1024]
    PREEMPTIVE_B_LIST = [10, 20, 50]



    def ensure_dirs(self) -> None:
        """Ensure output directories exist."""
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    def set_current_file(self, path: str) -> None:
        """Set the currently processed filename for naming outputs."""
        self.CURRENT_DATA_FILE = os.path.splitext(os.path.basename(path))[0]

    def prefix(self) -> str:
        """Return a short prefix for saving files."""
        return self.CURRENT_DATA_FILE or "unknown"


CFG = Config()
np.random.seed(CFG.RANDOM_SEED)
