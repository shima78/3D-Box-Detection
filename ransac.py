import numpy as np
from boxdetect.geometry import PlaneFitter

class RANSACPlaneDetector:
    """Detect a dominant plane using RANSAC estimators.

    Supported scoring / selection strategies:
      - "ransac": classic RANSAC (maximize number of inliers)
      - "mlesac": MLESAC-style cost (minimize truncated distance cost)

    Supported execution modes:
      - Standard: sequentially sample and score hypotheses for `num_iterations`
      - Preemptive: generate `M` hypotheses once and progressively prune them
        while scoring on batches of `B` points ("preemptive_ransac").

    Notes
    -----
    * In all modes we return the best plane and the indices of inliers
      (distance < `distance_threshold`) w.r.t. that plane.
    """

    def __init__(
        self,
        num_iterations: int = 100,
        distance_threshold: float = 0.02,
        min_inliers: int = 100,
        seed=None,
        *,
        score_method: str = "ransac",
        gamma: float | None = None,
        preemptive: bool = False,
        M: int = 256,
        B: int = 20,
    ):
        self.num_iterations = num_iterations
        self.distance_threshold = distance_threshold
        self.min_inliers = min_inliers
        self.rng = np.random.default_rng(seed)

        self.score_method = str(score_method).lower()
        if self.score_method not in {"ransac", "mlesac"}:
            raise ValueError(f"Unknown score_method: {score_method}")

        # MLESAC: constant outlier penalty (must be > threshold)
        if gamma is None:
            gamma = 2.0 * self.distance_threshold
        self.gamma = float(gamma)
        if self.gamma <= self.distance_threshold:
            self.gamma = float(self.distance_threshold) * 1.01


        # Preemptive settings
        self.preemptive = bool(preemptive)
        self.M = int(M)
        self.B = int(B)

    def _dists(self, points: np.ndarray, plane) -> np.ndarray:
        return np.abs(PlaneFitter.distances_to_plane(points, plane))

    def _mlesac_cost(self, dists: np.ndarray) -> float:
        # Eq. (2) in exercise sheet: sum(d_i if d_i < eps else gamma)
        eps = self.distance_threshold
        return float(np.sum(np.where(dists < eps, dists, self.gamma)))

    def _score_plane(self, points: np.ndarray, plane):
        dists = self._dists(points, plane)
        inliers = np.flatnonzero(dists < self.distance_threshold)

        if self.score_method == "ransac":
            # Higher is better
            return inliers.size, inliers
        else:  # mlesac
            # Lower is better; still enforce a minimum inlier count to avoid degeneracy
            return self._mlesac_cost(dists), inliers

    def detect_plane(self, points: np.ndarray):
        """Estimate plane parameters from a point set.

        Args:
            points: (N, 3) array of 3D points.

        Returns:
            (best_plane, inlier_indices) where inlier_indices are indices into `points`.
        """
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be of shape (N, 3)")
        N = points.shape[0]
        if N < 3:
            return None, np.empty(0, dtype=int)

        if self.preemptive:
            return self._detect_plane_preemptive(points)

        best_plane, best_inliers = None, np.empty(0, dtype=int)

        # For ransac we maximize score; for mlesac we minimize cost.
        if self.score_method == "ransac":
            best_score = -1
        else:
            best_score = np.inf

        for _ in range(self.num_iterations):
            idx = self.rng.choice(N, 3, replace=False)
            plane = PlaneFitter.fit_plane(points[idx])
            if plane is None:
                continue

            score, inliers = self._score_plane(points, plane)

            if inliers.size < self.min_inliers:
                continue

            if self.score_method == "ransac":
                if score > best_score:
                    best_score, best_plane, best_inliers = score, plane, inliers
            else:  # mlesac
                if score < best_score:
                    best_score, best_plane, best_inliers = score, plane, inliers

        return best_plane, best_inliers

    def _detect_plane_preemptive(self, points: np.ndarray):
        """Preemptive RANSAC (Niester 2003) with either RANSAC or MLESAC scoring.

        Algorithm:
          1) Sample M hypotheses.
          2) Score all hypotheses on the next batch of B points.
          3) Keep only the top f(i) hypotheses, where i is the number of points processed.
          4) Repeat until 1 hypothesis remains or all points processed.

        Preemption function (exercise sheet Eq. (3) with common interpretation):
            f(i) = floor(M * 2^{-floor(i / B)})

        Returns:
            (best_plane, inlier_indices) where inlier_indices are indices into `points`.
        """
        N = points.shape[0]
        M = max(1, min(self.M, 10_000))
        B = max(1, self.B)

        # Sample M valid hypotheses
        hyps = []
        tries = 0
        max_tries = M * 10
        while len(hyps) < M and tries < max_tries:
            tries += 1
            idx = self.rng.choice(N, 3, replace=False)
            plane = PlaneFitter.fit_plane(points[idx])
            if plane is None:
                continue
            hyps.append(plane)

        if not hyps:
            return None, np.empty(0, dtype=int)

        hyps = np.array(hyps, dtype=object)
        costs = np.zeros(len(hyps), dtype=float)

        # Evaluate points in a random order to reduce ordering bias
        order = self.rng.permutation(N)
        processed = 0

        while len(hyps) > 1 and processed < N:
            batch_idx = order[processed : min(N, processed + B)]
            batch_pts = points[batch_idx]

            # Score all remaining hypotheses on this batch
            for h in range(len(hyps)):
                d = self._dists(batch_pts, hyps[h])
                if self.score_method == "ransac":
                    # Classic RANSAC preemptive: count outliers in this batch (lower is better)
                    costs[h] += float(np.sum(d >= self.distance_threshold))
                else:
                    costs[h] += self._mlesac_cost(d)

            processed += len(batch_idx)

            # Preempt / keep top hypotheses according to f(i)
            k = int(np.floor(len(costs) * (2.0 ** (-np.floor(processed / B)))))
            k = max(1, k)

            keep = np.argsort(costs)[:k]
            hyps = hyps[keep]
            costs = costs[keep]

        # Best remaining hypothesis (lowest cost)
        best_plane = hyps[int(np.argmin(costs))]
        d_all = self._dists(points, best_plane)
        inliers = np.flatnonzero(d_all < self.distance_threshold)

        # Optional: enforce min_inliers (otherwise return what we have)
        if inliers.size < self.min_inliers:
            return best_plane, inliers

        return best_plane, inliers
