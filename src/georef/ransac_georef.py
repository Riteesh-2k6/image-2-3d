"""
RANSAC Georeferencing & Outlier Purging Module
==============================================
Robustly rejects GNSS multipath, satellite dropouts, and near-threshold
systematic GPS noise using iterative random sample consensus.
"""

from typing import List, Tuple, Optional
import numpy as np
import logging

from src.georef.types import UmeyamaTransform, GeodeticAnchor
from src.georef.umeyama import solve_umeyama_similarity

logger = logging.getLogger(__name__)


class RANSACGeoreferencer:
    """
    RANSAC 7-DoF similarity estimator with collinearity rejection
    and consensus inlier refitting.
    """

    def __init__(
        self,
        inlier_threshold_m: float = 5.0,
        max_iterations: int = 1000,
        min_inlier_ratio: float = 0.50,
        random_seed: Optional[int] = 42
    ):
        self.inlier_threshold_m = inlier_threshold_m
        self.max_iterations = max_iterations
        self.min_inlier_ratio = min_inlier_ratio
        self.random_seed = random_seed

    def fit(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        lat0: float = 0.0,
        lon0: float = 0.0,
        alt0: float = 0.0
    ) -> Tuple[UmeyamaTransform, np.ndarray, np.ndarray]:
        """
        Fits 7-DoF Umeyama similarity alignment using RANSAC.
        
        Returns:
            best_transform: UmeyamaTransform refit on consensus inliers
            inlier_mask: (N,) boolean mask of inliers
            residuals: (N,) float array of final alignment residuals in meters
        """
        n = len(src_points)
        if n < 4:
            # Fall back to direct solve if not enough points for RANSAC
            transform = solve_umeyama_similarity(src_points, dst_points, lat0, lon0, alt0)
            aligned = transform.transform_points(src_points)
            res = np.linalg.norm(aligned - dst_points, axis=1)
            return transform, np.ones(n, dtype=bool), res

        rng = np.random.default_rng(self.random_seed)
        best_inlier_count = 0
        best_inlier_mask = np.ones(n, dtype=bool)
        best_transform = None

        sample_size = 4

        for it in range(self.max_iterations):
            # Sample 4 random indices
            sample_idx = rng.choice(n, size=sample_size, replace=False)
            src_sample = src_points[sample_idx]
            dst_sample = dst_points[sample_idx]

            # Check for degenerate/collinear configuration
            # SVD singular values of centered source points
            src_c = src_sample - np.mean(src_sample, axis=0)
            _, s_vals, _ = np.linalg.svd(src_c)
            if s_vals[-1] < 1e-4 * s_vals[0]:
                continue  # Near-collinear sample, skip

            try:
                candidate_transform = solve_umeyama_similarity(
                    src_sample, dst_sample, lat0, lon0, alt0
                )
                if candidate_transform.scale <= 1e-3 or candidate_transform.scale > 1e4:
                    continue  # Unphysical scale
            except Exception:
                continue

            # Evaluate candidate transform on all points
            aligned = candidate_transform.transform_points(src_points)
            residuals = np.linalg.norm(aligned - dst_points, axis=1)
            inliers = residuals <= self.inlier_threshold_m
            inlier_count = int(np.sum(inliers))

            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inlier_mask = inliers
                best_transform = candidate_transform

                # Early stopping if nearly all points are inliers
                if inlier_count >= int(0.98 * n):
                    break

        if best_inlier_count < 3:
            logger.warning(f"RANSAC found only {best_inlier_count} inliers at {self.inlier_threshold_m}m threshold. Falling back to direct solve.")
            best_transform = solve_umeyama_similarity(src_points, dst_points, lat0, lon0, alt0)
            best_inlier_mask = np.ones(n, dtype=bool)

        # Refit final transform on all consensus inliers
        inlier_indices = np.where(best_inlier_mask)[0]
        final_transform = solve_umeyama_similarity(
            src_points[inlier_indices],
            dst_points[inlier_indices],
            lat0, lon0, alt0
        )

        final_aligned = final_transform.transform_points(src_points)
        final_residuals = np.linalg.norm(final_aligned - dst_points, axis=1)
        final_inlier_mask = final_residuals <= self.inlier_threshold_m

        logger.info(
            f"RANSAC Georeferencing: {np.sum(final_inlier_mask)}/{n} inliers "
            f"({100.0 * np.mean(final_inlier_mask):.1f}%) at threshold {self.inlier_threshold_m}m."
        )

        return final_transform, final_inlier_mask, final_residuals
