"""
Chapter 6 Pose Confidence & Verification Metrics Module
========================================================
Calculates the 4 required Chapter 6 metrics:
1. Pose Confidence Score (composite harmonic score [0, 1])
2. Reprojection Residual (RMS in pixels, target <= 1.5px)
3. Feature-Track Consistency (multi-view track stability [0, 1])
4. Camera Graph Connectivity (Laplacian Fiedler eigenvalue [0, 1])
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from src.sfm.types import CameraPose, CameraIntrinsics, SparsePointCloud, PoseConfidenceMetrics
from src.sfm.feature_tracker import FeatureTrack, PairwiseMatch


class PoseConfidenceCalculator:
    """Evaluates geometric quality and structural integrity of estimated camera poses."""

    def __init__(
        self,
        target_max_reproj_px: float = 1.5,
        min_connectivity_thresh: float = 0.5,
        min_confidence_thresh: float = 0.75,
    ):
        self.target_max_reproj_px = target_max_reproj_px
        self.min_connectivity_thresh = min_connectivity_thresh
        self.min_confidence_thresh = min_confidence_thresh

    def compute_reprojection_residuals(
        self,
        intrinsics: CameraIntrinsics,
        poses: Dict[int, CameraPose],
        tracks: List[FeatureTrack]
    ) -> Tuple[float, np.ndarray]:
        """
        Computes overall RMS reprojection error (in pixels) across all valid 3D points and 2D observations.
        Returns: (overall_rms_px, per_point_residuals)
        """
        K = intrinsics.K
        residuals = []
        point_errors = []

        for track in tracks:
            if track.point_3d is None:
                continue

            X_world = track.point_3d # (3,)
            X_h = np.array([X_world[0], X_world[1], X_world[2], 1.0], dtype=np.float64)

            track_errs = []
            for img_idx, (u_obs, v_obs) in track.observations.items():
                if img_idx not in poses:
                    continue
                pose = poses[img_idx]
                # P = K @ [R | t]
                P = K @ pose.world_to_camera_matrix[:3, :]
                x_proj = P @ X_h

                if x_proj[2] <= 1e-4:
                    continue # Behind camera

                u_proj = x_proj[0] / x_proj[2]
                v_proj = x_proj[1] / x_proj[2]

                err = np.sqrt((u_proj - u_obs)**2 + (v_proj - v_obs)**2)
                residuals.append(err)
                track_errs.append(err)

            if track_errs:
                mean_track_err = float(np.mean(track_errs))
                track.reprojection_error = mean_track_err
                point_errors.append(mean_track_err)

        if not residuals:
            return 0.0, np.zeros(0, dtype=np.float64)

        rms_px = float(np.sqrt(np.mean(np.square(residuals))))
        return rms_px, np.array(point_errors, dtype=np.float64)

    def compute_feature_track_consistency(self, tracks: List[FeatureTrack]) -> float:
        """
        Calculates track consistency ratio:
        Weighted ratio of tracks observed across >= 3 independent views,
        penalizing short or volatile 2-view matches.
        """
        if not tracks:
            return 0.0

        lengths = [t.track_length for t in tracks]
        multi_view_count = sum(1 for L in lengths if L >= 3)
        mean_length = np.mean(lengths)

        # Multi-view ratio weighted by track length coverage
        consistency = (multi_view_count / len(tracks)) * min(1.0, mean_length / 4.0)
        return float(np.clip(consistency, 0.0, 1.0))

    def compute_camera_graph_connectivity(
        self,
        num_cameras: int,
        matches: List[PairwiseMatch]
    ) -> float:
        """
        Calculates algebraic connectivity (Fiedler value of normalized Laplacian)
        of the multi-view essential matrix camera graph.
        """
        if num_cameras <= 1:
            return 1.0
        if not matches:
            return 0.0

        # Build adjacency matrix
        A = np.zeros((num_cameras, num_cameras), dtype=np.float64)
        for m in matches:
            i, j = m.src_idx, m.dst_idx
            if i < num_cameras and j < num_cameras:
                weight = m.inlier_ratio * len(m.matches_src)
                A[i, j] += weight
                A[j, i] += weight

        # Degree matrix
        d = np.sum(A, axis=1)
        isolated_nodes = np.sum(d == 0)
        if isolated_nodes > 0:
            # Graph is disconnected
            connected_ratio = (num_cameras - isolated_nodes) / num_cameras
            return float(0.2 * connected_ratio)

        # Normalized Laplacian L_norm = I - D^(-1/2) A D^(-1/2)
        d_inv_sqrt = np.power(d, -0.5, where=d > 0)
        d_inv_sqrt[d == 0] = 0.0
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L_norm = np.eye(num_cameras) - D_inv_sqrt @ A @ D_inv_sqrt

        # Eigenvalues
        try:
            eigenvalues = np.linalg.eigvalsh(L_norm)
            eigenvalues = np.sort(eigenvalues)
            # Fiedler eigenvalue is the second smallest eigenvalue lambda_2
            lambda_2 = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
            # Normalize lambda_2 (which ranges in [0, 2 * N/(N-1)])
            norm_fiedler = lambda_2 / (2.0 * num_cameras / max(1, num_cameras - 1))
            return float(np.clip(norm_fiedler * 3.0, 0.0, 1.0))
        except Exception:
            return 0.5

    def evaluate_metrics(
        self,
        intrinsics: CameraIntrinsics,
        poses: List[CameraPose],
        tracks: List[FeatureTrack],
        matches: List[PairwiseMatch],
        total_input_frames: int
    ) -> PoseConfidenceMetrics:
        """Computes all 4 metrics and synthesizes the global composite confidence score."""
        pose_dict = {p.frame_idx: p for p in poses}
        rms_reproj, point_errors = self.compute_reprojection_residuals(intrinsics, pose_dict, tracks)
        track_consistency = self.compute_feature_track_consistency(tracks)
        connectivity = self.compute_camera_graph_connectivity(len(poses), matches)

        # Mean inlier match ratio
        mean_inlier_ratio = float(np.mean([m.inlier_ratio for m in matches])) if matches else 0.0

        # Reprojection score: 1.0 if rms <= 0.5px, decays smoothly to 0.0 at 3.0px
        reproj_score = float(np.exp(-0.5 * (rms_reproj / self.target_max_reproj_px)**2))

        # Composite Pose Confidence Score (harmonic blend)
        reg_ratio = len(poses) / max(1, total_input_frames)
        composite_score = float(
            0.35 * reproj_score +
            0.25 * connectivity +
            0.20 * track_consistency +
            0.20 * reg_ratio
        )
        composite_score = float(np.clip(composite_score, 0.0, 1.0))

        # Update per-pose reprojection and confidence
        for p in poses:
            p.confidence = composite_score
            p.reprojection_error_px = rms_reproj

        return PoseConfidenceMetrics(
            pose_confidence_score=composite_score,
            reprojection_residual_px=rms_reproj,
            feature_track_consistency=track_consistency,
            camera_graph_connectivity=connectivity,
            inlier_match_ratio=mean_inlier_ratio,
            num_registered_frames=len(poses),
            total_frames=total_input_frames,
        )
