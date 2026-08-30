"""
Internal Levenberg-Marquardt Bundle Adjustment (BA) Solver
===========================================================
Pure Python/SciPy robust non-linear least-squares optimizer to polish
camera poses and 3D structure when feed-forward confidence is low.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
import logging
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from src.sfm.types import CameraPose, CameraIntrinsics, PoseEngineMode
from src.sfm.feature_tracker import FeatureTrack

logger = logging.getLogger(__name__)


class LocalBundleAdjuster:
    """
    Levenberg-Marquardt optimizer for joint refinement of camera extrinsics
    and 3D feature points using robust Huber loss.
    """

    def __init__(self, max_nfev: int = 50, loss_cutoff_px: float = 2.0, max_tracks: int = 3000):
        self.max_nfev = max_nfev
        self.loss_cutoff_px = loss_cutoff_px
        self.max_tracks = max_tracks

    @staticmethod
    def rotation_matrix_to_rodrigues(R: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to 3D Rodrigues axis-angle vector."""
        rvec, _ = cv2.Rodrigues(R.astype(np.float64))
        return rvec.ravel()

    @staticmethod
    def rodrigues_to_rotation_matrix(rvec: np.ndarray) -> np.ndarray:
        """Convert 3D Rodrigues axis-angle vector to 3x3 rotation matrix."""
        R, _ = cv2.Rodrigues(rvec.astype(np.float64))
        return R

    def optimize(
        self,
        intrinsics: CameraIntrinsics,
        poses: List[CameraPose],
        tracks: List[FeatureTrack]
    ) -> Tuple[List[CameraPose], List[FeatureTrack]]:
        """
        Executes Levenberg-Marquardt bundle adjustment:
        Minimizes sum of robust Huber reprojection errors across all observations.
        """
        # Filter valid tracks with triangulated 3D points
        valid_tracks = [t for t in tracks if t.point_3d is not None and t.track_length >= 2]
        if not valid_tracks or len(poses) <= 1:
            return poses, tracks

        # Select top longest multi-view tracks for fast and well-conditioned BA solve
        if len(valid_tracks) > self.max_tracks:
            valid_tracks = sorted(valid_tracks, key=lambda t: len(t.observations), reverse=True)[:self.max_tracks]

        n_cameras = len(poses)
        n_points = len(valid_tracks)

        # Index mapping
        camera_indices = []
        point_indices = []
        points_2d = []

        for p_idx, track in enumerate(valid_tracks):
            for cam_idx, (u, v) in track.observations.items():
                if cam_idx < n_cameras:
                    camera_indices.append(cam_idx)
                    point_indices.append(p_idx)
                    points_2d.append([u, v])

        camera_indices = np.array(camera_indices, dtype=np.int32)
        point_indices = np.array(point_indices, dtype=np.int32)
        points_2d = np.array(points_2d, dtype=np.float64)
        n_observations = len(points_2d)

        if n_observations < 20:
            return poses, tracks

        # Initial parameters vector:
        # For each camera i (except camera 0 to fix gauge): 3 params for rotation + 3 params for translation = 6 params
        # For each 3D point j: 3 params = 3 params
        camera_params = np.empty((n_cameras, 6), dtype=np.float64)
        for i, pose in enumerate(poses):
            camera_params[i, :3] = self.rotation_matrix_to_rodrigues(pose.R)
            camera_params[i, 3:] = pose.t

        # Gauge fixing: Camera 0 is fixed (6 DoF), Camera 1 translation is fixed to preserve metric scale gauge (1 DoF)
        # Parameters to optimize:
        # - Camera 1 rotation (3)
        # - Camera 2..N rotation and translation ((N-2) * 6)
        # - 3D Points (M * 3)
        cam1_rot = camera_params[1, :3]
        cam_rest = camera_params[2:].ravel() if n_cameras > 2 else np.empty(0, dtype=np.float64)
        points_3d = np.array([t.point_3d for t in valid_tracks], dtype=np.float64)
        
        x0 = np.hstack([cam1_rot, cam_rest, points_3d.ravel()])
        K = intrinsics.K
        fixed_c0 = camera_params[0]
        fixed_t1 = camera_params[1, 3:]

        # Pre-compute initial telemetry target step distances to anchor scale
        target_step_dists = []
        for i in range(1, n_cameras):
            c_p = -self.rodrigues_to_rotation_matrix(camera_params[i-1, :3]).T @ camera_params[i-1, 3:]
            c_c = -self.rodrigues_to_rotation_matrix(camera_params[i, :3]).T @ camera_params[i, 3:]
            target_step_dists.append(float(np.linalg.norm(c_c - c_p)))
        target_step_dists = np.array(target_step_dists, dtype=np.float64)

        def fun(params):
            """Residual cost function with 7-DoF gauge fixing and metric scale regularization."""
            c_params = np.empty((n_cameras, 6), dtype=np.float64)
            c_params[0] = fixed_c0
            c_params[1, :3] = params[:3]
            c_params[1, 3:] = fixed_t1 # Fixed scale gauge
            
            if n_cameras > 2:
                c_params[2:] = params[3 : 3 + (n_cameras - 2) * 6].reshape((n_cameras - 2, 6))

            p_offset = 3 + (n_cameras - 2) * 6 if n_cameras > 2 else 3
            p3d = params[p_offset :].reshape((n_points, 3))

            proj = np.empty((n_observations, 2), dtype=np.float64)

            # Vectorized projection grouped by camera
            for cam_i in range(n_cameras):
                cam_mask = camera_indices == cam_i
                if not np.any(cam_mask):
                    continue
                pts_i = p3d[point_indices[cam_mask]]
                rv = c_params[cam_i, :3]
                tv = c_params[cam_i, 3:]
                p2d, _ = cv2.projectPoints(pts_i, rv, tv, K, None)
                proj[cam_mask] = p2d.reshape(-1, 2)

            reproj_res = (proj - points_2d).ravel()

            # Metric scale prior (prevents optimizer from shrinking baseline scale)
            scale_res = np.empty(n_cameras - 1, dtype=np.float64)
            for i in range(1, n_cameras):
                # Camera centers C = -R^T * t
                R_prev = self.rodrigues_to_rotation_matrix(c_params[i-1, :3])
                R_curr = self.rodrigues_to_rotation_matrix(c_params[i, :3])
                cp = -R_prev.T @ c_params[i-1, 3:]
                cc = -R_curr.T @ c_params[i, 3:]
                d = np.linalg.norm(cc - cp)
                scale_res[i-1] = 0.5 * (d - target_step_dists[i-1])

            return np.hstack([reproj_res, scale_res])

        # Build Jacobian sparsity structure for fast sparse LM optimization
        n_params = len(x0)
        n_res = 2 * n_observations + (n_cameras - 1)
        A = lil_matrix((n_res, n_params), dtype=int)

        obs_indices = np.arange(n_observations)
        
        # Camera 1 (only rotation is optimized)
        mask1 = camera_indices == 1
        obs_cam1 = obs_indices[mask1]
        for param_idx in range(3):
            A[2 * obs_cam1, param_idx] = 1
            A[2 * obs_cam1 + 1, param_idx] = 1

        # Cameras 2..N
        for i in range(n_cameras - 2):
            cam_idx = i + 2
            mask = camera_indices == cam_idx
            obs_cam = obs_indices[mask]
            for param_idx in range(6):
                col = 3 + i * 6 + param_idx
                A[2 * obs_cam, col] = 1
                A[2 * obs_cam + 1, col] = 1

        cam_param_count = 3 + (n_cameras - 2) * 6 if n_cameras > 2 else 3
        for j in range(n_points):
            mask = point_indices == j
            obs_pt = obs_indices[mask]
            for param_idx in range(3):
                col = cam_param_count + j * 3 + param_idx
                A[2 * obs_pt, col] = 1
                A[2 * obs_pt + 1, col] = 1

        # Scale regularization rows
        for i in range(1, n_cameras):
            row_idx = 2 * n_observations + (i - 1)
            if i == 1:
                # Camera 1 rotation
                for p in range(3):
                    A[row_idx, p] = 1
            elif i == 2:
                # Camera 1 rot + Camera 2 (rot + trans)
                for p in range(3):
                    A[row_idx, p] = 1
                for p in range(6):
                    A[row_idx, 3 + p] = 1
            else:
                # Camera i-1 and Camera i
                col_prev = 3 + (i - 2) * 6
                col_curr = 3 + (i - 1) * 6
                for p in range(6):
                    A[row_idx, col_prev + p] = 1
                    A[row_idx, col_curr + p] = 1

        # Execute Levenberg-Marquardt optimization with Huber loss
        try:
            res = least_squares(
                fun,
                x0,
                jac_sparsity=A,
                verbose=0,
                x_scale="jac",
                ftol=1e-4,
                method="trf",
                loss="huber",
                f_scale=self.loss_cutoff_px,
                max_nfev=self.max_nfev
            )
            opt_params = res.x
        except Exception as e:
            logger.error(f"Levenberg-Marquardt bundle adjustment failed: {e}", exc_info=True)
            raise RuntimeError(f"Bundle Adjustment optimizer failed: {e}") from e

        # Validate that BA executed and produced non-trivial optimization
        param_shift = np.linalg.norm(opt_params - x0)
        logger.info(f"Bundle Adjustment completed (status: {res.status}, nfev: {res.nfev}, param shift: {param_shift:.4f})")

        # Update camera poses with optimized values
        opt_c_params = np.empty((n_cameras, 6), dtype=np.float64)
        opt_c_params[0] = fixed_c0
        opt_c_params[1, :3] = opt_params[:3]
        opt_c_params[1, 3:] = fixed_t1
        if n_cameras > 2:
            opt_c_params[2:] = opt_params[3 : 3 + (n_cameras - 2) * 6].reshape((n_cameras - 2, 6))

        for i, pose in enumerate(poses):
            rvec = opt_c_params[i, :3]
            tvec = opt_c_params[i, 3:]
            pose.R = self.rodrigues_to_rotation_matrix(rvec)
            pose.t = tvec
            pose.engine_mode = PoseEngineMode.LM_BUNDLE_ADJUSTMENT

        # Update 3D points
        p_offset = 3 + (n_cameras - 2) * 6 if n_cameras > 2 else 3
        opt_p3d = opt_params[p_offset :].reshape((n_points, 3))
        for idx, track in enumerate(valid_tracks):
            track.point_3d = opt_p3d[idx]

        return poses, tracks
