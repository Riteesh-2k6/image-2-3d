"""
Geodetically Constrained Joint Bundle Adjustment
================================================
Jointly optimizes camera extrinsics [R_i | t_i] and 3D point cloud X_j
with GPS spatial anchors to eliminate gauge drift and lock metric scale.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
import logging
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from src.sfm.types import CameraPose, CameraIntrinsics, PoseEngineMode
from src.sfm.feature_tracker import FeatureTrack
from src.georef.types import GeodeticAnchor

logger = logging.getLogger(__name__)


class GeodeticBundleAdjuster:
    """
    Levenberg-Marquardt optimizer with joint optical reprojection residuals
    and geodetic GPS position prior constraints.
    """

    def __init__(
        self,
        lambda_geo: float = 0.1,
        loss_cutoff_px: float = 2.0,
        huber_delta: float = 1.5,
        max_nfev: int = 40,
        max_tracks: int = 3000
    ):
        self.lambda_geo = lambda_geo
        self.loss_cutoff_px = loss_cutoff_px
        self.huber_delta = huber_delta
        self.max_nfev = max_nfev
        self.max_tracks = max_tracks

    @staticmethod
    def rotation_matrix_to_rodrigues(R: np.ndarray) -> np.ndarray:
        rvec, _ = cv2.Rodrigues(R.astype(np.float64))
        return rvec.ravel()

    @staticmethod
    def rodrigues_to_rotation_matrix(rvec: np.ndarray) -> np.ndarray:
        R, _ = cv2.Rodrigues(rvec.astype(np.float64))
        return R

    def optimize(
        self,
        intrinsics: CameraIntrinsics,
        poses: List[CameraPose],
        tracks: List[FeatureTrack],
        anchors: List[GeodeticAnchor],
        inlier_mask: Optional[np.ndarray] = None
    ) -> Tuple[List[CameraPose], List[FeatureTrack]]:
        """
        Executes Geodetically Constrained Bundle Adjustment.
        
        Minimizes:
            E_total = sum(rho_Huber(e_reproj)) + lambda_geo * sum(w_i * ||C_i - GPS_i||^2)
        """
        valid_tracks = [t for t in tracks if t.point_3d is not None and t.track_length >= 2]
        if not valid_tracks or len(poses) <= 1:
            return poses, tracks

        if len(valid_tracks) > self.max_tracks:
            valid_tracks = sorted(valid_tracks, key=lambda t: len(t.observations), reverse=True)[:self.max_tracks]

        n_cameras = len(poses)
        n_points = len(valid_tracks)

        # 1. Optical observation indexing
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

        # 2. GPS anchor indexing
        if inlier_mask is None:
            inlier_mask = np.ones(len(anchors), dtype=bool)

        anchor_map = {a.frame_idx: a for i, a in enumerate(anchors) if inlier_mask[i]}
        active_anchor_cams = [i for i, p in enumerate(poses) if p.frame_idx in anchor_map]
        n_anchors = len(active_anchor_cams)

        # 3. Parameter vector layout:
        # All N cameras are optimized (gauge is anchored by GPS positions):
        # 6 params per camera (3 rotation + 3 translation) + 3 params per 3D point
        cam_params = np.empty((n_cameras, 6), dtype=np.float64)
        for i, pose in enumerate(poses):
            cam_params[i, :3] = self.rotation_matrix_to_rodrigues(pose.R)
            cam_params[i, 3:] = pose.t

        point_params = np.array([t.point_3d for t in valid_tracks], dtype=np.float64)
        x0 = np.hstack([cam_params.ravel(), point_params.ravel()])

        fx, fy = intrinsics.fx, intrinsics.fy
        cx, cy = intrinsics.cx, intrinsics.cy
        k1, k2 = intrinsics.k1, intrinsics.k2
        geo_weight = float(np.sqrt(self.lambda_geo))

        # 4. Residual function
        def fun(params: np.ndarray) -> np.ndarray:
            c_params = params[: n_cameras * 6].reshape((n_cameras, 6))
            p_params = params[n_cameras * 6 :].reshape((n_points, 3))

            # Optical Reprojection Residuals
            c_idx = camera_indices
            p_idx = point_indices

            rvecs = c_params[c_idx, :3]
            tvecs = c_params[c_idx, 3:]
            pts = p_params[p_idx]

            # Vectorized Rodrigues rotation: R @ pt
            theta = np.linalg.norm(rvecs, axis=1, keepdims=True)
            k = np.divide(rvecs, theta, out=np.zeros_like(rvecs), where=theta > 1e-12)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            dot_kp = np.sum(k * pts, axis=1, keepdims=True)
            cross_kp = np.cross(k, pts)
            p_cam = pts * cos_t + cross_kp * sin_t + k * dot_kp * (1.0 - cos_t) + tvecs

            z = np.clip(p_cam[:, 2], 1e-3, 1e6)
            inv_z = 1.0 / z
            x_norm = p_cam[:, 0] * inv_z
            y_norm = p_cam[:, 1] * inv_z

            r2 = x_norm ** 2 + y_norm ** 2
            radial = 1.0 + k1 * r2 + k2 * (r2 ** 2)

            u_proj = fx * x_norm * radial + cx
            v_proj = fy * y_norm * radial + cy

            proj_res = np.empty((n_observations, 2), dtype=np.float64)
            proj_res[:, 0] = u_proj - points_2d[:, 0]
            proj_res[:, 1] = v_proj - points_2d[:, 1]
            reproj_flat = proj_res.ravel()

            # Geodetic Anchor Residuals: C_i = -R_i^T t_i vs GPS_i
            geo_res = []
            if geo_weight > 1e-6 and n_anchors > 0:
                for cam_i in active_anchor_cams:
                    rvec = c_params[cam_i, :3]
                    tvec = c_params[cam_i, 3:]
                    R_mat = GeodeticBundleAdjuster.rodrigues_to_rotation_matrix(rvec)
                    C_est = -R_mat.T @ tvec
                    
                    frame_idx = poses[cam_i].frame_idx
                    gps_target = anchor_map[frame_idx].enu_gt
                    w = anchor_map[frame_idx].weight
                    diff = geo_weight * w * (C_est - gps_target)
                    geo_res.extend(diff.tolist())

            if len(geo_res) > 0:
                return np.hstack([reproj_flat, np.array(geo_res, dtype=np.float64)])
            return reproj_flat

        # 5. Sparse Jacobian Sparsity Matrix
        n_params = len(x0)
        n_reproj_res = 2 * n_observations
        n_geo_res = 3 * n_anchors if geo_weight > 1e-6 else 0
        total_res = n_reproj_res + n_geo_res
        A = lil_matrix((total_res, n_params), dtype=int)

        obs_indices = np.arange(n_observations)
        for i in range(n_cameras):
            mask = camera_indices == i
            obs_cam = obs_indices[mask]
            for param_idx in range(6):
                col = i * 6 + param_idx
                A[2 * obs_cam, col] = 1
                A[2 * obs_cam + 1, col] = 1

        cam_param_count = n_cameras * 6
        for j in range(n_points):
            mask = point_indices == j
            obs_pt = obs_indices[mask]
            for param_idx in range(3):
                col = cam_param_count + j * 3 + param_idx
                A[2 * obs_pt, col] = 1
                A[2 * obs_pt + 1, col] = 1

        # Geodetic residual sparsity rows
        if n_geo_res > 0:
            for a_idx, cam_i in enumerate(active_anchor_cams):
                r_start = n_reproj_res + a_idx * 3
                for axis in range(3):
                    row = r_start + axis
                    for p in range(6):
                        A[row, cam_i * 6 + p] = 1

        # 6. Execute Levenberg-Marquardt optimization with Huber loss
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
            logger.error(f"Geodetic Bundle Adjustment failed: {e}", exc_info=True)
            raise RuntimeError(f"Geodetic Bundle Adjustment failed: {e}") from e

        param_shift = float(np.linalg.norm(opt_params - x0))
        logger.info(
            f"Geodetic BA complete (status: {res.status}, nfev: {res.nfev}, "
            f"lambda_geo: {self.lambda_geo}, param shift: {param_shift:.4f})"
        )

        # 7. Update camera poses and 3D tracks
        opt_c_params = opt_params[: n_cameras * 6].reshape((n_cameras, 6))
        opt_p_params = opt_params[n_cameras * 6 :].reshape((n_points, 3))

        for i, pose in enumerate(poses):
            rvec = opt_c_params[i, :3]
            tvec = opt_c_params[i, 3:]
            pose.R = self.rodrigues_to_rotation_matrix(rvec)
            pose.t = tvec
            pose.engine_mode = PoseEngineMode.LM_BUNDLE_ADJUSTMENT

        for j, track in enumerate(valid_tracks):
            track.point_3d = opt_p_params[j]

        return poses, valid_tracks
