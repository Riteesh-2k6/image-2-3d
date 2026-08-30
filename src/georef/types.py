"""
Georeferencing Data Models & Types
===================================
Defines 7-DoF similarity transformations, geodetic anchors,
and evaluation metrics for Chapter 7 Georeferencing.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

from src.sfm.types import CameraPose, SparsePointCloud, CameraIntrinsics


@dataclass
class UmeyamaTransform:
    """7-DoF Similarity Transformation: p_world = scale * (R @ p_local) + translation."""
    scale: float                            # Uniform scale factor s in R+
    R: np.ndarray                           # 3x3 rotation matrix in SO(3)
    translation: np.ndarray                 # 3D translation vector [tx, ty, tz] in meters
    lat0: float                             # Reference origin latitude in degrees
    lon0: float                             # Reference origin longitude in degrees
    alt0: float                             # Reference origin ellipsoidal height in meters

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Applies 7-DoF transformation to an (N, 3) point array."""
        if len(points) == 0:
            return np.empty((0, 3), dtype=np.float64)
        return self.scale * (points @ self.R.T) + self.translation

    def transform_camera_pose(self, pose: CameraPose) -> CameraPose:
        """Transforms a local camera pose into global georeferenced coordinates."""
        # Local optical center C_loc = -R_loc^T t_loc
        C_loc = pose.camera_center
        # Georeferenced optical center C_geo = scale * (R_align @ C_loc) + t_align
        C_geo = self.scale * (self.R @ C_loc) + self.translation
        
        # New camera rotation: R_cam_new = R_loc @ R_align^T
        # such that R_cam_new @ (C_geo - t_align) / scale = R_loc @ C_loc = -t_loc
        R_new = pose.R @ self.R.T
        t_new = -R_new @ C_geo

        return CameraPose(
            frame_idx=pose.frame_idx,
            timestamp_sec=pose.timestamp_sec,
            R=R_new,
            t=t_new,
            confidence=pose.confidence,
            image_name=pose.image_name,
            telemetry_gps=pose.telemetry_gps,
            telemetry_gimbal=pose.telemetry_gimbal,
            engine_mode=pose.engine_mode
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scale": float(self.scale),
            "rotation_matrix": self.R.tolist(),
            "translation_m": self.translation.tolist(),
            "origin_lat": float(self.lat0),
            "origin_lon": float(self.lon0),
            "origin_alt": float(self.alt0),
        }


@dataclass
class GeodeticAnchor:
    """Individual keyframe ground truth GNSS observation."""
    frame_idx: int
    timestamp: float
    enu_gt: np.ndarray                      # (3,) East-North-Up coordinates in meters
    lat: float
    lon: float
    alt_m: float
    num_satellites: int = 15
    weight: float = 1.0


@dataclass
class GeorefMetrics:
    """Quantitative evaluation of georeferencing accuracy."""
    horizontal_rmse_m: float
    vertical_rmse_m: float
    ate_3d_rmse_m: float
    ate_3d_median_m: float
    ate_3d_mean_m: float
    inlier_ratio: float
    num_inliers: int
    total_anchors: int
    scale_factor: float
    reprojection_rmse_px: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizontal_rmse_m": float(self.horizontal_rmse_m),
            "vertical_rmse_m": float(self.vertical_rmse_m),
            "ate_3d_rmse_m": float(self.ate_3d_rmse_m),
            "ate_3d_median_m": float(self.ate_3d_median_m),
            "ate_3d_mean_m": float(self.ate_3d_mean_m),
            "inlier_ratio": float(self.inlier_ratio),
            "num_inliers": int(self.num_inliers),
            "total_anchors": int(self.total_anchors),
            "scale_factor": float(self.scale_factor),
            "reprojection_rmse_px": float(self.reprojection_rmse_px)
        }


@dataclass
class GeorefResult:
    """Complete output of the Georeferencing Engine."""
    transform: UmeyamaTransform
    georeferenced_poses: List[CameraPose]
    georeferenced_cloud: SparsePointCloud
    metrics: GeorefMetrics
    inlier_indices: List[int]
    outlier_indices: List[int]
