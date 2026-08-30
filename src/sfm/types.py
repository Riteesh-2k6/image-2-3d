"""
SFM & VGGT Data Types & Representations
========================================
Defines camera intrinsics, camera extrinsics, 3D point cloud structures,
and Chapter 6 confidence metrics for feed-forward neural pose estimation.
"""

from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np


class PoseEngineMode(str, Enum):
    """Execution mode for pose estimation."""
    VGGT_FEEDFORWARD = "vggt_feedforward"
    LM_BUNDLE_ADJUSTMENT = "lm_bundle_adjustment"
    TELEMETRY_PRIOR = "telemetry_prior"


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic calibration matrix and distortion parameters."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    @property
    def K(self) -> np.ndarray:
        """3x3 calibration matrix K."""
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    @property
    def fov_x_deg(self) -> float:
        """Horizontal field of view in degrees."""
        return float(2.0 * np.arctan(self.width / (2.0 * self.fx)) * 180.0 / np.pi)

    @property
    def fov_y_deg(self) -> float:
        """Vertical field of view in degrees."""
        return float(2.0 * np.arctan(self.height / (2.0 * self.fy)) * 180.0 / np.pi)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "width": self.width,
            "height": self.height,
            "fov_x_deg": self.fov_x_deg,
            "fov_y_deg": self.fov_y_deg,
            "distortion": [self.k1, self.k2, self.p1, self.p2],
        }


@dataclass
class CameraPose:
    """Camera extrinsic pose in world coordinates [R | t]."""
    frame_idx: int
    timestamp_sec: float
    R: np.ndarray             # 3x3 rotation matrix (World to Camera)
    t: np.ndarray             # 3x1 translation vector (World to Camera)
    confidence: float = 1.0   # Per-frame pose confidence score [0, 1]
    reprojection_error_px: float = 0.0 # Mean reprojection residual
    engine_mode: PoseEngineMode = PoseEngineMode.VGGT_FEEDFORWARD
    image_name: Optional[str] = None
    telemetry_gps: Optional[Tuple[float, float, float]] = None # (lat, lon, alt)
    telemetry_gimbal: Optional[Tuple[float, float, float]] = None # (pitch, roll, yaw)

    @property
    def camera_center(self) -> np.ndarray:
        """Camera optical center C in world coordinates: C = -R^T * t."""
        return -self.R.T @ self.t

    @property
    def quaternion(self) -> np.ndarray:
        """Rotation converted to unit quaternion [w, x, y, z]."""
        # Hamilton convention
        m = self.R
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            w = 0.25 * S
            x = (m[2, 1] - m[1, 2]) / S
            y = (m[0, 2] - m[2, 0]) / S
            z = (m[1, 0] - m[0, 1]) / S
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w = (m[2, 1] - m[1, 2]) / S
            x = 0.25 * S
            y = (m[0, 1] + m[1, 0]) / S
            z = (m[0, 2] + m[2, 0]) / S
        elif m[1, 1] > m[2, 2]:
            S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            w = (m[0, 2] - m[2, 0]) / S
            x = (m[0, 1] + m[1, 0]) / S
            y = 0.25 * S
            z = (m[1, 2] + m[2, 1]) / S
        else:
            S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            w = (m[1, 0] - m[0, 1]) / S
            x = (m[0, 2] + m[2, 0]) / S
            y = (m[1, 2] + m[2, 1]) / S
            z = 0.25 * S
        q = np.array([w, x, y, z], dtype=np.float64)
        norm = np.linalg.norm(q)
        return q / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])

    @property
    def world_to_camera_matrix(self) -> np.ndarray:
        """4x4 homogeneous transformation matrix [R | t]."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "timestamp_sec": float(self.timestamp_sec),
            "camera_center": self.camera_center.tolist(),
            "rotation_matrix": self.R.tolist(),
            "translation_vector": self.t.tolist(),
            "quaternion_wxyz": self.quaternion.tolist(),
            "confidence": float(self.confidence),
            "reprojection_error_px": float(self.reprojection_error_px),
            "engine_mode": self.engine_mode.value,
            "image_name": self.image_name,
            "telemetry_gps": list(self.telemetry_gps) if self.telemetry_gps else None,
            "telemetry_gimbal": list(self.telemetry_gimbal) if self.telemetry_gimbal else None,
        }


@dataclass
class SparsePointCloud:
    """3D point cloud triangulated from multi-view keyframe observations."""
    points_3d: np.ndarray                   # (N, 3) float64 coordinates
    colors_rgb: np.ndarray                  # (N, 3) uint8 colors [0, 255]
    reprojection_errors: np.ndarray         # (N,) float64 mean error per point in px
    visibility_counts: np.ndarray           # (N,) int32 number of views observing each point
    point_ids: Optional[np.ndarray] = None  # (N,) int64 unique point IDs

    def __post_init__(self):
        if self.point_ids is None:
            self.point_ids = np.arange(len(self.points_3d), dtype=np.int64)

    @property
    def num_points(self) -> int:
        return len(self.points_3d)

    def filter_by_error(self, max_error_px: float = 2.5) -> "SparsePointCloud":
        """Filter out outlier 3D points with large reprojection error."""
        mask = self.reprojection_errors <= max_error_px
        return SparsePointCloud(
            points_3d=self.points_3d[mask],
            colors_rgb=self.colors_rgb[mask],
            reprojection_errors=self.reprojection_errors[mask],
            visibility_counts=self.visibility_counts[mask],
            point_ids=self.point_ids[mask] if self.point_ids is not None else None,
        )


@dataclass
class PoseConfidenceMetrics:
    """Chapter 6 Verification & Robustness Confidence Metrics."""
    pose_confidence_score: float        # Global composite confidence [0, 1]
    reprojection_residual_px: float     # Overall RMS reprojection error in pixels (target <= 1.5px)
    feature_track_consistency: float    # Multi-view inlier consistency ratio [0, 1]
    camera_graph_connectivity: float    # Normalized algebraic connectivity of view graph [0, 1]
    inlier_match_ratio: float = 1.0     # Mean epipolar inlier ratio across keyframe pairs
    num_registered_frames: int = 0      # Count of successfully posed frames
    total_frames: int = 0               # Total input keyframes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pose_confidence_score": float(self.pose_confidence_score),
            "reprojection_residual_px": float(self.reprojection_residual_px),
            "feature_track_consistency": float(self.feature_track_consistency),
            "camera_graph_connectivity": float(self.camera_graph_connectivity),
            "inlier_match_ratio": float(self.inlier_match_ratio),
            "num_registered_frames": self.num_registered_frames,
            "total_frames": self.total_frames,
            "registration_rate_pct": float(100.0 * self.num_registered_frames / max(1, self.total_frames)),
        }


@dataclass
class PoseEstimationResult:
    """Aggregated output from the VGGT pose estimation pipeline."""
    intrinsics: CameraIntrinsics
    poses: List[CameraPose]
    sparse_cloud: SparsePointCloud
    metrics: PoseConfidenceMetrics
    execution_time_sec: float
    engine_used: PoseEngineMode
