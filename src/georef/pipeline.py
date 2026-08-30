"""
Master Georeferencing Pipeline Orchestrator
===========================================
Coordinates RANSAC 7-DoF Umeyama alignment and Geodetically-Constrained
Bundle Adjustment to place visual reconstructions into metric ENU/ECEF coordinates.
"""

from typing import List, Dict, Tuple, Optional
import os
import json
import logging
import numpy as np

from src.sfm.types import PoseEstimationResult, CameraPose, SparsePointCloud
from src.sfm.telemetry_loader import TelemetryLoader
from src.geoprior.transforms import geodetic_to_ecef, ecef_to_enu
from src.georef.types import UmeyamaTransform, GeodeticAnchor, GeorefMetrics, GeorefResult
from src.georef.umeyama import solve_umeyama_similarity
from src.georef.ransac_georef import RANSACGeoreferencer
from src.georef.geodetic_ba_anchor import GeodeticBundleAdjuster
from src.sfm.confidence import PoseConfidenceCalculator

logger = logging.getLogger(__name__)


class GeoreferencingEngine:
    """
    Chapter 7 Georeferencing Engine:
    RANSAC Umeyama 7-DoF Alignment + Joint Geodetic-Visual BA Anchor.
    """

    def __init__(
        self,
        ransac_threshold_m: float = 5.0,
        lambda_geo: float = 0.1,
        enable_geodetic_ba: bool = True,
        max_ba_tracks: int = 3000
    ):
        self.ransac_threshold_m = ransac_threshold_m
        self.lambda_geo = lambda_geo
        self.enable_geodetic_ba = enable_geodetic_ba
        self.max_ba_tracks = max_ba_tracks
        self.ransac = RANSACGeoreferencer(inlier_threshold_m=self.ransac_threshold_m)
        self.geodetic_ba = GeodeticBundleAdjuster(lambda_geo=self.lambda_geo, max_tracks=self.max_ba_tracks)
        self.confidence_calc = PoseConfidenceCalculator()

    def build_geodetic_anchors(
        self,
        poses: List[CameraPose],
        telemetry_csv: str
    ) -> Tuple[List[GeodeticAnchor], float, float, float]:
        """Loads GPS telemetry and converts to Local ENU coordinates."""
        loader = TelemetryLoader(telemetry_csv)
        if not loader.records:
            raise ValueError(f"No telemetry records found in {telemetry_csv}")

        lat0 = loader.records[0].latitude
        lon0 = loader.records[0].longitude
        alt0 = loader.records[0].altitude_m
        ref_ecef0 = geodetic_to_ecef(lat0, lon0, alt0)

        anchors = []
        for pose in poses:
            rec = loader.get_interpolated_telemetry(pose.timestamp_sec)
            pt_ecef = geodetic_to_ecef(rec.latitude, rec.longitude, rec.altitude_m)
            enu = ecef_to_enu(pt_ecef, lat0, lon0, ref_ecef0)
            
            # Satellite lock weighting
            weight = 1.0
            if rec.gps_num_satellites < 10:
                weight = 0.3
            elif rec.gps_num_satellites < 15:
                weight = 0.7

            anchors.append(GeodeticAnchor(
                frame_idx=pose.frame_idx,
                timestamp=pose.timestamp_sec,
                enu_gt=np.array([enu.east, enu.north, enu.up], dtype=np.float64),
                lat=rec.latitude,
                lon=rec.longitude,
                alt_m=rec.altitude_m,
                num_satellites=rec.gps_num_satellites,
                weight=weight
            ))

        return anchors, lat0, lon0, alt0

    def georeference(
        self,
        pose_result: PoseEstimationResult,
        telemetry_csv: str,
        tracks: Optional[List] = None
    ) -> GeorefResult:
        """
        Georeferences local visual poses and 3D point cloud into metric ENU world coordinates.
        """
        local_poses = pose_result.poses
        intrinsics = pose_result.intrinsics

        # 1. Build ground-truth GNSS anchors
        anchors, lat0, lon0, alt0 = self.build_geodetic_anchors(local_poses, telemetry_csv)
        src_c = np.array([p.camera_center for p in local_poses], dtype=np.float64)
        dst_c = np.array([a.enu_gt for a in anchors], dtype=np.float64)

        # 2. RANSAC 7-DoF Umeyama Alignment
        transform, inlier_mask, residuals = self.ransac.fit(src_c, dst_c, lat0, lon0, alt0)

        # 3. Transform poses and 3D cloud into initial ENU space
        transformed_poses = [transform.transform_camera_pose(p) for p in local_poses]
        transformed_pts = transform.transform_points(pose_result.sparse_cloud.points_3d)

        # 4. Joint Geodetic Bundle Adjustment (Refining poses & 3D ray intersections)
        if self.enable_geodetic_ba and tracks is not None and len(tracks) > 0:
            # Update track points into ENU
            for t in tracks:
                if t.point_3d is not None:
                    t.point_3d = transform.transform_points(t.point_3d.reshape(1, 3))[0]

            optimized_poses, optimized_tracks = self.geodetic_ba.optimize(
                intrinsics=intrinsics,
                poses=transformed_poses,
                tracks=tracks,
                anchors=anchors,
                inlier_mask=inlier_mask
            )
            final_poses = optimized_poses
            final_pts = np.array([t.point_3d for t in optimized_tracks if t.point_3d is not None])
            
            # Reprojection RMS
            rms_px, _ = self.confidence_calc.compute_reprojection_residuals(
                intrinsics, {p.frame_idx: p for p in final_poses}, optimized_tracks
            )
        else:
            final_poses = transformed_poses
            final_pts = transformed_pts
            rms_px = float(pose_result.metrics.reprojection_residual_px)

        # 5. Compute Quantitative Metrics
        final_centers = np.array([p.camera_center for p in final_poses])
        inlier_indices = np.where(inlier_mask)[0]
        outlier_indices = np.where(~inlier_mask)[0]

        inlier_errors = np.linalg.norm(final_centers[inlier_indices] - dst_c[inlier_indices], axis=1)
        h_errors = np.linalg.norm(final_centers[inlier_indices, :2] - dst_c[inlier_indices, :2], axis=1)
        v_errors = np.abs(final_centers[inlier_indices, 2] - dst_c[inlier_indices, 2])

        metrics = GeorefMetrics(
            horizontal_rmse_m=float(np.sqrt(np.mean(h_errors ** 2))) if len(h_errors) > 0 else 0.0,
            vertical_rmse_m=float(np.sqrt(np.mean(v_errors ** 2))) if len(v_errors) > 0 else 0.0,
            ate_3d_rmse_m=float(np.sqrt(np.mean(inlier_errors ** 2))) if len(inlier_errors) > 0 else 0.0,
            ate_3d_median_m=float(np.median(inlier_errors)) if len(inlier_errors) > 0 else 0.0,
            ate_3d_mean_m=float(np.mean(inlier_errors)) if len(inlier_errors) > 0 else 0.0,
            inlier_ratio=float(len(inlier_indices) / max(1, len(local_poses))),
            num_inliers=int(len(inlier_indices)),
            total_anchors=int(len(local_poses)),
            scale_factor=float(transform.scale),
            reprojection_rmse_px=float(rms_px)
        )

        georef_cloud = SparsePointCloud(
            points_3d=final_pts,
            colors_rgb=pose_result.sparse_cloud.colors_rgb[: len(final_pts)],
            reprojection_errors=np.full(len(final_pts), rms_px),
            visibility_counts=pose_result.sparse_cloud.visibility_counts[: len(final_pts)]
        )

        return GeorefResult(
            transform=transform,
            georeferenced_poses=final_poses,
            georeferenced_cloud=georef_cloud,
            metrics=metrics,
            inlier_indices=inlier_indices.tolist(),
            outlier_indices=outlier_indices.tolist()
        )
