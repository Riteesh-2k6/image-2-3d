"""
VGGT Pose Estimation Pipeline Orchestrator
===========================================
Top-level pipeline coordinating VGGT feed-forward pose solving, telemetry synchronization,
Chapter 6 confidence metric evaluation, and dynamic bundle adjustment refinement.
"""

import time
import os
import json
from typing import List, Optional, Dict, Tuple, Any, Union
import numpy as np
import cv2

from src.sfm.types import (
    CameraIntrinsics,
    CameraPose,
    SparsePointCloud,
    PoseConfidenceMetrics,
    PoseEstimationResult,
    PoseEngineMode
)
from src.sfm.vggt_engine import VGGTEngine
from src.sfm.confidence import PoseConfidenceCalculator
from src.sfm.bundle_adjustment import LocalBundleAdjuster
from src.sfm.telemetry_loader import TelemetryLoader, TelemetryRecord
from src.sfm.feature_tracker import FeatureTracker


class VGGTPoseEstimator:
    """
    Master Hybrid Pose Estimation & Visual Geometry Pipeline.
    """

    def __init__(
        self,
        vggt_engine: Optional[VGGTEngine] = None,
        confidence_calculator: Optional[PoseConfidenceCalculator] = None,
        bundle_adjuster: Optional[LocalBundleAdjuster] = None,
        min_confidence_thresh: float = 0.75,
        max_reproj_err_px: float = 1.5,
    ):
        self.vggt_engine = vggt_engine or VGGTEngine()
        self.confidence_calc = confidence_calculator or PoseConfidenceCalculator(
            target_max_reproj_px=max_reproj_err_px,
            min_confidence_thresh=min_confidence_thresh
        )
        self.bundle_adjuster = bundle_adjuster or LocalBundleAdjuster()
        self.min_confidence_thresh = min_confidence_thresh
        self.max_reproj_err_px = max_reproj_err_px

    def estimate_poses(
        self,
        keyframes: List[Union[np.ndarray, str]],
        timestamps: Optional[List[float]] = None,
        image_names: Optional[List[str]] = None,
        telemetry_csv: Optional[str] = None,
        force_bundle_adjustment: bool = False,
        enable_rolling_shutter: bool = False
    ) -> PoseEstimationResult:
        """
        Executes end-to-end VGGT pose estimation and sparse point triangulation.
        
        Args:
            keyframes: List of loaded RGB/BGR image numpy arrays OR absolute filepaths to images.
            timestamps: Optional timestamps (in seconds) corresponding to each frame.
            image_names: Optional string identifiers for each image.
            telemetry_csv: Optional filepath to DJI flight log (e.g. videos/06.csv).
            force_bundle_adjustment: If True, always runs LM bundle adjustment polish.
            enable_rolling_shutter: If True, enables CMOS rolling-shutter scanline compensation (default: False).
        """
        start_time = time.time()
        num_frames = len(keyframes)
        if num_frames == 0:
            raise ValueError("No keyframes provided.")

        # Load images if filepaths provided
        loaded_images: List[np.ndarray] = []
        names: List[str] = []
        for i, item in enumerate(keyframes):
            if isinstance(item, str):
                img = cv2.imread(item)
                if img is None:
                    raise FileNotFoundError(f"Could not load keyframe image from path: {item}")
                loaded_images.append(img)
                names.append(os.path.basename(item))
            else:
                loaded_images.append(item)
                names.append(image_names[i] if image_names and i < len(image_names) else f"keyframe_{i:04d}")

        if timestamps is None:
            # Default to 1 sec intervals
            timestamps = [float(i) for i in range(num_frames)]

        # Load telemetry records if available
        telemetry_records: Optional[List[Optional[TelemetryRecord]]] = None
        if telemetry_csv and os.path.exists(telemetry_csv):
            telem_loader = TelemetryLoader(telemetry_csv)
            telemetry_records = [telem_loader.get_interpolated_telemetry(ts) for ts in timestamps]

        # Stage 1: VGGT Feed-Forward Pose Solve & Triangulation
        intrinsics, poses, tracks, matches = self.vggt_engine.solve_poses_feedforward(
            keyframes=loaded_images,
            timestamps=timestamps,
            image_names=names,
            telemetry=telemetry_records,
            enable_rolling_shutter=enable_rolling_shutter
        )
        engine_used = PoseEngineMode.VGGT_FEEDFORWARD

        # Stage 2: Evaluate 4 Chapter 6 Confidence Metrics
        metrics = self.confidence_calc.evaluate_metrics(
            intrinsics=intrinsics,
            poses=poses,
            tracks=tracks,
            matches=matches,
            total_input_frames=num_frames
        )

        # Stage 3: Dynamic Fallback Dispatch (Internal LM Bundle Adjustment)
        needs_refinement = (
            force_bundle_adjustment or
            metrics.pose_confidence_score < self.min_confidence_thresh or
            metrics.reprojection_residual_px > self.max_reproj_err_px
        )

        if needs_refinement:
            poses, tracks = self.bundle_adjuster.optimize(intrinsics, poses, tracks)
            engine_used = PoseEngineMode.LM_BUNDLE_ADJUSTMENT

            # Re-evaluate metrics after refinement
            metrics = self.confidence_calc.evaluate_metrics(
                intrinsics=intrinsics,
                poses=poses,
                tracks=tracks,
                matches=matches,
                total_input_frames=num_frames
            )

        # Stage 4: Compile Sparse 3D Point Cloud
        valid_points = []
        valid_colors = []
        valid_errors = []
        valid_visibilities = []

        for track in tracks:
            if track.point_3d is not None:
                valid_points.append(track.point_3d)
                valid_colors.append(track.color)
                valid_errors.append(track.reprojection_error)
                valid_visibilities.append(track.track_length)

        if valid_points:
            sparse_cloud = SparsePointCloud(
                points_3d=np.array(valid_points, dtype=np.float64),
                colors_rgb=np.array(valid_colors, dtype=np.uint8),
                reprojection_errors=np.array(valid_errors, dtype=np.float64),
                visibility_counts=np.array(valid_visibilities, dtype=np.int32)
            )
        else:
            sparse_cloud = SparsePointCloud(
                points_3d=np.empty((0, 3), dtype=np.float64),
                colors_rgb=np.empty((0, 3), dtype=np.uint8),
                reprojection_errors=np.empty((0,), dtype=np.float64),
                visibility_counts=np.empty((0,), dtype=np.int32)
            )

        execution_time = time.time() - start_time

        return PoseEstimationResult(
            intrinsics=intrinsics,
            poses=poses,
            sparse_cloud=sparse_cloud,
            metrics=metrics,
            execution_time_sec=execution_time,
            engine_used=engine_used
        )

    @staticmethod
    def export_results(
        result: PoseEstimationResult,
        output_dir: str,
        prefix: str = "06"
    ) -> Dict[str, str]:
        """Export poses JSON, sparse PLY cloud, and telemetry report."""
        os.makedirs(output_dir, exist_ok=True)

        poses_path = os.path.join(output_dir, f"{prefix}_poses.json")
        cloud_path = os.path.join(output_dir, f"{prefix}_sparse_cloud.ply")
        report_path = os.path.join(output_dir, f"{prefix}_pose_estimation_report.json")

        # 1. Poses JSON
        poses_data = {
            "intrinsics": result.intrinsics.to_dict(),
            "metrics": result.metrics.to_dict(),
            "execution_time_sec": result.execution_time_sec,
            "engine_used": result.engine_used.value,
            "num_cameras": len(result.poses),
            "num_points": result.sparse_cloud.num_points,
            "poses": [p.to_dict() for p in result.poses]
        }
        with open(poses_path, "w", encoding="utf-8") as f:
            json.dump(poses_data, f, indent=2)

        # 2. Sparse Point Cloud PLY
        cloud = result.sparse_cloud
        with open(cloud_path, "w", encoding="utf-8") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {cloud.num_points}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for pt, col in zip(cloud.points_3d, cloud.colors_rgb):
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {int(col[0])} {int(col[1])} {int(col[2])}\n")

        # 3. Report JSON
        report_data = {
            "stage": "Stage 2 (Pose Estimation)",
            "ticket": "[08] Hybrid Pose Estimation (VGGT)",
            "metrics": result.metrics.to_dict(),
            "runtime_sec": round(result.execution_time_sec, 3),
            "fps_throughput": round(len(result.poses) / max(0.001, result.execution_time_sec), 2),
            "sparse_point_count": cloud.num_points,
            "mean_reprojection_px": round(result.metrics.reprojection_residual_px, 3),
            "engine_used": result.engine_used.value
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        return {
            "poses_json": poses_path,
            "sparse_cloud_ply": cloud_path,
            "report_json": report_path
        }
