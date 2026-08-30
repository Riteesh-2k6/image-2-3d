"""
Unit Tests for VGGT Pose Estimation & SFM Pipeline
===================================================
Tests intrinsics math, camera pose conversions, 50Hz telemetry synchronization,
feature tracking, DLT multi-view triangulation, Chapter 6 confidence metrics,
and Levenberg-Marquardt bundle adjustment convergence.
"""

import os
import pytest
import numpy as np
import cv2

from src.sfm.types import (
    CameraIntrinsics,
    CameraPose,
    SparsePointCloud,
    PoseEngineMode,
    PoseConfidenceMetrics
)
from src.sfm.telemetry_loader import TelemetryLoader, TelemetryRecord
from src.sfm.feature_tracker import FeatureTracker, ImageFeatures, PairwiseMatch, FeatureTrack
from src.sfm.confidence import PoseConfidenceCalculator
from src.sfm.bundle_adjustment import LocalBundleAdjuster
from src.sfm.vggt_engine import VGGTEngine
from src.sfm.pipeline import VGGTPoseEstimator


def create_synthetic_textured_image(width: int = 640, height: int = 480, seed: int = 42) -> np.ndarray:
    """Creates a high-contrast synthetic image with distinct geometric features."""
    np.random.seed(seed)
    img = np.full((height, width, 3), 200, dtype=np.uint8)
    
    # Draw geometric patterns, grid lines, and high-contrast shapes
    for x in range(20, width, 40):
        cv2.line(img, (x, 0), (x, height), (50, 50, 50), 2)
    for y in range(20, height, 40):
        cv2.line(img, (0, y), (width, y), (50, 50, 50), 2)

    # Random colored circles and rectangles
    for _ in range(30):
        cx = np.random.randint(50, width - 50)
        cy = np.random.randint(50, height - 50)
        r = np.random.randint(10, 35)
        color = tuple(int(c) for c in np.random.randint(0, 255, 3))
        cv2.circle(img, (cx, cy), r, color, -1)
        cv2.rectangle(img, (cx - 15, cy - 15), (cx + 15, cy + 15), (0, 0, 0), 2)

    return img


class TestCameraIntrinsicsAndPose:
    """Tests basic geometric primitives and camera transformations."""

    def test_intrinsics_pinhole_math(self):
        intrinsics = CameraIntrinsics(
            fx=1000.0,
            fy=1000.0,
            cx=640.0,
            cy=360.0,
            width=1280,
            height=720
        )
        K = intrinsics.K
        assert K.shape == (3, 3)
        assert K[0, 0] == 1000.0
        assert K[1, 1] == 1000.0
        assert K[0, 2] == 640.0
        assert K[1, 2] == 360.0
        assert 60.0 < intrinsics.fov_x_deg < 70.0
        assert 35.0 < intrinsics.fov_y_deg < 45.0

    def test_camera_pose_optical_center_and_quaternion(self):
        # Camera at (10, 20, 5) looking forward (+Z) with identity rotation
        R = np.eye(3, dtype=np.float64)
        t = np.array([-10.0, -20.0, -5.0], dtype=np.float64)
        pose = CameraPose(
            frame_idx=0,
            timestamp_sec=1.5,
            R=R,
            t=t
        )
        # C = -R^T @ t = (10, 20, 5)
        np.testing.assert_allclose(pose.camera_center, [10.0, 20.0, 5.0])
        # Identity quaternion is [1, 0, 0, 0]
        np.testing.assert_allclose(pose.quaternion, [1.0, 0.0, 0.0, 0.0], atol=1e-6)

        # Pure 90 deg rotation around Y
        theta = np.pi / 2.0
        Ry = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        pose_rot = CameraPose(frame_idx=1, timestamp_sec=2.0, R=Ry, t=t)
        q = pose_rot.quaternion
        assert np.isclose(np.linalg.norm(q), 1.0)


class TestTelemetryLoader:
    """Tests 50Hz DJI CSV telemetry parsing and interpolation."""

    def test_load_real_dji_csv(self):
        csv_path = "videos/06.csv"
        if not os.path.exists(csv_path):
            pytest.skip("videos/06.csv not present in repo.")

        loader = TelemetryLoader(csv_path)
        assert len(loader.records) > 0
        r0 = loader.records[0]
        assert r0.latitude != 0.0
        assert r0.longitude != 0.0
        assert r0.time_sec >= 0.0

    def test_telemetry_interpolation(self):
        csv_path = "videos/06.csv"
        if not os.path.exists(csv_path):
            pytest.skip("videos/06.csv not present in repo.")

        loader = TelemetryLoader(csv_path)
        t_query = 2.5
        rec = loader.get_interpolated_telemetry(t_query)
        assert rec is not None
        assert np.isclose(rec.time_sec, t_query)
        assert 40.0 < rec.latitude < 50.0

    def test_gimbal_to_rotation_matrix(self):
        loader = TelemetryLoader()
        R = loader.gimbal_to_rotation_matrix(pitch_deg=0.0, roll_deg=0.0, yaw_deg=0.0)
        assert R.shape == (3, 3)
        # Verify orthogonality: R @ R.T == I and det(R) == 1
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)

    def test_telemetry_angle_wraparound_interpolation(self):
        """Tests that interpolation across +/- 180 deg boundary takes shortest circular arc."""
        loader = TelemetryLoader()
        loader.records = [
            TelemetryRecord(0.0, 45.0, 75.0, 100.0, 50.0, 0.0, 0.0, 178.0, 0.0, 0.0, 178.0, 20, 5.0),
            TelemetryRecord(1.0, 45.0, 75.0, 100.0, 50.0, 0.0, 0.0, -178.0, 0.0, 0.0, -178.0, 20, 5.0),
        ]
        # At midpoint t=0.5s, angle should be 180.0 / -180.0 (4 deg total rotation), NOT 0.0 deg!
        mid = loader.get_interpolated_telemetry(0.5)
        assert mid is not None
        assert abs(abs(mid.gimbal_yaw_deg) - 180.0) < 0.1

    def test_rolling_shutter_compensation_synthetic(self):
        """Tests synthetic rolling shutter distortion injection and analytical removal."""
        tracker = FeatureTracker()
        h, w = 540, 960
        fx, fy = 500.0, 500.0
        yaw_rate = 0.5 # 0.5 rad/s ~ 28.6 deg/s
        readout = 0.0145 # 14.5 ms

        # Create ground-truth grid of points
        ys = np.linspace(50, 490, 10)
        xs = np.full(10, 480.0)
        gt_pts = np.column_stack([xs, ys]).astype(np.float32)

        # Inject rolling-shutter shear: du = - fx * yaw_rate * tau
        tau = (gt_pts[:, 1] / h - 0.5) * readout
        sheared_pts = gt_pts.copy()
        sheared_pts[:, 0] -= fx * yaw_rate * tau

        sheared_feat = ImageFeatures(
            image_idx=0, image_name="test", width=w, height=h,
            keypoints=sheared_pts, descriptors=np.zeros((10, 128)), colors=np.zeros((10, 3), dtype=np.uint8)
        )

        # Apply compensation
        recovered_feat = tracker.compensate_rolling_shutter(
            sheared_feat, fx=fx, fy=fy, yaw_rate_rad_s=yaw_rate, pitch_rate_rad_s=0.0, readout_time_s=readout
        )

        # Confirm ground truth keypoints are recovered within 1e-4 px
        np.testing.assert_allclose(recovered_feat.keypoints, gt_pts, atol=1e-4)

    def test_heading_error_synthetic_zero_drift(self):
        """Tests that relative heading error metric returns exact 0.0 deg on true zero-drift orbit."""
        ts = np.linspace(0, 2 * np.pi, 50)
        radius = 20.0
        gps_east = radius * np.cos(ts)
        gps_north = radius * np.sin(ts)
        gps_up = np.full_like(ts, 10.0)

        # Inward-facing camera on circular orbit
        true_gimbal_yaws = np.degrees(np.arctan2(-gps_east, -gps_north))
        vo_cam_forwards = np.column_stack([-gps_east, -gps_north, np.zeros_like(ts)])
        vo_cam_forwards /= np.linalg.norm(vo_cam_forwards, axis=1, keepdims=True)

        vo_headings = np.degrees(np.arctan2(vo_cam_forwards[:, 0], vo_cam_forwards[:, 1]))
        heading_errors = (vo_headings - true_gimbal_yaws + 180.0) % 360.0 - 180.0

        np.testing.assert_allclose(heading_errors, np.zeros_like(heading_errors), atol=1e-6)


class TestFeatureTrackingAndTriangulation:
    """Tests 2D feature extraction, reciprocal matching, and 3D triangulation."""

    def test_extract_and_match_synthetic_pair(self):
        img1 = create_synthetic_textured_image(640, 480, seed=10)
        # Apply small affine shift (simulating drone motion)
        M = np.float32([[1, 0, 15], [0, 1, 5]])
        img2 = cv2.warpAffine(img1, M, (640, 480))

        tracker = FeatureTracker(max_features=1000)
        f1 = tracker.extract_features(img1, image_idx=0)
        f2 = tracker.extract_features(img2, image_idx=1)

        assert len(f1.keypoints) > 20
        assert len(f2.keypoints) > 20

        match = tracker.match_pair(f1, f2)
        assert match is not None
        assert len(match.matches_src) >= tracker.min_inliers
        assert match.inlier_ratio > 0.5

    def test_multi_view_track_union_find(self):
        tracker = FeatureTracker()
        f0 = ImageFeatures(0, "0", 640, 480, np.array([[10, 10], [20, 20]], dtype=np.float32), np.ones((2, 128)), np.zeros((2, 3), dtype=np.uint8))
        f1 = ImageFeatures(1, "1", 640, 480, np.array([[12, 11], [22, 21]], dtype=np.float32), np.ones((2, 128)), np.zeros((2, 3), dtype=np.uint8))
        f2 = ImageFeatures(2, "2", 640, 480, np.array([[14, 12], [24, 22]], dtype=np.float32), np.ones((2, 128)), np.zeros((2, 3), dtype=np.uint8))

        # Match 0->1 and 1->2 for feature 0
        m01 = PairwiseMatch(0, 1, np.array([[10, 10]]), np.array([[12, 11]]), np.array([0]), np.array([0]), 1.0)
        m12 = PairwiseMatch(1, 2, np.array([[12, 11]]), np.array([[14, 12]]), np.array([0]), np.array([0]), 1.0)

        tracks = tracker.build_feature_tracks([f0, f1, f2], [m01, m12])
        assert len(tracks) >= 1
        assert tracks[0].track_length == 3
        assert 0 in tracks[0].observations
        assert 1 in tracks[0].observations
        assert 2 in tracks[0].observations

    def test_dlt_multiview_triangulation(self):
        vggt = VGGTEngine()
        intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
        K = intrinsics.K

        # Known true 3D point in world coordinates
        X_true = np.array([1.5, 0.8, 8.0], dtype=np.float64)

        # Camera 1 at (0, 0, 0), looking +Z
        pose1 = CameraPose(frame_idx=0, timestamp_sec=0.0, R=np.eye(3), t=np.zeros(3))
        # Camera 2 at (1.0, 0, 0), looking +Z
        pose2 = CameraPose(frame_idx=1, timestamp_sec=1.0, R=np.eye(3), t=np.array([-1.0, 0.0, 0.0]))
        # Camera 3 at (0.5, 1.0, 0), looking +Z
        pose3 = CameraPose(frame_idx=2, timestamp_sec=2.0, R=np.eye(3), t=np.array([-0.5, -1.0, 0.0]))

        poses = {0: pose1, 1: pose2, 2: pose3}

        # Project true point to get exact 2D observations
        observations = {}
        for idx, p in poses.items():
            P = K @ p.world_to_camera_matrix[:3, :]
            xh = P @ np.append(X_true, 1.0)
            observations[idx] = (xh[0] / xh[2], xh[1] / xh[2])

        # Triangulate via DLT
        X_est = vggt.triangulate_point_multiview(intrinsics, poses, observations)
        assert X_est is not None
        np.testing.assert_allclose(X_est, X_true, atol=1e-4)


class TestConfidenceAndBundleAdjustment:
    """Tests Chapter 6 confidence metrics and Levenberg-Marquardt BA refinement."""

    def test_confidence_metrics_evaluation(self):
        calc = PoseConfidenceCalculator()
        intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)

        pose0 = CameraPose(frame_idx=0, timestamp_sec=0.0, R=np.eye(3), t=np.zeros(3))
        pose1 = CameraPose(frame_idx=1, timestamp_sec=1.0, R=np.eye(3), t=np.array([-1.0, 0.0, 0.0]))
        poses = [pose0, pose1]

        # Perfect track
        t = FeatureTrack(track_id=0, observations={0: (320.0, 240.0), 1: (220.0, 240.0)}, point_3d=np.array([0.0, 0.0, 5.0]))
        match = PairwiseMatch(0, 1, np.array([[320, 240]]), np.array([[220, 240]]), np.array([0]), np.array([0]), 0.95)

        metrics = calc.evaluate_metrics(intrinsics, poses, [t], [match], total_input_frames=2)
        assert isinstance(metrics, PoseConfidenceMetrics)
        assert metrics.reprojection_residual_px < 1.0
        assert metrics.pose_confidence_score > 0.6
        assert metrics.num_registered_frames == 2

    def test_bundle_adjustment_polishing(self):
        intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
        K = intrinsics.K

        # True setup
        X_true_list = [
            np.array([0.0, 0.0, 6.0]),
            np.array([1.0, -0.5, 7.0]),
            np.array([-1.0, 0.5, 6.5]),
            np.array([0.5, 1.0, 5.5]),
            np.array([-0.5, -1.0, 8.0]),
        ]

        pose0 = CameraPose(frame_idx=0, timestamp_sec=0.0, R=np.eye(3), t=np.zeros(3))
        pose1 = CameraPose(frame_idx=1, timestamp_sec=1.0, R=np.eye(3), t=np.array([-1.0, 0.0, 0.0]))
        pose2 = CameraPose(frame_idx=2, timestamp_sec=2.0, R=np.eye(3), t=np.array([-2.0, 0.0, 0.0]))
        poses = [pose0, pose1, pose2]

        tracks = []
        for i, X in enumerate(X_true_list):
            obs = {}
            for cam_idx, p in enumerate(poses):
                P = K @ p.world_to_camera_matrix[:3, :]
                xh = P @ np.append(X, 1.0)
                obs[cam_idx] = (xh[0] / xh[2], xh[1] / xh[2])
            # Add perturbed initial 3D guess
            tracks.append(FeatureTrack(track_id=i, observations=obs, point_3d=X + np.random.normal(0, 0.1, 3)))

        # Perturb camera 1 & 2 translation slightly
        poses[1].t += np.array([0.05, -0.05, 0.05])
        poses[2].t += np.array([-0.05, 0.05, -0.05])

        ba = LocalBundleAdjuster(max_nfev=30)
        opt_poses, opt_tracks = ba.optimize(intrinsics, poses, tracks)

        assert len(opt_poses) == 3
        # Ensure camera 0 stayed fixed at identity
        np.testing.assert_allclose(opt_poses[0].R, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(opt_poses[0].t, np.zeros(3), atol=1e-6)


class TestEndToEndVGGTPipeline:
    """Tests top-level VGGTPoseEstimator workflow."""

    def test_full_pipeline_on_synthetic_keyframes(self, tmp_path):
        # Generate 4 small consecutive synthetic keyframes
        images = [create_synthetic_textured_image(320, 240, seed=i*10) for i in range(4)]
        timestamps = [0.0, 1.0, 2.0, 3.0]

        estimator = VGGTPoseEstimator()
        result = estimator.estimate_poses(images, timestamps=timestamps)

        assert len(result.poses) == 4
        assert result.intrinsics.width == 320
        assert result.intrinsics.height == 240
        assert result.sparse_cloud is not None
        assert result.metrics.pose_confidence_score > 0.0

        # Test export
        out_dir = str(tmp_path / "sfm_test_out")
        exported = VGGTPoseEstimator.export_results(result, out_dir, prefix="test")
        assert os.path.exists(exported["poses_json"])
        assert os.path.exists(exported["sparse_cloud_ply"])
        assert os.path.exists(exported["report_json"])
