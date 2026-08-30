"""
Unit & Integration Tests for Chapter 7 Georeferencing Module
============================================================
Validates 7-DoF Umeyama exact recovery, reflection handling,
RANSAC multipath rejection (both isolated and clustered 3-8m bursts),
and Geodetic Bundle Adjustment convergence.
"""

import os
import pytest
import numpy as np
import cv2

from src.georef.types import UmeyamaTransform, GeodeticAnchor
from src.georef.umeyama import solve_umeyama_similarity
from src.georef.ransac_georef import RANSACGeoreferencer
from src.georef.geodetic_ba_anchor import GeodeticBundleAdjuster
from src.georef.pipeline import GeoreferencingEngine
from src.sfm.types import CameraPose, CameraIntrinsics, SparsePointCloud, PoseEstimationResult, PoseConfidenceMetrics, PoseEngineMode
from src.sfm.feature_tracker import FeatureTrack


class TestUmeyamaClosedForm:
    """Validates 7-DoF closed-form similarity alignment."""

    def test_umeyama_exact_recovery_synthetic(self):
        """Validates exact recovery of scale, rotation, and translation within < 1e-6."""
        # 1. Ground truth source points (e.g. 50 random 3D points)
        rng = np.random.default_rng(123)
        src = rng.uniform(-50.0, 50.0, size=(50, 3))

        # 2. Ground truth transformation parameters
        gt_scale = 1.845
        rvec = np.array([0.35, -0.22, 0.48]) # Rodrigues axis-angle
        gt_R, _ = cv2.Rodrigues(rvec)
        gt_t = np.array([125.4, -84.2, 42.1])

        # 3. Compute target points: dst = s * (src @ R.T) + t
        dst = gt_scale * (src @ gt_R.T) + gt_t

        # 4. Recover with Umeyama
        recovered = solve_umeyama_similarity(src, dst)

        # 5. Assertions
        assert np.isclose(recovered.scale, gt_scale, atol=1e-6)
        np.testing.assert_allclose(recovered.R, gt_R, atol=1e-6)
        np.testing.assert_allclose(recovered.translation, gt_t, atol=1e-6)
        assert np.isclose(np.linalg.det(recovered.R), 1.0, atol=1e-6)

    def test_umeyama_reflection_handling(self):
        """Ensures that reflection (det(R) = -1) is properly corrected to proper rotation in SO(3)."""
        src = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0]
        ])
        # Invert Z axis (reflection)
        dst = src.copy()
        dst[:, 2] *= -1.0

        recovered = solve_umeyama_similarity(src, dst)
        assert np.isclose(np.linalg.det(recovered.R), 1.0, atol=1e-6)


class TestRANSACGeoreferencing:
    """Validates robust GPS outlier rejection under isolated and clustered multipath."""

    def test_ransac_rejects_large_gps_outliers(self):
        """Injects 20% random large GPS outliers (>20m) and confirms 100% rejection."""
        rng = np.random.default_rng(42)
        n = 60
        src = rng.uniform(-30.0, 30.0, size=(n, 3))
        
        gt_scale = 2.1
        gt_R = np.eye(3)
        gt_t = np.array([10.0, 20.0, 5.0])
        dst = gt_scale * (src @ gt_R.T) + gt_t

        # Inject 12 large outliers (20m - 50m)
        outlier_indices = [5, 12, 18, 25, 33, 40, 44, 49, 52, 55, 57, 59]
        for idx in outlier_indices:
            dst[idx] += rng.uniform(20.0, 50.0, size=3)

        ransac = RANSACGeoreferencer(inlier_threshold_m=4.0, max_iterations=500, random_seed=42)
        transform, inlier_mask, residuals = ransac.fit(src, dst)

        # Check that all injected outliers were identified and rejected
        for idx in outlier_indices:
            assert not inlier_mask[idx], f"Outlier at index {idx} was not rejected by RANSAC!"

        assert np.isclose(transform.scale, gt_scale, atol=0.05)

    def test_ransac_rejects_clustered_near_threshold_multipath(self):
        """
        Injects a burst of 3-8m correlated systematic offsets over 10 consecutive frames
        (simulating building multipath occlusion) and confirms RANSAC purges the burst.
        """
        rng = np.random.default_rng(99)
        n = 80
        src = np.column_stack([
            np.linspace(0, 100, n),
            np.sin(np.linspace(0, 4*np.pi, n)) * 20.0,
            np.full(n, 15.0)
        ])

        gt_scale = 1.0
        gt_R = np.eye(3)
        gt_t = np.array([50.0, 50.0, 0.0])
        dst = gt_scale * (src @ gt_R.T) + gt_t

        # Add small nominal GPS noise (0.3m std)
        dst += rng.normal(0.0, 0.3, size=(n, 3))

        # Inject correlated multipath burst between frames 30 and 40 (3.5m - 6.5m offset)
        burst_indices = list(range(30, 41))
        multipath_drift = np.array([4.5, 4.0, -2.0]) # 6.3m norm
        for idx in burst_indices:
            dst[idx] += multipath_drift + rng.normal(0.0, 0.2, size=3)

        # Run RANSAC with a tight 3.0m threshold
        ransac = RANSACGeoreferencer(inlier_threshold_m=3.0, max_iterations=500, random_seed=42)
        transform, inlier_mask, residuals = ransac.fit(src, dst)

        # Check that the multipath burst is successfully rejected
        rejected_burst_count = sum(not inlier_mask[i] for i in burst_indices)
        assert rejected_burst_count >= len(burst_indices) - 1, f"Expected multipath burst to be rejected, got {rejected_burst_count}/{len(burst_indices)}"


class TestGeodeticBundleAdjustment:
    """Validates joint visual-geodetic bundle adjustment."""

    def test_geodetic_ba_anchor_convergence(self):
        """Validates that Geodetic BA optimizes joint visual reprojection and GPS spatial prior."""
        intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=480.0, cy=270.0, width=960, height=540)
        
        # 3 cameras along an arc
        poses = [
            CameraPose(frame_idx=0, R=np.eye(3), t=np.array([0.0, 0.0, 0.0]), timestamp_sec=0.0),
            CameraPose(frame_idx=1, R=np.eye(3), t=np.array([2.0, 0.0, 0.0]), timestamp_sec=1.0),
            CameraPose(frame_idx=2, R=np.eye(3), t=np.array([4.0, 0.0, 0.0]), timestamp_sec=2.0),
        ]

        # 3D points
        pts = np.array([
            [0.0, 0.0, 10.0],
            [1.0, 0.5, 12.0],
            [2.0, -0.5, 11.0],
            [3.0, 0.2, 10.5]
        ])

        tracks = []
        for p_idx, pt in enumerate(pts):
            obs = {}
            for c_idx, pose in enumerate(poses):
                p_cam = pose.R @ pt + pose.t
                u = intrinsics.fx * (p_cam[0] / p_cam[2]) + intrinsics.cx
                v = intrinsics.fy * (p_cam[1] / p_cam[2]) + intrinsics.cy
                obs[c_idx] = (float(u), float(v))
            tracks.append(FeatureTrack(track_id=p_idx, observations=obs, point_3d=pt.copy()))

        anchors = [
            GeodeticAnchor(frame_idx=0, timestamp=0.0, enu_gt=np.array([0.0, 0.0, 0.0]), lat=0, lon=0, alt_m=0),
            GeodeticAnchor(frame_idx=1, timestamp=1.0, enu_gt=np.array([2.0, 0.0, 0.0]), lat=0, lon=0, alt_m=0),
            GeodeticAnchor(frame_idx=2, timestamp=2.0, enu_gt=np.array([4.0, 0.0, 0.0]), lat=0, lon=0, alt_m=0),
        ]

        # Perturb pose 1
        poses[1].t += np.array([0.2, -0.1, 0.1])
        init_err = float(np.linalg.norm(np.array([0.2, -0.1, 0.1])))

        gba = GeodeticBundleAdjuster(lambda_geo=0.1, max_nfev=20)
        opt_poses, opt_tracks = gba.optimize(intrinsics, poses, tracks, anchors)

        # Verify optimization corrected the perturbed pose towards ground truth
        shift = np.linalg.norm(opt_poses[1].t - np.array([2.0, 0.0, 0.0]))
        assert shift < init_err, f"Expected pose error to decrease from {init_err:.3f}m, got {shift:.3f}m"
