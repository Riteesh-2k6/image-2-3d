"""
Unit Test Suite for SSAKS 3-Stage Cascade & Cruise Fallback
===========================================================
Verifies Stage 1 quality filtering, Stage 2 visual novelty clustering,
ADR 0003 30-second emergency fallback, and pipeline determinism.

Chapter References:
- Chapter 5: Semantic Scene-Aware Keyframe Selection (SSAKS)
- ADR 0003: SSAKS Emergency Fallback (DD-07)
"""

import pytest
import numpy as np
import cv2
from src.ssaks.types import (
    FrameTelemetry,
    FlightState,
    SSAKSFallbackMode,
    SSAKSConfig,
)
from src.ssaks.stage1_motion_quality import MotionQualityFilter
from src.ssaks.stage2_visual_novelty import VisualNoveltyClusterer
from src.ssaks.cruise_detector import CruiseDetector
from src.ssaks.cascade import SSAKSCascade


class TestStage1QualityFilter:
    """Verify Laplacian blur, exposure, and optical flow gating."""

    def test_reject_blurry_frame(self):
        filter_s1 = MotionQualityFilter()
        
        # Sharp synthetic checkerboard image
        sharp_img = np.zeros((256, 256, 3), dtype=np.uint8)
        sharp_img[::16, :] = 255
        sharp_img[:, ::16] = 255
        
        # Heavy Gaussian blurred image
        blurry_img = cv2.GaussianBlur(sharp_img, (51, 51), 0)

        # Sharp should pass, blurry should fail
        sharp_pass, sharp_blur, _ = filter_s1.evaluate_frame(sharp_img)
        blurry_pass, blurry_blur, _ = filter_s1.evaluate_frame(blurry_img)

        assert sharp_blur > 65.0
        assert blurry_blur < 65.0
        assert blurry_pass is False

    def test_reject_overexposed_frame(self):
        filter_s1 = MotionQualityFilter()
        overexposed = np.full((256, 256, 3), 250, dtype=np.uint8)
        is_valid, _, _ = filter_s1.evaluate_frame(overexposed)
        assert is_valid is False


class TestStage2VisualNovelty:
    """Verify embedding-based viewpoint deduplication."""

    def test_reject_redundant_viewpoints(self):
        clusterer = VisualNoveltyClusterer()
        
        frame1 = np.full((128, 128, 3), 100, dtype=np.uint8)
        frame1[:64, :64] = 200 # Pattern
        
        # First frame is always novel
        novel_1, _ = clusterer.evaluate_novelty(frame1)
        assert novel_1 is True

        # Exact same frame is rejected as redundant
        novel_2, sim = clusterer.evaluate_novelty(frame1)
        assert novel_2 is False
        assert sim > 0.99


class TestCruiseDetectionAndADR0003Fallback:
    """Verify DD-02 cruise detection and ADR 0003 30-second fallback."""

    def test_cruise_detection_on_stable_flight(self):
        detector = CruiseDetector()
        
        # Feed 30 stable cruise telemetry frames
        state = FlightState.TAKEOFF
        for idx in range(30):
            t = FrameTelemetry(
                frame_idx=idx,
                timestamp_sec=idx * 0.1,
                altitude_m=50.0 + np.sin(idx * 0.05) * 0.2, # Stable ~50m
                velocity_mps=(8.0, 0.0, 0.0),
                imu_accel=(0.0, 0.0, 9.81),
            )
            state = detector.update_telemetry(t)
            
        assert state == FlightState.CRUISE
        assert detector.cruise_detected is True
        assert detector.current_mode == SSAKSFallbackMode.STANDARD_CASCADE

    def test_adr0003_emergency_fallback_trigger_after_30s(self):
        detector = CruiseDetector(config=SSAKSConfig(cruise_timeout_sec=30.0))
        
        # Feed 35 seconds of erratic/hovering telemetry (high altitude variance, unstable IMU)
        state = FlightState.TAKEOFF
        for sec in range(36):
            t = FrameTelemetry(
                frame_idx=sec * 10,
                timestamp_sec=float(sec),
                altitude_m=5.0 + (sec % 4) * 8.0, # Rapid erratic altitude changes
                velocity_mps=(0.0, 0.0, 0.0),
                imu_accel=(2.5 * np.sin(sec), 3.0 * np.cos(sec), 9.81 + 2.0 * np.sin(sec)),
            )
            state = detector.update_telemetry(t)

        # After 30s timeout, ADR 0003 triggers fallback mode
        assert detector.cruise_detected is False
        assert detector.current_mode == SSAKSFallbackMode.MOTION_ADAPTIVE_FALLBACK
        assert detector.fallback_warning_emitted is True


class TestSSAKSCascadeEndToEnd:
    """Verify complete cascade execution and pipeline determinism."""

    def test_pipeline_determinism(self):
        # Generate 50 synthetic video frames
        frames = []
        for i in range(50):
            f = np.zeros((128, 128, 3), dtype=np.uint8)
            cv2.circle(f, (30 + i * 2, 64), 20, (255, 255, 255), -1) # Moving white circle
            frames.append(f)

        # Run 1
        cascade_1 = SSAKSCascade()
        selected_1 = []
        for idx, frame in enumerate(frames):
            kf = cascade_1.process_frame(frame, idx)
            if kf:
                selected_1.append(kf.frame_idx)

        # Run 2
        cascade_2 = SSAKSCascade()
        selected_2 = []
        for idx, frame in enumerate(frames):
            kf = cascade_2.process_frame(frame, idx)
            if kf:
                selected_2.append(kf.frame_idx)

        # Must be 100% deterministic
        assert selected_1 == selected_2
        assert len(selected_1) > 0
        assert cascade_1.reduction_percentage > 0.0
