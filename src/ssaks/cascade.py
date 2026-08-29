"""
SSAKS 3-Stage Cascade Orchestrator
===================================
Coordinates Stage 1 (Quality/Motion) -> Stage 2 (Visual Novelty) -> Stage 3 (Semantic Coverage)
with per-flight cruise calibration and motion-adaptive fallback.

Chapter References:
- Chapter 5: Semantic Scene-Aware Keyframe Selection (SSAKS)
- ADR 0003: SSAKS Emergency Fallback (DD-07)
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Generator
from src.ssaks.types import (
    SSAKSKeyframe,
    FrameTelemetry,
    SSAKSConfig,
    SSAKSFallbackMode,
    FlightState,
)
from src.ssaks.stage1_motion_quality import MotionQualityFilter
from src.ssaks.stage2_visual_novelty import VisualNoveltyClusterer
from src.ssaks.cruise_detector import CruiseDetector


class SSAKSCascade:
    """End-to-end Semantic Scene-Aware Keyframe Selection cascade."""

    def __init__(self, config: Optional[SSAKSConfig] = None):
        self.cfg = config or SSAKSConfig()
        self.stage1_filter = MotionQualityFilter(self.cfg)
        self.stage2_clusterer = VisualNoveltyClusterer(self.cfg)
        self.cruise_detector = CruiseDetector(self.cfg)
        self.selected_keyframes: List[SSAKSKeyframe] = []
        self.total_frames_processed: int = 0

    def process_frame(
        self,
        bgr_frame: np.ndarray,
        frame_idx: int,
        telemetry: Optional[FrameTelemetry] = None,
    ) -> Optional[SSAKSKeyframe]:
        """
        Process a single incoming video frame through the cascade.
        Returns SSAKSKeyframe if selected, otherwise None.
        """
        self.total_frames_processed += 1
        timestamp = telemetry.timestamp_sec if telemetry else (frame_idx / 30.0)

        # Update flight state & cruise calibration
        if telemetry:
            self.cruise_detector.update_telemetry(telemetry)

        # Stage 1: Quality & Motion Check
        stage1_pass, blur_score, flow_mag = self.stage1_filter.evaluate_frame(bgr_frame)
        if not stage1_pass:
            return None

        # Check if in ADR 0003 Emergency Fallback Mode
        if self.cruise_detector.current_mode == SSAKSFallbackMode.MOTION_ADAPTIVE_FALLBACK:
            # Fallback accepts high-quality motion frames directly without Stage 2 gating
            keyframe = SSAKSKeyframe(
                frame_idx=frame_idx,
                timestamp_sec=timestamp,
                blur_score=blur_score,
                motion_magnitude=flow_mag,
                selection_reason="motion_adaptive_fallback_adr0003",
            )
            self.selected_keyframes.append(keyframe)
            return keyframe

        # Stage 2: Visual Novelty & Feature Diversity Check
        stage2_pass, max_sim = self.stage2_clusterer.evaluate_novelty(bgr_frame)
        if not stage2_pass:
            return None

        # Selected Keyframe
        keyframe = SSAKSKeyframe(
            frame_idx=frame_idx,
            timestamp_sec=timestamp,
            blur_score=blur_score,
            motion_magnitude=flow_mag,
            novelty_score=1.0 - max_sim,
            selection_reason="standard_cascade",
        )
        self.selected_keyframes.append(keyframe)
        return keyframe

    @property
    def reduction_percentage(self) -> float:
        """Calculate frame reduction ratio achieved by the cascade."""
        if self.total_frames_processed == 0:
            return 0.0
        selected = len(self.selected_keyframes)
        return (1.0 - (selected / self.total_frames_processed)) * 100.0
