"""
SSAKS Core Module
=================
Semantic Scene-Aware Keyframe Selection for drone photogrammetry.
"""

from src.ssaks.types import (
    FlightState,
    SSAKSFallbackMode,
    FrameTelemetry,
    SSAKSKeyframe,
    SSAKSConfig,
)
from src.ssaks.stage1_motion_quality import MotionQualityFilter
from src.ssaks.stage2_visual_novelty import VisualNoveltyClusterer
from src.ssaks.cruise_detector import CruiseDetector
from src.ssaks.cascade import SSAKSCascade

__all__ = [
    "FlightState",
    "SSAKSFallbackMode",
    "FrameTelemetry",
    "SSAKSKeyframe",
    "SSAKSConfig",
    "MotionQualityFilter",
    "VisualNoveltyClusterer",
    "CruiseDetector",
    "SSAKSCascade",
]
