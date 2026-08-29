"""
SSAKS Data Types & Telemetry Models
====================================
Defines keyframe selection representations, telemetry structures,
and flight state classifications.

ADR References:
- ADR 0003: SSAKS Emergency Fallback (DD-07)
"""

from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field


class FlightState(str, Enum):
    """Drone flight phase classification."""
    TAKEOFF = "takeoff"
    CRUISE = "cruise"
    ERRATIC = "erratic"
    MANEUVERING = "maneuvering"
    LANDING = "landing"


class SSAKSFallbackMode(str, Enum):
    """Fallback operational mode if cruise calibration fails."""
    STANDARD_CASCADE = "standard_cascade"
    MOTION_ADAPTIVE_FALLBACK = "motion_adaptive_fallback" # ADR 0003: Optical flow + blur only
    FIXED_FPS_EMERGENCY = "fixed_fps_emergency"           # Last resort emergency fallback


@dataclass
class FrameTelemetry:
    """Per-frame drone flight telemetry record."""
    frame_idx: int
    timestamp_sec: float
    altitude_m: float
    velocity_mps: Tuple[float, float, float] = (0.0, 0.0, 0.0) # (vx, vy, vz)
    imu_accel: Tuple[float, float, float] = (0.0, 0.0, 9.81)    # (ax, ay, az)


@dataclass
class SSAKSKeyframe:
    """Selected keyframe metadata emitted by the SSAKS cascade."""
    frame_idx: int
    timestamp_sec: float
    blur_score: float
    motion_magnitude: float
    novelty_score: float = 1.0
    semantic_coverage_score: float = 1.0
    selection_reason: str = "standard_cascade"


@dataclass
class SSAKSConfig:
    """Configurable hyperparameters for SSAKS 3-stage cascade."""
    min_blur_laplacian: float = 65.0       # Minimum Laplacian variance (sharpness)
    min_flow_magnitude: float = 0.20       # Minimum pixel displacement to prevent static frames
    max_flow_magnitude: float = 140.0      # Maximum pixel displacement (motion blur threshold)
    dino_similarity_thresh: float = 0.88   # Cosine similarity cutoff for visual novelty
    cruise_timeout_sec: float = 30.0       # ADR 0003: 30-second timeout before fallback
    target_reduction_pct: float = 90.0     # Nominal frame reduction target (85-97%)
