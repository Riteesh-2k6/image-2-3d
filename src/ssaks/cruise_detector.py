"""
SSAKS Cruise Calibration & Emergency Fallback Detector
======================================================
Monitors drone telemetry for cruise phase stabilization (DD-02) and executes
the 30-second emergency calibration fallback to motion-adaptive sampling (ADR 0003 / DD-07).

ADR References:
- ADR 0003: SSAKS Emergency Fallback (DD-07)
"""

import logging
from typing import List, Optional
from collections import deque
import numpy as np
from src.ssaks.types import FrameTelemetry, FlightState, SSAKSFallbackMode, SSAKSConfig

logger = logging.getLogger("ssaks.cruise")


class CruiseDetector:
    """Detects stabilized flight cruise mode and triggers ADR 0003 emergency fallback."""

    def __init__(self, config: Optional[SSAKSConfig] = None):
        self.cfg = config or SSAKSConfig()
        self.telemetry_history: deque = deque(maxlen=60) # Last 60 telemetry samples (~2-3 sec)
        self.cruise_detected: bool = False
        self.first_timestamp: Optional[float] = None
        self.current_mode: SSAKSFallbackMode = SSAKSFallbackMode.STANDARD_CASCADE
        self.fallback_warning_emitted: bool = False

    def update_telemetry(self, telemetry: FrameTelemetry) -> FlightState:
        """Process incoming frame telemetry and update flight state."""
        if self.first_timestamp is None:
            self.first_timestamp = telemetry.timestamp_sec

        self.telemetry_history.append(telemetry)
        elapsed_sec = telemetry.timestamp_sec - self.first_timestamp

        # Check if already in cruise
        if len(self.telemetry_history) < 15:
            return FlightState.TAKEOFF

        altitudes = [t.altitude_m for t in self.telemetry_history]
        alt_std = float(np.std(altitudes))
        
        accels = [np.linalg.norm(t.imu_accel) for t in self.telemetry_history]
        accel_variance = float(np.var(accels))

        # Cruise criteria: Altitude variance < 1.5m, IMU acceleration stability
        if alt_std < 1.5 and accel_variance < 2.0 and altitudes[-1] > 10.0:
            self.cruise_detected = True
            return FlightState.CRUISE

        # Check ADR 0003 Emergency Fallback Condition (30s timeout without cruise)
        if not self.cruise_detected and elapsed_sec >= self.cfg.cruise_timeout_sec:
            if self.current_mode != SSAKSFallbackMode.MOTION_ADAPTIVE_FALLBACK:
                self.current_mode = SSAKSFallbackMode.MOTION_ADAPTIVE_FALLBACK
                if not self.fallback_warning_emitted:
                    logger.warning(
                        f"🚨 [ADR 0003 / DD-07] Cruise calibration timeout reached ({elapsed_sec:.1f}s >= {self.cfg.cruise_timeout_sec}s). "
                        "Switching to Motion-Adaptive Sampling Fallback."
                    )
                    self.fallback_warning_emitted = True
            return FlightState.ERRATIC

        return FlightState.MANEUVERING
