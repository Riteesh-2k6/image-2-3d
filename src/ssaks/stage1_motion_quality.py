"""
SSAKS Stage 1: Motion & Quality Filtering
=========================================
Filters blurry, static, overexposed, or excessive motion frames using CPU-based
Laplacian variance, OpenCV DIS optical flow, and exposure histograms.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from src.ssaks.types import SSAKSConfig


class MotionQualityFilter:
    """Stage 1 filter rejecting uninformative and corrupted video frames."""

    def __init__(self, config: Optional[SSAKSConfig] = None):
        self.cfg = config or SSAKSConfig()
        self.dis_flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self.prev_gray: Optional[np.ndarray] = None

    def compute_blur_score(self, gray_frame: np.ndarray) -> float:
        """Calculate Laplacian variance as an image sharpness metric."""
        return float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())

    def check_exposure(self, gray_frame: np.ndarray) -> bool:
        """Reject severe under-exposure (<5% mean) or over-exposure (>95% mean)."""
        mean_val = float(np.mean(gray_frame))
        return 12.0 <= mean_val <= 243.0

    def compute_optical_flow_magnitude(self, gray_frame: np.ndarray) -> float:
        """Compute mean optical flow displacement magnitude against previous frame."""
        if self.prev_gray is None:
            self.prev_gray = gray_frame.copy()
            return self.cfg.min_flow_magnitude + 5.0 # Pass initial frame

        # Downsample for ultra-fast CPU optical flow computation (128x128)
        small_prev = cv2.resize(self.prev_gray, (128, 128), interpolation=cv2.INTER_AREA)
        small_curr = cv2.resize(gray_frame, (128, 128), interpolation=cv2.INTER_AREA)
        
        flow = self.dis_flow.calc(small_prev, small_curr, None)
        magnitude = float(np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)))

        self.prev_gray = gray_frame.copy()
        return magnitude

    def evaluate_frame(self, bgr_frame: np.ndarray) -> Tuple[bool, float, float]:
        """
        Evaluate frame against Stage 1 quality criteria.
        Returns: (is_valid, blur_score, flow_magnitude)
        """
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Blur Check
        blur = self.compute_blur_score(gray)
        if blur < self.cfg.min_blur_laplacian:
            return False, blur, 0.0

        # 2. Exposure Check
        if not self.check_exposure(gray):
            return False, blur, 0.0

        # 3. Motion Flow Check
        flow_mag = self.compute_optical_flow_magnitude(gray)
        if not (self.cfg.min_flow_magnitude <= flow_mag <= self.cfg.max_flow_magnitude):
            return False, blur, flow_mag

        return True, blur, flow_mag
