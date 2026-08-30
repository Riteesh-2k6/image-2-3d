"""
SFM & Pose Estimation Module
=============================
Provides Visual Geometry Grounded Transformer (VGGT) pose estimation,
telemetry synchronization, confidence scoring, and sparse triangulation.
"""

from src.sfm.types import (
    CameraIntrinsics,
    CameraPose,
    SparsePointCloud,
    PoseConfidenceMetrics,
    PoseEstimationResult,
    PoseEngineMode,
)
from src.sfm.vggt_engine import VGGTEngine
from src.sfm.feature_tracker import FeatureTracker, FeatureTrack, PairwiseMatch, ImageFeatures
from src.sfm.confidence import PoseConfidenceCalculator
from src.sfm.bundle_adjustment import LocalBundleAdjuster
from src.sfm.telemetry_loader import TelemetryLoader, TelemetryRecord
from src.sfm.pipeline import VGGTPoseEstimator

__all__ = [
    "CameraIntrinsics",
    "CameraPose",
    "SparsePointCloud",
    "PoseConfidenceMetrics",
    "PoseEstimationResult",
    "PoseEngineMode",
    "VGGTEngine",
    "FeatureTracker",
    "FeatureTrack",
    "PairwiseMatch",
    "ImageFeatures",
    "PoseConfidenceCalculator",
    "LocalBundleAdjuster",
    "TelemetryLoader",
    "TelemetryRecord",
    "VGGTPoseEstimator",
]
