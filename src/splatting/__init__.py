"""
Stage 3: 3D Gaussian Splatting with GeoPrior Structural Anchors.
"""

from src.splatting.types import CameraInfo, RenderOutput, SplatMetrics
from src.splatting.gaussian_model import GaussianModel
from src.splatting.rasterizer import SpeedySplatRasterizer
from src.splatting.prior_anchor import PriorAnchorManager
from src.splatting.loss import CombinedSplatLoss, ssim_loss
from src.splatting.density_controller import DensityController

__all__ = [
    "CameraInfo",
    "RenderOutput",
    "SplatMetrics",
    "GaussianModel",
    "SpeedySplatRasterizer",
    "PriorAnchorManager",
    "CombinedSplatLoss",
    "ssim_loss",
    "DensityController"
]
