"""
Data types and definitions for 3D Gaussian Splatting (Stage 3).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import torch
import numpy as np


@dataclass
class CameraInfo:
    """Camera specification for rendering a single viewpoint."""
    frame_idx: int
    image_name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R: torch.Tensor          # (3, 3) World-to-Camera rotation matrix
    t: torch.Tensor          # (3,) World-to-Camera translation vector
    image_tensor: Optional[torch.Tensor] = None # (3, H, W) normalized [0, 1] RGB on GPU
    camera_center: Optional[torch.Tensor] = None # (3,) Camera optical center in World frame

    @property
    def world_view_transform(self) -> torch.Tensor:
        """4x4 World-to-Camera transformation matrix [R | t]."""
        W = torch.eye(4, device=self.R.device, dtype=self.R.dtype)
        W[:3, :3] = self.R
        W[:3, 3] = self.t
        return W

    @property
    def K_matrix(self) -> torch.Tensor:
        """3x3 Camera Intrinsics matrix."""
        K = torch.zeros((3, 3), device=self.R.device, dtype=self.R.dtype)
        K[0, 0] = self.fx
        K[1, 1] = self.fy
        K[0, 2] = self.cx
        K[1, 2] = self.cy
        K[2, 2] = 1.0
        return K


@dataclass
class RenderOutput:
    """Output from differentiable Gaussian rasterization."""
    rgb: torch.Tensor               # (3, H, W) rendered image
    alpha: torch.Tensor             # (1, H, W) accumulated opacity
    depth: Optional[torch.Tensor] = None # (1, H, W) rendered depth map
    radii: Optional[torch.Tensor] = None # (N,) 2D screen-space radius of each Gaussian
    visibility_filter: Optional[torch.Tensor] = None # (N,) boolean mask of Gaussians visible in view


@dataclass
class SplatMetrics:
    """Reconstruction evaluation metrics."""
    psnr_db: float
    ssim: float
    lpips: float = 0.0
    num_gaussians: int = 0
    training_vram_mb: float = 0.0
    eval_vram_mb: float = 0.0
    rendering_fps: float = 0.0
    iteration: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "psnr_db": round(self.psnr_db, 3),
            "ssim": round(self.ssim, 4),
            "lpips": round(self.lpips, 4),
            "num_gaussians": self.num_gaussians,
            "training_vram_mb": round(self.training_vram_mb, 1),
            "eval_vram_mb": round(self.eval_vram_mb, 1),
            "rendering_fps": round(self.rendering_fps, 1)
        }
