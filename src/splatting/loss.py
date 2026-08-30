"""
Loss functions for 3D Gaussian Splatting with GeoPrior Anchoring.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def ssim_loss(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, channel: int = 3) -> torch.Tensor:
    """Computes Structural Similarity Index (SSIM) between two (3, H, W) images."""
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    # 1D Gaussian kernel
    sigma = 1.5
    gauss = torch.tensor([
        np_gauss(x - window_size // 2, sigma) for x in range(window_size)
    ], device=img1.device, dtype=img1.dtype)
    gauss = gauss / gauss.sum()

    # 2D Gaussian window
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def np_gauss(x: float, sigma: float) -> float:
    import math
    return math.exp(-(x ** 2) / (2 * sigma ** 2))


class CombinedSplatLoss(nn.Module):
    """
    Combined multi-objective loss:
    L = (1 - lambda_dssim) * L1 + lambda_dssim * (1 - SSIM) + lambda_prior * L_anchor + lambda_scale * L_scale
    """
    def __init__(
        self,
        lambda_dssim: float = 0.2,
        lambda_scale: float = 0.01,
        max_scale_ratio: float = 20.0
    ):
        super().__init__()
        self.lambda_dssim = lambda_dssim
        self.lambda_scale = lambda_scale
        self.max_scale_ratio = max_scale_ratio

    def forward(
        self,
        rendered_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        scales: Optional[torch.Tensor] = None,
        anchor_loss: Optional[torch.Tensor] = None,
        lambda_prior: float = 0.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. Photometric L1 Loss
        l1 = F.l1_loss(rendered_rgb, target_rgb)

        # 2. D-SSIM Loss
        ssim_val = ssim_loss(rendered_rgb, target_rgb)
        d_ssim = 1.0 - ssim_val

        total_loss = (1.0 - self.lambda_dssim) * l1 + self.lambda_dssim * d_ssim

        # 3. Scale Anisotropy Regularization (penalizes extreme needle-like floaters)
        scale_loss_val = 0.0
        if scales is not None and self.lambda_scale > 0:
            s_max = torch.max(scales, dim=-1)[0]
            s_min = torch.clamp(torch.min(scales, dim=-1)[0], min=1e-5)
            ratio = s_max / s_min
            scale_loss = F.relu(ratio - self.max_scale_ratio).mean()
            total_loss = total_loss + self.lambda_scale * scale_loss
            scale_loss_val = scale_loss.item()

        # 4. GeoPrior Structural Anchor Loss
        prior_loss_val = 0.0
        if anchor_loss is not None and lambda_prior > 0:
            total_loss = total_loss + lambda_prior * anchor_loss
            prior_loss_val = anchor_loss.item()

        loss_breakdown = {
            "total_loss": total_loss.item(),
            "l1": l1.item(),
            "ssim": ssim_val.item(),
            "d_ssim": d_ssim.item(),
            "scale_reg": scale_loss_val,
            "prior_anchor": prior_loss_val
        }

        return total_loss, loss_breakdown
