"""
Adaptive Density Controller for 3D Gaussian Splatting (Stage 3).
Handles splitting over-sized Gaussians, cloning under-reconstructed areas, and pruning transparent floaters.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.splatting.gaussian_model import GaussianModel


class DensityController:
    """
    Adaptive Gaussian densification & pruning engine:
    - Splits large Gaussians with high positional gradients
    - Clones small Gaussians in under-reconstructed regions
    - Prunes transparent Gaussians (opacity < min_opacity) and excessively large screen-space floaters
    """
    def __init__(
        self,
        tau_grad: float = 0.0002,
        tau_scale: float = 0.2,
        min_opacity: float = 0.005,
        max_screen_radius: float = 200.0,
        densify_interval: int = 100,
        opacity_reset_interval: int = 3000
    ):
        self.tau_grad = tau_grad
        self.tau_scale = tau_scale
        self.min_opacity = min_opacity
        self.max_screen_radius = max_screen_radius
        self.densify_interval = densify_interval
        self.opacity_reset_interval = opacity_reset_interval

    def record_step(
        self,
        model: GaussianModel,
        radii: torch.Tensor,
        visibility_mask: torch.Tensor
    ):
        """Records view-dependent 2D screen-space gradients and radii."""
        if model.get_means.grad is None:
            return
        
        # 2D screen-space gradient norm approximation
        grad_norm = torch.norm(model.get_means.grad[visibility_mask, :2], dim=-1, keepdim=True)
        model.xyz_gradient_accum[visibility_mask] += grad_norm
        model.denom[visibility_mask] += 1
        model.max_radii2D[visibility_mask] = torch.max(model.max_radii2D[visibility_mask], radii[visibility_mask])

    def densify_and_prune(
        self,
        model: GaussianModel,
        optimizer: torch.optim.Optimizer,
        scene_extent: float = 50.0
    ) -> Dict[str, int]:
        """Performs split, clone, and prune operations on Gaussian parameters."""
        device = model.device
        grads = model.xyz_gradient_accum / torch.clamp(model.denom, min=1.0)
        grads[torch.isnan(grads)] = 0.0

        scales = model.get_scales
        max_scale = torch.max(scales, dim=-1)[0]

        # 1. Identify Clone Candidates: High gradient & Small scale
        clone_mask = (grads.squeeze(-1) >= self.tau_grad) & (max_scale <= self.tau_scale * scene_extent)

        # 2. Identify Split Candidates: High gradient & Large scale
        split_mask = (grads.squeeze(-1) >= self.tau_grad) & (max_scale > self.tau_scale * scene_extent)

        # Extract parameters for cloning
        new_means = []
        new_scales = []
        new_rots = []
        new_opacities = []
        new_shs = []

        if clone_mask.sum() > 0:
            new_means.append(model._means[clone_mask].detach())
            new_scales.append(model._log_scales[clone_mask].detach())
            new_rots.append(model._rotations[clone_mask].detach())
            new_opacities.append(model._logit_opacities[clone_mask].detach())
            new_shs.append(model._sh_coeffs[clone_mask].detach())

        # Extract parameters for splitting (split 1 Gaussian into 2 smaller ones)
        if split_mask.sum() > 0:
            split_scales = model.get_scales[split_mask] / 1.6
            split_log_scales = torch.log(torch.clamp(split_scales, min=1e-5))
            
            # Sample perturbed offsets from Gaussian covariance
            stds = split_scales
            samples = torch.randn((split_mask.sum(), 3), device=device) * stds
            
            new_means.append((model._means[split_mask] + samples).detach())
            new_means.append((model._means[split_mask] - samples).detach())

            new_scales.append(split_log_scales.detach())
            new_scales.append(split_log_scales.detach())

            new_rots.append(model._rotations[split_mask].detach())
            new_rots.append(model._rotations[split_mask].detach())

            new_opacities.append(model._logit_opacities[split_mask].detach())
            new_opacities.append(model._logit_opacities[split_mask].detach())

            new_shs.append(model._sh_coeffs[split_mask].detach())
            new_shs.append(model._sh_coeffs[split_mask].detach())

        # 3. Identify Prune Mask: Remove split parents + transparent Gaussians + over-sized screen radii
        prune_mask = (model.get_opacities.squeeze(-1) < self.min_opacity) | (model.max_radii2D > self.max_screen_radius)
        prune_mask = prune_mask | split_mask # remove original split parents

        keep_mask = ~prune_mask

        # Re-assemble kept parameters
        kept_means = model._means[keep_mask].detach()
        kept_scales = model._log_scales[keep_mask].detach()
        kept_rots = model._rotations[keep_mask].detach()
        kept_opacities = model._logit_opacities[keep_mask].detach()
        kept_shs = model._sh_coeffs[keep_mask].detach()

        if new_means:
            all_means = torch.cat([kept_means] + new_means, dim=0)
            all_scales = torch.cat([kept_scales] + new_scales, dim=0)
            all_rots = torch.cat([kept_rots] + new_rots, dim=0)
            all_opacities = torch.cat([kept_opacities] + new_opacities, dim=0)
            all_shs = torch.cat([kept_shs] + new_shs, dim=0)
        else:
            all_means = kept_means
            all_scales = kept_scales
            all_rots = kept_rots
            all_opacities = kept_opacities
            all_shs = kept_shs

        # Update model parameters
        model._means = nn.Parameter(all_means)
        model._log_scales = nn.Parameter(all_scales)
        model._rotations = nn.Parameter(all_rots)
        model._logit_opacities = nn.Parameter(all_opacities)
        model._sh_coeffs = nn.Parameter(all_shs)

        N_new = all_means.shape[0]
        model.xyz_gradient_accum = torch.zeros((N_new, 1), device=device)
        model.denom = torch.zeros((N_new, 1), device=device)
        model.max_radii2D = torch.zeros(N_new, device=device)

        return {
            "num_cloned": int(clone_mask.sum().item()),
            "num_split": int(split_mask.sum().item()),
            "num_pruned": int(prune_mask.sum().item()),
            "total_gaussians": N_new
        }
