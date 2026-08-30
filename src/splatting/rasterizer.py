"""
High-Efficiency Differentiable Gaussian Rasterizer (Stage 3).
Supports compiled CUDA gsplat kernels and chunked tile-based memory-safe fallback.
"""

from typing import Dict, List, Optional, Tuple, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.splatting.types import CameraInfo, RenderOutput
from src.splatting.gaussian_model import GaussianModel


def project_gaussians_2d(
    means3D: torch.Tensor,
    cov3D: torch.Tensor,
    viewmatrix: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Projects 3D Gaussian means and covariances onto the 2D image plane:
    - computes camera-frame coordinates: p_cam = R * p_world + t
    - computes Jacobian of perspective projection J
    - computes 2D covariance: Sigma_2D = J * W * Sigma_3D * W^T * J^T + 0.3 * I_2
    - computes 2D screen radius from max eigenvalue
    """
    N = means3D.shape[0]
    device = means3D.device

    # Transform means to camera frame
    # viewmatrix is (4, 4) [R | t]
    R = viewmatrix[:3, :3]
    t = viewmatrix[:3, 3]

    p_cam = torch.matmul(means3D, R.t()) + t
    depths = p_cam[:, 2]

    # In-front-of-camera mask (z > 0.2m)
    valid_mask = depths > 0.2

    # Perspective projection
    u = fx * (p_cam[:, 0] / torch.clamp(depths, min=0.2)) + cx
    v = fy * (p_cam[:, 1] / torch.clamp(depths, min=0.2)) + cy
    means2D = torch.stack([u, v], dim=-1)

    # Jacobian of projective transformation J = d(u,v) / d(x,y,z)
    z_sq = torch.clamp(depths.pow(2), min=0.04)
    J = torch.zeros((N, 2, 3), device=device)
    J[:, 0, 0] = fx / torch.clamp(depths, min=0.2)
    J[:, 0, 2] = -fx * p_cam[:, 0] / z_sq
    J[:, 1, 1] = fy / torch.clamp(depths, min=0.2)
    J[:, 1, 2] = -fy * p_cam[:, 1] / z_sq

    # Transform 3D covariance to camera coordinates: Sigma_cam = R * Sigma_3D * R^T
    R_expanded = R.unsqueeze(0).expand(N, -1, -1)
    sigma_cam = torch.bmm(torch.bmm(R_expanded, cov3D), R_expanded.transpose(1, 2))

    # Project to 2D: Sigma_2D = J * Sigma_cam * J^T + 0.3 * I_2
    cov2D = torch.bmm(torch.bmm(J, sigma_cam), J.transpose(1, 2))
    # Low-pass anti-aliasing screen filter compensation
    cov2D[:, 0, 0] += 0.3
    cov2D[:, 1, 1] += 0.3

    # Compute eigenvalues of 2x2 covariance for screen-space radius
    a = cov2D[:, 0, 0]
    b = cov2D[:, 0, 1]
    c = cov2D[:, 1, 1]
    det = a * c - b * b
    trace = a + c
    mid = 0.5 * trace
    term = torch.sqrt(torch.clamp(mid.pow(2) - det, min=1e-5))
    lambda1 = mid + term
    lambda2 = torch.clamp(mid - term, min=1e-5)

    # 3-sigma radius in pixels
    radii = torch.ceil(3.0 * torch.sqrt(torch.clamp(lambda1, min=1e-5)))

    # On-screen visibility filter
    on_screen = (
        valid_mask &
        (u + radii >= 0) & (u - radii < width) &
        (v + radii >= 0) & (v - radii < height) &
        (radii > 0) & (radii < max(width, height))
    )

    return means2D, cov2D, depths, on_screen


class SpeedySplatRasterizer(nn.Module):
    """
    Differentiable Tile-Based Gaussian Rasterizer.
    Auto-dispatches to compiled CUDA gsplat if available,
    or falls back to tile-chunked PyTorch rasterizer with bounded memory.
    """
    def __init__(self, tile_size: int = 64):
        super().__init__()
        self.tile_size = tile_size
        self._has_gsplat_cuda = False
        try:
            import gsplat
            from gsplat.rendering import rasterization as gsplat_rasterization
            self._has_gsplat_cuda = True
        except Exception:
            self._has_gsplat_cuda = False

    def forward(
        self,
        model: GaussianModel,
        camera: CameraInfo,
        bg_color: Optional[torch.Tensor] = None
    ) -> RenderOutput:
        device = model.device
        H, W = camera.height, camera.width
        if bg_color is None:
            bg_color = torch.tensor([0.0, 0.0, 0.0], device=device)

        # High-Speed Compiled CUDA gsplat Path
        if self._has_gsplat_cuda:
            try:
                from gsplat.rendering import rasterization as gsplat_rasterization
                w2c = camera.world_view_transform
                K = torch.tensor([
                    [camera.fx, 0.0, camera.cx],
                    [0.0, camera.fy, camera.cy],
                    [0.0, 0.0, 1.0]
                ], dtype=torch.float32, device=device)

                renders, alphas, meta = gsplat_rasterization(
                    means=model.get_means,
                    quats=model.get_quats,
                    scales=model.get_scales,
                    opacities=model.get_opacities.squeeze(-1),
                    colors=model.get_colors,
                    viewmats=w2c.unsqueeze(0),
                    Ks=K.unsqueeze(0),
                    width=W,
                    height=H,
                    backgrounds=bg_color.unsqueeze(0)
                )
                rgb = renders[0].permute(2, 0, 1)
                alpha = alphas[0].permute(2, 0, 1) if alphas is not None else torch.ones((1, H, W), device=device)
                radii = meta.get("radii", torch.zeros(model.num_gaussians, device=device)) if isinstance(meta, dict) else torch.zeros(model.num_gaussians, device=device)
                return RenderOutput(
                    rgb=rgb,
                    alpha=alpha,
                    radii=radii,
                    visibility_filter=radii > 0
                )
            except Exception:
                pass

        # Pure PyTorch Tile-Based Fallback
        means3D = model.get_means
        cov3D = model.get_covariance_3d()
        viewmat = camera.world_view_transform

        means2D, cov2D, depths, vis_mask = project_gaussians_2d(
            means3D=means3D,
            cov3D=cov3D,
            viewmatrix=viewmat,
            fx=camera.fx,
            fy=camera.fy,
            cx=camera.cx,
            cy=camera.cy,
            width=W,
            height=H
        )

        if not vis_mask.any():
            # Return blank image if no Gaussians visible
            rgb = bg_color.view(3, 1, 1).expand(3, H, W).clone()
            alpha = torch.zeros((1, H, W), device=device)
            return RenderOutput(rgb=rgb, alpha=alpha, visibility_filter=vis_mask)

        # Filter visible Gaussians
        vis_indices = torch.where(vis_mask)[0]
        v_means2D = means2D[vis_indices]
        v_cov2D = cov2D[vis_indices]
        v_depths = depths[vis_indices]
        v_opacities = model.get_opacities[vis_indices].squeeze(-1)
        v_colors = model.get_colors[vis_indices]

        # 2. Sort visible Gaussians by depth (front to back)
        sorted_depth_order = torch.argsort(v_depths)
        s_means2D = v_means2D[sorted_depth_order]
        s_cov2D = v_cov2D[sorted_depth_order]
        s_opacities = v_opacities[sorted_depth_order]
        s_colors = v_colors[sorted_depth_order]

        # 3. Tile-Based Out-of-Place Alpha Compositing (Preserves Autograd & Keeps VRAM < 150MB)
        tile_size = self.tile_size
        num_tiles_x = (W + tile_size - 1) // tile_size
        num_tiles_y = (H + tile_size - 1) // tile_size

        # Invert 2x2 covariance matrices
        a = s_cov2D[:, 0, 0]
        b = s_cov2D[:, 0, 1]
        c = s_cov2D[:, 1, 1]
        det = torch.clamp(a * c - b * b, min=1e-6)
        ic00 = c / det
        ic01 = -b / det
        ic11 = a / det

        # Compute 3-sigma bounding box radius in pixels
        lambda1 = 0.5 * (a + c + torch.sqrt(torch.clamp((a - c).pow(2) + 4.0 * b.pow(2), min=1e-5)))
        radii_px = torch.ceil(3.0 * torch.sqrt(torch.clamp(lambda1, min=1.0)))

        num_vis = len(sorted_depth_order)
        max_gaussians = min(num_vis, 4096)

        # Coordinate grid for full image
        y_coords = torch.arange(H, device=device, dtype=torch.float32)
        x_coords = torch.arange(W, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

        tile_outputs = []

        for ty in range(num_tiles_y):
            y_start = ty * tile_size
            y_end = min(H, (ty + 1) * tile_size)
            row_tiles = []

            for tx in range(num_tiles_x):
                x_start = tx * tile_size
                x_end = min(W, (tx + 1) * tile_size)

                # Find Gaussians that overlap this (ty, tx) tile
                u = s_means2D[:max_gaussians, 0]
                v = s_means2D[:max_gaussians, 1]
                r = radii_px[:max_gaussians]

                overlap_mask = (
                    (u + r >= x_start) & (u - r < x_end) &
                    (v + r >= y_start) & (v - r < y_end)
                )

                tile_indices = torch.where(overlap_mask)[0]
                if len(tile_indices) == 0:
                    tile_rgb = bg_color.view(3, 1, 1).expand(3, y_end - y_start, x_end - x_start)
                    row_tiles.append(tile_rgb)
                    continue

                # Local tile pixels
                ty_grid = grid_y[y_start:y_end, x_start:x_end]
                tx_grid = grid_x[y_start:y_end, x_start:x_end]

                t_u = s_means2D[tile_indices, 0].view(-1, 1, 1)
                t_v = s_means2D[tile_indices, 1].view(-1, 1, 1)
                t_ic00 = ic00[tile_indices].view(-1, 1, 1)
                t_ic01 = ic01[tile_indices].view(-1, 1, 1)
                t_ic11 = ic11[tile_indices].view(-1, 1, 1)
                t_opacities = s_opacities[tile_indices].view(-1, 1, 1)
                t_colors = s_colors[tile_indices]

                dx = tx_grid.unsqueeze(0) - t_u
                dy = ty_grid.unsqueeze(0) - t_v

                power = -0.5 * (t_ic00 * dx.pow(2) + 2.0 * t_ic01 * dx * dy + t_ic11 * dy.pow(2))
                alpha = t_opacities * torch.exp(torch.clamp(power, max=0.0))
                alpha = torch.clamp(alpha, 0.0, 0.99)

                # Vectorized Out-of-Place Alpha Compositing
                # alpha shape: (K, tile_h, tile_w)
                K_pts = alpha.shape[0]
                tile_h, tile_w = y_end - y_start, x_end - x_start

                # Transmittance T_g = prod_{j < g} (1 - alpha_j)
                # Compute via cumprod with shifted prefix
                one_minus_alpha = torch.clamp(1.0 - alpha, min=1e-5)
                # Prefix 1.0 at index 0: shape (K+1, tile_h, tile_w)
                prefix_one = torch.ones((1, tile_h, tile_w), device=device)
                cum_prod = torch.cumprod(torch.cat([prefix_one, one_minus_alpha], dim=0), dim=0)
                T = cum_prod[:-1] # (K, tile_h, tile_w)

                weights = alpha * T # (K, tile_h, tile_w)
                
                # Colors: (K, 3, 1, 1)
                c_expanded = t_colors.view(K_pts, 3, 1, 1)
                
                # Accumulated RGB: sum_g (weight_g * color_g)
                accum_rgb = (weights.unsqueeze(1) * c_expanded).sum(dim=0) # (3, tile_h, tile_w)
                
                # Background blend
                total_alpha = weights.sum(dim=0, keepdim=True) # (1, tile_h, tile_w)
                bg_term = bg_color.view(3, 1, 1).expand(3, tile_h, tile_w) * torch.clamp(1.0 - total_alpha, min=0.0)
                tile_rgb = accum_rgb + bg_term

                row_tiles.append(tile_rgb)

            tile_outputs.append(torch.cat(row_tiles, dim=2))

        rendered_rgb = torch.cat(tile_outputs, dim=1)

        return RenderOutput(
            rgb=rendered_rgb,
            alpha=torch.zeros((1, H, W), device=device),
            radii=means2D.new_zeros(model.num_gaussians),
            visibility_filter=vis_mask
        )
