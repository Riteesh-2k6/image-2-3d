"""
3D Gaussian Scene Representation Model (Stage 3).
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianModel(nn.Module):
    """
    Parameterizes 3D Gaussian primitives:
    - means: 3D positions (x, y, z)
    - log_scales: log 3D scaling (sx, sy, sz)
    - rotations: unit quaternions (w, x, y, z)
    - logit_opacities: raw opacity before sigmoid
    - sh_coeffs: spherical harmonics radiance coefficients
    """
    def __init__(
        self,
        sh_degree: int = 0,
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ):
        super().__init__()
        self.sh_degree = sh_degree
        self.device = device
        self.max_sh_channels = (sh_degree + 1) ** 2

        # Model parameters
        self._means = nn.Parameter(torch.empty(0, 3, device=device))
        self._log_scales = nn.Parameter(torch.empty(0, 3, device=device))
        self._rotations = nn.Parameter(torch.empty(0, 4, device=device))
        self._logit_opacities = nn.Parameter(torch.empty(0, 1, device=device))
        self._sh_coeffs = nn.Parameter(torch.empty(0, self.max_sh_channels, 3, device=device))

        # Density control tracking state
        self.xyz_gradient_accum = torch.empty(0, 1, device=device)
        self.denom = torch.empty(0, 1, device=device)
        self.max_radii2D = torch.empty(0, device=device)

    @property
    def num_gaussians(self) -> int:
        return self._means.shape[0]

    @property
    def get_means(self) -> torch.Tensor:
        return self._means

    @property
    def get_scales(self) -> torch.Tensor:
        return torch.exp(self._log_scales)

    @property
    def get_rotations(self) -> torch.Tensor:
        return F.normalize(self._rotations, dim=-1)

    @property
    def get_quats(self) -> torch.Tensor:
        return self.get_rotations

    @property
    def get_opacities(self) -> torch.Tensor:
        return torch.sigmoid(self._logit_opacities)

    @property
    def get_colors(self) -> torch.Tensor:
        """Returns direct RGB colors (SH degree 0) activated via sigmoid."""
        # SH degree 0 factor: C0 = 0.28209479177387814
        C0 = 0.28209479177387814
        rgb = self._sh_coeffs[:, 0, :] * C0 + 0.5
        return torch.clamp(rgb, 0.0, 1.0)

    def initialize_from_points(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        init_scale: float = 0.1,
        init_opacity: float = 0.5
    ):
        """Initializes Gaussian parameters from a point cloud."""
        N = len(points)
        pts_tensor = torch.tensor(points, dtype=torch.float32, device=self.device)
        
        # Scales initialized from nearest neighbor distance or constant
        scales = torch.full((N, 3), init_scale, dtype=torch.float32, device=self.device)
        log_scales = torch.log(torch.clamp(scales, min=1e-5))

        # Rotations initialized to identity quaternion [1, 0, 0, 0]
        rots = torch.zeros((N, 4), dtype=torch.float32, device=self.device)
        rots[:, 0] = 1.0

        # Opacities initialized via inverse sigmoid (logit)
        logit_opacities = torch.full((N, 1), float(np.log(init_opacity / (1.0 - init_opacity))), dtype=torch.float32, device=self.device)

        # Spherical harmonics
        shs = torch.zeros((N, self.max_sh_channels, 3), dtype=torch.float32, device=self.device)
        if colors is not None:
            c_norm = torch.tensor(colors / 255.0, dtype=torch.float32, device=self.device)
            C0 = 0.28209479177387814
            shs[:, 0, :] = (c_norm - 0.5) / C0

        self._means = nn.Parameter(pts_tensor)
        self._log_scales = nn.Parameter(log_scales)
        self._rotations = nn.Parameter(rots)
        self._logit_opacities = nn.Parameter(logit_opacities)
        self._sh_coeffs = nn.Parameter(shs)

        self.xyz_gradient_accum = torch.zeros((N, 1), device=self.device)
        self.denom = torch.zeros((N, 1), device=self.device)
        self.max_radii2D = torch.zeros(N, device=self.device)

    def get_covariance_3d(self) -> torch.Tensor:
        """Computes 3D covariance matrix Sigma = R * S * S^T * R^T for all Gaussians."""
        scales = self.get_scales
        rots = self.get_rotations

        # Construct rotation matrix from quaternion
        w, x, y, z = rots[:, 0], rots[:, 1], rots[:, 2], rots[:, 3]
        R = torch.zeros((self.num_gaussians, 3, 3), device=self.device)
        R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
        R[:, 0, 1] = 2 * (x*y - w*z)
        R[:, 0, 2] = 2 * (x*z + w*y)
        R[:, 1, 0] = 2 * (x*y + w*z)
        R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
        R[:, 1, 2] = 2 * (y*z - w*x)
        R[:, 2, 0] = 2 * (x*z - w*y)
        R[:, 2, 1] = 2 * (y*z + w*x)
        R[:, 2, 2] = 1 - 2 * (x**2 + y**2)

        # Scale matrix S
        S = torch.diag_embed(scales)
        RS = torch.bmm(R, S)
        sigma = torch.bmm(RS, RS.transpose(1, 2))
        return sigma

    def export_ply(self, filename: str):
        """Exports Gaussians into standard 3DGS PLY format."""
        import os
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        means = self.get_means.detach().cpu().numpy()
        scales = self.get_scales.detach().cpu().numpy()
        rots = self.get_rotations.detach().cpu().numpy()
        opacities = self.get_opacities.detach().cpu().numpy()
        colors = (self.get_colors.detach().cpu().numpy() * 255.0).astype(np.uint8)

        num_pts = len(means)
        with open(filename, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {num_pts}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("property float opacity\n")
            f.write("property float scale_0\n")
            f.write("property float scale_1\n")
            f.write("property float scale_2\n")
            f.write("property float rot_0\n")
            f.write("property float rot_1\n")
            f.write("property float rot_2\n")
            f.write("property float rot_3\n")
            f.write("end_header\n")
            for i in range(num_pts):
                f.write(
                    f"{means[i,0]:.4f} {means[i,1]:.4f} {means[i,2]:.4f} "
                    f"{colors[i,0]} {colors[i,1]} {colors[i,2]} "
                    f"{opacities[i,0]:.4f} "
                    f"{scales[i,0]:.4f} {scales[i,1]:.4f} {scales[i,2]:.4f} "
                    f"{rots[i,0]:.4f} {rots[i,1]:.4f} {rots[i,2]:.4f} {rots[i,3]:.4f}\n"
                )
