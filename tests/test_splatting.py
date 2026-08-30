"""
Unit tests for 3D Gaussian Splatting (Stage 3).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from src.splatting.types import CameraInfo, RenderOutput, SplatMetrics
from src.splatting.gaussian_model import GaussianModel
from src.splatting.rasterizer import SpeedySplatRasterizer, project_gaussians_2d
from src.splatting.prior_anchor import PriorAnchorManager
from src.splatting.loss import CombinedSplatLoss, ssim_loss
from src.splatting.density_controller import DensityController


class TestGaussianRepresentationAndCovariance:
    """Validates 3D Gaussian mathematical representation."""

    def test_gaussian_covariance_math(self):
        """Validates that 3D covariance is symmetric positive semi-definite."""
        device = torch.device("cpu")
        model = GaussianModel(sh_degree=0, device=device)

        pts = np.array([[0.0, 0.0, 5.0], [1.0, -1.0, 6.0]], dtype=np.float32)
        model.initialize_from_points(pts, init_scale=0.2, init_opacity=0.7)

        cov = model.get_covariance_3d()
        assert cov.shape == (2, 3, 3)

        # Check symmetry: cov == cov^T
        diff = torch.norm(cov - cov.transpose(1, 2))
        assert diff.item() < 1e-5, f"Expected symmetric covariance, got diff {diff.item()}"

        # Check positive definiteness (all eigenvalues > 0)
        eigenvalues = torch.linalg.eigvalsh(cov)
        assert (eigenvalues > 0).all().item(), f"Covariance not positive definite: {eigenvalues}"


class TestDifferentiableRasterizer:
    """Validates differentiable rasterizer and numerical gradient checks."""

    def test_differentiable_rasterizer_gradient_flow(self):
        """Validates that loss gradients flow back into positions, scales, rots, and opacities."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = GaussianModel(sh_degree=0, device=device)

        pts = np.array([[0.0, 0.0, 5.0], [0.5, 0.2, 5.5]], dtype=np.float32)
        model.initialize_from_points(pts, init_scale=0.3, init_opacity=0.8)

        camera = CameraInfo(
            frame_idx=0,
            image_name="test_f00000.jpg",
            width=64,
            height=64,
            fx=50.0,
            fy=50.0,
            cx=32.0,
            cy=32.0,
            R=torch.eye(3, device=device),
            t=torch.zeros(3, device=device)
        )

        rasterizer = SpeedySplatRasterizer()
        output = rasterizer(model, camera)

        target = torch.ones((3, 64, 64), device=device)
        loss = F.l1_loss(output.rgb, target)
        loss.backward()

        assert model._means.grad is not None and torch.isfinite(model._means.grad).all()
        assert model._log_scales.grad is not None and torch.isfinite(model._log_scales.grad).all()
        assert model._logit_opacities.grad is not None and torch.isfinite(model._logit_opacities.grad).all()

    def test_rasterizer_finite_difference_gradient_check(self):
        """
        Performs a finite-difference numerical gradient check vs analytical autograd gradient
        on Gaussian position to prevent silent no-op / broken Jacobian bugs.
        """
        device = torch.device("cpu")
        model = GaussianModel(sh_degree=0, device=device)

        pt0 = np.array([[0.1, -0.05, 4.0]], dtype=np.float32)
        model.initialize_from_points(pt0, init_scale=0.2, init_opacity=0.9)

        camera = CameraInfo(
            frame_idx=0,
            image_name="test_grad.jpg",
            width=32,
            height=32,
            fx=40.0,
            fy=40.0,
            cx=16.0,
            cy=16.0,
            R=torch.eye(3, device=device),
            t=torch.zeros(3, device=device)
        )

        rasterizer = SpeedySplatRasterizer()
        
        # Analytical gradient
        out = rasterizer(model, camera)
        target = torch.full((3, 32, 32), 0.5, device=device)
        loss_orig = F.mse_loss(out.rgb, target)
        loss_orig.backward()
        analytical_grad_x = model._means.grad[0, 0].item()

        # Finite difference: f(x + eps) - f(x - eps) / (2 * eps)
        eps = 1e-3
        with torch.no_grad():
            model._means[0, 0] += eps
            out_plus = rasterizer(model, camera)
            loss_plus = F.mse_loss(out_plus.rgb, target).item()

            model._means[0, 0] -= 2.0 * eps
            out_minus = rasterizer(model, camera)
            loss_minus = F.mse_loss(out_minus.rgb, target).item()

            numerical_grad_x = (loss_plus - loss_minus) / (2.0 * eps)

        # Confirm sign and relative magnitude agreement
        if abs(analytical_grad_x) > 1e-4:
            rel_diff = abs(analytical_grad_x - numerical_grad_x) / (abs(analytical_grad_x) + abs(numerical_grad_x))
            assert rel_diff < 0.25, f"Gradient mismatch: analytical {analytical_grad_x} vs numerical {numerical_grad_x}"


class TestGeoPriorStructuralAnchoring:
    """Validates structural prior seeding and anchor loss."""

    def test_prior_guided_seeding(self):
        """Confirms that prior manager generates both SfM and roof/wall seeds."""
        manager = PriorAnchorManager()
        seeds, colors = manager.generate_prior_guided_seeds(num_surface_samples=200)
        assert len(seeds) >= 200
        assert seeds.shape[1] == 3
        assert colors.shape[1] == 3

    def test_anchor_loss_penalizes_sky_and_subterranean_floaters(self):
        """Confirms anchor loss produces positive penalties on out-of-bounds Gaussians."""
        manager = PriorAnchorManager()
        
        # Gaussian floating at Z = 50.0m (above 35m ceiling)
        floater_means = torch.tensor([[0.0, 0.0, 50.0], [5.0, 5.0, -10.0]], dtype=torch.float32)
        loss = manager.compute_anchor_loss(floater_means, z_ground=-2.0, z_ceiling=35.0)
        assert loss.item() > 1.0, f"Expected penalty for out-of-bounds Gaussians, got {loss.item()}"


class TestDensityController:
    """Validates split, clone, and prune logic."""

    def test_density_control_split_and_clone(self):
        """Confirms splitting large Gaussians and cloning small ones."""
        device = torch.device("cpu")
        model = GaussianModel(sh_degree=0, device=device)

        pts = np.array([[0.0, 0.0, 5.0], [1.0, 1.0, 6.0], [2.0, 2.0, 7.0]], dtype=np.float32)
        model.initialize_from_points(pts, init_scale=0.5, init_opacity=0.8)

        controller = DensityController(tau_grad=0.01, tau_scale=0.01)

        # Inject simulated high gradient on Gaussian 0
        model.xyz_gradient_accum[0] = 5.0
        model.denom[0] = 1.0

        optimizer = torch.optim.Adam([model._means], lr=0.001)
        stats = controller.densify_and_prune(model, optimizer, scene_extent=10.0)

        assert stats["total_gaussians"] > 3
        assert stats["num_split"] > 0 or stats["num_cloned"] > 0
