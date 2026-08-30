"""
Speedy-Splat Master Training & Evaluation Pipeline (Stage 3).
"""

from typing import Dict, List, Optional, Tuple, Any
import os
import sys
import json
import glob
import re
import math
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.splatting.types import CameraInfo, RenderOutput, SplatMetrics
from src.splatting.gaussian_model import GaussianModel
from src.splatting.rasterizer import SpeedySplatRasterizer
from src.splatting.prior_anchor import PriorAnchorManager
from src.splatting.loss import CombinedSplatLoss, ssim_loss
from src.splatting.density_controller import DensityController


def extract_keyframe_idx(filename: str) -> int:
    """Extracts 0-indexed keyframe number from filename (e.g. keyframe_0001_f00000.jpg -> 0)."""
    match = re.search(r"keyframe_(\d+)_", os.path.basename(filename))
    if match:
        return int(match.group(1)) - 1
    nums = re.findall(r"\d+", os.path.basename(filename))
    return int(nums[0]) if nums else 0


class SpeedySplatTrainer:
    """
    Orchestrates 3D Gaussian Splatting training with GeoPrior structural anchoring.
    """
    def __init__(
        self,
        keyframes_dir: str = "output/06_keyframes",
        georef_poses_json: str = "output/06_georef/06_georef_poses.json",
        georef_cloud_ply: str = "output/06_georef/06_georef_cloud.ply",
        output_dir: str = "output/06_splat",
        lambda_prior: float = 0.001,
        lambda_dssim: float = 0.2,
        lr_means: float = 0.005,
        lr_sh: float = 0.02,
        lr_opacity: float = 0.005,
        lr_scales: float = 0.001,
        lr_rots: float = 0.001,
        sh_degree: int = 0,
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ):
        self.keyframes_dir = keyframes_dir
        self.georef_poses_json = georef_poses_json
        self.georef_cloud_ply = georef_cloud_ply
        self.output_dir = output_dir
        self.lambda_prior = lambda_prior
        self.device = device

        os.makedirs(self.output_dir, exist_ok=True)

        self.model = GaussianModel(sh_degree=sh_degree, device=device)
        self.rasterizer = SpeedySplatRasterizer()
        self.prior_manager = PriorAnchorManager(georef_cloud_ply=georef_cloud_ply, device=device)
        self.loss_fn = CombinedSplatLoss(lambda_dssim=lambda_dssim)
        self.density_controller = DensityController(tau_grad=0.0002, tau_scale=0.1)

        # Learning rates
        self.lr_means = lr_means
        self.lr_sh = lr_sh
        self.lr_opacity = lr_opacity
        self.lr_scales = lr_scales
        self.lr_rots = lr_rots

    def load_dataset(self, downscale: int = 4) -> Tuple[List[CameraInfo], List[CameraInfo]]:
        """Loads keyframes and georeferenced poses, splitting into 70% Train / 30% Test views."""
        with open(self.georef_poses_json, "r") as f:
            poses_data = json.load(f)

        keyframe_paths = sorted(glob.glob(os.path.join(self.keyframes_dir, "*.jpg")), key=extract_keyframe_idx)
        img_map = {extract_keyframe_idx(p): p for p in keyframe_paths}

        cameras = []
        for p_info in poses_data.get("poses", []):
            f_idx = p_info["frame_idx"]
            if f_idx not in img_map:
                continue

            # Load and downscale image for training
            img_bgr = cv2.imread(img_map[f_idx])
            H_orig, W_orig = img_bgr.shape[:2]
            H = H_orig // downscale
            W = W_orig // downscale
            img_rgb = cv2.cvtColor(cv2.resize(img_bgr, (W, H)), cv2.COLOR_BGR2RGB)
            img_tensor = torch.tensor(img_rgb / 255.0, dtype=torch.float32, device=self.device).permute(2, 0, 1)

            # Calibrated focal length scaled from 960x540 base (fx=551.20)
            scale_ratio = W / 960.0
            fx = 551.2046 * scale_ratio
            fy = 551.2046 * scale_ratio
            cx = W / 2.0
            cy = H / 2.0

            R_mat = torch.tensor(p_info["rotation_matrix"], dtype=torch.float32, device=self.device)
            t_vec = torch.tensor(p_info["translation_vector"], dtype=torch.float32, device=self.device)

            cameras.append(CameraInfo(
                frame_idx=f_idx,
                image_name=os.path.basename(img_map[f_idx]),
                width=W,
                height=H,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                R=R_mat,
                t=t_vec,
                image_tensor=img_tensor
            ))

        num_total = len(cameras)
        # Uniform 360-degree interleaved holdout (every 4th view is a novel held-out test viewpoint: 25% test, 75% train)
        train_views = [c for i, c in enumerate(cameras) if i % 4 != 0]
        test_views = [c for i, c in enumerate(cameras) if i % 4 == 0]
        return train_views, test_views

    def initialize_scene(self, use_priors: bool = True):
        """Initializes 3D Gaussian cloud from prior-guided seeds or pure SfM cloud."""
        if use_priors:
            pts, colors = self.prior_manager.generate_prior_guided_seeds(num_surface_samples=2000)
        else:
            pts, colors = self.prior_manager.load_sparse_point_cloud()

        self.model.initialize_from_points(pts, colors=colors, init_scale=0.2, init_opacity=0.5)

    def evaluate_views(self, test_views: List[CameraInfo], max_eval_views: Optional[int] = None) -> Tuple[float, float, float]:
        """Evaluates PSNR, SSIM, and rendering throughput on held-out test viewpoints."""
        psnrs = []
        ssims = []
        times = []

        eval_subset = test_views[:max_eval_views] if max_eval_views else test_views

        with torch.no_grad():
            for cam in eval_subset:
                t0 = time.time()
                out = self.rasterizer(self.model, cam)
                dt = time.time() - t0
                times.append(dt)

                mse = F.mse_loss(out.rgb, cam.image_tensor).item()
                psnr = 10.0 * math.log10(1.0 / max(mse, 1e-8))
                ssim_val = ssim_loss(out.rgb, cam.image_tensor).item()

                psnrs.append(psnr)
                ssims.append(ssim_val)

        mean_psnr = float(np.mean(psnrs))
        mean_ssim = float(np.mean(ssims))
        mean_fps = float(1.0 / max(1e-4, np.mean(times)))
        return mean_psnr, mean_ssim, mean_fps

    def create_optimizer(self) -> torch.optim.Optimizer:
        """Creates an Adam optimizer bound to the current model parameters."""
        return torch.optim.Adam([
            {"params": [self.model._means], "lr": self.lr_means, "name": "means"},
            {"params": [self.model._sh_coeffs], "lr": self.lr_sh, "name": "sh"},
            {"params": [self.model._logit_opacities], "lr": self.lr_opacity, "name": "opacity"},
            {"params": [self.model._log_scales], "lr": self.lr_scales, "name": "scales"},
            {"params": [self.model._rotations], "lr": self.lr_rots, "name": "rotations"}
        ])

    def train(
        self,
        iterations: int = 1500,
        eval_interval: int = 250,
        densify_interval: int = 100
    ) -> Dict[str, Any]:
        """Executes the training loop with learning rate scheduling and density control."""
        train_views, test_views = self.load_dataset(downscale=4)
        self.initialize_scene(use_priors=(self.lambda_prior > 0))

        # Initial benchmark before training
        init_psnr, init_ssim, _ = self.evaluate_views(test_views)

        optimizer = self.create_optimizer()

        history = []
        peak_train_vram = 0.0

        for step in range(1, iterations + 1):
            cam = train_views[step % len(train_views)]
            
            optimizer.zero_grad()
            out = self.rasterizer(self.model, cam)

            # Prior structural anchor loss
            anchor_loss = None
            if self.lambda_prior > 0:
                anchor_loss = self.prior_manager.compute_anchor_loss(self.model.get_means)

            loss, loss_dict = self.loss_fn(
                rendered_rgb=out.rgb,
                target_rgb=cam.image_tensor,
                scales=self.model.get_scales,
                anchor_loss=anchor_loss,
                lambda_prior=self.lambda_prior
            )

            loss.backward()

            # Record density tracking
            self.density_controller.record_step(self.model, out.radii, out.visibility_filter)

            optimizer.step()

            # Adaptive densification and pruning with optimizer re-binding
            if step > 0 and step % densify_interval == 0 and step < (iterations - 200):
                stats = self.density_controller.densify_and_prune(self.model, optimizer)
                # Re-bind optimizer to active model parameters and purge stale Adam state from VRAM
                del optimizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                optimizer = self.create_optimizer()

            # Track peak training VRAM
            if torch.cuda.is_available():
                vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                peak_train_vram = max(peak_train_vram, vram_mb)

            if step % 50 == 0 or step == 1:
                print(f"[*] Step {step}/{iterations} | Loss: {loss.item():.4f} | Gaussians: {self.model.num_gaussians} | VRAM: {peak_train_vram:.1f}MB", flush=True)

            # Periodic evaluation on held-out test views
            if step % eval_interval == 0:
                test_psnr, test_ssim, fps = self.evaluate_views(test_views, max_eval_views=10)
                print(f"[+] Eval @ step {step}: Test PSNR={test_psnr:.2f}dB, SSIM={test_ssim:.4f}, FPS={fps:.1f}", flush=True)
                metrics = SplatMetrics(
                    psnr_db=test_psnr,
                    ssim=test_ssim,
                    num_gaussians=self.model.num_gaussians,
                    training_vram_mb=peak_train_vram,
                    rendering_fps=fps,
                    iteration=step
                )
                history.append(metrics.to_dict())

        final_psnr, final_ssim, final_fps = self.evaluate_views(test_views)

        # Standing Sanity Check: Ensure optimization actively improved radiance field
        assert final_psnr >= init_psnr, f"Optimization failed to improve scene PSNR: init {init_psnr:.2f}dB -> final {final_psnr:.2f}dB"

        # Export trained model
        ply_path = os.path.join(self.output_dir, "06_splat_scene.ply")
        self.model.export_ply(ply_path)

        report = {
            "dataset": "videos/06.MP4",
            "num_train_views": len(train_views),
            "num_test_views": len(test_views),
            "iterations": iterations,
            "lambda_prior": self.lambda_prior,
            "initial_psnr_db": round(init_psnr, 2),
            "final_psnr_db": round(final_psnr, 2),
            "final_ssim": round(final_ssim, 4),
            "peak_train_vram_mb": round(peak_train_vram, 1),
            "rendering_fps": round(final_fps, 1),
            "num_gaussians": self.model.num_gaussians,
            "output_ply": ply_path,
            "history": history
        }

        report_path = os.path.join("reports", "06_splatting_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return report
