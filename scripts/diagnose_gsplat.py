import sys
import os
sys.path.insert(0, ".")
sys.path.insert(0, "/workspace")
import traceback
import torch
import numpy as np
import cv2

from src.splatting.gaussian_model import GaussianModel
from src.splatting.types import CameraInfo
from src.splatting.trainer import SpeedySplatTrainer
from src.splatting.rasterizer import SpeedySplatRasterizer

print("Python version:", sys.version)
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainer = SpeedySplatTrainer(
    keyframes_dir="output/06_keyframes",
    georef_poses_json="output/06_georef/06_georef_poses.json",
    georef_cloud_ply="output/06_georef/06_georef_cloud.ply",
    output_dir="output/test_diagnose",
    lambda_prior=0.001
)

train_views, test_views = trainer.load_dataset(downscale=4)
print(f"Loaded {len(train_views)} train views and {len(test_views)} test views.")
cam = train_views[0]

print(f"Camera width={cam.width}, height={cam.height}, fx={cam.fx}, fy={cam.fy}, cx={cam.cx}, cy={cam.cy}")
print(f"Model Gaussians: {trainer.model.num_gaussians}")

try:
    print("Testing direct gsplat rasterization...")
    from gsplat.rendering import rasterization
    
    means = trainer.model.get_means
    quats = trainer.model.get_quats
    scales = trainer.model.get_scales
    opacities = trainer.model.get_opacities.squeeze(-1)
    colors = trainer.model.get_colors
    
    w2c = cam.world_view_transform
    K = torch.tensor([
        [cam.fx, 0.0, cam.cx],
        [0.0, cam.fy, cam.cy],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32, device=device)
    
    print("Means shape:", means.shape, "dtype:", means.dtype)
    print("Quats shape:", quats.shape, "dtype:", quats.dtype)
    print("Scales shape:", scales.shape, "dtype:", scales.dtype)
    print("Opacities shape:", opacities.shape, "dtype:", opacities.dtype)
    print("Colors shape:", colors.shape, "dtype:", colors.dtype)
    print("Viewmat shape:", w2c.shape, "dtype:", w2c.dtype)
    print("K shape:", K.shape, "dtype:", K.dtype)
    
    renders, alphas, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=w2c.unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=cam.width,
        height=cam.height
    )
    print("Render successful! Shape:", renders.shape)
    print("Alphas shape:", alphas.shape)
    print("Render RGB min/max/mean:", renders.min().item(), renders.max().item(), renders.mean().item())
    
    # Save test render vs GT image
    rendered_np = (renders[0].detach().cpu().numpy() * 255).astype(np.uint8)
    gt_np = (cam.image_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    
    os.makedirs("output/diagnostics", exist_ok=True)
    cv2.imwrite("output/diagnostics/render_f0.png", cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite("output/diagnostics/gt_f0.png", cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR))
    print("Saved diagnostic images to output/diagnostics/")
    
except Exception as e:
    print("GSPLAT EXECUTION FAILED:")
    traceback.print_exc()
