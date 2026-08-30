"""
Production Training Runner for Speedy-Splat (Stage 3).
======================================================
Trains a 3D Gaussian Splatting scene with GeoPrior structural anchoring,
benchmarks held-out test viewpoints, and exports 3DGS PLY and JSON reports.
"""

import os
import sys
import argparse
import json
import torch

sys.path.insert(0, ".")

from src.splatting.trainer import SpeedySplatTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="GeoPrior Speedy-Splat 3D Gaussian Reconstruction")
    parser.add_argument("--keyframes-dir", type=str, default="output/06_keyframes", help="Directory containing SSAKS keyframes")
    parser.add_argument("--georef-poses", type=str, default="output/06_georef/06_georef_poses.json", help="Path to georeferenced poses JSON")
    parser.add_argument("--georef-cloud", type=str, default="output/06_georef/06_georef_cloud.ply", help="Path to georeferenced sparse point cloud PLY")
    parser.add_argument("--output-dir", type=str, default="output/06_splat", help="Output directory for trained Gaussian scene")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--lambda-prior", type=float, default=0.05, help="Weight for GeoPrior structural anchor loss")
    parser.add_argument("--eval-interval", type=int, default=200, help="Interval for evaluating held-out test views")
    parser.add_argument("--densify-interval", type=int, default=100, help="Interval for adaptive Gaussian densification")
    return parser.parse_args()


def main():
    args = parse_args()

    print("+" + "=" * 76 + "+")
    print("| GeoPrior Stage 3: Prior-Guided 3D Gaussian Splatting (Speedy-Splat)        |")
    print("| Chapters 8-11: Radiance Fields + GeoPrior Structural Anchor Regularization |")
    print("+" + "=" * 76 + "+")

    trainer = SpeedySplatTrainer(
        keyframes_dir=args.keyframes_dir,
        georef_poses_json=args.georef_poses,
        georef_cloud_ply=args.georef_cloud,
        output_dir=args.output_dir,
        lambda_prior=args.lambda_prior
    )

    print(f"[*] Initializing training for {args.iterations} iterations (lambda_prior={args.lambda_prior})...")
    report = trainer.train(
        iterations=args.iterations,
        eval_interval=args.eval_interval,
        densify_interval=args.densify_interval
    )

    print("\n" + "=" * 90)
    print("       STAGE 3 3D RECONSTRUCTION VERIFICATION vs ACCEPTANCE TARGETS")
    print("=" * 90)
    psnr_status = "Pass" if report['final_psnr_db'] >= 24.0 else f"❌ Fail (Gap: {report['final_psnr_db'] - 24.0:.2f} dB)"
    ssim_status = "Pass" if report['final_ssim'] >= 0.800 else f"❌ Fail (Gap: {report['final_ssim'] - 0.800:.4f})"
    vram_status = "✅ Pass" if report['peak_train_vram_mb'] <= 4500.0 else "❌ Fail"
    fps_status = "✅ Pass" if report['rendering_fps'] >= 30.0 else f"❌ Fail ({report['rendering_fps']:.1f} FPS)"
    n_gauss = str(report['num_gaussians'])

    print(f"{'Held-Out Test View PSNR':<35} | {'>= 24.0 dB':<22} | {report['final_psnr_db']:.2f} dB ({psnr_status})")
    print(f"{'Held-Out Test View SSIM':<35} | {'>= 0.800':<22} | {report['final_ssim']:.4f} ({ssim_status})")
    print(f"{'Peak Training VRAM Footprint':<35} | {'<= 4500 MB (6GB GPU)':<22} | {report['peak_train_vram_mb']:.1f} MB ({vram_status})")
    print(f"{'Real-Time Rendering Throughput':<35} | {'>= 30.0 FPS':<22} | {report['rendering_fps']:.1f} FPS ({fps_status})")
    print(f"{'Total 3D Gaussian Primitives':<35} | {'N/A':<22} | {n_gauss}")
    print("=" * 90)

    print("\n+" + "=" * 77 + "+")
    print(f"| * Trained Gaussian Scene PLY: {report['output_ply']:<44} |")
    print(f"| * Splatting Benchmark Report: reports/06_splatting_report.json              |")
    print("+" + "=" * 77 + "+")


if __name__ == "__main__":
    main()
