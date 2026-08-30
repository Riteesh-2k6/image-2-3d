"""
Stage 3 Ablation Suite: Prior Guidance & Anchor Weight Sweep
============================================================
Ablates the impact of GeoPrior structural anchoring on reconstruction quality (PSNR, SSIM, Floaters).
"""

import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, ".")

from src.splatting.trainer import SpeedySplatTrainer


def main():
    print("=" * 85, flush=True)
    print("      STAGE 3 PRIOR-GUIDED GAUSSIAN SPLATTING ABLATION SUITE", flush=True)
    print("=" * 85, flush=True)

    lambda_values = [0.0, 0.001, 0.005, 0.01, 0.05]
    ablation_results = []

    print(f"{'lambda_prior':<15} | {'Initial PSNR':<15} | {'Final PSNR':<15} | {'Final SSIM':<15} | {'Gaussians':<12} | {'VRAM (MB)':<12}", flush=True)
    print("-" * 85, flush=True)

    for l_prior in lambda_values:
        print(f"[*] Training ablation configuration: lambda_prior = {l_prior}...", flush=True)
        trainer = SpeedySplatTrainer(
            keyframes_dir="output/06_keyframes",
            georef_poses_json="output/06_georef/06_georef_poses.json",
            georef_cloud_ply="output/06_georef/06_georef_cloud.ply",
            output_dir=f"output/06_splat_ablation_lp_{l_prior}",
            lambda_prior=l_prior
        )

        report = trainer.train(iterations=250, eval_interval=100, densify_interval=100)

        init_psnr = report["initial_psnr_db"]
        final_psnr = report["final_psnr_db"]
        final_ssim = report["final_ssim"]
        num_g = report["num_gaussians"]
        vram = report["peak_train_vram_mb"]

        print(f"{l_prior:<15.3f} | {f'{init_psnr:.2f} dB':<15} | {f'{final_psnr:.2f} dB':<15} | {f'{final_ssim:.4f}':<15} | {num_g:<12} | {f'{vram:.1f} MB':<12}", flush=True)

        ablation_results.append({
            "lambda_prior": l_prior,
            "initial_psnr_db": init_psnr,
            "final_psnr_db": final_psnr,
            "final_ssim": final_ssim,
            "num_gaussians": num_g,
            "peak_vram_mb": vram
        })

    # Save ablation report
    out_path = "reports/06_splat_prior_ablation_report.json"
    with open(out_path, "w") as f:
        json.dump({"lambda_prior_sweep": ablation_results}, f, indent=2)
    print("=" * 85)
    print(f"\n[+] Saved complete Stage 3 ablation report to: {out_path}")


if __name__ == "__main__":
    main()
