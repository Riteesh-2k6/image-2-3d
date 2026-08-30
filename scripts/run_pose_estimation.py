"""
Real Flight Evaluation: VGGT Pose Estimation & 3D Pointmap Triangulation
========================================================================
Runs the end-to-end VGGT pose estimation pipeline on real DJI Mini 3 Pro 4K keyframes,
synchronizes with 50Hz telemetry, evaluates Chapter 6 confidence metrics, and exports
metric camera poses + sparse 3D point cloud PLY.
"""

import os
import sys
import glob
import re
import argparse
import time
import cv2
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sfm import VGGTPoseEstimator


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Pose Estimation on DJI Keyframes")
    parser.add_argument("--keyframes-dir", type=str, default="output/06_keyframes", help="Directory containing extracted keyframe images")
    parser.add_argument("--telemetry", type=str, default="videos/06.csv", help="Path to DJI flight telemetry CSV")
    parser.add_argument("--output-dir", type=str, default="output/06_sfm", help="Directory to save poses.json and sparse_cloud.ply")
    parser.add_argument("--report-dir", type=str, default="reports", help="Directory to save benchmark report JSON")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional limit on number of keyframes to process")
    parser.add_argument("--downscale", type=int, default=2, help="Downscale factor for feature matching (e.g. 2 for 1080p, 1 for raw 4K)")
    parser.add_argument("--rolling-shutter", action=argparse.BooleanOptionalAction, default=False, help="Enable CMOS scanline rolling-shutter compensation (default: False)")
    return parser.parse_args()


def extract_frame_idx(filename: str) -> int:
    """Extract frame index from filename like 'keyframe_0005_f00090.jpg'."""
    m = re.search(r"_f(\d+)", filename)
    if m:
        return int(m.group(1))
    m2 = re.search(r"keyframe_(\d+)", filename)
    if m2:
        return int(m2.group(1))
    return 0


def main():
    console = Console()
    args = parse_args()

    console.print(Panel.fit(
        "[bold cyan]GeoPrior Stage 2: VGGT Pose Estimation & 3D Triangulation[/bold cyan]\n"
        "[dim]Visual Geometry Grounded Transformer + Chapter 6 Confidence Verification[/dim]",
        border_style="cyan"
    ))

    # 1. Discover keyframe images
    if not os.path.exists(args.keyframes_dir):
        console.print(f"[red]Error: Keyframes directory '{args.keyframes_dir}' not found. Run SSAKS first![/red]")
        sys.exit(1)

    image_paths = sorted(
        glob.glob(os.path.join(args.keyframes_dir, "*.jpg")) + glob.glob(os.path.join(args.keyframes_dir, "*.png")),
        key=extract_frame_idx
    )

    if not image_paths:
        console.print(f"[red]No keyframes found in '{args.keyframes_dir}'[/red]")
        sys.exit(1)

    if args.max_frames:
        image_paths = image_paths[: args.max_frames]

    num_keyframes = len(image_paths)
    console.print(f"[+] Loaded [bold green]{num_keyframes}[/bold green] SSAKS keyframes from [cyan]{args.keyframes_dir}[/cyan]")
    console.print(f"[+] Telemetry log: [cyan]{args.telemetry}[/cyan] (Exists: {os.path.exists(args.telemetry)})")

    # 2. Pre-load images & calculate timestamps (assuming 30 FPS video source)
    console.print("[-] Ingesting keyframes and computing timestamps...")
    loaded_images = []
    timestamps = []
    image_names = []

    for path in image_paths:
        frame_idx = extract_frame_idx(path)
        ts = float(frame_idx) / 30.0 # 30 FPS
        timestamps.append(ts)
        image_names.append(os.path.basename(path))

        img = cv2.imread(path)
        if img is None:
            continue
        if args.downscale > 1:
            h, w = img.shape[:2]
            img = cv2.resize(img, (w // args.downscale, h // args.downscale), interpolation=cv2.INTER_AREA)
        loaded_images.append(img)

    console.print(f"[*] Processed image resolution: [bold]{loaded_images[0].shape[1]} x {loaded_images[0].shape[0]}[/bold] px")

    # 3. Instantiate VGGT Estimator & Run
    console.print("\n[bold yellow]>>> Running VGGT Feed-Forward Pose Solve & Multi-View Triangulation...[/bold yellow]")
    estimator = VGGTPoseEstimator(
        min_confidence_thresh=0.70,
        max_reproj_err_px=2.0
    )

    t0 = time.time()
    result = estimator.estimate_poses(
        keyframes=loaded_images,
        timestamps=timestamps,
        image_names=image_names,
        telemetry_csv=args.telemetry,
        enable_rolling_shutter=args.rolling_shutter
    )
    total_time = time.time() - t0

    # 4. Export artifacts
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)
    export_paths = VGGTPoseEstimator.export_results(result, args.output_dir, prefix="06")

    # Also copy report to reports dir
    main_report_path = os.path.join(args.report_dir, "06_pose_estimation_report.json")
    with open(export_paths["report_json"], "r") as f_src, open(main_report_path, "w") as f_dst:
        f_dst.write(f_src.read())

    # 5. Display Quantitative Summary Table
    table = Table(title="VGGT Camera Pose Estimation & Verification Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric / Property", style="cyan", no_wrap=True)
    table.add_column("Measured Value", style="green")
    table.add_column("Target / Tolerance", style="dim")

    m = result.metrics
    table.add_row("Input Keyframes", str(m.total_frames), "193 frames")
    table.add_row("Registered Camera Poses", f"{m.num_registered_frames} ({100.0 * m.num_registered_frames / max(1, m.total_frames):.1f}%)", ">= 95.0%")
    table.add_row("Triangulated 3D Points", f"{result.sparse_cloud.num_points:,} points", ">= 1,000 points")
    table.add_row("Mean Reprojection Error", f"{m.reprojection_residual_px:.3f} px", "<= 1.50 px")
    table.add_row("Pose Confidence Score", f"{m.pose_confidence_score:.3f}", ">= 0.70")
    table.add_row("Feature-Track Consistency", f"{m.feature_track_consistency:.3f}", ">= 0.50")
    table.add_row("Camera Graph Connectivity", f"{m.camera_graph_connectivity:.3f}", ">= 0.50")
    table.add_row("Mean Epipolar Inlier Ratio", f"{m.inlier_match_ratio * 100:.1f}%", ">= 60.0%")
    table.add_row("Execution Engine", result.engine_used.value, "vggt_feedforward")
    table.add_row("Total Runtime", f"{total_time:.2f} s ({len(loaded_images) / max(0.001, total_time):.1f} FPS)", "< 30.0 s")

    console.print("\n", table, "\n")

    console.print(Panel(
        f"[bold green]Pose Estimation & Sparse Pointmap Triangulation Complete![/bold green]\n\n"
        f"* Camera Poses JSON: [cyan]{export_paths['poses_json']}[/cyan]\n"
        f"* Sparse Point Cloud PLY: [cyan]{export_paths['sparse_cloud_ply']}[/cyan]\n"
        f"* Telemetry Report JSON: [cyan]{main_report_path}[/cyan]\n\n"
        f"[dim]Ready for Ticket [09] (Georeferencing Module: Umeyama 7-DoF + RANSAC)[/dim]",
        title="Artifacts Emitted",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
