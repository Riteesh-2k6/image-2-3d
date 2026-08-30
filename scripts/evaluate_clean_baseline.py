"""
Unified Stage 2 Verification & Diagnostic Suite
===============================================
1. Reconciles Raw Feedforward VO vs. Bundle Adjusted VO Tail-ATE.
2. Evaluates the true relative heading error (Camera-to-Velocity Angle VO vs GPS).
3. Evaluates 5-Fold Contiguous Block CV, Tail-Holdout, RPE, and Reprojection percentiles.
4. Exports canonical baseline report: reports/06_clean_baseline_report.json.
"""

import os
import sys
import json
import glob
import re
import cv2
import numpy as np

sys.path.insert(0, ".")

from src.sfm.vggt_engine import VGGTEngine
from src.sfm.pipeline import VGGTPoseEstimator
from src.sfm.telemetry_loader import TelemetryLoader
from src.sfm.confidence import PoseConfidenceCalculator
from src.geoprior.transforms import geodetic_to_ecef, ecef_to_enu


def extract_frame_idx(filename: str) -> int:
    m = re.search(r"_f(\d+)", filename)
    return int(m.group(1)) if m else 0


def solve_umeyama(src: np.ndarray, dst: np.ndarray):
    """Solves 7-DoF Umeyama similarity alignment: dst = s * (src @ R.T) + t"""
    mu_src = np.mean(src, axis=0)
    mu_dst = np.mean(dst, axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    var_src = np.sum(src_c ** 2) / len(src)
    scale = float(np.sum(S) / (len(src) * var_src))
    t = mu_dst - scale * (R @ mu_src)
    return R, t, scale


def main():
    # 1. Load clean keyframes
    keyframe_paths = sorted(glob.glob("output/06_keyframes/*.jpg"), key=extract_frame_idx)
    num_frames = len(keyframe_paths)
    print(f"[*] Ingesting {num_frames} pristine keyframes...")

    frame_numbers = [extract_frame_idx(p) for p in keyframe_paths]
    timestamps = np.array([fn / 30.0 for fn in frame_numbers])
    images = [cv2.resize(cv2.imread(p), (960, 540)) for p in keyframe_paths]

    # 2. Ingest 50Hz ground truth GPS telemetry
    loader = TelemetryLoader("videos/06.csv")
    telem_records = [loader.get_interpolated_telemetry(ts) for ts in timestamps]

    lat0, lon0, alt0 = loader.records[0].latitude, loader.records[0].longitude, loader.records[0].altitude_m
    ref_ecef0 = geodetic_to_ecef(lat0, lon0, alt0)

    enu_gps = []
    gimbal_yaws = []
    for ts in timestamps:
        rec = loader.get_interpolated_telemetry(ts)
        pt_ecef = geodetic_to_ecef(rec.latitude, rec.longitude, rec.altitude_m)
        enu = ecef_to_enu(pt_ecef, lat0, lon0, ref_ecef0)
        enu_gps.append([enu.east, enu.north, enu.up])
        gimbal_yaws.append(rec.gimbal_yaw_deg)
    enu_gps = np.array(enu_gps)
    gimbal_yaws = np.array(gimbal_yaws)

    # =========================================================================
    # PART 1: COMPUTE BOTH POSE SETS UNDER IDENTICAL CONDITIONS (RS OFF)
    # =========================================================================
    print("[*] Generating Pose Set A: Raw Feedforward VO (Telemetry Step Integrated)...")
    vggt = VGGTEngine()
    intrinsics_ff, poses_ff, tracks_ff, _ = vggt.solve_poses_feedforward(
        images, timestamps.tolist(), telemetry=telem_records, enable_rolling_shutter=False
    )
    c_ff = np.array([p.camera_center for p in poses_ff])

    print("[*] Generating Pose Set B: Levenberg-Marquardt Bundle Adjusted VO...")
    estimator = VGGTPoseEstimator(min_confidence_thresh=0.70, max_reproj_err_px=2.0)
    ba_result = estimator.estimate_poses(
        keyframes=images, timestamps=timestamps.tolist(), telemetry_csv="videos/06.csv", enable_rolling_shutter=False
    )
    c_ba = np.array([p.camera_center for p in ba_result.poses])
    tracks_ba = ba_result.sparse_cloud.points_3d

    # Compute trajectory path lengths
    gps_path_len = float(np.sum(np.linalg.norm(np.diff(enu_gps, axis=0), axis=1)))
    ff_path_len = float(np.sum(np.linalg.norm(np.diff(c_ff, axis=0), axis=1)))
    ba_path_len = float(np.sum(np.linalg.norm(np.diff(c_ba, axis=0), axis=1)))

    # =========================================================================
    # PART 2: TAIL-ATE RECONCILIATION
    # =========================================================================
    train_idx = np.arange(0, 135)
    test_idx = np.arange(135, num_frames)

    # Pose Set A (Feedforward) Tail Fit
    R_ff_t, t_ff_t, s_ff_t = solve_umeyama(c_ff[train_idx], enu_gps[train_idx])
    c_ff_tail_aligned = s_ff_t * (c_ff[test_idx] @ R_ff_t.T) + t_ff_t
    errs_ff_tail = np.linalg.norm(c_ff_tail_aligned - enu_gps[test_idx], axis=1)

    # Pose Set B (Bundle Adjusted) Tail Fit
    R_ba_t, t_ba_t, s_ba_t = solve_umeyama(c_ba[train_idx], enu_gps[train_idx])
    c_ba_tail_aligned = s_ba_t * (c_ba[test_idx] @ R_ba_t.T) + t_ba_t
    errs_ba_tail = np.linalg.norm(c_ba_tail_aligned - enu_gps[test_idx], axis=1)

    # Global Fit on all frames
    R_ff_g, t_ff_g, s_ff_g = solve_umeyama(c_ff, enu_gps)
    errs_ff_global = np.linalg.norm((s_ff_g * (c_ff @ R_ff_g.T) + t_ff_g) - enu_gps, axis=1)

    R_ba_g, t_ba_g, s_ba_g = solve_umeyama(c_ba, enu_gps)
    errs_ba_global = np.linalg.norm((s_ba_g * (c_ba @ R_ba_g.T) + t_ba_g) - enu_gps, axis=1)

    # 5-Fold Contiguous Block CV for both
    def run_5fold_cv(c_poses):
        fold_size = num_frames // 5
        fold_errs = []
        for f in range(5):
            te_start = f * fold_size
            te_end = (f + 1) * fold_size if f < 4 else num_frames
            te_idx = np.arange(te_start, te_end)
            tr_idx = np.setdiff1d(np.arange(num_frames), te_idx)
            R_f, t_f, s_f = solve_umeyama(c_poses[tr_idx], enu_gps[tr_idx])
            aligned_te = s_f * (c_poses[te_idx] @ R_f.T) + t_f
            fold_errs.append(np.median(np.linalg.norm(aligned_te - enu_gps[te_idx], axis=1)))
        return float(np.mean(fold_errs)), float(np.std(fold_errs))

    cv_ff_mean, cv_ff_std = run_5fold_cv(c_ff)
    cv_ba_mean, cv_ba_std = run_5fold_cv(c_ba)

    # =========================================================================
    # PART 3: TRUE CAMERA HEADING DRIFT (ENU ALIGNED VS GIMBAL YAW)
    # =========================================================================
    vo_headings = []
    for pose in ba_result.poses:
        fwd_vo = pose.R.T @ np.array([0, 0, 1])
        fwd_enu = R_ba_g @ fwd_vo
        bearing = np.degrees(np.arctan2(fwd_enu[0], fwd_enu[1]))
        vo_headings.append(bearing)
    vo_headings = np.array(vo_headings)

    heading_errors = (vo_headings - gimbal_yaws + 180.0) % 360.0 - 180.0

    # Fit linear drift slope (deg/sec)
    drift_slope_deg_per_sec, _ = np.polyfit(timestamps, np.abs(heading_errors), 1)

    # Reprojection percentile distribution
    calc = PoseConfidenceCalculator()
    rms_px, pt_errs = calc.compute_reprojection_residuals(
        intrinsics_ff, {p.frame_idx: p for p in ba_result.poses}, tracks_ff
    )

    # =========================================================================
    # PART 4: PRINT STRUCTURED RECONCILIATION REPORT
    # =========================================================================
    print("\n" + "="*85)
    print("       STAGE 2 CANONICAL VERIFICATION & RECONCILIATION REPORT (RS OFF)")
    print("="*85)
    print("1. TAIL-ATE & TRAJECTORY METRIC RECONCILIATION")
    print("-" * 85)
    print(f"{'Property / Metric':<35} | {'Feedforward VO (A)':<22} | {'Bundle Adjusted VO (B)':<22}")
    print("-" * 85)
    print(f"{'Path Length (GPS GT: ' + f'{gps_path_len:.1f}m)':<35} | {ff_path_len:<22.1f} m | {ba_path_len:<22.1f} m")
    print(f"{'5-Fold Contiguous Block CV':<35} | {f'{cv_ff_mean:.2f} ± {cv_ff_std:.2f} m':<22} | {f'{cv_ba_mean:.2f} ± {cv_ba_std:.2f} m':<22}")
    print(f"{'Tail-Holdout Median ATE (135-192)':<35} | {np.median(errs_ff_tail):<22.2f} m | {np.median(errs_ba_tail):<22.2f} m")
    print(f"{'Tail-Holdout Mean ATE (135-192)':<35} | {np.mean(errs_ff_tail):<22.2f} m | {np.mean(errs_ba_tail):<22.2f} m")
    print(f"{'Tail-Holdout Max ATE':<35} | {np.max(errs_ff_tail):<22.2f} m | {np.max(errs_ba_tail):<22.2f} m")
    print(f"{'Whole-Flight Global Median ATE':<35} | {np.median(errs_ff_global):<22.2f} m | {np.median(errs_ba_global):<22.2f} m")
    print(f"{'Umeyama Train Scale Factor (s_fit)':<35} | {s_ff_t:<22.3f}   | {s_ba_t:<22.3f}")
    print("="*85)

    print("\n2. CAMERA HEADING ERROR & DRIFT (ENU-ALIGNED OPTICAL AXIS VS GIMBAL YAW)")
    print("-" * 85)
    print(f"Heading Error (|Aligned VO Heading - Gimbal Yaw|):")
    print(f"  * Train Segment (Frames 0-134):  Mean = {np.mean(np.abs(heading_errors[:135])):.2f}°, Median = {np.median(np.abs(heading_errors[:135])):.2f}°")
    print(f"  * Tail Segment (Frames 135-192): Mean = {np.mean(np.abs(heading_errors[135:])):.2f}°, Median = {np.median(np.abs(heading_errors[135:])):.2f}°")
    print(f"  * Full Flight:                   Mean = {np.mean(np.abs(heading_errors)):.2f}°, Median = {np.median(np.abs(heading_errors)):.2f}°")
    print(f"  * Measured Drift Rate:           {drift_slope_deg_per_sec:.3f} deg/sec over flight")
    print("-" * 85)
    pose_diffs = np.linalg.norm(c_ff - c_ba, axis=1)
    print(f"Pose-Level Difference (Feedforward vs Bundle Adjusted Centers):")
    print(f"  * Mean Positional Shift: {np.mean(pose_diffs):.4f} m, Max Shift: {np.max(pose_diffs):.4f} m")
    print("="*85)

    print("\n3. GEOMETRIC REPROJECTION ACCURACY (RS OFF)")
    print("-" * 85)
    print(f"Triangulated 3D Points: {len(tracks_ff):,d} inliers")
    print(f"Reprojection Residual:  Mean = {np.mean(pt_errs):.3f} px, Median = {np.median(pt_errs):.3f} px, RMS = {rms_px:.3f} px")
    print(f"Percentile Distribution: P90 = {np.percentile(pt_errs, 90):.3f} px, P95 = {np.percentile(pt_errs, 95):.3f} px, P99 = {np.percentile(pt_errs, 99):.3f} px")
    print("="*85)

    # Save to canonical report JSON
    canonical_report = {
        "report_version": "v4_clean_canonical",
        "dataset": "videos/06.MP4 + videos/06.csv",
        "num_keyframes": num_frames,
        "rolling_shutter_correction": "OFF (Default)",
        "feedforward_vo_metrics": {
            "path_length_m": ff_path_len,
            "cv_5fold_mean_ate_m": cv_ff_mean,
            "cv_5fold_std_ate_m": cv_ff_std,
            "tail_holdout_median_ate_m": float(np.median(errs_ff_tail)),
            "tail_holdout_mean_ate_m": float(np.mean(errs_ff_tail)),
            "global_median_ate_m": float(np.median(errs_ff_global))
        },
        "bundle_adjusted_vo_metrics": {
            "path_length_m": ba_path_len,
            "cv_5fold_mean_ate_m": cv_ba_mean,
            "cv_5fold_std_ate_m": cv_ba_std,
            "tail_holdout_median_ate_m": float(np.median(errs_ba_tail)),
            "tail_holdout_mean_ate_m": float(np.mean(errs_ba_tail)),
            "global_median_ate_m": float(np.median(errs_ba_global))
        },
        "heading_geometric_drift": {
            "train_mean_error_deg": float(np.mean(np.abs(heading_errors[:135]))),
            "tail_mean_error_deg": float(np.mean(np.abs(heading_errors[135:]))),
            "full_flight_mean_error_deg": float(np.mean(np.abs(heading_errors))),
            "drift_rate_deg_per_sec": float(drift_slope_deg_per_sec)
        },
        "reprojection_percentiles_px": {
            "mean": float(np.mean(pt_errs)),
            "median": float(np.median(pt_errs)),
            "rms": float(rms_px),
            "p90": float(np.percentile(pt_errs, 90)),
            "p95": float(np.percentile(pt_errs, 95)),
            "p99": float(np.percentile(pt_errs, 99))
        }
    }

    report_path = "reports/06_clean_baseline_report.json"
    with open(report_path, "w") as f:
        json.dump(canonical_report, f, indent=2)
    print(f"\n[+] Saved canonical report to: {report_path}")

if __name__ == "__main__":
    main()
