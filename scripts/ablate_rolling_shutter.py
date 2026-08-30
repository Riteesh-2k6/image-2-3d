import os
import sys
import json
import numpy as np
import cv2
import glob
import re

sys.path.insert(0, ".")

from src.sfm.vggt_engine import VGGTEngine
from src.sfm.telemetry_loader import TelemetryLoader
from src.sfm.confidence import PoseConfidenceCalculator
from src.geoprior.transforms import geodetic_to_ecef, ecef_to_enu

def extract_frame_idx(filename: str) -> int:
    m = re.search(r"_f(\d+)", filename)
    return int(m.group(1)) if m else 0

def solve_umeyama_alignment(src_points: np.ndarray, dst_points: np.ndarray):
    mu_src = np.mean(src_points, axis=0)
    mu_dst = np.mean(dst_points, axis=0)
    src_c = src_points - mu_src
    dst_c = dst_points - mu_dst
    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    var_src = np.sum(src_c ** 2) / len(src_points)
    scale = float(np.sum(S) / (len(src_points) * var_src))
    t = mu_dst - scale * (R @ mu_src)
    return R, t, scale

def evaluate_trajectory(c_est: np.ndarray, enu_gps: np.ndarray, timestamps: np.ndarray):
    # 5-fold CV
    n_frames = len(c_est)
    fold_size = n_frames // 5
    fold_errors = []
    for f in range(5):
        test_start = f * fold_size
        test_end = (f + 1) * fold_size if f < 4 else n_frames
        test_idx = np.arange(test_start, test_end)
        train_idx = np.setdiff1d(np.arange(n_frames), test_idx)
        
        R_f, t_f, s_f = solve_umeyama_alignment(c_est[train_idx], enu_gps[train_idx])
        c_aligned_test = (s_f * (c_est[test_idx] @ R_f.T)) + t_f
        errs = np.linalg.norm(c_aligned_test - enu_gps[test_idx], axis=1)
        fold_errors.append(np.median(errs))
    
    cv_mean = float(np.mean(fold_errors))
    cv_std = float(np.std(fold_errors))

    # Tail holdout (0-135 train, 135-end test)
    train_idx = np.arange(0, 135)
    test_idx = np.arange(135, n_frames)
    R_t, t_t, s_t = solve_umeyama_alignment(c_est[train_idx], enu_gps[train_idx])
    c_aligned_tail = (s_t * (c_est[test_idx] @ R_t.T)) + t_t
    tail_errs = np.linalg.norm(c_aligned_tail - enu_gps[test_idx], axis=1)
    tail_median = float(np.median(tail_errs))

    # 1-step and 5-step RPE (m/s)
    dt1 = np.maximum(0.01, timestamps[1:] - timestamps[:-1])
    rpe1 = np.linalg.norm((c_est[1:] - c_est[:-1]) - (enu_gps[1:] - enu_gps[:-1]), axis=1) / dt1
    rpe1_mean = float(np.mean(rpe1))

    # Whole-flight global alignment
    R_g, t_g, s_g = solve_umeyama_alignment(c_est, enu_gps)
    c_aligned_all = (s_g * (c_est @ R_g.T)) + t_g
    global_errs = np.linalg.norm(c_aligned_all - enu_gps, axis=1)
    global_median = float(np.median(global_errs))

    return {
        "cv_mean_m": cv_mean,
        "cv_std_m": cv_std,
        "tail_median_m": tail_median,
        "global_median_m": global_median,
        "rpe_1step_mps": rpe1_mean,
    }

def main():
    image_paths = sorted(glob.glob("output/06_keyframes/*.jpg"), key=extract_frame_idx)
    images = [cv2.resize(cv2.imread(p), (960, 540)) for p in image_paths]
    frame_numbers = [extract_frame_idx(p) for p in image_paths]
    timestamps = [fn / 30.0 for fn in frame_numbers]

    loader = TelemetryLoader("videos/06.csv")
    telem_records = [loader.get_interpolated_telemetry(ts) for ts in timestamps]

    lat0, lon0, alt0 = loader.records[0].latitude, loader.records[0].longitude, loader.records[0].altitude_m
    ref_ecef0 = geodetic_to_ecef(lat0, lon0, alt0)

    enu_gps = []
    for ts in timestamps:
        rec = loader.get_interpolated_telemetry(ts)
        pt_ecef = geodetic_to_ecef(rec.latitude, rec.longitude, rec.altitude_m)
        enu = ecef_to_enu(pt_ecef, lat0, lon0, ref_ecef0)
        enu_gps.append([enu.east, enu.north, enu.up])
    enu_gps = np.array(enu_gps)
    timestamps = np.array(timestamps)

    # 1. RUN WITH ROLLING SHUTTER CORRECTION ON
    print("\n--- Running with RS CORRECTION ON ---")
    vggt_on = VGGTEngine() # RS is enabled by default
    intrinsics_on, poses_on, tracks_on, _ = vggt_on.solve_poses_feedforward(images, timestamps, telemetry=telem_records)
    c_est_on = np.array([p.camera_center for p in poses_on])
    metrics_on = evaluate_trajectory(c_est_on, enu_gps, timestamps)

    calc = PoseConfidenceCalculator()
    rms_on, _ = calc.compute_reprojection_residuals(intrinsics_on, {p.frame_idx: p for p in poses_on}, tracks_on)

    # 2. RUN WITH ROLLING SHUTTER CORRECTION OFF (telemetry=None for feature compensation)
    print("\n--- Running with RS CORRECTION OFF ---")
    vggt_off = VGGTEngine()
    intrinsics_off, poses_off, tracks_off, _ = vggt_off.solve_poses_feedforward(images, timestamps, telemetry=None)
    c_est_off = np.array([p.camera_center for p in poses_off])
    metrics_off = evaluate_trajectory(c_est_off, enu_gps, timestamps)

    rms_off, _ = calc.compute_reprojection_residuals(intrinsics_off, {p.frame_idx: p for p in poses_off}, tracks_off)

    print("\n" + "="*80)
    print("ROLLING SHUTTER CORRECTION ABLATION SUMMARY (True Mild Flight Dynamics)")
    print("="*80)
    print(f"{'Metric':<35} | {'RS OFF':<20} | {'RS ON':<20} | {'Delta':<15}")
    print("-" * 80)
    print(f"{'Reprojection RMS (px)':<35} | {rms_off:<20.3f} | {rms_on:<20.3f} | {rms_on - rms_off:+.3f} px")
    print(f"{'5-Fold Block CV ATE (m)':<35} | {metrics_off['cv_mean_m']:<20.2f} | {metrics_on['cv_mean_m']:<20.2f} | {metrics_on['cv_mean_m'] - metrics_off['cv_mean_m']:+.2f} m")
    print(f"{'Tail-Holdout Median ATE (m)':<35} | {metrics_off['tail_median_m']:<20.2f} | {metrics_on['tail_median_m']:<20.2f} | {metrics_on['tail_median_m'] - metrics_off['tail_median_m']:+.2f} m")
    print(f"{'Whole-Flight Global Median (m)':<35} | {metrics_off['global_median_m']:<20.2f} | {metrics_on['global_median_m']:<20.2f} | {metrics_on['global_median_m'] - metrics_off['global_median_m']:+.2f} m")
    print(f"{'1-Step RPE Drift (m/s)':<35} | {metrics_off['rpe_1step_mps']:<20.2f} | {metrics_on['rpe_1step_mps']:<20.2f} | {metrics_on['rpe_1step_mps'] - metrics_off['rpe_1step_mps']:+.2f} m/s")
    print("="*80)

if __name__ == "__main__":
    main()
