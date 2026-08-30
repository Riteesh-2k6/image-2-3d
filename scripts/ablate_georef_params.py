"""
Systematic Georeferencing Parameter Ablation Suite
==================================================
1. RANSAC Inlier Threshold Sensitivity: [3.0m, 5.0m, 7.5m, 10.0m]
2. Joint-BA lambda_geo Multi-Objective Sweep: [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
   Evaluates trade-off between GPS ATE error and Optical Reprojection RMS.
"""

import os
import sys
import glob
import re
import json
import cv2
import numpy as np

sys.path.insert(0, ".")

from src.sfm.pipeline import VGGTPoseEstimator
from src.sfm.vggt_engine import VGGTEngine
from src.sfm.telemetry_loader import TelemetryLoader
from src.sfm.feature_tracker import FeatureTrack
from src.georef.pipeline import GeoreferencingEngine
from src.georef.ransac_georef import RANSACGeoreferencer
from src.georef.geodetic_ba_anchor import GeodeticBundleAdjuster
from src.geoprior.transforms import geodetic_to_ecef, ecef_to_enu
from src.sfm.confidence import PoseConfidenceCalculator


def extract_frame_idx(filename: str) -> int:
    m = re.search(r"_f(\d+)", filename)
    return int(m.group(1)) if m else 0


def main():
    print("=" * 85)
    print("      CHAPTER 7 GEOREFERENCING PARAMETER ABLATION & SENSITIVITY SUITE")
    print("=" * 85)

    # 1. Ingest clean keyframes & telemetry
    keyframe_paths = sorted(glob.glob("output/06_keyframes/*.jpg"), key=extract_frame_idx)
    num_frames = len(keyframe_paths)
    frame_numbers = [extract_frame_idx(p) for p in keyframe_paths]
    timestamps = np.array([fn / 30.0 for fn in frame_numbers])
    images = [cv2.resize(cv2.imread(p), (960, 540)) for p in keyframe_paths]

    loader = TelemetryLoader("videos/06.csv")
    lat0, lon0, alt0 = loader.records[0].latitude, loader.records[0].longitude, loader.records[0].altitude_m
    ref_ecef0 = geodetic_to_ecef(lat0, lon0, alt0)

    enu_gps = []
    for ts in timestamps:
        rec = loader.get_interpolated_telemetry(ts)
        pt_ecef = geodetic_to_ecef(rec.latitude, rec.longitude, rec.altitude_m)
        enu = ecef_to_enu(pt_ecef, lat0, lon0, ref_ecef0)
        enu_gps.append([enu.east, enu.north, enu.up])
    enu_gps = np.array(enu_gps)

    # Solve initial feedforward poses and tracks
    print("[*] Computing baseline visual poses and multi-view tracks...")
    vggt = VGGTEngine()
    telem_records = [loader.get_interpolated_telemetry(ts) for ts in timestamps]
    intrinsics, poses_raw, tracks_raw, _ = vggt.solve_poses_feedforward(
        images, timestamps.tolist(), telemetry=telem_records, enable_rolling_shutter=False
    )
    src_c = np.array([p.camera_center for p in poses_raw])

    # =========================================================================
    # PART 1: RANSAC INLIER THRESHOLD SENSITIVITY
    # =========================================================================
    print("\n" + "-" * 85)
    print("1. RANSAC INLIER DISTANCE THRESHOLD SENSITIVITY SWEEP")
    print("-" * 85)
    print(f"{'Threshold (d_thresh)':<22} | {'Inlier Count':<15} | {'Inlier Ratio':<15} | {'Median ATE':<15} | {'Scale Factor (s)':<15}")
    print("-" * 85)

    thresholds = [3.0, 5.0, 7.5, 10.0]
    ransac_results = {}
    for d_th in thresholds:
        ransac = RANSACGeoreferencer(inlier_threshold_m=d_th, max_iterations=1000, random_seed=42)
        transform, inlier_mask, residuals = ransac.fit(src_c, enu_gps, lat0, lon0, alt0)
        inliers_cnt = int(np.sum(inlier_mask))
        inliers_pct = 100.0 * inliers_cnt / num_frames
        med_ate = float(np.median(residuals[inlier_mask]))
        print(f"{f'{d_th:.1f} meters':<22} | {f'{inliers_cnt}/{num_frames}':<15} | {f'{inliers_pct:.1f}%':<15} | {f'{med_ate:.2f} m':<15} | {transform.scale:<15.3f}")
        ransac_results[d_th] = {
            "inlier_count": inliers_cnt,
            "inlier_ratio_pct": inliers_pct,
            "median_ate_m": med_ate,
            "scale_factor": float(transform.scale)
        }

    # =========================================================================
    # PART 2: JOINT-BA LAMBDA_GEO MULTI-OBJECTIVE ABLATION SWEEP
    # =========================================================================
    print("\n" + "-" * 85)
    print("2. JOINT-BA LAMBDA_GEO MULTI-OBJECTIVE ABLATION SWEEP (GPS ATE vs Optical RMS)")
    print("-" * 85)
    print(f"{'lambda_geo':<12} | {'Tail ATE (135-192)':<20} | {'5-Fold CV ATE':<18} | {'Reproj RMS (px)':<18} | {'Scale Ratio':<12}")
    print("-" * 85)

    lambda_values = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    sweep_results = []

    # Filter top 2500 tracks
    valid_tracks = [t for t in tracks_raw if t.point_3d is not None and t.track_length >= 2]
    top_tracks = sorted(valid_tracks, key=lambda t: len(t.observations), reverse=True)[:2500]

    # Pre-compute initial 8.0m RANSAC alignment (matching production baseline)
    best_ransac = RANSACGeoreferencer(inlier_threshold_m=8.0, max_iterations=1000, random_seed=42)
    init_transform, inlier_mask, _ = best_ransac.fit(src_c, enu_gps, lat0, lon0, alt0)

    # Build geodetic anchors
    engine = GeoreferencingEngine()
    anchors, _, _, _ = engine.build_geodetic_anchors(poses_raw, "videos/06.csv")

    calc = PoseConfidenceCalculator()

    train_idx = np.arange(0, 135)
    test_idx = np.arange(135, num_frames)

    for l_geo in lambda_values:
        # Clone initial transformed poses
        t_poses = [init_transform.transform_camera_pose(p) for p in poses_raw]
        t_tracks = []
        for t in top_tracks:
            t_tracks.append(FeatureTrack(
                track_id=t.track_id,
                observations=dict(t.observations),
                point_3d=init_transform.transform_points(t.point_3d.reshape(1, 3))[0]
            ))

        if l_geo > 1e-6:
            gba = GeodeticBundleAdjuster(lambda_geo=l_geo, max_nfev=30, max_tracks=2500)
            opt_poses, opt_tracks = gba.optimize(
                intrinsics=intrinsics,
                poses=t_poses,
                tracks=t_tracks,
                anchors=anchors,
                inlier_mask=inlier_mask
            )
        else:
            opt_poses = t_poses
            opt_tracks = t_tracks

        opt_centers = np.array([p.camera_center for p in opt_poses])

        # Tail-Holdout Median ATE on unseen 135-192
        tail_errors = np.linalg.norm(opt_centers[test_idx] - enu_gps[test_idx], axis=1)
        tail_med_ate = float(np.median(tail_errors))

        # 5-Fold Contiguous Block CV
        fold_size = num_frames // 5
        fold_errs = []
        for f in range(5):
            te_start = f * fold_size
            te_end = (f + 1) * fold_size if f < 4 else num_frames
            te_i = np.arange(te_start, te_end)
            fold_errs.append(np.median(np.linalg.norm(opt_centers[te_i] - enu_gps[te_i], axis=1)))
        cv_mean = float(np.mean(fold_errs))

        # Reprojection RMS
        rms_px, pt_errs = calc.compute_reprojection_residuals(
            intrinsics, {p.frame_idx: p for p in opt_poses}, opt_tracks
        )

        opt_path_len = float(np.sum(np.linalg.norm(np.diff(opt_centers, axis=0), axis=1)))
        gps_path_len = float(np.sum(np.linalg.norm(np.diff(enu_gps, axis=0), axis=1)))
        scale_ratio = opt_path_len / max(1e-3, gps_path_len)

        print(f"{l_geo:<12.3f} | {f'{tail_med_ate:.2f} m':<20} | {f'{cv_mean:.2f} m':<18} | {f'{rms_px:.3f} px (med {np.median(pt_errs):.2f})':<18} | {scale_ratio:<12.3f}")

        sweep_results.append({
            "lambda_geo": l_geo,
            "tail_holdout_median_ate_m": tail_med_ate,
            "cv_5fold_mean_ate_m": cv_mean,
            "reprojection_rms_px": float(rms_px),
            "reprojection_median_px": float(np.median(pt_errs)),
            "path_scale_ratio": float(scale_ratio)
        })

    print("=" * 85)

    # Save ablation report
    ablation_report = {
        "dataset": "videos/06.MP4 + videos/06.csv",
        "num_keyframes": num_frames,
        "ransac_threshold_sensitivity": ransac_results,
        "lambda_geo_ablation_sweep": sweep_results
    }
    with open("reports/06_georef_ablation_report.json", "w") as f:
        json.dump(ablation_report, f, indent=2)
    print("\n[+] Saved complete ablation report to: reports/06_georef_ablation_report.json")


if __name__ == "__main__":
    main()
