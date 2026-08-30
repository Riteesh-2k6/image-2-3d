import json
import sys
import os
sys.path.insert(0, ".")
import numpy as np
from src.sfm.telemetry_loader import TelemetryLoader
from src.geoprior.transforms import geodetic_to_ecef, ecef_to_enu
from src.sfm.pipeline import VGGTPoseEstimator
import cv2
import glob
import os
import re

def extract_frame_idx(filename: str) -> int:
    m = re.search(r"_f(\d+)", filename)
    return int(m.group(1)) if m else 0

# 1. Load clean keyframes
image_paths = sorted(glob.glob("output/06_keyframes/*.jpg"), key=extract_frame_idx)
print(f"Total keyframe images: {len(image_paths)}")

frame_numbers = [extract_frame_idx(p) for p in image_paths]
timestamps = [fn / 30.0 for fn in frame_numbers]

loader = TelemetryLoader("videos/06.csv")
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

# 2. Measure actual heading divergence and lever arm on existing poses
poses_data = json.load(open("output/06_sfm/06_poses.json"))["poses"]
c_est = np.array([p["camera_center"] for p in poses_data])
rot_matrices = [np.array(p["rotation_matrix"]) for p in poses_data]

# Visual heading from R (camera optical axis projected on ENU ground plane)
vis_headings_deg = []
for R in rot_matrices:
    # Camera +Z forward vector in world: v = R.T @ [0, 0, 1]
    fwd = R.T @ np.array([0, 0, 1])
    heading = np.degrees(np.arctan2(fwd[0], fwd[1])) # East=X, North=Y
    vis_headings_deg.append(heading)
vis_headings_deg = np.array(vis_headings_deg)

# Umeyama fit on 0-135 (train)
train_idx = np.arange(0, 135)
test_idx = np.arange(135, len(c_est))

mu_src = c_est[train_idx].mean(axis=0)
mu_dst = enu_gps[train_idx].mean(axis=0)
src_c = c_est[train_idx] - mu_src
dst_c = enu_gps[train_idx] - mu_dst

H = src_c.T @ dst_c
U, S, Vt = np.linalg.svd(H)
R_fit = Vt.T @ U.T
if np.linalg.det(R_fit) < 0:
    Vt[-1, :] *= -1
    R_fit = Vt.T @ U.T
scale_fit = float(np.sum(S) / (len(train_idx) * (np.sum(src_c**2)/len(train_idx))))
t_fit = mu_dst - scale_fit * (R_fit @ mu_src)

c_aligned = scale_fit * (c_est @ R_fit.T) + t_fit

# Lever arm distance from centroid of train set (mu_dst) to tail test points
lever_arms = np.linalg.norm(enu_gps[test_idx] - mu_dst, axis=1)

# Measured spatial errors on tail
tail_errors = np.linalg.norm(c_aligned[test_idx] - enu_gps[test_idx], axis=1)

# Actual measured heading divergence between aligned visual heading and GPS trajectory tangent
gps_tangents = np.diff(enu_gps, axis=0)
gps_headings = np.degrees(np.arctan2(gps_tangents[:, 0], gps_tangents[:, 1]))
gps_headings = np.append(gps_headings, gps_headings[-1])

aligned_vis_fwd = [(R_fit @ R.T @ np.array([0, 0, 1])) for R in rot_matrices]
aligned_vis_headings = np.array([np.degrees(np.arctan2(f[0], f[1])) for f in aligned_vis_fwd])

heading_diffs = (aligned_vis_headings - gps_headings + 180.0) % 360.0 - 180.0

print("==================================================================")
print("EMPIRICAL LEVER-ARM & HEADING DIVERGENCE MEASUREMENTS")
print("==================================================================")
print(f"Centroid of Train Set (0-135): East={mu_dst[0]:.2f}m, North={mu_dst[1]:.2f}m, Up={mu_dst[2]:.2f}m")
print(f"Lever Arm Distances to Tail (135-193):")
print(f"  Start of Tail (Frame 135): {lever_arms[0]:.2f} m")
print(f"  Mid of Tail (Frame 164):   {lever_arms[len(lever_arms)//2]:.2f} m")
print(f"  End of Tail (Frame 193):   {lever_arms[-1]:.2f} m")
print(f"  Mean Lever Arm across Tail: {np.mean(lever_arms):.2f} m (Max: {np.max(lever_arms):.2f} m)")
print(f"\nMeasured Heading Divergence (Visual Heading vs GPS Path Heading):")
print(f"  Train Set (0-135): Mean Abs = {np.mean(np.abs(heading_diffs[:135])):.2f} deg, Median = {np.median(np.abs(heading_diffs[:135])):.2f} deg")
print(f"  Boundary (Frame 135): {abs(heading_diffs[135]):.2f} deg")
print(f"  Tail Set (135-193):  Mean Abs = {np.mean(np.abs(heading_diffs[135:])):.2f} deg, Max Abs = {np.max(np.abs(heading_diffs[135:])):.2f} deg")
print(f"\nEmpirical Tail Errors:")
print(f"  Tail Start Error (Frame 135): {tail_errors[0]:.2f} m")
print(f"  Tail Mid Error (Frame 164):   {tail_errors[len(tail_errors)//2]:.2f} m")
print(f"  Tail End Error (Frame 193):   {tail_errors[-1]:.2f} m")
print(f"  Tail Median Error:            {np.median(tail_errors):.2f} m")
print(f"  Theoretical Lever-Arm Prediction: L * sin(theta) = {np.mean(lever_arms):.1f}m * sin({np.mean(np.abs(heading_diffs[135:])):.1f} deg) = {np.mean(lever_arms) * np.sin(np.radians(np.mean(np.abs(heading_diffs[135:])))):.2f} m")
print("==================================================================")
