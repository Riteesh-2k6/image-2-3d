"""
DJI Telemetry & Flight Log Ingestion Module
============================================
Parses 50Hz DJI drone telemetry logs (CSV) to extract camera positions,
gimbal orientations, and IMU attitudes synchronized with video frame timestamps.
"""

import csv
import os
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class TelemetryRecord:
    """Individual flight telemetry sample at a discrete timestamp."""
    time_sec: float
    latitude: float
    longitude: float
    altitude_m: float
    height_m: float
    gimbal_pitch_deg: float
    gimbal_roll_deg: float
    gimbal_yaw_deg: float
    drone_pitch_deg: float
    drone_roll_deg: float
    drone_yaw_deg: float
    gps_num_satellites: int = 0
    h_speed_mps: float = 0.0


class TelemetryLoader:
    """Ingests and interpolates 50Hz DJI Mini 3 Pro flight telemetry logs."""

    def __init__(self, csv_path: Optional[str] = None):
        self.records: List[TelemetryRecord] = []
        if csv_path and os.path.exists(csv_path):
            self.load_csv(csv_path)

    def load_csv(self, csv_path: str) -> List[TelemetryRecord]:
        """Parse DJI telemetry CSV format."""
        self.records = []
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        # Find the header row (skipping leading 'sep=,' if present)
        start_idx = 0
        for i, line in enumerate(lines):
            if "OSD.latitude" in line or "OSD.flyTime" in line:
                start_idx = i
                break

        reader = csv.DictReader(lines[start_idx:])
        for row in reader:
            try:
                # Time
                fly_time_str = row.get("OSD.flyTime [s]", "0").strip()
                time_sec = float(fly_time_str) if fly_time_str else 0.0

                # GPS Coordinates
                lat_str = row.get("OSD.latitude", "").strip()
                lon_str = row.get("OSD.longitude", "").strip()
                if not lat_str or not lon_str:
                    continue
                lat = float(lat_str)
                lon = float(lon_str)

                # Altitudes (convert ft to meters: 1 ft = 0.3048 m)
                alt_ft = float(row.get("OSD.altitude [ft]", "0") or 0.0)
                height_ft = float(row.get("OSD.height [ft]", "0") or 0.0)
                alt_m = alt_ft * 0.3048
                height_m = height_ft * 0.3048

                # Gimbal angles
                g_pitch = float(row.get("GIMBAL.pitch", "0") or 0.0)
                g_roll = float(row.get("GIMBAL.roll", "0") or 0.0)
                g_yaw = float(row.get("GIMBAL.yaw", "0") or 0.0)

                # Drone IMU angles
                d_pitch = float(row.get("OSD.pitch", "0") or 0.0)
                d_roll = float(row.get("OSD.roll", "0") or 0.0)
                d_yaw = float(row.get("OSD.yaw", "0") or 0.0)

                # Satellites & Speed
                sats = int(row.get("OSD.gpsNum", "0") or 0)
                speed_mph = float(row.get("OSD.hSpeed [MPH]", "0") or 0.0)
                speed_mps = speed_mph * 0.44704

                record = TelemetryRecord(
                    time_sec=time_sec,
                    latitude=lat,
                    longitude=lon,
                    altitude_m=alt_m,
                    height_m=height_m,
                    gimbal_pitch_deg=g_pitch,
                    gimbal_roll_deg=g_roll,
                    gimbal_yaw_deg=g_yaw,
                    drone_pitch_deg=d_pitch,
                    drone_roll_deg=d_roll,
                    drone_yaw_deg=d_yaw,
                    gps_num_satellites=sats,
                    h_speed_mps=speed_mps,
                )
                self.records.append(record)
            except (ValueError, TypeError):
                continue

        # Sort chronologically
        self.records.sort(key=lambda r: r.time_sec)
        return self.records

    def get_interpolated_telemetry(self, timestamp_sec: float) -> Optional[TelemetryRecord]:
        """Linearly interpolate telemetry parameters at exact query timestamp."""
        if not self.records:
            return None

        if timestamp_sec <= self.records[0].time_sec:
            return self.records[0]
        if timestamp_sec >= self.records[-1].time_sec:
            return self.records[-1]

        # Binary search for interval
        times = [r.time_sec for r in self.records]
        idx = np.searchsorted(times, timestamp_sec)
        r0 = self.records[idx - 1]
        r1 = self.records[idx]

        dt = r1.time_sec - r0.time_sec
        if dt <= 1e-6:
            return r0

        alpha = (timestamp_sec - r0.time_sec) / dt

        def interpolate_angle(a0: float, a1: float, a: float) -> float:
            """Interpolates angle along the shortest circular arc on S1."""
            diff = (a1 - a0 + 180.0) % 360.0 - 180.0
            interp = a0 + a * diff
            return (interp + 180.0) % 360.0 - 180.0

        return TelemetryRecord(
            time_sec=timestamp_sec,
            latitude=r0.latitude + alpha * (r1.latitude - r0.latitude),
            longitude=r0.longitude + alpha * (r1.longitude - r0.longitude),
            altitude_m=r0.altitude_m + alpha * (r1.altitude_m - r0.altitude_m),
            height_m=r0.height_m + alpha * (r1.height_m - r0.height_m),
            gimbal_pitch_deg=interpolate_angle(r0.gimbal_pitch_deg, r1.gimbal_pitch_deg, alpha),
            gimbal_roll_deg=interpolate_angle(r0.gimbal_roll_deg, r1.gimbal_roll_deg, alpha),
            gimbal_yaw_deg=interpolate_angle(r0.gimbal_yaw_deg, r1.gimbal_yaw_deg, alpha),
            drone_pitch_deg=interpolate_angle(r0.drone_pitch_deg, r1.drone_pitch_deg, alpha),
            drone_roll_deg=interpolate_angle(r0.drone_roll_deg, r1.drone_roll_deg, alpha),
            drone_yaw_deg=interpolate_angle(r0.drone_yaw_deg, r1.drone_yaw_deg, alpha),
            gps_num_satellites=r1.gps_num_satellites,
            h_speed_mps=r0.h_speed_mps + alpha * (r1.h_speed_mps - r0.h_speed_mps),
        )

    def gimbal_to_rotation_matrix(self, pitch_deg: float, roll_deg: float, yaw_deg: float) -> np.ndarray:
        """
        Convert gimbal Euler angles (pitch, roll, yaw in degrees) to a 3x3 camera rotation matrix.
        Camera frame: +X right, +Y down, +Z forward (standard CV photogrammetry).
        World frame: East-North-Up (+X East, +Y North, +Z Up).
        """
        pitch = np.radians(pitch_deg)
        roll = np.radians(roll_deg)
        yaw = np.radians(yaw_deg)

        # Yaw around Z
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        # Pitch around X
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ])
        # Roll around Y
        Ry = np.array([
            [np.cos(roll), 0, np.sin(roll)],
            [0, 1, 0],
            [-np.sin(roll), 0, np.cos(roll)]
        ])

        # World to Camera alignment transformation
        R_world_drone = Rz @ Ry @ Rx
        # Convert ENU to OpenCV Camera convention
        R_enu_to_cv = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ], dtype=np.float64)

        R_cam = R_enu_to_cv @ R_world_drone.T
        return R_cam
