"""
Geodetic Coordinate Transformations (WGS84 <-> ENU)
====================================================
High-precision WGS84 ellipsoidal transformations to local East-North-Up (ENU)
tangent plane coordinates for 3D Gaussian Splatting and mesh alignment.
"""

import math
import numpy as np
from typing import Tuple, List
from src.geoprior.types import ENUCoordinate


# WGS84 Ellipsoid Constants
WGS84_A = 6378137.0         # Semi-major axis (meters)
WGS84_F = 1.0 / 298.257223563 # Flattening
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2 # Square of eccentricity


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
    """Convert geodetic latitude, longitude, and ellipsoidal height to ECEF (meters)."""
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)
    
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)
    
    # Radius of curvature in prime vertical
    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)
    
    x = (N + alt_m) * cos_lat * cos_lon
    y = (N + alt_m) * cos_lat * sin_lon
    z = (N * (1.0 - WGS84_E2) + alt_m) * sin_lat
    
    return np.array([x, y, z], dtype=np.float64)


def ecef_to_enu(ecef_point: np.ndarray, ref_lat_deg: float, ref_lon_deg: float, ref_ecef: np.ndarray) -> ENUCoordinate:
    """Convert an ECEF point to local East-North-Up (ENU) coordinates relative to a reference origin."""
    lat_rad = math.radians(ref_lat_deg)
    lon_rad = math.radians(ref_lon_deg)
    
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)
    
    # ECEF displacement vector
    dx = ecef_point[0] - ref_ecef[0]
    dy = ecef_point[1] - ref_ecef[1]
    dz = ecef_point[2] - ref_ecef[2]
    
    # Rotation Matrix from ECEF to Local ENU
    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    
    return ENUCoordinate(east=float(east), north=float(north), up=float(up))


def wgs84_polygon_to_enu(polygon_wgs84: List[Tuple[float, float]], origin_wgs84: Tuple[float, float, float], base_alt_m: float = 0.0) -> List[ENUCoordinate]:
    """Transform a WGS84 polygon [(lat, lon), ...] to local ENU coordinates."""
    ref_lat, ref_lon, ref_alt = origin_wgs84
    ref_ecef = geodetic_to_ecef(ref_lat, ref_lon, ref_alt)
    
    enu_coords = []
    for lat, lon in polygon_wgs84:
        pt_ecef = geodetic_to_ecef(lat, lon, base_alt_m)
        enu_pt = ecef_to_enu(pt_ecef, ref_lat, ref_lon, ref_ecef)
        enu_coords.append(enu_pt)
        
    return enu_coords
