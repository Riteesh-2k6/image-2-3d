"""
7-DoF Umeyama Closed-Form Similarity Alignment
==============================================
Estimates the optimal rigid scale, rotation (SO(3)), and translation
between visual camera centers and ground-truth GNSS ENU positions.
"""

from typing import Tuple
import numpy as np

from src.georef.types import UmeyamaTransform


def solve_umeyama_similarity(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    lat0: float = 0.0,
    lon0: float = 0.0,
    alt0: float = 0.0
) -> UmeyamaTransform:
    """
    Solves the least-squares 7-DoF similarity transformation:
        dst = scale * (src @ R.T) + translation
    
    Parameters:
        src_points: (N, 3) float array of local camera centers
        dst_points: (N, 3) float array of target ENU ground-truth coordinates
        lat0, lon0, alt0: Geodetic reference origin coordinates
        
    Returns:
        UmeyamaTransform containing scale, R in SO(3), translation, and origin.
    """
    if len(src_points) < 3 or len(dst_points) < 3:
        raise ValueError(f"Umeyama requires at least 3 non-collinear point pairs, got {len(src_points)}.")

    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)

    n, m = src.shape
    if m != 3 or dst.shape != (n, 3):
        raise ValueError(f"Points must have shape (N, 3), got src={src.shape}, dst={dst.shape}")

    # Compute centroids
    mu_src = np.mean(src, axis=0)
    mu_dst = np.mean(dst, axis=0)

    # Center the point sets
    src_c = src - mu_src
    dst_c = dst - mu_dst

    # Compute variance of source points
    var_src = np.sum(src_c ** 2) / n
    if var_src < 1e-12:
        raise ValueError("Degenerate source points: point variance is virtually zero.")

    # Cross-covariance matrix
    H = (src_c.T @ dst_c) / n

    # SVD of covariance matrix
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    # Reflection check to enforce proper rotation in SO(3) with det(R) = +1
    d = np.linalg.det(V @ U.T)
    S_det = np.ones(m, dtype=np.float64)
    if d < 0:
        S_det[-1] = -1.0

    # Compute optimal rotation matrix
    R = V @ np.diag(S_det) @ U.T

    # Compute optimal scale factor s
    scale = float(np.sum(S * S_det) / var_src)

    # Compute optimal translation vector t
    translation = mu_dst - scale * (R @ mu_src)

    return UmeyamaTransform(
        scale=scale,
        R=R,
        translation=translation,
        lat0=lat0,
        lon0=lon0,
        alt0=alt0
    )
