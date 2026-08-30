"""
GeoPrior Georeferencing & World Alignment Module (Chapter 7)
============================================================
Provides 7-DoF Umeyama similarity alignment, RANSAC outlier purging,
and joint visual-geodetic bundle adjustment.
"""

from src.georef.types import UmeyamaTransform, GeodeticAnchor, GeorefMetrics, GeorefResult
from src.georef.umeyama import solve_umeyama_similarity
from src.georef.ransac_georef import RANSACGeoreferencer
from src.georef.geodetic_ba_anchor import GeodeticBundleAdjuster
from src.georef.pipeline import GeoreferencingEngine

__all__ = [
    "UmeyamaTransform",
    "GeodeticAnchor",
    "GeorefMetrics",
    "GeorefResult",
    "solve_umeyama_similarity",
    "RANSACGeoreferencer",
    "GeodeticBundleAdjuster",
    "GeoreferencingEngine",
]
