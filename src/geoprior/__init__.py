"""
GeoPrior Core Module
====================
Provides geographic prior ingestion, geodetic transformations,
and primitive-level provenance tracking for 3D reconstruction.
"""

from src.geoprior.types import (
    ProvenanceType,
    GeoPriorSource,
    ProvenanceMetadata,
    BoundingBoxWGS84,
    ENUCoordinate,
    BuildingFootprint,
    TerrainElevationGrid,
    GeoPriorScene,
)
from src.geoprior.transforms import (
    geodetic_to_ecef,
    ecef_to_enu,
    wgs84_polygon_to_enu,
)
from src.geoprior.providers.engine import GeoPriorProviderEngine

__all__ = [
    "ProvenanceType",
    "GeoPriorSource",
    "ProvenanceMetadata",
    "BoundingBoxWGS84",
    "ENUCoordinate",
    "BuildingFootprint",
    "TerrainElevationGrid",
    "GeoPriorScene",
    "geodetic_to_ecef",
    "ecef_to_enu",
    "wgs84_polygon_to_enu",
    "GeoPriorProviderEngine",
]
