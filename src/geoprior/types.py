"""
GeoPrior Data Types & Provenance Metadata Schema
================================================
Defines immutable provenance tracking types, geometric structures,
and coordinate representations for geographic prior integration.

ADR References:
- ADR 0002: Per-Primitive Provenance Tracking (DD-06)
- ADR 0005: Provider Independence (DD-09)
"""

import time
import uuid
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np


class ProvenanceType(str, Enum):
    """Primitive-level provenance classification."""
    OBSERVED = "observed"          # Directly imaged by drone camera with photometric evidence
    PRIOR_GUIDED = "prior_guided"  # Initialized/constrained by external geospatial GIS data
    AI_INFERRED = "ai_inferred"    # Synthesized by geometric heuristics or generative priors


class GeoPriorSource(str, Enum):
    """External data sources for geographic priors."""
    OVERTURE = "overture_maps"
    OSM = "openstreetmap"
    CESIUM_TERRAIN = "cesium_world_terrain"
    SRTM = "srtm_dem"
    SYNTHETIC_FLAT = "synthetic_flat_datum"


@dataclass(frozen=True)
class ProvenanceMetadata:
    """Immutable provenance metadata attached to every geometric primitive and prior."""
    primitive_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    provenance: ProvenanceType = ProvenanceType.PRIOR_GUIDED
    observation_confidence: float = 0.0 # 0.0 for pure priors, scales to 1.0 with multi-view coverage
    geo_prior_source: Optional[GeoPriorSource] = GeoPriorSource.OVERTURE
    ai_inference: bool = False
    creation_timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primitive_id": self.primitive_id,
            "provenance": self.provenance.value,
            "observation_confidence": self.observation_confidence,
            "geo_prior_source": self.geo_prior_source.value if self.geo_prior_source else None,
            "ai_inference": self.ai_inference,
            "creation_timestamp": self.creation_timestamp,
        }


@dataclass
class BoundingBoxWGS84:
    """Geographic bounding box in WGS84 degrees."""
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float

    def validate(self):
        if not (-90.0 <= self.min_lat <= self.max_lat <= 90.0):
            raise ValueError(f"Invalid latitude bounds: [{self.min_lat}, {self.max_lat}]")
        if not (-180.0 <= self.min_lon <= self.max_lon <= 180.0):
            raise ValueError(f"Invalid longitude bounds: [{self.min_lon}, {self.max_lon}]")

    @property
    def center(self) -> Tuple[float, float]:
        return (self.min_lat + self.max_lat) / 2.0, (self.min_lon + self.max_lon) / 2.0


@dataclass
class ENUCoordinate:
    """Local Cartesian coordinate in East-North-Up (meters) relative to a reference origin."""
    east: float
    north: float
    up: float

    def to_numpy(self) -> np.ndarray:
        return np.array([self.east, self.north, self.up], dtype=np.float32)


@dataclass
class BuildingFootprint:
    """Georeferenced building entity with 2D/3D polygon and immutable provenance."""
    id: str
    polygon_wgs84: List[Tuple[float, float]] # [(lat, lon), ...]
    polygon_enu: List[ENUCoordinate]         # [(east, north, up), ...]
    height_meters: float
    base_elevation_meters: float = 0.0
    levels: Optional[int] = None
    building_type: str = "building"
    provenance: ProvenanceMetadata = field(default_factory=ProvenanceMetadata)


@dataclass
class TerrainElevationGrid:
    """Rasterized digital elevation model in local ENU frame."""
    bounds: BoundingBoxWGS84
    resolution_meters: float
    grid_shape: Tuple[int, int] # (rows, cols)
    elevation_matrix: np.ndarray # 2D float32 array in meters
    provenance: ProvenanceMetadata = field(default_factory=lambda: ProvenanceMetadata(geo_prior_source=GeoPriorSource.CESIUM_TERRAIN))


@dataclass
class GeoPriorScene:
    """Complete aggregated geographic prior scene ready for 3DGS initialization."""
    origin_wgs84: Tuple[float, float, float] # (lat, lon, altitude)
    terrain: TerrainElevationGrid
    buildings: List[BuildingFootprint]
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
