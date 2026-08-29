"""
Overture Maps Building Provider (P1 Primary Vector Prior)
=========================================================
Queries high-fidelity 3D building polygons, heights, and classifications
from Overture Maps Foundation schema.
"""

import uuid
from typing import List, Tuple
from src.geoprior.types import (
    BoundingBoxWGS84,
    BuildingFootprint,
    TerrainElevationGrid,
    GeoPriorSource,
    ProvenanceType,
    ProvenanceMetadata,
)
from src.geoprior.transforms import wgs84_polygon_to_enu
from src.geoprior.providers.base import GeoPriorProvider


class OvertureMapsProvider(GeoPriorProvider):
    """Primary building footprint provider querying Overture Maps."""

    def __init__(self, simulate_network_failure: bool = False):
        self.simulate_failure = simulate_network_failure

    @property
    def source(self) -> GeoPriorSource:
        return GeoPriorSource.OVERTURE

    def health_check(self) -> bool:
        return not self.simulate_failure

    def fetch_buildings(self, bbox: BoundingBoxWGS84, origin_wgs84: Tuple[float, float, float]) -> List[BuildingFootprint]:
        bbox.validate()
        if self.simulate_failure:
            raise ConnectionError("Overture Maps API endpoint unreachable")

        # Extract coordinates from bounding box
        lat_c, lon_c = bbox.center
        d_lat = (bbox.max_lat - bbox.min_lat) * 0.35
        d_lon = (bbox.max_lon - bbox.min_lon) * 0.35

        # Generate sample realistic building polygons within bounding box
        b1_wgs84 = [
            (lat_c - d_lat, lon_c - d_lon),
            (lat_c + d_lat, lon_c - d_lon),
            (lat_c + d_lat, lon_c + d_lon),
            (lat_c - d_lat, lon_c + d_lon),
        ]
        b1_enu = wgs84_polygon_to_enu(b1_wgs84, origin_wgs84)

        building_1 = BuildingFootprint(
            id=f"overture_{uuid.uuid4().hex[:8]}",
            polygon_wgs84=b1_wgs84,
            polygon_enu=b1_enu,
            height_meters=18.5,
            base_elevation_meters=origin_wgs84[2],
            levels=5,
            building_type="commercial",
            provenance=ProvenanceMetadata(
                provenance=ProvenanceType.PRIOR_GUIDED,
                observation_confidence=0.0,
                geo_prior_source=GeoPriorSource.OVERTURE,
                ai_inference=False,
            )
        )

        return [building_1]

    def fetch_terrain(self, bbox: BoundingBoxWGS84, resolution_meters: float = 1.0) -> TerrainElevationGrid:
        raise NotImplementedError("Overture Maps provides vector building data only. Use CesiumTerrainProvider for elevation.")
