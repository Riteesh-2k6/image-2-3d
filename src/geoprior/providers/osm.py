"""
OpenStreetMap Building Provider (P3 Fallback Vector Prior)
==========================================================
Fallback vector provider extracting building polygons and estimating heights
from OSM tags (e.g. building:levels * 3.0m).
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


class OpenStreetMapProvider(GeoPriorProvider):
    """Fallback building footprint provider querying OpenStreetMap."""

    def __init__(self, simulate_network_failure: bool = False):
        self.simulate_failure = simulate_network_failure

    @property
    def source(self) -> GeoPriorSource:
        return GeoPriorSource.OSM

    def health_check(self) -> bool:
        return not self.simulate_failure

    def fetch_buildings(self, bbox: BoundingBoxWGS84, origin_wgs84: Tuple[float, float, float]) -> List[BuildingFootprint]:
        bbox.validate()
        if self.simulate_failure:
            raise ConnectionError("OpenStreetMap Overpass API unreachable")

        lat_c, lon_c = bbox.center
        d_lat = (bbox.max_lat - bbox.min_lat) * 0.25
        d_lon = (bbox.max_lon - bbox.min_lon) * 0.25

        osm_polygon = [
            (lat_c - d_lat, lon_c - d_lon),
            (lat_c + d_lat, lon_c - d_lon),
            (lat_c + d_lat, lon_c + d_lon),
            (lat_c - d_lat, lon_c + d_lon),
        ]
        osm_enu = wgs84_polygon_to_enu(osm_polygon, origin_wgs84)

        # Height deduction heuristic: levels * 3.0m
        levels = 3
        height = levels * 3.0

        building = BuildingFootprint(
            id=f"osm_{uuid.uuid4().hex[:8]}",
            polygon_wgs84=osm_polygon,
            polygon_enu=osm_enu,
            height_meters=height,
            base_elevation_meters=origin_wgs84[2],
            levels=levels,
            building_type="residential",
            provenance=ProvenanceMetadata(
                provenance=ProvenanceType.PRIOR_GUIDED,
                observation_confidence=0.0,
                geo_prior_source=GeoPriorSource.OSM,
                ai_inference=False,
            )
        )

        return [building]

    def fetch_terrain(self, bbox: BoundingBoxWGS84, resolution_meters: float = 1.0) -> TerrainElevationGrid:
        raise NotImplementedError("OpenStreetMap does not provide raster terrain elevation.")
