"""
GeoPrior Provider Engine & Fallback Orchestrator
================================================
Aggregates heterogeneous geospatial providers (Overture, OSM, Cesium Terrain)
into the unified GeoPrior schema in WGS84 and local Tangent ENU coordinates.

ADR References:
- ADR 0002: Per-Primitive Provenance Tracking (DD-06)
- ADR 0005: Provider Independence (DD-09)
"""

import logging
from typing import Tuple, List, Optional
from src.geoprior.types import (
    BoundingBoxWGS84,
    BuildingFootprint,
    TerrainElevationGrid,
    GeoPriorScene,
    ProvenanceType,
    GeoPriorSource,
)
from src.geoprior.providers.overture import OvertureMapsProvider
from src.geoprior.providers.osm import OpenStreetMapProvider
from src.geoprior.providers.cesium import CesiumTerrainProvider, SyntheticFlatTerrainProvider

logger = logging.getLogger("geoprior.engine")


class GeoPriorProviderEngine:
    """
    Main aggregator orchestrating provider queries, failover chains,
    local coordinate transformations, and strict provenance verification.
    """

    def __init__(
        self,
        overture_provider: Optional[OvertureMapsProvider] = None,
        osm_provider: Optional[OpenStreetMapProvider] = None,
        cesium_provider: Optional[CesiumTerrainProvider] = None,
    ):
        self.overture = overture_provider or OvertureMapsProvider()
        self.osm = osm_provider or OpenStreetMapProvider()
        self.cesium = cesium_provider or CesiumTerrainProvider()
        self.synthetic_terrain = SyntheticFlatTerrainProvider()

    def fetch_scene(
        self,
        bbox: BoundingBoxWGS84,
        origin_wgs84: Optional[Tuple[float, float, float]] = None,
        terrain_resolution_m: float = 1.0,
    ) -> GeoPriorScene:
        """
        Aggregate terrain and building priors for a bounding box into a unified GeoPriorScene.
        Executes automatic fallback if primary providers fail.
        """
        bbox.validate()
        
        # Default origin to center of bounding box at altitude 0.0
        if origin_wgs84 is None:
            c_lat, c_lon = bbox.center
            origin_wgs84 = (c_lat, c_lon, 0.0)

        # 1. Fetch Buildings (Overture -> OSM -> Empty)
        buildings: List[BuildingFootprint] = []
        try:
            logger.info("Querying primary vector provider (Overture Maps)...")
            buildings = self.overture.fetch_buildings(bbox, origin_wgs84)
        except Exception as e_overture:
            logger.warning(f"Overture Maps query failed ({e_overture}). Falling back to OpenStreetMap...")
            try:
                buildings = self.osm.fetch_buildings(bbox, origin_wgs84)
            except Exception as e_osm:
                logger.error(f"OpenStreetMap query failed ({e_osm}). No building priors available.")
                buildings = []

        # 2. Fetch Terrain Elevation (Cesium -> Synthetic Flat)
        terrain: TerrainElevationGrid
        try:
            logger.info("Querying primary elevation provider (Cesium World Terrain)...")
            terrain = self.cesium.fetch_terrain(bbox, resolution_meters=terrain_resolution_m)
        except Exception as e_cesium:
            logger.warning(f"Cesium Terrain query failed ({e_cesium}). Falling back to Synthetic Flat Datum...")
            terrain = self.synthetic_terrain.fetch_terrain(bbox, resolution_meters=terrain_resolution_m)

        # 3. Strict Provenance Invariant Verification
        self._verify_provenance_invariants(buildings, terrain)

        return GeoPriorScene(
            origin_wgs84=origin_wgs84,
            terrain=terrain,
            buildings=buildings,
        )

    def _verify_provenance_invariants(self, buildings: List[BuildingFootprint], terrain: TerrainElevationGrid):
        """
        Asserts the mathematical and architectural invariant that 100% of emitted
        geographic priors carry provenance: prior_guided and ai_inference: false.
        """
        # Verify Terrain Provenance
        assert terrain.provenance.provenance == ProvenanceType.PRIOR_GUIDED, (
            f"Terrain provenance must be '{ProvenanceType.PRIOR_GUIDED.value}', found '{terrain.provenance.provenance.value}'"
        )
        assert terrain.provenance.ai_inference is False, (
            "Terrain prior must have ai_inference=False to prevent hallucinated geometry contamination"
        )

        # Verify Buildings Provenance
        for idx, b in enumerate(buildings):
            assert b.provenance.provenance == ProvenanceType.PRIOR_GUIDED, (
                f"Building [{idx}] provenance must be '{ProvenanceType.PRIOR_GUIDED.value}', found '{b.provenance.provenance.value}'"
            )
            assert b.provenance.ai_inference is False, (
                f"Building [{idx}] prior must have ai_inference=False"
            )
            assert b.provenance.observation_confidence == 0.0, (
                f"Building [{idx}] unobserved prior confidence must be 0.0 until verified by drone camera"
            )
            assert len(b.polygon_enu) == len(b.polygon_wgs84), (
                f"Building [{idx}] polygon ENU and WGS84 point counts do not match"
            )
