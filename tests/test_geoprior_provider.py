"""
Unit Test Suite for GeoPrior Provider Engine & Provenance Invariants
====================================================================
Verifies WGS84-to-ENU coordinate transforms, provider fallback hierarchies,
and strict primitive-level provenance tracking.

Chapter References:
- Chapter 4: Geographic Prior Provider Layer
- Chapter 11: Provenance Tracking Engine
- ADR 0002: Per-Primitive Provenance Tracking (DD-06)
- ADR 0005: Provider Independence (DD-09)
"""

import pytest
import numpy as np
from src.geoprior.types import (
    BoundingBoxWGS84,
    ProvenanceType,
    GeoPriorSource,
    BuildingFootprint,
    TerrainElevationGrid,
)
from src.geoprior.transforms import (
    geodetic_to_ecef,
    ecef_to_enu,
    wgs84_polygon_to_enu,
)
from src.geoprior.providers.overture import OvertureMapsProvider
from src.geoprior.providers.osm import OpenStreetMapProvider
from src.geoprior.providers.cesium import CesiumTerrainProvider, SyntheticFlatTerrainProvider
from src.geoprior.providers.engine import GeoPriorProviderEngine


class TestGeodeticTransformations:
    """Verify high-precision WGS84 ellipsoidal to local tangent ENU math."""

    def test_geodetic_to_ecef_equator(self):
        # Lat=0, Lon=0, Alt=0 should yield X = WGS84_A (~6378137.0), Y=0, Z=0
        ecef = geodetic_to_ecef(0.0, 0.0, 0.0)
        assert np.isclose(ecef[0], 6378137.0, atol=1e-3)
        assert np.isclose(ecef[1], 0.0, atol=1e-3)
        assert np.isclose(ecef[2], 0.0, atol=1e-3)

    def test_ecef_to_enu_identity_at_origin(self):
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 30.0
        ref_ecef = geodetic_to_ecef(ref_lat, ref_lon, ref_alt)
        
        enu = ecef_to_enu(ref_ecef, ref_lat, ref_lon, ref_ecef)
        assert np.isclose(enu.east, 0.0, atol=1e-4)
        assert np.isclose(enu.north, 0.0, atol=1e-4)
        assert np.isclose(enu.up, 0.0, atol=1e-4)

    def test_polygon_enu_conversion(self):
        origin = (12.9716, 77.5946, 920.0) # Bangalore, India
        poly_wgs84 = [
            (12.9715, 77.5945),
            (12.9717, 77.5945),
            (12.9717, 77.5947),
            (12.9715, 77.5947),
        ]
        poly_enu = wgs84_polygon_to_enu(poly_wgs84, origin)
        assert len(poly_enu) == 4
        # Dimensions should be roughly tens of meters
        assert abs(poly_enu[1].north - poly_enu[0].north) > 10.0


class TestProviderFallbackHierarchy:
    """Verify robust failover chains across Overture, OSM, and Cesium."""

    @pytest.fixture
    def sample_bbox(self):
        return BoundingBoxWGS84(
            min_lat=12.9700,
            min_lon=77.5900,
            max_lat=12.9750,
            max_lon=77.5950,
        )

    def test_primary_pipeline_success(self, sample_bbox):
        engine = GeoPriorProviderEngine()
        scene = engine.fetch_scene(sample_bbox)
        
        assert len(scene.buildings) > 0
        assert scene.buildings[0].provenance.geo_prior_source == GeoPriorSource.OVERTURE
        assert scene.terrain.provenance.geo_prior_source == GeoPriorSource.CESIUM_TERRAIN

    def test_overture_failure_fallbacks_to_osm(self, sample_bbox):
        failing_overture = OvertureMapsProvider(simulate_network_failure=True)
        working_osm = OpenStreetMapProvider(simulate_network_failure=False)
        engine = GeoPriorProviderEngine(overture_provider=failing_overture, osm_provider=working_osm)
        
        scene = engine.fetch_scene(sample_bbox)
        assert len(scene.buildings) > 0
        assert scene.buildings[0].provenance.geo_prior_source == GeoPriorSource.OSM
        assert scene.buildings[0].height_meters == 9.0 # 3 levels * 3.0m

    def test_all_vector_providers_failing_returns_empty_gracefully(self, sample_bbox):
        failing_overture = OvertureMapsProvider(simulate_network_failure=True)
        failing_osm = OpenStreetMapProvider(simulate_network_failure=True)
        engine = GeoPriorProviderEngine(overture_provider=failing_overture, osm_provider=failing_osm)
        
        scene = engine.fetch_scene(sample_bbox)
        assert len(scene.buildings) == 0
        assert isinstance(scene.terrain, TerrainElevationGrid)

    def test_cesium_failure_fallbacks_to_synthetic_flat(self, sample_bbox):
        failing_cesium = CesiumTerrainProvider(simulate_network_failure=True)
        engine = GeoPriorProviderEngine(cesium_provider=failing_cesium)
        
        scene = engine.fetch_scene(sample_bbox)
        assert scene.terrain.provenance.geo_prior_source == GeoPriorSource.SYNTHETIC_FLAT
        assert np.all(scene.terrain.elevation_matrix == 0.0)


class TestProvenanceInvariants:
    """Verify strict adherence to ADR 0002 and ADR 0005."""

    def test_100_percent_priors_tagged_prior_guided(self):
        bbox = BoundingBoxWGS84(min_lat=37.77, min_lon=-122.42, max_lat=37.78, max_lon=-122.41)
        engine = GeoPriorProviderEngine()
        scene = engine.fetch_scene(bbox)

        # 1. Verify Terrain
        assert scene.terrain.provenance.provenance == ProvenanceType.PRIOR_GUIDED
        assert scene.terrain.provenance.ai_inference is False
        assert scene.terrain.provenance.observation_confidence == 0.0

        # 2. Verify Every Single Building Primitive
        for b in scene.buildings:
            assert b.provenance.provenance == ProvenanceType.PRIOR_GUIDED
            assert b.provenance.ai_inference is False
            assert b.provenance.observation_confidence == 0.0
            assert b.provenance.primitive_id is not None

    def test_invalid_bounding_box_raises_error(self):
        with pytest.raises(ValueError):
            BoundingBoxWGS84(min_lat=50.0, min_lon=10.0, max_lat=40.0, max_lon=20.0).validate()
