"""
GeoPrior Abstract Base Provider Interface
=========================================
Defines the standard query protocol for geographic prior data providers.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from src.geoprior.types import BoundingBoxWGS84, BuildingFootprint, TerrainElevationGrid, GeoPriorSource


class GeoPriorProvider(ABC):
    """Abstract base class for all geographic metadata providers."""

    @property
    @abstractmethod
    def source(self) -> GeoPriorSource:
        """Returns the specific GeoPriorSource identifier."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Verify connectivity and operational status."""
        pass

    @abstractmethod
    def fetch_buildings(self, bbox: BoundingBoxWGS84, origin_wgs84: Tuple[float, float, float]) -> List[BuildingFootprint]:
        """Query building footprints within the bounding box."""
        pass

    @abstractmethod
    def fetch_terrain(self, bbox: BoundingBoxWGS84, resolution_meters: float = 1.0) -> TerrainElevationGrid:
        """Query digital elevation model for the bounding box."""
        pass
