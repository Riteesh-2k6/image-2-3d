# Provider Independence

Phase 1 GeoPrior will use only Overture Maps, OpenStreetMap, and Cesium Terrain as its geometric priors. Google Maps integration is explicitly removed from Phase 1 and made an optional, non-blocking feature flag for Phase 3.

This removes a "Pending Legal Review" dependency on Google Maps from the critical path, ensuring the core pipeline has zero dependencies on Google and can ship within the aggressive 22-week timeline.
