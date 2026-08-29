# Memory Budget Fallback

We will maintain a strict 6 GB hardware target for v1 to maximize deployability, rather than bumping the hardware requirement to 8 GB. When a scene exceeds this budget, we will employ a progressive fallback hierarchy:
1. Reduce input resolution
2. Cap Gaussian densification
3. Spatial tiling (only for scenes that still exceed VRAM after the first two steps)

Tiling is kept as an absolute last resort because it significantly complicates the mesh merging process during the extraction phase.
