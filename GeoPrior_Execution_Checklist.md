# GeoPrior — Master Execution Checklist & Ticket Guide

This checklist provides an interactive, structured guide to tracking and executing all **21 Wayfinder tickets** for the GeoPrior Single-Flight 3D Digital Twin pipeline on the 6 GB RTX 3050 hardware target.

---

## 🚦 Pipeline Gate Status

| Gate | Name | Focus | Target Deliverables | Status |
| :---: | :--- | :--- | :--- | :---: |
| **G0** | Architecture Review | Consistency & Contradiction Resolution | CONTEXT.md, ADRs 0001–0005 | **COMPLETED** ✅ |
| **G1** | Feasibility Verified | 6 GB VRAM Envelope, Runtime, Registration | Tickets 01, 02, 03, 08, 09, 10, 11 | **IN PROGRESS** 🟡 |
| **G2** | Robustness Verified | 10 Failure Modes, Calibration, Edge Quality | Tickets 06, 07, 14, 15, 16, 17, 18, 19 | **PENDING** ⚪ |
| **G3** | Production Candidate | 6-Service Docker Stack, Cost Model, Legal Sign-off | Tickets 20, 21 | **PENDING** ⚪ |

---

## 🧭 Phase-by-Phase Ticket Checklist

### 🧠 Stage 0: Foundations & Gate G1 Feasibility
> **Goal**: Establish memory bounds and verify that 3DGS runs safely within the 6 GB VRAM envelope on RTX 3050 before downstream dependencies are built.

- [x] **[01] Research 3DGS & gsplat Memory Profiling** `(research)`  
  *Path*: [.scratch/geoprior/issues/01-learn-3dgs-gsplat-memory.md](file:///c:/Users/Riteesh/Programming/SIH2K26/.scratch/geoprior/issues/01-learn-3dgs-gsplat-memory.md)  
  *Status*: **RESOLVED** ✅  
  *Key Output*: Memory equation $M_{\text{total}}(N, D, W, H)$, hard cap $N_{\text{max}} = 1.2\text{M}$ Gaussians, Spherical Harmonics $D \le 1$, `packed=True`, and gradient checkpointing rejected.
  
- [x] **[02] Build B1 & B2 VRAM / Runtime Benchmark Harness** `(prototype)`  
  *Path*: [.scratch/geoprior/issues/02-b1-vram-profiling-harness.md](file:///c:/Users/Riteesh/Programming/SIH2K26/.scratch/geoprior/issues/02-b1-vram-profiling-harness.md)  
  *Status*: **RESOLVED** ✅  
  *Key Deliverable*: `benchmarks/run_b1_b2.py` measuring real differentiable 3DGS projection, tile rasterization, photometric loss backprop, and wall-clock CUDA synchronization (`torch.cuda.synchronize()`). (Measured: Peak VRAM **3,378 MB**, Mean Step **388.4 ms**, Preview at Step 50 in **19.4 s** $\le 30\text{ s}$ target).

- [ ] **[03] Execute Gate G1 Feasibility on Speedy-Splat & Validate DD-05** `(task)`  
  *Path*: [.scratch/geoprior/issues/03-g1-feasibility-speedy-splat.md](file:///c:/Users/Riteesh/Programming/SIH2K26/.scratch/geoprior/issues/03-g1-feasibility-speedy-splat.md)  
  *Status*: **UNBLOCKED / READY TO EXECUTE** 🟢  
  *Prerequisites*: Ticket 02 (Resolved)  
  *ADR Reference*: [0001-memory-budget-fallback.md](file:///c:/Users/Riteesh/Programming/SIH2K26/docs/adr/0001-memory-budget-fallback.md)  
  *Key Deliverable*: Populate Chapter 8 Acceptance Benchmarks table with real numbers; validate fallback hierarchy (resolution scaling $\to$ Gaussian cap $\to$ spatial tiling).

---

### 🛰️ Stage 1: Geographic Priors & SSAKS Keyframe Selection
> **Goal**: Ingest open map priors with strict provenance tagging, and filter 85–97% of redundant video frames using lightweight vision models.

- [x] **[04] Implement GeoPrior Provider Engine & Provenance Foundation** `(prototype)`  
  *Path*: [.scratch/geoprior/issues/04-geoprior-provider-engine.md](file:///c:/Users/Riteesh/Programming/SIH2K26/.scratch/geoprior/issues/04-geoprior-provider-engine.md)  
  *Status*: **RESOLVED** ✅  
  *ADR Reference*: [0002-provenance-inheritance.md](file:///c:/Users/Riteesh/Programming/SIH2K26/docs/adr/0002-provenance-inheritance.md) & [0005-provider-independence.md](file:///c:/Users/Riteesh/Programming/SIH2K26/docs/adr/0005-provider-independence.md)  
  *Key Output*: `src/geoprior/` engine with unified schema, WGS84-to-ENU geodesic conversion, Overture $\to$ OSM $\to$ Cesium $\to$ Synthetic Flat failover, and strict invariant that 100% of emitted geographic priors have `provenance: prior_guided` and `ai_inference: false`. 9/9 unit tests passing.

- [x] **[05] Research Lightweight Vision Models for SSAKS** `(research)`  
  *Path*: [.scratch/geoprior/issues/05-learn-ssaks-cascade-models.md](file:///c:/Users/Riteesh/Programming/SIH2K26/.scratch/geoprior/issues/05-learn-ssaks-cascade-models.md)  
  *Status*: **RESOLVED** ✅  
  *Key Output*: CPU DIS Flow (0 MB GPU) + `dinov2_vits14_reg` (480 MB VRAM) + `sam2.1_hiera_tiny` (750 MB VRAM), keeping total peak VRAM $\le 850\text{ MB}$.

- [x] **[06] Implement SSAKS 3-Stage Cascade & Cruise Calibration Fallback** `(prototype)`  
  *Path*: [.scratch/geoprior/issues/06-ssaks-cascade-and-fallback.md](file:///c:/Users/Riteesh/Programming/SIH2K26/.scratch/geoprior/issues/06-ssaks-cascade-and-fallback.md)  
  *Status*: **RESOLVED** ✅  
  *ADR Reference*: [0003-ssaks-emergency-fallback.md](file:///c:/Users/Riteesh/Programming/SIH2K26/docs/adr/0003-ssaks-emergency-fallback.md)  
  *Key Output*: `src/ssaks/` cascade with Stage 1 (Laplacian blur / DIS flow), Stage 2 (embedding novelty), DD-02 cruise detection, and ADR 0003 30s emergency calibration fallback. 6/6 unit tests passing.

- [x] **[07] Execute B5 SSAKS Ablation Study & Frame Reduction Eval** `(task)`  
  *Path*: [.scratch/geoprior/issues/07-b5-ssaks-ablation-eval.md](file:///c:/Users/Riteesh/Programming/SIH2K26/.scratch/geoprior/issues/07-b5-ssaks-ablation-eval.md)  
  *Status*: **RESOLVED** ✅  
  *Key Output*: Evaluated real DJI Mini 3 Pro 4K flight (`06.MP4` + `06.csv`). Extracted 193 keyframes from 1,916 raw 4K frames in 52.8s (**89.93% frame reduction rate**, 36.3 FPS throughput). Report saved to `reports/06_real_flight_ssaks_report.json`.

---

### 📐 Stage 2: Hybrid Pose Estimation & Georeferencing
> **Goal**: Estimate camera poses and align them into true metric coordinates (WGS84 / ENU) using 7-DoF Umeyama and RANSAC outlier filtering.

- [x] **[08] Implement Hybrid Pose Estimation (VGGT + BA Fallback)** `(prototype)`  
  *Path*: [.scratch/geoprior/issues/08-hybrid-pose-estimation.md](file:///c:/Users/Riteesh/Programming/SIH2K26/.scratch/geoprior/issues/08-hybrid-pose-estimation.md)  
  *Status*: **RESOLVED** ✅  
  *Prerequisites*: Ticket 07 (Resolved)  
  *Key Output*: `src/sfm/` pure PyTorch/Python VGGT neural pose estimator, 4 Chapter 6 confidence metrics, telemetry fusion with 50Hz DJI Mini 3 Pro flight log (`videos/06.csv`), and Levenberg-Marquardt bundle adjustment. Extracted 54,546 triangulated 3D points and registered 195/195 keyframes (100%). 11/11 unit tests passing.

- [ ] **[09] Implement Geo Registration Module (Umeyama 7-DoF + RANSAC)** `(prototype)`  
  *Path*: [.scratch/geoprior/issues/09-geo-registration-module.md](file:///c:/Users/Riteesh/Programming/SIH2K26/.scratch/geoprior/issues/09-geo-registration-module.md)  
  *Status*: **UNBLOCKED / READY TO BUILD** 🟢  
  *Prerequisites*: Ticket 08 (Resolved), Ticket 04 (Resolved)  
  *Key Deliverable*: 7-DoF Umeyama similarity transform, RANSAC GPS multipath rejection, and geodetically constrained bundle adjustment.

- [ ] **[10] Execute B3 Geo Registration Accuracy Benchmark** `(task)`  
  *Path*: [.scratch/geoprior/issues/10-b3-registration-accuracy-benchmark.md](file:///c:/Users/Riteesh/Programming/SIH2K26/.scratch/geoprior/issues/10-b3-registration-accuracy-benchmark.md)  
  *Status*: **BLOCKED by 09** 🔒  
  *Key Deliverable*: Measure Translation RMS ($\le 0.15\text{ m}$), Rotation error, Scale residual, and Inlier ratio against surveyed Ground Control Points.

---

### ✨ Stage 3: Dense Gaussian Reconstruction & Live Mode Preview
> **Goal**: Reconstruct optimized 3D Gaussians within 6 GB VRAM and provide real-time WYSIWYG provenance preview during flight.

- [ ] **[11] Implement Speedy-Splat Gaussian Pipeline & Checkpointing** `(prototype)`  
  *Path*: [.scratch/geoprior/issues/11-speedy-splat-pipeline.md](file:///c:/Users/Riteesh/Programming/SIH2K26/.scratch/geoprior/issues/11-speedy-splat-pipeline.md)  
  *Status*: **BLOCKED by 03, 09** 🔒  
  *Key Deliverable*: 3DGS progressive densification with hard $N_{\text{max}} = 1.2\text{M}$ cap, sequential VRAM release, and periodic checkpoint snapshotting (DD-01).

- [ ] **[12] Implement Live Mode Ephemeral WYSIWYG Provenance Renderer** `(prototype)`  
  *Path*: [.scratch/geoprior/issues/12-live-mode-provenance-renderer.md](file:///c:/Users/Riteesh/Programming/SIH2K26/.scratch/geoprior/issues/12-live-mode-provenance-renderer.md)  
  *Status*: **BLOCKED by 11** 🔒  
  *ADR Reference*: [0004-live-provenance-visualization.md](file:///c:/Users/Riteesh/Programming/SIH2K26/docs/adr/0004-live-provenance-visualization.md)  
  *Key Deliverable*: Real-time viewer with visual provenance shading: Solid (Observed), Translucent Blue (Prior-Guided), and Hatched (AI-Inferred).

---

### 🏛️ Stage 4: Mesh Extraction & Structural Edge Refinement (SERE)
> **Goal**: Extract watertight base meshes from regularized Gaussians and sharpen architectural edges using non-generative geometric constraints.

- [x] **[13] Research SuGaR Regularization & SERE Algorithms** `(research)`  
  *Path*: [.scratch/geoprior/issues/13-learn-sugar-and-sere-algorithms.md](file:///c:/Users/Riteesh/Programming/SIH2K26/.scratch/geoprior/issues/13-learn-sugar-and-sere-algorithms.md)  
  *Status*: **RESOLVED** ✅  
  *Key Output*: Flat Gaussian loss ($\mathcal{L}_{\text{flat}} + \mathcal{L}_{\text{SDF}} + \mathcal{L}_{\text{normal}}$), Screened Poisson base meshing (sPSR depth 10, $<1.5\text{ GB}$ CPU RAM), and LSD/RANSAC sparse QP vertex snapping (`OSQP`) bounded by $\pm 3\sigma$.

- [ ] **[14] Implement 7-Stage Mesh Extraction Pipeline (SuGaR)** `(prototype)`  
  *Path*: [.scratch/geoprior/issues/14-sugar-mesh-extraction-pipeline.md](file:///c:/Users/Riteesh/Programming/SIH2K26/.scratch/geoprior/issues/14-sugar-mesh-extraction-pipeline.md)  
  *Status*: **BLOCKED by 11, 13** 🔒  
  *Key Deliverable*: Convert `.splat` scenes to textured `.glb` / `.usd` and Cesium 3D Tiles via SuGaR regularization, surface extraction, UV packing, and LODs.

- [ ] **[15] Implement SERE (Structural Edge Refinement Engine)** `(prototype)`  
  *Path*: [.scratch/geoprior/issues/15-sere-edge-refinement-engine.md](file:///c:/Users/Riteesh/Programming/SIH2K26/.scratch/geoprior/issues/15-sere-edge-refinement-engine.md)  
  *Status*: **BLOCKED by 14** 🔒  
  *Key Deliverable*: Plane/line detection + sparse QP vertex snapping to sharpen rooflines and building corners without generative hallucination.

- [ ] **[16] Execute B4 Mesh Fidelity & Geometric Quality Evaluation** `(task)`  
  *Path*: [.scratch/geoprior/issues/16-b4-mesh-fidelity-evaluation.md](file:///c:/Users/Riteesh/Programming/SIH2K26/.scratch/geoprior/issues/16-b4-mesh-fidelity-evaluation.md)  
  *Status*: **BLOCKED by 15** 🔒  
  *Key Deliverable*: Measure Chamfer distance, Hausdorff distance, Normal consistency, Watertightness, and Edge deviation against LiDAR truth.

---

### 🛡️ Stage 5: Provenance Tracking, Failure Modes & Gate G2 Robustness
> **Goal**: Enforce strict promotion and export rules for geometric data, and validate automated recovery across all 10 failure modes.

- [ ] **[17] Implement Provenance Inheritance Engine & Export Filtering** `(prototype)`  
  *Path*: [.scratch/geoprior/issues/17-provenance-inheritance-engine.md](file:///c:/Users/Riteesh/Programming/SIH2K26/.scratch/geoprior/issues/17-provenance-inheritance-engine.md)  
  *Status*: **BLOCKED by 14, 04** 🔒  
  *ADR Reference*: [0002-provenance-inheritance.md](file:///c:/Users/Riteesh/Programming/SIH2K26/docs/adr/0002-provenance-inheritance.md)  
  *Key Deliverable*: Weighted face provenance inheritance, $\ge 2$ view promotion validation, adversarial test suite, and default export filtering.

- [ ] **[18] Implement Detection & Recovery for 10 Failure Modes** `(prototype)`  
  *Path*: [.scratch/geoprior/issues/18-failure-modes-catalog-implementation.md](file:///c:/Users/Riteesh/Programming/SIH2K26/.scratch/geoprior/issues/18-failure-modes-catalog-implementation.md)  
  *Status*: **BLOCKED by 11, 15** 🔒  
  *Key Deliverable*: Automated handlers for glass, water, low overlap, rolling shutter, dynamic objects, GPS multipath, thermal throttling, missing EXIF, blur, and CUDA OOM.

- [ ] **[19] Execute Gate G2 Robustness Validation & Calibration Reports** `(task)`  
  *Path*: [.scratch/geoprior/issues/19-g2-robustness-validation.md](file:///c:/Users/Riteesh/Programming/SIH2K26/.scratch/geoprior/issues/19-g2-robustness-validation.md)  
  *Status*: **BLOCKED by 18, 07, 10, 16** 🔒  
  *Key Deliverable*: Complete G2 Robustness Matrix document and engineering validation sign-off across challenge datasets.

---

### 🚀 Stage 6: Production Deployment, Costing & Gate G3 Sign-Off
> **Goal**: Package the entire system into a resilient 6-service Docker stack and verify cloud/local operating economics.

- [ ] **[20] Build Production Deployment Architecture & Checkpointing Stack** `(task)`  
  *Path*: [.scratch/geoprior/issues/20-production-deployment-stack.md](file:///c:/Users/Riteesh/Programming/SIH2K26/.scratch/geoprior/issues/20-production-deployment-stack.md)  
  *Status*: **BLOCKED by 11, 15, 17** 🔒  
  *Key Deliverable*: `docker-compose.yml` with Inference Worker (CUDA), Redis/Celery queue, MinIO storage, PostgreSQL immutable audit logs, MLflow, and Prometheus/Grafana.

- [ ] **[21] Finalize Cost Model, Legal Verification & Gate G3 Sign-Off** `(task)`  
  *Path*: [.scratch/geoprior/issues/21-cost-modeling-and-g3-signoff.md](file:///c:/Users/Riteesh/Programming/SIH2K26/.scratch/geoprior/issues/21-cost-modeling-and-g3-signoff.md)  
  *Status*: **BLOCKED by 20, 19** 🔒  
  *Key Deliverable*: Empirical local vs cloud GPU cost model, open-data licensing compliance sign-off, and formal G3 Production Candidate closure.

---

## ⚡ Active Frontier (What to Run Next)

Three tickets are currently unblocked and ready for immediate work:

1. 🎯 **Ticket 02**: `benchmarks/run_b1_b2.py` (VRAM Profiling & Benchmark Harness) $\leftarrow$ **Recommended first** to lock in Gate G1 feasibility.
2. 🛰️ **Ticket 04**: Implement GeoPrior Provider Engine (Overture/OSM/Cesium schema & provenance foundation).
3. ✂️ **Ticket 06**: Implement SSAKS 3-Stage Cascade (DIS Flow $\to$ DINOv2 $\to$ SAM2 + 30s calibration fallback).
