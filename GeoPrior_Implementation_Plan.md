# GeoPrior — Implementation Plan v2 (Fortified Against v1.0 Spec)

## What Changed From the Previous Plan

The uploaded v1.0 document (Ch. 1–15) is a real step forward — it operationalizes every DD from the review thread with actual schemas, metrics, and a 10-item failure catalogue. This version reconciles the previous risk-ordered plan against that spec's exact artifacts, and fixes one structural problem in the spec's own roadmap: **Chapter 15's 12-week roadmap puts "Verification benchmarks" in Phase 6 (weeks 11–12) — last.** Chapter 12's entire Verification Matrix and Chapter 8's entire Acceptance Benchmarks table are marked "Pending" with no earlier checkpoint. That means the plan as written would spend 10 weeks building SSAKS → VGGT → Geo Registration → Speedy-Splat before finding out whether Speedy-Splat fits in 6GB at all. This is the exact ordering risk this whole review thread exists to catch — the fix is to pull G1 forward, not to add more design chapters.

**Also missing from the 12-week roadmap entirely:** Chapters 13 (Failure Modes), 14 (Production Deployment), and 15 (Cost/Licensing) have no weeks allocated to them anywhere in the Phase 1–6 table, despite Chapter 12's own "Pending Validation Register" assigning them named owners (Engineering, Research, Product/Infra) as required deliverables. A roadmap that specs 10 failure modes and a 6-service deployment stack but budgets zero weeks for either isn't a 12-week plan — it's a 12-week plan for chapters 1–10 with three unscheduled chapters bolted on.

Below is the reconciled plan: same risk-first ordering as before, now using the spec's real gates, schemas, and metrics.

---

## Gate Structure (from Ch. 15, now wired into the timeline)

| Gate | Definition | This plan's position |
|---|---|---|
| G0 | Architecture Complete | Already done — this is what the uploaded v1.0 doc represents |
| **G1** | **Feasibility Verified** (VRAM, runtime, registration benchmarks) | **Moved to Weeks 1–2**, not deferred to the end |
| G2 | Robustness Verified (failure-mode + calibration reports) | Week 14–17, running continuously once keyframes/pose exist |
| G3 | Production Candidate (infra, monitoring, legal review) | Weeks 18–22, parallel administrative + infra track |

---

## Phase 0 (= pulling G1 forward) — Weeks 1–2

**Goal:** answer Chapter 8's "Acceptance Benchmarks" table and Chapter 12's B1/B2 benchmark suite *before* anything is built on top of Speedy-Splat.

**Build:** the exact benchmark harness Ch. 8 already specifies — Peak allocated/reserved VRAM, mean/P95 training runtime, Gaussian-count-vs-memory curve, first-preview latency — run on real RTX 3050 hardware against a small public dataset, at the B2 suite's own three flight lengths (1/5/10 min equivalent).

**Learning required:** 3DGS internals (Kerbl et al.), gsplat memory profiling (`torch.cuda.memory_stats`), Speedy-Splat paper.

**Known pitfalls (unchanged from earlier review):** don't rely on "gradient checkpointing" as a memory technique — it doesn't map cleanly onto per-primitive Gaussian optimization; use gsplat-native pruning/densification caps/mixed precision instead, and measure them, don't assume them.

**Exit criteria:** Chapter 8's Acceptance Benchmarks table filled in with real numbers, not "Pending." If 6GB doesn't hold at target scene sizes, decide the scope change *now* — this decision changes Phases 4–5 below.

---

## Phase 1 — GeoPrior Provider Architecture (Ch. 4) — Weeks 2–3

**Build against the spec's actual provider priority** (already correctly reordered from the original doc): P1 Overture Maps, P2 Cesium World Terrain, P3 OpenStreetMap, P4 Google (metadata-enrichment only). Implement the exact unified schema from Ch. 4 (`coordinate_system`, `terrain.elevation_grid`, `buildings.polygons`, `imagery.source`, `confidence.terrain`, `confidence.buildings`).

**Provenance starts here**, using the Ch. 11 schema exactly:
```
primitive_id, provenance, observation_confidence, source_views, geo_prior_source, ai_inference
```
Every object GeoPrior emits gets `provenance: prior_guided`, `geo_prior_source` set, `ai_inference: false` from creation.

**Exit criteria:** schema conformance tests; fallback chain verified under simulated provider failure; every emitted object carries a valid, complete provenance record per the Ch. 11 schema.

---

## Phase 2 — SSAKS (Ch. 5) — Weeks 3–4

**Build the exact three-stage cascade specified**: Stage 1 (blur/optical-flow/exposure) → Stage 2 (DINOv2 novelty clustering) → Stage 3 (SAM2 semantic coverage verification). Calibration (DD-02) per Ch. 5: triggered from the first *stabilized cruise segment* (IMU stability + altitude convergence + motion variance), not a fixed time window — already correctly specified in this version, carry it through unchanged.

**Run Ch. 12's B5 ablation study now, not later**: Fixed-FPS baseline vs. optical-flow-only vs. flow+DINOv2 vs. full cascade. This answers the open question from the earlier review ("is SSAKS better than fixed FPS sampling?") with real numbers instead of leaving it as an assumption.

**Exit criteria:** 85–97% frame reduction target (Ch. 5's own stated range) verified per real flight, not the illustrative 18,000→550 example; ablation report comparing all four configurations.

---

## Phase 3 — Hybrid Pose Estimation + Geo Registration (Ch. 6–7) — Weeks 4–6

**Build Ch. 6's condition table exactly**: VGGT alone at high confidence; VGGT-init + COLMAP bundle adjustment at low confidence; DUSt3R depth prior + COLMAP refinement for poor-GPS/texture-poor scenes. Instrument all four Ch. 6 confidence metrics (pose confidence score, reprojection residual, feature-track consistency, camera graph connectivity) — these need real thresholds defined during this phase, not left implicit.

**Geo Registration Module (Ch. 7)**: implement the exact 5-step pipeline (GPS-tagged camera centers → Umeyama 7-DoF → RANSAC outlier rejection → global bundle adjustment → georeferenced poses + point cloud). Test against synthetic ground truth first, then real surveyed points, per Ch. 12's B3 suite (translation RMS, rotation error, scale residual, inlier ratio).

**Exit criteria:** Ch. 7's Verification Metrics filled with real numbers; Ch. 12's B3 benchmark suite complete.

---

## Phase 4 — Gaussian Reconstruction (Ch. 8) — Weeks 6–9

**Build:** wire Phase 0's already-benchmarked config into the real pipeline stages Ch. 8 defines (Initialization → Optimization → Checkpointing → Final Scene). Checkpointing stage explicitly supports Live Mode per Ch. 8 — implement Live/Offline exactly as Ch. 3's DD-01 table specifies: Live Mode is ephemeral/stateless, Offline Mode is canonical and always restarts from raw video + SSAKS keyframes, never from Live Mode's optimizer state.

**Exit criteria:** re-run Phase 0's benchmarks with *real* SSAKS-selected keyframes (not the clean benchmark dataset) — flag any divergence; Ch. 8's full Acceptance Benchmarks table (VRAM, runtime, Gaussian-count-vs-memory, preview latency) now reflects production-representative data.

---

## Phase 5 — Mesh Extraction (Ch. 9) + SERE (Ch. 10) — Weeks 9–12

**Mesh extraction now has more steps specified than previously assumed** — Ch. 9's pipeline is: Canonical Gaussian Scene → SuGaR regularization → surface extraction → **topology cleanup → UV generation → texture baking → LOD generation**. The earlier plan treated this as roughly one stage; it's five, and UV/texture-baking/LOD are each their own source of bugs (seam artifacts, texture bleeding, LOD popping) that need independent testing.

**Run Ch. 12's B4 suite**: Chamfer Distance, Hausdorff Distance, normal consistency, watertightness — against LiDAR or manual measurement per Ch. 9's Quality Metrics table.

**SERE (Ch. 10)**: plane detection (RANSAC), line detection (LSD + Hough — note LSD, not just generic "deep line detection" as in the earlier draft, is now specified), vertex snapping via constraint optimization, topology repair. Feed it the Phase 3 registration confidence — Ch. 10's input is explicitly "SuGaR mesh + confidence map," so this dependency is now spec'd, not something to remember to add.

**Exit criteria:** Ch. 9's Quality Metrics table and Ch. 12's B4 suite both populated with real values; SERE output includes edge-quality metadata per Ch. 10's stated output.

---

## Phase 6 — Provenance Engine Completion (Ch. 11) — Weeks 8–9 (parallel with Phase 4)

**Build the exact reclassification rule from Ch. 11**: promotion from Prior-Guided → Observed requires BOTH (1) evidence from ≥2 independent camera views AND (2) observation-confidence exceeding a calibrated threshold derived from reprojection consistency. Write the adversarial unit test flagged in the earlier review: a Gaussian nudged by gradient descent but still primarily constrained by the prior must not promote.

**Export filtering policy (Ch. 11, now exact):** Observed → exported by default; Prior-Guided → excluded unless visualization mode enabled; AI-Inferred → exported only with inference mode enabled and metadata tagged.

**Exit criteria:** promotion-rule test suite passes including the adversarial case; export filter integration test (all-Prior-Guided scene exports empty by default).

---

## Phase 7 — Failure Mode Implementation (Ch. 13) — Weeks 12–17

Chapter 13 already specifies detection + recovery for all 10 failure modes — this phase is "implement exactly what's written," not "design it." Build in residual-risk order (High → Medium → Low), since that's the order that most affects export trustworthiness:

| Priority | Failure mode | Detection → Recovery (as specified) |
|---|---|---|
| High | Reflective glass | Depth inconsistency → confidence masking |
| High | Water surfaces | Optical flow instability → exclude low-confidence regions |
| High | Low-overlap flight | Coverage score → warn operator / preserve uncertainty |
| Medium | Rolling shutter | IMU inconsistency + line distortion → correction before SSAKS |
| Medium | Dynamic vehicles/people | SAM2 segmentation → mask before reconstruction (reuse Phase 2's SAM2 pass, don't re-run it) |
| Medium | GPS multipath | High registration residual → RANSAC + bundle adjustment (already built in Phase 3 — this is a *detection trigger* on existing output, not new registration logic) |
| Medium | Thermal throttling | GPU telemetry → reduce tile size |
| Medium | Missing EXIF/telemetry | Metadata validation → pose-only reconstruction |
| Low | Motion blur | Blur score during SSAKS → discard/down-weight (already built in Phase 2 Stage 1 — verify it's wired to this recovery behavior) |
| Low | CUDA OOM | GPU memory monitor → resume from checkpoint |

**Exit criteria:** each row has a test case (synthetic or curated real footage) and a filled-in Residual Risk Matrix entry (Ch. 13), replacing "Pending Validation" with measured outcomes. This is the deliverable the Ch. 12 register assigns to "Engineering" as a "Robustness matrix" — treat that as the literal exit artifact.

---

## Phase 8 — Production Deployment (Ch. 14) — Weeks 17–20

**Build the exact stack Ch. 14 specifies** — no further design needed here, it's already concrete:

| Component | Technology | Notes |
|---|---|---|
| Inference Worker | Docker + CUDA | SSAKS, VGGT, Speedy-Splat execution |
| Artifact Storage | MinIO/S3 | Videos, splats, meshes, metadata |
| Queue | Redis + Celery | Async reconstruction jobs |
| Model Registry | MLflow | Checkpoint/model versioning |
| Monitoring | Prometheus + Grafana | GPU, runtime, failures |
| Metadata DB | PostgreSQL | Flight metadata + provenance records |

**Checkpointing boundaries** are specified in Ch. 14: after SSAKS, pose estimation, Gaussian optimization, mesh extraction, and refinement — implement resume-from-checkpoint at each of these five points, matching the CUDA OOM recovery behavior from Phase 7.

**Security principles from Ch. 14** to implement as code, not just policy: API keys outside containers, immutable provenance metadata (enforce at the DB/storage layer, not just convention), checksum verification for downloaded models, read-only export pipeline for provenance metadata.

**Exit criteria:** a killed-mid-training job resumes correctly from the nearest checkpoint; provenance metadata is verified immutable (attempt-to-modify test fails as expected).

---

## Phase 9 — Cost Model + Licensing (Ch. 15) — Weeks 20–22 (parallel administrative track)

**Cost model**: fill in Ch. 15's table (GPU compute, storage, geo providers, monitoring — local vs. cloud) with real figures once Phase 0/4 benchmarks give actual GPU-hour and storage-per-scene numbers.

**Licensing**: Ch. 15 states the policy correctly (Observed exported by default, Prior-Guided non-exportable by default, AI-Inferred requires retained metadata) — this is now a *software enforcement* task (Phase 6) plus a *legal sign-off* task. The "Pending Legal Review" status next to Google Maps Platform in Ch. 15's cost table is the same open item flagged throughout this whole review thread — get it actually reviewed by counsel here, don't let it inherit "Pending" status into a production release.

**Exit criteria:** G3 gate closed — cost model has real numbers, legal review status changes from "Pending Legal Review" to an actual sign-off or an explicit decision to drop Google as a provider.

---

## Reconciled Timeline

| Weeks | Phase | Ch. reference | Gate |
|---|---|---|---|
| 1–2 | 0: Benchmark harness (pulled forward) | Ch. 8, Ch. 12 B1/B2 | **G1 starts here, not week 11** |
| 2–3 | 1: GeoPrior + provenance foundation | Ch. 4, Ch. 11 | |
| 3–4 | 2: SSAKS + B5 ablation | Ch. 5, Ch. 12 B5 | |
| 4–6 | 3: Pose estimation + Geo Registration | Ch. 6–7, Ch. 12 B3 | G1 continues |
| 6–9 | 4: Gaussian reconstruction + Live/Offline | Ch. 3 DD-01, Ch. 8 | G1 closes |
| 8–9 | 6: Provenance engine (parallel with 4) | Ch. 11 | |
| 9–12 | 5: Mesh extraction + SERE | Ch. 9–10, Ch. 12 B4 | |
| 12–17 | 7: Failure modes | Ch. 13 | G2 |
| 17–20 | 8: Production deployment | Ch. 14 | G3 (infra half) |
| 20–22 | 9: Cost + legal | Ch. 15 | G3 (legal/cost half) |

**~22 weeks to G3**, vs. the spec's own 12-week roadmap — the difference is entirely the three unscheduled chapters (13–15) plus moving G1 to the front where a failed feasibility check is cheap instead of catastrophic. This isn't padding the timeline; it's the same total work the v1.0 document already committed to delivering (it names owners and deliverables for all of it in the Pending Validation Register) — this version just puts weeks next to it.
