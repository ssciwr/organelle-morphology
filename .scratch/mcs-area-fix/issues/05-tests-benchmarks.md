# 05 — Validation: analytic ground truth & benchmarks

Status: ready-for-agent
Spec: `../spec.md` · ADR: `docs/adr/0002-mcs-per-organelle-surface-distance.md`
Depends on: 01–04 (geometry tests may start alongside 01)

Replace tests that hardcode the flawed algorithm's outputs
(`tests/test_distance_calculations.py:204–244` min_dist 88.9775,
`:339–355` mean_area≈240.60 / n_contacts==7) with an analytic accuracy
contract, and benchmark the speed goal.

## Scope — accuracy

- **Parallel planes**: two triangulated squares (side L), gap g inside the
  band → contact area L² per side (boundary effects excluded by construction).
- **Spheres**: radii r_A, r_B, center distance D; distance from a point on A
  at polar angle θ to B's surface = `sqrt(D² + r_A² − 2·D·r_A·cosθ) − r_B`;
  in-band zone area closed-form in cosθ → exact expected area per side.
- **Missed-pair regression**: 0.05-gap icospheres, max_dist 0.1 → pair found,
  both sides' areas within subdivision tolerance of analytic.
- **Refinement convergence**: subdividing the input meshes converges the
  area (error shrinks with face size; no systematic drift).
- **Swap symmetry**: (A, B) vs (B, A) identical per-side results.
- **U-case**: sphere 1 MCS, deep U 2 MCS (counts differ across sides).
- **Clipping**: clipped fixture → rim contacts included, `truncated` set;
  unclipped control → not set.
- **Orientation**: check outward winding fraction on merged/qslim real-data
  meshes (synthetic verified 100%). If real meshes are inverted, fix the
  merging path — do not reintroduce global repair.

## Scope — speed

- Benchmark on `example_analysis` with demo.py settings (s2, clipped,
  mito×ER; `src/scripts/demo.py:103–124`): old vs new end-to-end and per-pair
  timings; report AABB cache memory.
- Target: new MCS stage no slower than old end-to-end (point-to-surface
  queries measured 2.4× faster than the old KDTree path; subdivision adds
  work only on boundary faces). If missed, profile before tuning defaults
  (subdivision depth is the first knob).

## Acceptance

- All analytic cases pass at the documented tolerance; no test asserts a
  value produced by the old algorithm.
- Benchmark numbers recorded in this ticket's Comments (baseline + new).
- `pytest` green across the suite.

## Comments
