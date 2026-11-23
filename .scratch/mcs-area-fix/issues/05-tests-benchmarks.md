# 05 — Validation: analytic ground truth & benchmarks

Status: resolved
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

- 2026-11-23 — Resolved on integration branch `fix/mcs-area-fix` (merge `7735bf4`; branch `fix/mcs-area-fix-05` @ `a852ae5`). `tests/test_mcs_accuracy.py` replaces the old hardcoded expectations with an analytic contract: parallel planes exact (`rel=1e-9`), equal/unequal spheres and an annular band within 2% at 10 242-vertex icospheres (measured 0.04–0.7%), refinement convergence (55% → 13% → 2.6% → 0.66% for subdivisions 2–5), swap symmetry, and the 0.05-gap missed-pair regression at `max_dist=0.1`. The unused `tests/assets/mcs_df.csv` was briefly removed during this ticket and restored by the review fix; no test or doc asserts old-algorithm values (the ADR's `0.224` is the rejected old behaviour).

  **Benchmark** (`example_analysis`, demo.py settings: s2, clipping `[[0.6,0.5,0.5],[1,1,1]]`, mito×ER, `search_mcs(0.05)`, 8 workers, cold remesh; old baseline measured on this machine before the fix):

  | stage | old | new |
  |---|---:|---:|
  | project load | 1.7 s | 2.1 s |
  | mesh dispatch / compute | 9.7 s / 2.7 s | 15.4 s / 2.4 s |
  | distance matrix (1507², max 0.05) | 227.9 s | **118.8 s** |
  | MCS stage (`search_mcs`) | 4.8 s | **40.5 s** |
  | end-to-end | 254.5 s | **179.2 s** (0.70×) |
  | parent peak RSS | 1.68 GB | 1.63 GB |

  Target met: new MCS stage 40.5 s ≪ old end-to-end 254.5 s; new end-to-end is also 1.42× faster. The new MCS stage is 8.4× the old MCS stage; accepted because the spec target is the end-to-end comparison and the cost is the documented depth-3 refinement on one 370k-face apposition (32.7 s of 48.5 s in-process; all other 93 gate pairs ≤ 1 s). Default subdivision depth unchanged. Gate: 94 candidates → 65 pairs with patches → 68 records, 1 846 patches, 6.87 µm². AABB/geometry cache: 86.6 MB cached arrays after the MCS stage (1 573 meshes, 1.33M faces); worker RSS max 1.30 GB.

  **Orientation**: `example_analysis/exported_meshes` contains chunk fragments (161 glb; 19 non-empty, only 1 watertight), not merged organelle meshes as the ticket assumed. All 19 non-empty meshes are winding-consistent, the watertight one has positive signed volume, and a local raw-segmentation occupancy test finds 19/19 ≥ 95.9% outward (mean 98.5%). Spot-checked benchmark organelles are consistent (watertight mito volume +0.0319 µm³). No inversion → merging path untouched; no global repair. Data-dependent tests skip cleanly when the external data is absent.
