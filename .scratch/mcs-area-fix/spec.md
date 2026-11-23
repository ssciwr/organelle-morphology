# MCS area calculation fix

Status: agreed — grilling session 2026-07-28, all decisions settled (Q1–Q20).
ADR: `docs/adr/0002-mcs-per-organelle-surface-distance.md`
Glossary terms: **MCS**, **Contact area**, **Distance band**, **Clipping**.

Goal: speed and accuracy of membrane contact site area calculation. An MCS is
defined by a distance band [min_distance, max_distance] between two organelle
meshes; one pair can form several MCSs; organelles may be clipped (open
meshes), very big, or very small.

## Problem

Current pipeline: `MembraneContactSiteCalculator` in
`src/organelle_morphology/distance_calculations.py`.

1. **Order-dependent, one-directional** (:184–295): a vertex-count /
   watertightness heuristic picks one "target" mesh; both sides' results are
   derived from target-vertex → nearest-source-vertex queries. Swapping the
   pair changes the answer.
2. **Vertex-to-vertex distance** (:239–242): overestimates surface distance,
   resolution-dependent. Measured: two 10k-vertex icospheres, true gap 0.05 →
   reported min 0.224. At `max_dist=0.1` the pair is missed entirely.
3. **Whole-face area** (:323–331): every face touching any in-band vertex is
   summed in full, on both sides, from different vertex sets → asymmetric,
   inflated areas.
4. **One record per pair**: `Organelle.add_mcs`
   (`organelle.py:261–277`) stores one entry per (label, partner); a second
   call overwrites. `McsData.n_contacts` counts partners, not patches.
5. **Fragile normal handling** (:265–295, :387–391): global dot-product-sign
   heuristic chosen by mean distance; volume-based `fix_normals` on open
   meshes.
6. **Pair gate conflates band with closest approach** (`generate_mcs`
   :609–611): pairs whose global minimum distance < min_distance are dropped
   even when other regions appose inside the band.
7. **Python-loop filters** (:272–295, :311–315, :337–345): per-index list
   comprehensions over arrays.

## Domain semantics (settled)

- **MCS** = edge-connected component of the in-band, facing patch on **one**
  organelle's own membrane. Per-organelle records; the two sides of an
  apposition have independent counts and areas (Q1a, Q3a, Q10, Q16).
- **Distance band** filters surface points locally, never pairs (Q9). Pairs
  are gated by `max_distance` only, via a conservative lower bound (Q6a).
- **Distance** = exact point-to-surface (closest point on partner triangles),
  computed in **both directions**, each side classifying its own points
  against the partner surface. Symmetry is a protocol property (Q2).
- **Facing test**: local, per sample point — face normal must point toward
  the partner's closest point: `dot(n, C − p) > 0`. No global sign heuristic
  (Q4b). Mesher winding is trusted; global repair dropped (Q18).
- **Area**: per-side contact area; boundary faces adaptively subdivided
  (midpoint, depth cap 3, configurable), (sub-)face membership by centroid
  distance + facing test; interior faces counted whole (Q11b, Q16, Q19).
- **Clipping**: meshes are open at clip planes, no cap faces. Rim contacts
  are included but the MCS is flagged `truncated` when its patch lies within
  `max_distance` of an open boundary edge (Q5, Q12b).
- **Cross-side link**: each MCS stores `partner_patch_ids` (byproduct of the
  closest-point map); scene coloring is per patch, not per partner (Q17).
- **Clean break**: new record schema with version; pre-fix YAML rejected on
  load; adjacent aggregation bugs fixed in the same pass; aggregate-only
  persistence kept (Q7a, Q13).
- **Constraints**: CPU only, laptop → HPC; libigl already a dependency
  (conda env `morph`, pip); no new dependencies; dask kept for distribution
  (Q8).

## Design

### Geometry core (ticket 01)

New module-level geometry, replacing `MembraneContactSiteCalculator`:

```
per organelle: cache igl.AABB tree over (V, F)   [~9.5 ms / 10k faces, reused]
per pair (A, B), each side S ∈ {A, B} with partner P:
  d, I, C = tree_P.squared_distance(V_P, F_P, sample_points_S)   # exact
  facing  = dot(face_normal_S, C − p) > 0
  member  = (min_dist ≤ sqrt(d) ≤ max_dist) and facing
  subdivide faces straddling a membership boundary (vertex vs centroid
    classification differs across a band edge or facing sign), depth ≤ 3;
    classify children by centroid
  area    = sum of member (sub-)face areas
  MCSs    = edge-connected components of member (sub-)faces
            (sibling edges included; report parent-face indices for rendering)
  truncated = component has a (sub-)face within max_distance of an open
            boundary edge (edges with one incident face)
  partner_patch_ids = partner components hit by the closest-face map I
  patch_id = deterministic, e.g. f"{self_id}~{partner_id}~{k}", components
            ordered by smallest parent face index
```

- Both directions use the same code path; no source/target roles.
- Per-mesh **sampling radius** `h`: max over faces of the covering radius of
  its 3 vertices (circumradius if circumcenter inside the face, else half the
  longest edge). Used by the gate; compute once per mesh, cache with the tree.
- All filters vectorized (boolean masks, no per-index Python loops).
- Mesh repair: drop `fix_inversion`/`fix_normals`; at most a local
  winding-consistency pass. Face normals as meshed.

### Pair gate & distance matrix (ticket 02)

- `get_min_dist` → bidirectional vertex-to-surface minima via cached AABB:
  `d_AB` (A vertices → B surface), `d_BA`; store `min(d_AB, d_BA)` — an upper
  bound on the true minimum, error ≤ max(h_A, h_B). Document as such.
- Rigorous lower bound: `max(d_AB − h_A, d_BA − h_B) ≤ true minimum`.
- `generate_mcs` pair filter: keep pair iff lower bound ≤ max_distance.
  **min_distance removed from the pair filter** (applied locally in ticket 01).
- Domain decomposition, cache keys (`max_distance_computed`), and public API
  kept. Vectorize `_check_overlap` / mask loops where cheap.

### Records & schema (ticket 03)

- `Organelle._mcs[label][partner]` → **list** of patch entries:
  `{patch_id, faces_index, distances, area, truncated, partner_patch_ids, meta}`.
- `McsData`: `n_contacts` = number of own patches (across partners, per label);
  add `n_partners`; area/distance aggregates over patches (distance mean
  area-weighted); `min_dist_per_org`/`max_dist_per_org` remain per partner
  (min/max over that partner's patches); normalizations unchanged.
- `McsMetadata`: add `schema_version` (current = 2); populate `filter_1`/
  `filter_2` from `mcs_meta`; take min/max from the label parameters, not the
  last loop entry.
- Record load: reject records with missing/old `schema_version` (warn + list).
  Clean break — no migration of old areas.
- Persistence stays aggregate-only (raw patch geometry in memory, as today).
  Per-patch persistence explicitly out of scope (separate ticket if needed
  for HPC→local patch transfer).

### Analysis & UI (ticket 04)

- `Mcs_Analysis.get_mcs_properties`: one row per (label, organelle) aggregate,
  as today; add `get_mcs_patches()` in-memory per-patch table.
- `get_mcs_overview`: weighting by patch counts; fix partner dedup to use
  sets/frozensets instead of mutating `partners` lists.
- `analysis.get_contact_properties`: include new fields (`n_partners`, …).
- `get_dataframe`: replace `rsplit("-", 1)` column parsing (breaks on labels
  containing `-`) with structured columns/MultiIndex.
- `Organelle.get_mesh_mcs_colored`: color per patch (tab20, i % 20).
- `src/app/ui.py`: verify tables/scene render with the new shape; no
  structural redesign.

### Validation & performance (ticket 05)

Accuracy contract — analytic ground truth, replacing the hardcoded values of
the flawed algorithm (`min_dist 88.9775`, `mean_area≈240.60`, `n_contacts==7`):

- Parallel triangulated planes, gap g in band → exact known area per side.
- Two spheres (radii r_A, r_B, center distance D): distance from a point on A
  at polar angle θ to B's surface is `sqrt(D² + r_A² − 2·D·r_A·cosθ) − r_B`;
  in-band zone area has a closed form in cosθ → exact expected areas.
- Regression: 0.05-gap icospheres at max_dist 0.1 must be found (old code
  misses them).
- Tessellation refinement: area converges under subdivision (no drift).
- Swap symmetry: (A, B) and (B, A) give identical per-side results.
- U-case fixture: sphere gets 1 MCS, deep U gets 2.
- Clipped fixture: rim contacts included, `truncated` set correctly.
- Orientation: merged/qslim real-data meshes checked for outward winding
  (synthetic verified 100%; if real data inverted, fix merging, not repair).

Performance: benchmark on `example_analysis` (demo.py settings: s2, clipped,
mito×ER) — baseline old vs new end-to-end and per-pair; target: new MCS stage
no slower than old end-to-end (queries are 2.4× faster; subdivision adds work
only on boundary faces). Memory: one AABB tree + h per organelle, cached.

## Established facts (do not re-derive)

- Benchmark (conda `morph`, 10k-vertex icospheres, 10k queries):
  KDTree vertex-vertex 36.7 ms / min 0.224; `igl.point_mesh_squared_distance`
  15.6 ms / min 0.050 (exact); `igl.AABB.init` 9.5 ms (reusable);
  `trimesh.nearest.on_surface` 694 ms (unusable).
  `igl.simplex_simplex_squared_distance` exists (single triangle pair).
- zmesh `close=False` winding is outward (100% on synthetic ball);
  `trimesh.intersections.slice_mesh_plane` preserves it; clipped mesh open
  (156 boundary edges on the fixture); current repair pipeline is a no-op on
  clean open meshes but volume-based and unsafe in general.
- Clipping never caps (`block_mesher.py:100–132`, no `cap=True`); no clip
  marker stored; open boundary edges identify the rim exactly.
- s0–s3 are data pyramid levels, not mesh simplification; meshes are qslim-
  simplified via `project.simplify` (default 0.5) in `merge_fragments_into_meshes`.
- Records: YAML under `analysis/McsData/{hash}.yaml`, dedup on meta only
  (data excluded, `records.py:289–305`), no version field anywhere today.
- `McsMetadata.filter_1/filter_2` never populated; min/max taken from last
  loop entry (`organelle.py:320,375–376`).
- Tests asserting flawed outputs: `tests/test_distance_calculations.py`
  (:204–244, :339–355), `tests/test_mcs_analysis.py`, `tests/test_analysis.py`.

## Tickets

| # | File | Scope | Depends on |
|---|------|-------|------------|
| 01 | `issues/01-geometry-core.md` | point-to-surface, facing, subdivision, components, areas, truncated, correspondence | — |
| 02 | `issues/02-gate-distance-matrix.md` | bidirectional minima, sampling radius, conservative gate | 01 (shared geometry) |
| 03 | `issues/03-schema-records.md` | patch lists, McsData/McsMetadata, versioning, invalidation | 01 (patch shape) |
| 04 | `issues/04-analysis-ui.md` | aggregation fixes, per-patch table/coloring, column parsing | 03 |
| 05 | `issues/05-tests-benchmarks.md` | analytic tests, regression, benchmarks, orientation check | 01–04 (geometry tests may start with 01) |

Worktree per issue under `../worktrees` (see `docs/agents/domain.md`).

## Out of scope

- Per-patch persistence / HPC→local patch-geometry transfer.
- GPU, new native dependencies, embree.
- Minimum patch-size filter or gap-based clustering (rejected as overkill).
- Symmetrizing the two sides' areas or counts (deliberately not done, ADR 0002).
- Distance-matrix replacement by spatial-index neighbor search (gate kept).
