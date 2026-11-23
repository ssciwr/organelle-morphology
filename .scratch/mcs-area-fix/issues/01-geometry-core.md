# 01 — Geometry core: point-to-surface MCS patches

Status: ready-for-agent
Spec: `../spec.md` · ADR: `docs/adr/0002-mcs-per-organelle-surface-distance.md`
Depends on: —

Replace `MembraneContactSiteCalculator`
(`src/organelle_morphology/distance_calculations.py:183–391`) with a
symmetric, per-side geometry core.

## Scope

- Per-organelle cached `igl.AABB` tree over (V, F); reuse across all partners.
  Build once per mesh (lazy, invalidated with the mesh cache).
- Per pair, each side S classifies its own sample points against the partner
  surface P: `sqrD, I, C = tree_P.squared_distance(V_P, F_P, points)`;
  `d = sqrt(sqrD)`. Both directions, same code path, no source/target roles.
- Facing test per sample point: `dot(face_normal_S, C − p) > 0`. Face normals
  as meshed; drop `fix_inversion`/`fix_normals` (keep at most a local
  winding-consistency pass).
- Membership: `min_distance ≤ d ≤ max_distance` and facing. Band filters
  surface points only.
- Adaptive midpoint subdivision (4 children) of faces straddling a membership
  boundary (vertex vs centroid classification differs across a band edge or
  facing sign), depth cap 3, configurable. Children classified by centroid.
  Interior faces counted whole.
- Contact area = sum of member (sub-)face areas, per side.
- MCS = edge-connected component of member (sub-)faces (sibling edges
  included). Report parent-face indices for rendering. No size filter.
- `truncated` flag: component has a (sub-)face within `max_distance` of an
  open boundary edge (edge with one incident face).
- `partner_patch_ids`: map closest-face indices I onto partner components.
- Deterministic `patch_id`: components ordered by smallest parent face index,
  `f"{self_id}~{partner_id}~{k}"`.
- Per-mesh sampling radius h (max over faces: circumradius if circumcenter
  inside, else half longest edge) — computed here, consumed by ticket 02.
- Vectorize everything: boolean masks, no per-index Python list comprehensions.

## Acceptance

- Two 10k-vertex icospheres, gap 0.05, band [0, 0.1]: contact found on both
  sides; areas match the analytic spherical-zone formula (ticket 05) within
  subdivision tolerance.
- Swapping (A, B) → (B, A) yields identical per-side results.
- U-case fixture: sphere 1 MCS, deep U 2 MCS.
- Clipped fixture: rim contacts included, `truncated` set.
- No `fix_inversion`/`fix_normals` calls remain in the MCS path.

## Comments
