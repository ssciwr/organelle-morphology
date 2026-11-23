# 02 — Pair gate & distance matrix: conservative surface bounds

Status: ready-for-agent
Spec: `../spec.md` · ADR: `docs/adr/0002-mcs-per-organelle-surface-distance.md`
Depends on: 01 (shared geometry: AABB cache, sampling radius)

Make the pair gate never miss a contact, and stop conflating the distance band
with the pair's closest approach.

## Scope

- `get_min_dist` (`distance_calculations.py:28–32`): bidirectional
  vertex-to-surface minima via cached AABB trees — `d_AB` (A vertices → B
  surface), `d_BA`; store `min(d_AB, d_BA)`. This is an **upper bound** on the
  true surface minimum (error ≤ max(h_A, h_B)); document it as such wherever
  the matrix is exposed.
- Rigorous lower bound per pair: `max(d_AB − h_A, d_BA − h_B)` (h from
  ticket 01).
- `generate_mcs` pair filter (:609–611): keep pair iff lower bound ≤
  max_distance. **Remove min_distance from the pair filter** — the band is
  applied locally by the geometry core.
- Keep domain decomposition (`make_domains`, `chunk_domain_masks`), cache
  keys (`distance_matrix`, `max_distance_computed`), and the public API
  (`generate_distance_matrix`, `project.distance_matrix`).
- Vectorize cheap loops (`_check_overlap`, mask building) where it doesn't
  change behavior; leave the process-pool sizing alone unless benchmarked.

## Acceptance

- 0.05-gap icospheres at max_dist 0.1: pair present in the matrix and passed
  to the MCS stage (regression for the missed-pair bug).
- Pair with global minimum 0.002 and other regions at 0.05, band [0.01, 0.1]:
  pair kept, 0.05 region reported (min_distance no longer drops pairs).
- Matrix symmetric; cached values identical on recompute; DD and non-DD
  paths agree (existing test structure, new expected values).
- Lower bound never exceeds the true minimum on analytic fixtures.

## Comments
