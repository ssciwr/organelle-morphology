# MCS: per-organelle patches, point-to-surface distance

The old contact-site calculator derived both organelles' contact areas from a
single query direction (a vertex-count/watertightness heuristic picked one
"target" mesh), measured vertex-to-vertex distances, and summed whole faces
touching any in-band vertex. Results depended on argument order, areas were
inflated, each pair yielded exactly one record, and pairs could be missed
entirely: two 10k-vertex spheres with a true surface gap of 0.05 report a
minimum distance of 0.224 under vertex-to-vertex, so a 0.1 threshold skips
them. We decided that an MCS is an edge-connected patch on **one** organelle's
own membrane, classified by exact point-to-surface distance to the partner
(libigl AABB), with each side computed independently and identically;
symmetry is a property of the protocol, not of the individual query.

## Considered Options

- **Vertex-to-vertex KDTree (status quo)**: rejected — resolution-dependent
  overestimate (missed-pair example above), and slower than libigl
  point-to-surface (36.7 ms vs 15.6 ms per 10k queries, tree build included).
- **Union graph over both sides' faces (one shared two-sided MCS)**: rejected —
  an MCS is a property of a single organelle; a U-shaped ER facing a sphere
  must yield one MCS on the sphere and two on the U (given the U is deep enough
  that its arm patches do not connect). A shared entity forces equal counts
  and merges biologically distinct patches.
- **Exact triangle-triangle distance sweep for the pair gate**: rejected as
  unnecessary — bidirectional vertex-to-surface minima minus each mesh's
  sampling radius form a rigorous lower bound on the true minimum, from the
  same queries, cheaper.
- **Whole-face area inclusion**: rejected — inflates patch boundaries by up to
  a face each; boundary faces are adaptively subdivided instead (depth ≤ 3).

## Consequences

- The two sides of one apposition legitimately differ in MCS count and area
  (a bump's spherical cap vs the wider in-band disk on the apposed sheet).
  This is geometry, not a bug; do not "fix" it by symmetrizing.
- Distance-matrix values are upper bounds on the true surface minimum (error
  ≤ sampling radius). They gate pairs conservatively (rigorous lower bound
  ≤ max_distance); min_distance filters surface points, never pairs.
- Pre-fix McsData records are invalid: McsMetadata carries a schema version
  and old YAML is rejected on load. Record dedup ignores data, so without
  this, wrong areas would silently persist alongside correct ones.
- Global mesh repair (`fix_inversion`/`fix_normals`) is removed: zmesh winding
  is outward and survives clipping (verified on synthetic closed and clipped
  meshes); volume-based repair is unsafe on open meshes. Clip rims participate
  in MCS, but records whose patch touches the rim zone are flagged `truncated`.
