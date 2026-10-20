# Axis convention: the library speaks array order

n5/XML attributes on disk are written in disk order (x, y, z), but z5py hands
volumes back reversed (x = first array dimension), while passing `resolution`
and `downsamplingFactors` attributes through unreversed; the loader mixed the
two, corrupting every physical measurement on anisotropic data and swapping
axis labels in the position analysis. We decided that everything in the
library — metadata tuples, index-based axis arguments, the mesh/scene/export
frame, and user-facing axis names — uses array order, with disk order
converted to array order exactly once inside metadata load.

## Considered Options

- **Canonical physical (disk/BDV) xyz order**: rejected because every
  array-touching site (regionprops spacing, centroid, mesh scaling, density
  binning, distance-domain decomposition, domain/clipping boxes) would need a
  per-site reversal — a large, easy-to-incomplete set.
- **Physical-xyz scene frame** (reorder mesh vertices at the mesher
  boundary): rejected because the reorder ripples into clipping corners,
  distance decomposition, and mesh-bound tests for no analysis benefit; the
  scene stays a dimension world (red/green/blue = first/second/third
  dimension).

## Consequences

- The library's "x" is the first array dimension, which for bdv-written data
  corresponds to the disk's z. User documentation must state this explicitly
  so nobody maps "x" onto the specimen's physical x by reflex.
- Guard assertions that compare XML and n5 metadata must compare them in disk
  order (both unreversed); they are the only line of defense against
  anisotropic fixtures regressing silently.
