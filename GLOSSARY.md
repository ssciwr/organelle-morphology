# Organelle Morphology

Analysis of segmented 3D microscopy images loaded from CebraEM/Mobie (bdv-n5)
projects. Volumes live on disk in n5/XML attribute conventions that differ from
the order the library works in, so axis order is a first-class concept here.

## Language

**Array order**:
The order x, y, z = first, second, third dimension of a volume as z5py hands
it back. Everything inside the library uses array order: metadata tuples
(`size`, `resolution`, `downsampling`), index-based axis arguments, the
mesh/scene/export coordinate frame, and user-facing axis names. The library's
"x" is the first array dimension, which is not necessarily the physical x of
the specimen.
_Avoid_: zyx, row-major, "physical order"

**Disk order**:
The order of n5 and XML attributes on disk (`size`, `resolution`,
`downsamplingFactors`). z5py reverses array shapes and chunks relative to disk
when loading, but passes attributes through unreversed, so disk order is the
reverse of array order. Disk order is converted to array order exactly once,
at metadata load.
_Avoid_: xyz (ambiguous: often used for both orders), "on-disk order"
