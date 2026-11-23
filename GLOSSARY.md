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

**Membrane contact site (MCS)**:
A connected region on one organelle's membrane whose distance to a partner
organelle's membrane lies within a distance band. An MCS is a property of a
single organelle: the two sides of an apposition are separate MCSs, and their
counts can differ.
_Avoid_: "contact" (unqualified), "contact pair" (an MCS is a region, not a pair of organelles)

**Contact area**:
The area of an MCS's patch on its own organelle's membrane. The partner's
apposed patch has its own, generally different contact area; neither depends
on the order the two organelles are given to the calculation.
_Avoid_: "symmetric area" (the two sides are not forced equal)

**Distance band**:
The interval [min_distance, max_distance] applied to local surface-to-surface
distance: a surface point belongs to an MCS iff its distance to the partner
membrane lies in the band. It filters surface points, not organelle pairs.
_Avoid_: "distance threshold" (ambiguous which end)

**Clipping**:
Truncation of a source volume to a fractional bounding box. Organelle meshes
are cut open at the clip planes — no cap faces are created — so the cut rim is
artificial surface, not membrane.
_Avoid_: "cropping"
