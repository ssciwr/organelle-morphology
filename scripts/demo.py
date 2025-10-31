# ruff: noqa
# fmt: off
# %%
%load_ext rich
from pathlib import Path

from zmesh import Mesh, Mesher
from organelle_morphology import Project
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from organelle_morphology.source import _block_mesher

from dask import compute
# %%
p = Path.cwd() / "example_analysis"
p = Project(p, compression_level="s2", loglevel="DEBUG", clipping=((0.4,0.4,0.4),(0.6,0.6,0.6)))
p.add_source("../data/Interphase_4T/er_it00_b0_7_stitched.xml", "er")
s = p.sources["er_it00_b0_7_stitched"]

# %%
label = 64400066
meshes = s.get_meshes_from_id(label)
mmesh = s.merge_meshes(meshes)


# %%
cm = mpl.colormaps.get("tab10")
ax = plt.figure().add_subplot(projection='3d')
for i, m in enumerate(compute(*meshes)):
    ax.plot_trisurf(
        m.vertices[:,0,],
        m.vertices[:,1,],
        m.vertices[:,2,],
        triangles=m.faces,
        color=cm.colors[i%10]

    )
plt.show(block=False)

cm = mpl.colormaps.get("tab10")
ax = plt.figure().add_subplot(projection='3d')
ax.plot_trisurf(
    mmesh.vertices[:,0,],
    mmesh.vertices[:,1,],
    mmesh.vertices[:,2,],
    triangles=mmesh.faces,
    color=cm.colors[0]

)
plt.show(block=False)

# %%
# From raw data: closed! Chunking issue, missing faces
# TODO: This is best approach, but stacking necessary
chunks = s.ids_to_chunks[label]
d1 = s.data.blocks[chunks[0]].compute()
d2 = s.data.blocks[chunks[1]].compute()
block = np.concatenate([d1,d2])
meshes, labels = _block_mesher(block, (0, 0, 0))

m = meshes[label].compute()

cm = mpl.colormaps.get("tab10")
ax = plt.figure().add_subplot(projection='3d')
ax.plot_trisurf(
    m.vertices[:,0,],
    m.vertices[:,1,],
    m.vertices[:,2,],
    triangles=m.faces,
    color=cm.colors[0]

)
plt.show(block=False)

# %%
# chunked meshes miss faces
ids_to_fix = {k:v for k,v in s.ids_to_chunks.items() if len(v) > 1}
chunks = ids_to_fix[label]

# %% merge overlap
tmesh.vertices

import trimesh
tmesh = trimesh.Trimesh(mmesh.vertices, mmesh.faces, process=False)
tmesh.merge_vertices()
tmesh.show()

# %% duplicates?
np.unique(tmesh.vertices, axis=0, return_counts=True)

# %% fix while merging??
# fixing meshes does not work as there are less verts than raw points
label = 64400066
meshes = s.get_meshes_from_id(label)
mmesh = s.merge_meshes(meshes)
verts, faces = get_faces_verts(compute(*meshes))
m = Mesh(vertices=verts, faces=faces)

offset = np.min(verts, axis=0)
upper_corner = np.max(verts, axis=0) # inclusive

shape = ((upper_corner+0.5) - offset)*2  # round up, and include max corner
shape = np.array(np.round(shape), dtype=int)
space = np.zeros(shape)
idxs = np.array(np.round((verts-offset)*2), dtype=int)
idxs = [tuple(idx) for idx in idxs]
for idx in idxs:
    space[idx] = label

mesher = Mesher((1,1,1))
mesher.mesh(space, close=False)
new_mesh = mesher.get(mesher.ids()[0],
                      normals=False,
                    reduction_factor=0,
                      voxel_centered=False,
                      max_error=None,
                      )
new_mesh.viewer()
