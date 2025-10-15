# ruff: noqa
# fmt: off
# %%
%load_ext rich
from pathlib import Path
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
d1 = s.data.blocks[3,0,2].compute()
d2 = s.data.blocks[4,0,2].compute()
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

