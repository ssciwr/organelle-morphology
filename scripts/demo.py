# ruff: noqa
# fmt: off
# %%
%load_ext rich
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm
import trimesh
from trimesh import Trimesh
from zmesh import Mesh, Mesher
from organelle_morphology import Project
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from dask import compute
from organelle_morphology.util import disk_cache

viridis = mpl.colormaps.get("viridis")
# %%
project_path = Path.cwd() / "example_analysis"
p = Project(project_path, compression_level="s2", loglevel="DEBUG", clipping=((0.4,0.4,0.4),(0.6,0.6,0.6)))
p.add_source("../data/Interphase_4T/mito_it00_b0_7_stitched.xml", "mito")
# p.add_source("../data/Interphase_4T/er_it00_b0_7_stitched.xml", "er")
# s = p.sources["er_it00_b0_7_stitched"]
s = p.sources["mito_it00_b0_7_stitched"]

# %%
l = int(s.labels[20])
d = s.meshes[l]
d.compute().show()


# %% clear memory, load from cache
s.clear_memory_cache()
print(len(s.labels))


# %% change compression
p.clipping = None
p.compression_level = "s3"
print(len(s.labels))

meshes = compute(*s.meshes.values())
mmesh = Trimesh()
for i, mesh in enumerate(tqdm(meshes)):
    # mesh.visual.face_colors = trimesh.visual.random_color()
    mesh.visual.face_colors = viridis.resampled(len(mesh.faces)).colors
    mmesh += mesh
mmesh.show()

# %%
broken = trimesh.repair.broken_faces(mmesh, color=(255,0,255,255))
mmesh.show()

mask = np.ones(mmesh.faces.shape[0], dtype=bool)
mask[broken] = False
mmesh.update_faces(mask)
mmesh.show()

# %% TODO: WHY IS THIS NECESSARY?????
# should be done while merging. after combining into one necessary again
mmesh.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=2, digits_norm=2, digits_uv=2)
broken = trimesh.repair.broken_faces(mmesh, color=(255,0,255,255))
mmesh.show()

# %%

# %%

# %%
# %%
old_labels = s.labels
p.compression_level = "s2"
p.clipping = None
p.clipping = (0.5,0,0),(1,1,1)
s.clear_memory_cache()

print(len(s.labels))

meshes = compute(*s.meshes.values())

for mesh in meshes:
    mesh.visual.face_colors = trimesh.visual.random_color()
mmesh = Trimesh()
for mesh in tqdm(meshes[:]):
    mmesh += mesh
mmesh.show()

_n_chunk_ids = defaultdict(list)
for label, chunks in s._ids_to_chunks.items():
    _n_chunk_ids[len(chunks)].append(label)


# %% slice and glue approach
p.compression_level = "s3"
l = int(s.labels[20])
d = s.meshes[l]
m = d.compute()
mean = np.mean(m.vertices, axis=0)
dir = (1,1,0)
mm = m.slice_plane(mean, dir)

## slicing now implemented in _block_mesher



# %%
cm = mpl.colormaps.get("tab10")
ax = plt.figure().add_subplot(projection='3d')
for i, m in enumerate(meshes[:1]):
    ax.plot_trisurf(
        m.vertices[:,0,],
        m.vertices[:,1,],
        m.vertices[:,2,],
        triangles=m.faces,
        color=cm.colors[i%10]

    )
plt.show(block=False)


# %% ## Blender ##
mmesh
import bpy

er = bpy.data.meshes.new(name="er")      
# er.vertices.foreach_set("co", verts) would be faster
er.from_pydata(mmesh.vertices, mmesh.edges, mmesh.faces)
er.update()
er_obj = bpy.data.objects.new("er_mesh", er)
bpy.context.scene.collection.objects.link(er_obj) 

bpy.ops.object.select_all(action='DESELECT')
er_obj.select_set(True)
bpy.context.selected_objects

# %%
bpy.ops.object.editmode_toggle()
bpy.ops.mesh.remove_doubles(threshold=0.001)
bpy.ops.object.editmode_toggle()

verts = np.empty(len(er.vertices)*3, dtype=np.float64)
er.vertices.foreach_get("co", verts)
verts = verts.reshape((-1, 3))

faces = np.empty(len(er.polygons)*3, dtype=np.int32)
er.polygons.foreach_get("vertices", faces)
faces = faces.reshape((-1, 3))

# %%
bmesh = Trimesh(vertices=verts, faces=faces)
bmesh.show()

# %%
with open("/home/kriedmiller/test/mmesh_overlap.stl", "wb") as f: 
    f.write(trimesh.exchange.stl.export_stl(mmesh))


# %% ### Merge seams ###
s.calculate_mesh(reduction_factor=0, overlap=True)
mmesh = Trimesh() # all meshes, but should be only the overlapping
for mesh in tqdm(meshes[:]):
    mmesh += mesh
mmesh.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=1, digits_norm=1, digits_uv=1)
np.unique(mmesh.unique_faces(),return_counts=True)
mmesh.update_faces(mmesh.unique_faces())

# %%

