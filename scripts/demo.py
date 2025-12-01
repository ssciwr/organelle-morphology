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
from organelle_morphology import Project, merge_meshes
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import tempfile

from dask import compute

viridis = mpl.colormaps.get("viridis")
# %%
project_path = Path.cwd() / ".." / "example_analysis"
p = Project(project_path, compression_level="s2", loglevel="DEBUG", clipping=((0.4,0.4,0.4),(0.6,0.6,0.6)))
p.add_source("../data/Interphase_4T/mito_it00_b0_7_stitched.xml", "mito")
s = p.sources["mito_it00_b0_7_stitched"]
p.add_source("../data/Interphase_4T/er_it00_b0_7_stitched.xml", "er")
s = p.sources["er_it00_b0_7_stitched"]

# %%
l = int(s.labels[20])
d = s.meshes[l]
d.compute().show()


# %% clear memory, load from cache
s.clear_memory_cache()
print(len(s.labels))
c = p.get_caches()


# %% change compression
p.clipping = None
p.clipping = [[0.6,0,0], [1,1,1]]
p.compression_level = "s3"
print(len(s.labels))

mmesh = merge_meshes(list(s.meshes.values()), color=1).compute()
mmesh.show()

# %% debug colors
s = p.sources["mito_it00_b0_7_stitched"]
p.clipping = [[0.6,0,0], [1,1,1]]
s.calculate_mesh(debug_color=2)
mmesh = merge_meshes(list(s.meshes.values()), color=0).compute()
mmesh.show()

# %% weired cubes in the middle
s = p.sources["mito_it00_b0_7_stitched"]
p.clipping = [[0.3,0,0], [0.7,1,1]]
p.compression_level = "s2" # also on other levels
s.calculate_mesh(debug_color=1)
mmesh = merge_meshes(list(s.meshes.values()), color=0).compute()
mmesh.show()

# %% high-res
p.clipping = [[0.6,0.3,0.4], [0.8,0.8,1]]
p.compression_level = "s1"
print(len(s.labels))
mmesh = merge_meshes(list(s.meshes.values()), color=0).compute()
mmesh.show()

# %% Project API
p.clipping = [[0.6,0,0], [1,1,1]]
p.compression_level = "s3"

mmesh = p.merged_meshes(color=1)
mmesh.show()

# $$$$$$$$$$$$$
# %% Organelles
p.clipping = [[0.6,0,0], [1,1,1]]
p.compression_level = "s3"
s = p.sources["mito_it00_b0_7_stitched"]
label = 88000113
id_ = f"mito_{label}"
o = s.get_organelles(ids=id_)[0]

# %%
# p.skeletonize_vertex_clusters(id_)
p.skeletonize_wavefront(id_)
o.skeleton.show()

o.mesh_properties
o.geometric_data

o.curvature_map
cm = mpl.colormaps["coolwarm"].copy()
norm = mpl.colors.Normalize(vmin=o.curvature_map.min(), vmax=o.curvature_map.max())
colors = cm(norm(o.curvature_map))
mesh = o.mesh.compute()
mesh.visual.vertex_colors = colors
mesh.show()

# %% Curvature
curv, meshes = s.get_curvature(label)
curv, meshes = s.get_curvature(None)






# %% ## Blender ##
# uses mmesh
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

