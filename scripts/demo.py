# ruff: noqa
# fmt: off
# %%
%load_ext rich
from collections import defaultdict
from pathlib import Path
from time import time

from dask.delayed import delayed
from tqdm import tqdm
import trimesh
from trimesh import Trimesh
from zmesh import Mesh, Mesher
from organelle_morphology import Project, merge_meshes
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

from dask.base import compute
from dask.distributed import LocalCluster, Client
import organelle_morphology as om
from organelle_morphology.distance_calculations import generate_distance_matrix
from organelle_morphology.util import color_delayed_trimesh_rgba, show
from organelle_morphology.position import Position_Analysis
from organelle_morphology.analysis import Mcs_Analysis

viridis = mpl.colormaps.get("viridis")
# %%
project_path = Path.cwd() / ".." / "example_analysis"
p = Project(
    project_path,
    compression_level="s2",
    loglevel="DEBUG",
    clipping=((0.4,0.4,0.4),(0.6,0.6,0.6)),
    n_workers=4,
)
p.add_source("../data/old_cebraEM/Interphase_4T/mito_it00_b0_7_stitched.xml", "mito")
s = p.sources["mito_it00_b0_7_stitched"]
p.add_source("../data/old_cebraEM/Interphase_4T/er_it00_b0_7_stitched.xml", "er")
s = p.sources["er_it00_b0_7_stitched"]

# %%
l = int(s.labels[20])
d = s.meshes[l]
d.compute().show()


# %% clear memory, load from cache
s.clear_memory_cache()
print(len(s.labels))
c = p.get_caches()


# %% benchmark
p.clear_caches(True)
p.clipping = [[0.6,0.5,0.5], [1,1,1]]
p.compression_level = "s2"
t0 = time()
merge_meshes(list(s.meshes.values())).compute()
print("Done in: ", time() - t0)

# %% bench domain decomp
p.clear_caches(True)
p.clipping = [[0.6,0.5,0.5], [1,1,1]]
p.compression_level = "s2"
p.max_distance = 0.01
t0 = time()
generate_distance_matrix(project=p,domain_decomposition=True)
t_dd = time() - t0
print("Done in: ", t_dd )

p.clear_caches(True)
p.clipping = [[0.6,0.5,0.5], [1,1,1]]
p.compression_level = "s2"
p.max_distance = 0.01
t0 = time()
generate_distance_matrix(project=p,domain_decomposition=False,chunk_dd=True)
t_chunks = time() - t0
print("chunks: ", t_chunks )
print("domains: ", t_dd )


p.clear_caches(True)
p.clipping = [[0.60,0.45,0.5], [0.75,0.8,0.6]]
p.compression_level = "s1"
p.max_distance = 0.01
t0 = time()
generate_distance_matrix(project=p,domain_decomposition=True)
t_dd_s1 = time() - t0
print("Done in: ", t_dd_s1 )

p.clear_caches(True)
p.clipping = [[0.60,0.45,0.5], [0.75,0.8,0.6]]
p.compression_level = "s1"
p.max_distance = 0.01
t0 = time()
generate_distance_matrix(project=p,domain_decomposition=False,chunk_dd=True)
t_chunks_s1 = time() - t0
print("chunks: ", t_chunks_s1 )
print("domains: ", t_dd_s1 )


# %% benchmark
p.clear_caches(True)
p.clipping = [[0.6,0.5,0.5], [1,1,1]]
p.compression_level = "s2"
p.calculate_meshes()
t0 = time()
compute(list(s.meshes.values()))
print("First: ", t1:=time() - t0)
t0 = time()
compute(list(s.meshes.values()))
print("Second: ", t2:=time() - t0)
t0 = time()
compute(list(s.meshes.values()))
print("Third: ", t3:=time() - t0)

t0 = time()
p.search_mcs(0.05, ids_filter_1="mito*", ids_filter_2="er*")
print("mcs: ", t4:=time() - t0)
print("\n", t1)
print(t2)
print(t3)
print(t4)

# %% change compression
p.clipping = None
p.clipping = [[0.6,0.5,0.3], [0.8,1,0.8]]
p.compression_level = "s3"
print(len(s.labels))

mmesh = merge_meshes(list(s.meshes.values()), color=1).compute()
show(mmesh)

# %% debug colors
s = p.sources["mito_it00_b0_7_stitched"]
p.clipping = [[0.6,0,0], [1,1,1]]
s.calculate_mesh(debug_color=2)
mmesh = merge_meshes(list(s.meshes.values()), color=0).compute()
show(mmesh)

# %% debug colors -- correct merging
s = p.sources["mito_it00_b0_7_stitched"]
p.clipping = [[0.6,0,0], [1,1,1]]
s.calculate_mesh(debug_color=0)
mmesh = merge_meshes(list(s.meshes.values()), color=2).compute()
show(mmesh)

# %% weired cubes in the middle
s = p.sources["mito_it00_b0_7_stitched"]
p.clipping = [[0.3,0,0], [0.7,1,1]]
p.compression_level = "s2" # also on other levels
s.calculate_mesh(debug_color=0)
mmesh = merge_meshes(list(s.meshes.values()), color=0).compute()
show(mmesh)

# %% high-res
p.clipping = [[0.67,0.45,0.5], [0.73,0.8,0.6]]
p.compression_level = "s1"
print(len(s.labels))
mmesh = merge_meshes(list(s.meshes.values()), color=0).compute()
show(mmesh)

# %% Position analysis
p.clipping = None
p.clipping = [[0.6,0.5,0.3], [0.8,1,0.8]]
p.compression_level = "s3"

pa = Position_Analysis(p)
density1d = pa.density1D(s, bin_resolution=(0.1,0.1,0.1), axis=0)



# %% Project API
p.clipping = [[0.6,0.5,0], [1,1,1]]
p.compression_level = "s3"

mmesh = p.merged_meshes(color=1)
show(mmesh)

# %%
p.show()
p.show(curvature=True)
p.show(box=((0.6,0.4,0.3),(0.9,0.8,0.6)))
p.show(skeleton=True)

# $$$$$$$$$$$$$
# %% Organelles
p.clipping = [[0.6,0,0], [1,1,1]]
p.compression_level = "s3"
s = p.sources["mito_it00_b0_7_stitched"]
label = 88000113
id_ = f"mito_{label}"
o = s.get_organelles(ids=id_)[0]

# %%
orgs = p.skeletonize_wavefront("mito_*")
orgs = p.skeletonize_vertex_clusters("mito_101800051", recompute=True)
o = orgs[0]
o.skeleton.skeleton.show()
p.show(skeleton=True)

# %%
p.skeleton_info
o.mesh_properties
o.geometric_data

# %% mcs
p.clear_caches(True)
# %% mcs
%%time
p.clipping = [[0.5,0.6,0.4], [0.6,0.7,1]]
p.compression_level = "s1"
p.search_mcs(0.1)

# %%

o = s.organelles[100]
mesh = o.get_mesh_mcs_colored("0.0-0.1,-")
mesh.compute().show()

p.show(mcs_max=0.02)


# %% MCS queries
p.clipping = [[0.5,0.6,0.5], [0.6,0.7,1]]
p.compression_level = "s2"

p.max_distance = 0.2
p.distance_analysis()
p.search_mcs(0.05, ids_filter_1="mito*", ids_filter_2="er*")
p.get_mcs_overview()

# %%
p.clipping = None
p.compression_level = "s3"
p.search_mcs(0.05, ids_filter_1="mito*", ids_filter_2="er*")

# %% benchmark mcs
p.clipping = [[0.5,0.6,0.5], [1,1,1]]
p.compression_level = "s2"
p.search_mcs(0.05, ids_filter_1="mito*", ids_filter_2="er*")

# %% Curvature
o.curvature_map
color_delayed_trimesh_rgba(o.mesh, o.curvature_map).compute().show()



# %% distances

m1 = o.mesh.compute()
m2 = s.meshes[s.labels[0]].compute()
m1.nearest.vertex(m2.vertices)

for m in tqdm(s.meshes.values()):
    m1.nearest.vertex(m.compute().vertices)


p.distance_analysis()
id1 = "mito_37800009"
id2 = "mito_37900033"
id3 = "er_69500010"

o1= p.get_organelles(id1)[0]
o2= p.get_organelles(id2)[0]
o3= p.get_organelles(id3)[0]

m1 = o1.mesh.compute()
m2 = o2.mesh.compute()
m3 = o3.mesh.compute()
(m1+m3).show()

p.search_mcs("my_search", 5)
p.get_mcs_overview()
p.get_mcs_properties()


b = trimesh.primitives.Box(bounds=((2,2,2), (100,200,500))).as_outline()
(mmesh+b).show()

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
# import tracemalloc
# tracemalloc.start()
cluster = LocalCluster(dashboard_address=":8789", memory_limit="8GiB", n_workers=6)
client = Client(cluster)
project_path = Path.cwd() / ".." / "example_analysis"
p = Project(project_path, compression_level="s2", loglevel="DEBUG", clipping=((0.4,0.4,0.4),(0.6,0.6,0.6)), client=client)
p.add_source("../data/interphase_4T/mito_it00_b0_7_stitched.xml", "mito")
s = p.sources["mito_it00_b0_7_stitched"]
p.add_source("../data/interphase_4T/er_it00_b0_7_stitched.xml", "er")
s = p.sources["er_it00_b0_7_stitched"]

p.clear_caches(True)
p.clipping = [[0.6,0.5,0.4],[0.8,1,0.5]]
p.compression_level = "s1"
p.max_distance = 0.05

# snap1 = tracemalloc.take_snapshot()

p.distance_matrix

# snap2 = tracemalloc.take_snapshot()
