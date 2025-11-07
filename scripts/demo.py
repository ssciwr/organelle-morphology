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
from organelle_morphology.util import disk_cache
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
d.compute()




# %% caching issue
cache_name = 'meshes_mito_it00_b0_7_stitched.xml_s2'

def check_cached_labels():
    with disk_cache(p.path, cache_name) as cache:
        print("Labels? ", "labels" in cache)
        print("All: ", sum([label in cache for label in s.labels]))

# %% clean up
with disk_cache(p.path, cache_name) as cache:
    cache.clear()
    cache.close()
check_cached_labels()

# %%
# computing
print(len(s.labels))

# %%
check_cached_labels()

# %%
# from cache
s._meshes = None
print(len(s.labels))

# %%


# %% caching issue


# %% caching issue
with disk_cache(p.path, cache_name) as cache:
    print("Labels? ", "labels" in cache)
    print(list(cache)


# %%
label = 65500028 # crossing two chunks in er_it00_b0_7_stitched
mmesh = s.meshes[label].compute()
mmesh

# %%
label = 64400066 # crossing two chunks in er_it00_b0_7_stitched
label = s.labels[1]
mmesh = s.meshes[label].compute()
mmesh.show()

# %% # caching

s.meshes[s.labels[10]].compute()
s._meshes = None
s.meshes[s.labels[5]].compute()

# %%

# meshes = [s.meshes[s.labels[i]] for i in range(100)]
meshes = [s.meshes[s.labels[i]] for i in range(len(s.labels))]
meshes = compute(*meshes)
mmesh = sum(meshes)
mmesh.show()

