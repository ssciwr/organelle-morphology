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
project_path = Path.cwd() / "example_analysis"
p = Project(project_path, compression_level="s2", loglevel="DEBUG", clipping=((0.4,0.4,0.4),(0.6,0.6,0.6)))
p.add_source("../data/Interphase_4T/er_it00_b0_7_stitched.xml", "er")
s = p.sources["er_it00_b0_7_stitched"]

# %%
s.meshes[s.labels[20]].compute()

# %%
label = 64400066 # crossing two chunks
mmesh = s.meshes[label].compute()
mmesh.show()

# %% # caching

s.meshes[s.labels[10]].compute()
s._meshes = None
s.meshes[s.labels[5]].compute()

# %%

meshes = [s.meshes[s.labels[i]] for i in range(100)]
meshes = compute(*meshes)
mmesh = sum(meshes)
mmesh.show()

