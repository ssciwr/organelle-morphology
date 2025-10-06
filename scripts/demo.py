# ruff: noqa
# fmt: off
# %%
%load_ext rich
from pathlib import Path
from organelle_morphology import Project

# %%
pp = Path.cwd() / "example_analysis"
p = Project(pp, compression_level="s2", loglevel="DEBUG", clipping=((0.4,0.4,0.4),(0.6,0.6,0.6)))
p.add_source("../data/Interphase_4T/er_it00_b0_7_stitched.xml", "er")

# %%
p.calculate_meshes()

# %%



# %%
