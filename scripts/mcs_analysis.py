# %%
from organelle_morphology import Project
from pathlib import Path

# %%
project_path = Path.cwd() / ".." / "mcs_analysis"
p = Project(project_path, compression_level="s2", loglevel="DEBUG", clipping=None)
p.add_source("../data/interphase_4T/er_it00_b0_7_stitched.xml", "mito")
s = p.sources["er_it00_b0_7_stitched"]
p.add_source("../data/interphase_4T/mito_it00_b0_7_stitched.xml", "er")
s = p.sources["mito_it00_b0_7_stitched"]

# %%
p.clipping = [[0.5,0.6,0.5], [0.6,0.7,1]]
p.compression_level = "s2"
p.max_distance = 0.2
p.distance_analysis()

# %%
p.search_mcs(0.05, ids_filter_1="mito*", ids_filter_2="er*")
p.get_mcs_overview()
