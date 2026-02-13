from pathlib import Path
from time import time
from organelle_morphology import Project, merge_meshes

def main():

    project_path = Path.cwd() / ".." / "example_analysis"
    p = Project(project_path, compression_level="s2", loglevel="DEBUG", clipping=((0.4,0.4,0.4),(0.6,0.6,0.6)))
    p.add_source("../data/old_cebraEM/Interphase_4T/mito_it00_b0_7_stitched.xml", "mito")
    s = p.sources["mito_it00_b0_7_stitched"]
    p.add_source("../data/old_cebraEM/Interphase_4T/er_it00_b0_7_stitched.xml", "er")
    s = p.sources["er_it00_b0_7_stitched"]

    p.clipping = [[0.6,0.5,0.3], [0.8,1,0.8]]
    p.compression_level = "s3"
    p.clear_caches(True)

    t0 = time()
    p.distance_matrix
    print("#### DURATION:", time() - t0)
    p.client.shutdown()



if __name__ == "__main__":
    main()

