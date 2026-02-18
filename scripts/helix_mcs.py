from pathlib import Path
from organelle_morphology import Project
from dask_jobqueue.slurm import SLURMRunner

def main():
    project_path = Path.cwd() / "mcs_analysis"
    cluster = SLURMCluster(cores=256,
                       processes=4,
                       memory="16GB",
                       account="woodshole",
                       walltime="01:00:00",
                       queue="normal")
    client = cluster.client()
    p = Project(project_path, compression_level="s2", loglevel="DEBUG", clipping=None, n_workers=32, client=client )
    p.add_source("interphase_4T/mito_it00_b0_7_stitched.xml", "mito")




    s = p.sources["mito_it00_b0_7_stitched"]
    p.add_source("../data/old_cebraEM/Interphase_4T/er_it00_b0_7_stitched.xml", "er")
    s = p.sources["er_it00_b0_7_stitched"]



if __name__ == "__main__":
    main()
