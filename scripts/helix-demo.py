import argparse
from pathlib import Path
from time import time
from dask.distributed import LocalCluster, Client
from organelle_morphology import Project

def main():
    parser = argparse.ArgumentParser(description="Run organelle-morphology benchmark")
    parser.add_argument("--workers", type=int, default=4, help="Number of Dask workers (processes)")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads per worker")
    parser.add_argument("--data", type=str, required=True, help="Path to the directory containing the xml files")
    args = parser.parse_args()

    start_time = time()

    cluster = LocalCluster(
        n_workers=args.workers,
        threads_per_worker=args.threads,
    )
    client = Client(cluster)

    print("#### DASK CLUSTER INFO ####")
    print(client)
    print(client.cluster)
    print("###########################")

    project_path = Path.cwd() / ".." / "example_analysis"
    p = Project(
        project_path,
        compression_level="s2",
        loglevel="DEBUG",
        clipping=((0.4, 0.4, 0.4), (0.6, 0.6, 0.6)),
        client=client
    )

    data_path = Path(args.data)

    p.add_source(data_path / "cell_it01_b0_7_stitched.xml", "cell")
    p.add_source(data_path / "er_it00_b0_7_stitched.xml", "er")

    p.clear_caches(True)
    p.clipping = [[0.6, 0.5, 0.4], [0.8, 1, 0.5]]
    p.compression_level = "s1"
    p.max_distance = 0.05

    print("Starting distance matrix calculation...")
    p.distance_matrix

    total_time = time() - start_time
    print(f"#### TOTAL TIME NEEDED: {total_time:.2f} seconds")

    client.shutdown()

if __name__ == "__main__":
    main()
