import argparse
from pathlib import Path
from time import time
from dask.distributed import LocalCluster, Client
from dask.base import compute
from organelle_morphology import Project
from organelle_morphology.position import Position_Analysis

def main():
    parser = argparse.ArgumentParser(description="Run full organelle-morphology benchmark")
    parser.add_argument("--workers", type=int, default=16, help="Number of Dask workers")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads per worker")
    parser.add_argument("--data", type=str, required=True, help="Path to XML directory")
    args = parser.parse_args()

    start_time_total = time()

    cluster = LocalCluster(
        n_workers=args.workers,
        threads_per_worker=args.threads,
    )
    client = Client(cluster)

    print("#### DASK CLUSTER INFO ####")
    print(client)
    print("###########################\n")

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
    s_cell = p.sources["cell_it01_b0_7_stitched"]
    p.add_source(data_path / "er_it00_b0_7_stitched.xml", "er")
    s_er = p.sources["er_it00_b0_7_stitched"]

    # Meshing & Cache
    print("\n--- Running Benchmark 1: Mesh Computation ---")
    p.clear_caches(True)
    p.clipping = [[0.6, 0.5, 0.5], [1, 1, 1]]
    p.compression_level = "s2"
    p.calculate_meshes()

    t0 = time()
    compute(list(s_cell.meshes.values()) + list(s_er.meshes.values()))
    print(f"First compute (uncached): {time() - t0:.2f}s")

    t0 = time()
    compute(list(s_cell.meshes.values()) + list(s_er.meshes.values()))
    print(f"Second compute (cached): {time() - t0:.2f}s")

    # Distance Matrix
    print("\n--- Running Benchmark 2: Distance Matrix ---")
    p.clear_caches(True)
    p.clipping = [[0.6, 0.5, 0.4], [0.8, 1, 0.5]]
    p.compression_level = "s1"
    p.max_distance = 0.05

    t0 = time()
    p.distance_matrix
    print(f"Distance Matrix Duration: {time() - t0:.2f}s")

    # Membrane Contact Sites (MCS)
    print("\n--- Running Benchmark 3: MCS Search ---")
    p.clipping = [[0.5, 0.6, 0.5], [0.6, 0.7, 1]]
    p.compression_level = "s2"
    t0 = time()
    p.search_mcs(0.05, ids_filter_1="cell*", ids_filter_2="er*")
    print(f"MCS Search Duration: {time() - t0:.2f}s")

    # Position Analysis
    print("\n--- Running Benchmark 4: Position Analysis ---")
    p.clipping = [[0.6, 0.5, 0.3], [0.8, 1, 0.8]]
    p.compression_level = "s3"
    t0 = time()
    pa = Position_Analysis(p)
    density1d = pa.density1D(s_cell, bin_resolution=(0.1, 0.1, 0.1), axis=0)
    print(f"Position Analysis Duration: {time() - t0:.2f}s")

    # Skeletonization
    print("\n--- Running Benchmark 5: Skeletonization ---")
    p.clipping = [[0.6, 0, 0], [1, 1, 1]]
    p.compression_level = "s3"
    t0 = time()
    orgs = p.skeletonize_wavefront("cell_*")
    print(f"Skeletonization Duration: {time() - t0:.2f}s")

    total_time = time() - start_time_total
    print(f"\n#### TOTAL TIME NEEDED: {total_time:.2f} seconds")

    client.shutdown()

if __name__ == "__main__":
    main()
