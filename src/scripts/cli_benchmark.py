import argparse
from pathlib import Path
from time import time
from dask.distributed import LocalCluster, Client
from dask_mpi import initialize
from dask.base import compute
from organelle_morphology.project import Project
from organelle_morphology.position import Position_Analysis


def main():
    parser = argparse.ArgumentParser(
        description="Run full organelle-morphology benchmark"
    )
    parser.add_argument(
        "--workers", type=int, default=16, help="Number of Dask workers"
    )
    parser.add_argument(
        "--threads", type=int, default=4, help="Number of threads per worker"
    )
    parser.add_argument(
        "-d", "--data", type=Path, required=True, help="Path to XML directory of interphase_4T"
    )
    parser.add_argument(
        "-p",
        "--projectpath",
        type=Path,
        required=True,
        help="Path to project directory",
    )
    parser.add_argument("--mpi", action="store_true", help="Enable MPI support")
    args = parser.parse_args()

    # Initialize dask_mpi on all ranks, but only run main benchmark on rank 0
    if args.mpi:
        initialize(nthreads=args.threads)

    start_time_total = time()

    if args.mpi:
        client = Client()
    else:
        cluster = LocalCluster(
            n_workers=args.workers,
            threads_per_worker=args.threads,
        )
        client = Client(cluster)

    p = Project(
        args.projectpath,
        compression_level="s2",
        loglevel="DEBUG",
        clipping=((0.4, 0.4, 0.4), (0.6, 0.6, 0.6)),
        client=client,
    )

    p.logger.info("#### DASK CLUSTER INFO ####")
    p.logger.info(client)
    p.logger.info("###########################\n")

    data_path = args.data

    p.add_source(data_path / "cell_it01_b0_7_stitched.xml", "cell")
    s_cell = p.sources["cell_it01_b0_7_stitched"]
    p.add_source(data_path / "er_it00_b0_7_stitched.xml", "er")
    s_er = p.sources["er_it00_b0_7_stitched"]

    # Meshing & Cache
    p.logger.info("\n--- Running Benchmark 1: Mesh Computation ---")
    p.clear_caches(True)
    p.clipping = [[0.6, 0.5, 0.5], [1, 0.6, 1]]
    p.compression_level = "s2"
    p.calculate_meshes()

    t0 = time()
    compute(list(s_cell.meshes.values()) + list(s_er.meshes.values()))
    p.logger.info(f"First compute (uncached): {time() - t0:.2f}s")

    t0 = time()
    compute(list(s_cell.meshes.values()) + list(s_er.meshes.values()))
    p.logger.info(f"Second compute (cached): {time() - t0:.2f}s")

    # Distance Matrix
    p.logger.info("\n--- Running Benchmark 2: Distance Matrix ---")
    p.clear_caches(True)
    p.clipping = [[0.6, 0.5, 0.5], [1, 0.6, 1]]
    p.compression_level = "s2"
    p.max_distance = 0.05

    t0 = time()
    p.distance_matrix
    p.logger.info(f"Distance Matrix Duration: {time() - t0:.2f}s")

    # Membrane Contact Sites (MCS)
    p.logger.info("\n--- Running Benchmark 3: MCS Search ---")
    p.clipping = [[0.6, 0.5, 0.5], [1, 0.6, 1]]
    p.compression_level = "s2"
    t0 = time()
    p.search_mcs(0.05, ids_filter_1="cell*", ids_filter_2="er*")
    p.logger.info(f"MCS Search Duration: {time() - t0:.2f}s")

    # Position Analysis
    p.logger.info("\n--- Running Benchmark 4: Position Analysis ---")
    p.clipping = [[0.6, 0.5, 0.5], [1, 0.6, 1]]
    p.compression_level = "s2"
    t0 = time()
    pa = Position_Analysis(p)
    pa.density1D(s_cell, bin_resolution=(0.1, 0.1, 0.1), axis=0)
    p.logger.info(f"Position Analysis Duration: {time() - t0:.2f}s")

    # Skeletonization
    p.logger.info("\n--- Running Benchmark 5: Skeletonization ---")
    p.clipping = [[0.6, 0.5, 0.5], [1, 0.6, 1]]
    p.compression_level = "s2"
    t0 = time()
    p.skeletonize_wavefront("cell_*")
    p.logger.info(f"Skeletonization Duration: {time() - t0:.2f}s")

    total_time = time() - start_time_total
    p.logger.info(f"\n#### TOTAL TIME NEEDED: {total_time:.2f} seconds")

    client.shutdown()


if __name__ == "__main__":
    main()
