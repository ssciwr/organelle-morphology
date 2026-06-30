from time import time
from dask.base import compute
from organelle_morphology.project import Project
from organelle_morphology.position import Position_Analysis


def main():
    p, data_path = Project.from_args()

    # ######################################
    # Set the data sources, clipping, and compression
    # ######################################
    p.add_source(data_path / "cell_it01_b0_7_stitched.xml", "cell")
    p.add_source(data_path / "er_it00_b0_7_stitched.xml", "er")

    # Simplify the mesh by 50%
    p.simplify = 0.5

    # clipping: tuple of (lower corner, upper corner) or None
    p.clipping = ((0.4, 0.4, 0.4), (0.6, 0.6, 0.6))

    # compression: usually s0-s3, bigger more compressed, s0 uncompressed
    p.compression_level = "s2"

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

    p.logger.info("Finished Calculations!")


if __name__ == "__main__":
    main()
