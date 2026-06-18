import argparse
from pathlib import Path
from dask.distributed import LocalCluster, Client
from dask_mpi import initialize
from organelle_morphology import Project
from organelle_morphology.position import Position_Analysis
from organelle_morphology.profile_calculations import ProfileCalculator
from organelle_morphology.skeleton_analysis import Skeleton_Analysis


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
        "-d", "--data", type=Path, required=True, help="Path to XML directory"
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

    if args.mpi:
        initialize(nthreads=args.threads)
        client = Client()
    else:
        cluster = LocalCluster(
            n_workers=args.workers,
            threads_per_worker=args.threads,
        )
        client = Client(cluster)


    # ######################################
    # Define the project directory
    # ######################################
    p = Project(
        args.projectpath,
        # compression: usually s0-s3, bigger more compressed
        compression_level="s2",
        # clipping: tuple of (lower corner, upper corner) or None
        clipping=((0.4, 0.4, 0.4), (0.6, 0.6, 0.6)),
        client=client,
        loglevel="INFO",
    )

    # ######################################
    # Set the data sources
    # ######################################
    data_path = args.data
    p.add_source(data_path / "cell_it01_b0_7_stitched.xml", "cell")
    p.add_source(data_path / "er_it00_b0_7_stitched.xml", "er")

    # Simplify the mesh by 50%
    p.simplify = 0.5


    # ######################################
    # Exclude organelles
    # ######################################
    p.blacklist_by_volume(max_vol=0.001) 


    # ######################################
    # Membrane contact sites
    # ######################################
    p.search_mcs(
        max_distance=0.05,
        min_distance=0.0,
        ids_filter_1="cell*",
        ids_filter_2="er*",
    )
    p.registry.save_all_to_yaml()
    

    # ######################################
    # Position Analysis
    # ######################################
    pos_analysis = Position_Analysis(p)
    pos_analysis.density1D(
        source="cell",
        bin_resolution=(0.1, 0.1, 0.1),
        axis=0,
        rot_angle=0,
        rot_axis=0,
    )
    pos_analysis.density2D(
        source="er",
        bin_resolution=(0.1, 0.1, 0.1),
        marginal_axis=0,
        rot_angle=0,
        rot_axis=0,
    )
    pos_analysis.save_records()


    # ######################################
    # Skeletonization
    # ######################################
    skel_analysis = Skeleton_Analysis(p)
    p.skeletonize_vertex_clusters(ids="*", theta=0.4, epsilon=0.1, path_sample_dist=0.1, recompute=False)
    p.skeletonize_wavefront(ids="*", waves=1, theta=0.4, path_sample_dist=0.1, recompute=False)
    skel_analysis.save_records()


    # ######################################
    # Profile Analysis
    # ######################################
    prof_analysis = ProfileCalculator(p)
    # slices perpendicular to axis:
    prof_analysis.calculate_profile_lengths(ids="er_*", axis="z", num_slices=20)
    # randomly oriented slices:
    prof_analysis.calculate_random_profiles(ids="mito_0007", num_planes=20, seed=42)
    # slices perpendicular to the skeleton
    prof_analysis.calculate_skeleton_profiles(ids="mito_0007")


    # ######################################
    # Misc snippets, can be used anywhere
    # ######################################
    p.clear_caches(clear_disk=True)
    p.clipping = ((0.6, 0.5, 0.5), (1, 1, 1))
    p.compression_level = "s2"


    client.shutdown()


if __name__ == "__main__":
    main()
