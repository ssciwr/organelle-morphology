import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from sys import platform
from typing import Optional

import numpy as np
import pandas as pd
import trimesh
from dask.base import compute
from dask.delayed import Delayed
from dask.distributed import Client, LocalCluster
from trimesh import Trimesh

from organelle_morphology.distance_calculations import (
    generate_distance_matrix,
    generate_mcs,
)
from organelle_morphology.organelle import Organelle
from organelle_morphology.records import PropertyBlock, RecordRegistry
from organelle_morphology.source import DataSource
from organelle_morphology.util import (
    Cache,
    color_delayed_trimesh,
    color_delayed_trimesh_vertices,
    corners_to_edges,
    export_delayed_trimesh,
    merge_mesh_dict_values,
    merge_meshes,
    setup_logging,
    show,
)

clipping_type = tuple[tuple[float, float, float], tuple[float, float, float]]


@dataclass(frozen=True)
class ProjectMetadata(PropertyBlock):
    path: Path
    name: str
    clipping: Optional[clipping_type]
    compression: str
    sources: tuple[str, ...]
    blacklist: tuple[str, ...]
    whitelist: tuple[str, ...]


class Project:
    def __init__(
        self,
        project_path: Path | str,
        clipping: Optional[clipping_type] = None,
        compression_level: str = "s0",
        loglevel: Optional[str] = None,
        client: Optional[Client] = None,
        n_workers=4,
    ):
        """Instantiate an EM project

        The given path is expected to contain either a CebraEM or Mobie
        project JSON file. This is a lazy operation. No data except metadata
        is loaded until it is required.

        Args:
            project_path: The location of the project.

            clipping: If not None, the data is clipped with the given lower
                left and the given upper right corner as the bounding box of
                the clipping. Coordinates are expected to be in micrometer.

            compression_level: The compression level at which we operate.
                This is used to determine the resolution of the data that we
                work with. The default of 0 corresponds to the highest resolution.
        """

        self._project_path = Path(project_path)
        self.clear_memory_cache()

        # records can be loaded before sources -> init registry here
        self.registry = RecordRegistry(self)
        self._simplify = 0.5

        self.path.mkdir(exist_ok=True)

        log_file = self.path / "om2.log"
        setup_logging(loglevel or "INFO", log_file)

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"\n ---- New Project {self.path} loaded ----\n")

        if not self.path.exists():
            self.path.mkdir()

        # The dictionary of data sources that we have added
        self.sources: dict[str, DataSource] = {}

        self._clipping = None
        if clipping is not None:
            self.clipping = clipping

        # this permanent filter can be used to soft drop organelles from the project
        # for example if they are too small to be considered
        self.permanent_whitelist = []
        self.permanent_blacklist = []

        # The compression level at which we operate
        self.compression_level = compression_level

        # callables will be updated on demand
        self._cache_settings = {
            "project_name": lambda: self.path.name,
            "clipping": lambda: str(self.clipping).replace("\n", ""),
            "level": lambda: str(self.compression_level),
            "simplify": lambda: str(self.simplify),
            "disk": True,
            "cache_root": lambda: self.path,
            "cache_meshes": True,
            "cache_fragments": False,
        }

        # debug help
        self.use_cache = True
        self.debug = False

        self.cluster = client.cluster if client else LocalCluster(n_workers=n_workers)
        self.client = client if client else Client(self.cluster)
        self.n_workers = n_workers

    def __str__(self):
        return f"Project at {self.path}"

    @staticmethod
    def from_args():
        """Create a project from command line arguments.

        This is mostly intended for use on HPC clusters with a submit
        script controlling some parameters.

        Returns:
            Project: The created project instance
            Path: Path to the data as provided through the cli.
        """
        parser = argparse.ArgumentParser(
            description="Run full organelle-morphology benchmark"
        )
        parser.add_argument(
            "--workers",
            type=int,
            default=16,
            help="Number of Dask workers, only relevant without mpi",
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
        parser.add_argument(
            "-l",
            "--loglevel",
            type=str,
            choices=["INFO", "DEBUG", "WARNING"],
            default="INFO",
            help="Set the loglevel.",
        )
        args = parser.parse_args()

        if args.mpi:
            from dask_mpi import initialize

            initialize(nthreads=args.threads)
            client = Client()
        else:
            cluster = LocalCluster(
                n_workers=args.workers,
                threads_per_worker=args.threads,
            )
            client = Client(cluster)

        p = Project(
            args.projectpath,
            client=client,
            loglevel="INFO",
        )
        return p, args.data

    def recreate_client(self):
        self.client = Client(self.cluster)

    def set_loglevel(self, loglevel: Optional[str]):
        if loglevel:
            root_logger = logging.getLogger()
            root_logger.setLevel(getattr(logging, loglevel.upper()))
            self.logger.debug(f"Set logging level to: {loglevel}")

    @property
    def metadata(self) -> ProjectMetadata:
        clip = None
        if self.clipping is not None:
            clip = tuple([tuple(c) for c in self.clipping.tolist()])

        return ProjectMetadata(
            path=self.path,
            name=self.path.name,
            clipping=clip,
            compression=self.compression_level,
            sources=tuple(self.sources.keys()),
            blacklist=tuple(self.permanent_blacklist),
            whitelist=tuple(self.permanent_whitelist),
        )

    @property
    def path(self) -> Path:
        return self._project_path.resolve()

    @property
    def cache_settings(self):
        d = dict()
        for k, v in self._cache_settings.items():
            if callable(v):
                v = v()
            d[k] = v
        return d

    @property
    def cache(self):
        """Get the cache for this project.

        Unique for sum of sources, compression level and clipping setting.
        Returns the same cache object on consecutive calls.
        On clipping or level change, this cache must be invalidated by calling
        `project.clear_caches()`
        """

        if self._cache is None:
            cs = self.cache_settings
            active_sources = sorted(list(self.sources.keys()))
            name = f"cache_{cs['project_name']}/proj_{active_sources}/{cs['level']}-{cs['simplify']}/{cs['clipping']}"
            cache = Cache(cache_name=name, disk=cs["disk"], cache_root=cs["cache_root"])
            self._cache = cache
        return self._cache

    @property
    def simplify(self):
        return self._simplify

    @simplify.setter
    def simplify(self, simplify: float):
        self.logger.info(f"Setting simplify to {simplify}")
        if 0.0 < simplify > 1.0:
            raise ValueError(
                "Simplify value must be between 0.0 and 1.0. "
                "It is a percent value of how much to simplify."
            )
        self._simplify = simplify
        self.clear_caches()

    def add_source(
        self,
        xml_path: Path | str,
        organelle: str,
        background_label: int = 0,
    ) -> DataSource:
        """Connect a data source in the project with an organelle type

        Parameters
        ----------
        source_path
            The path to the xml source to add.
        organelle
            The name of the organelle that is labelled in the data source.
            Must be on the strings returned by organelle_morphology.organelle_types
        background_label
            The label in the data source that is used to encode the background.
            Assumed to be 0.

        Returns
        -------
        DataSource
            Also accessable as `project[xml_name]`

        Raises
        ------
        ValueError
            Source already loaded
        ValueError
            Compression level of project not available
        """
        self.logger.info(f"Project: {self}")
        xml_path = Path(xml_path)
        if xml_path.suffix != ".xml":
            xml_path = xml_path.with_suffix(".xml")

        if xml_path.stem in self.sources:
            raise ValueError("Source already loaded!")

        # resolve relative to project path
        if not xml_path.exists() and not xml_path.is_absolute():
            if (self.path / xml_path).exists():
                xml_path = self.path / xml_path

        # Instantiate the new source
        source_obj = DataSource(
            self,
            xml_path,
            organelle,
            background_label,
        )

        self.logger.info(f"Adding source {source_obj.metadata.name}")

        # Double-check that it provides the current compression level
        if self.compression_level not in source_obj.metadata.levels:
            raise ValueError(
                f"Compression level {self.compression_level} is not available "
                f"for source {source_obj.metadata.name}.\n Available levels:"
                f"{source_obj.metadata.levels}"
            )
        self.sources[xml_path.stem] = source_obj

        return source_obj

    @property
    def resolution(self):
        res = None
        for s in self.sources.values():
            if res is None:
                res = s.resolution
            elif s.resolution != res:
                raise RuntimeError(
                    "Sources have different resolutions which is not supported!"
                )
        return res

    def skeletonize_wavefront(
        self,
        ids: str = "*",
        theta: float = 0.4,
        waves: int = 1,
        step_size: int = 2,
        path_sample_dist: float = 0.1,
        recompute=False,
    ):
        """Note that some meshes will be skipped if they are too small to be skeletonized."""
        orgs = self.get_organelles(ids=ids)

        self.logger.info(
            f"Starting Skeleton wavefront generation for {len(orgs)} organelles. "
        )

        org_per_source: dict[DataSource, list[Organelle]] = defaultdict(list)
        for o in orgs:
            org_per_source[o.source].append(o)
        for s, o_s in org_per_source.items():
            labels = [o.label for o in o_s]
            s.generate_skeletons(
                labels=labels,
                skeletonization_type="wavefront",
                theta=theta,
                waves=waves,
                step_size=step_size,
                path_sample_dist=path_sample_dist,
                recompute=recompute,
            )
        self.logger.info("Skeletonization done!")

    def skeletonize_vertex_clusters(
        self,
        ids: str = "*",
        theta: float = 0.4,
        epsilon: float = 0.1,
        sampling_dist: float = 0.1,
        path_sample_dist: float = 0.1,
        recompute: bool = False,
    ):
        orgs = self.get_organelles(ids=ids)

        self.logger.info(
            f"Starting Skeleton vertex cluster generation for {len(orgs)} organelles. "
        )

        org_per_source: dict[DataSource, list[Organelle]] = defaultdict(list)
        for o in orgs:
            org_per_source[o.source].append(o)
        for s, o_s in org_per_source.items():
            labels = [o.label for o in o_s]
            s.generate_skeletons(
                labels=labels,
                skeletonization_type="vertex_clusters",
                theta=theta,
                epsilon=epsilon,
                sampling_dist=sampling_dist,
                path_sample_dist=path_sample_dist,
                recompute=recompute,
            )
        self.logger.info("Skeletonization done!")

    def show(
        self,
        ids: str = "*",
        ids_highlight: Optional[str] = None,
        box: Optional[
            tuple[tuple[float, float, float], tuple[float, float, float]]
        ] = None,
        clipping_box=True,
        domain_box=True,
        curvature=False,
        skeleton=False,
        curv_log=True,
        color_instances=False,
        mcs_min: Optional[float] = None,
        mcs_max: Optional[float] = None,
        axis: bool = True,
        rot_axis: Optional[str] = None,
        rot_angle: Optional[float] = None,
        volume: Optional[float] = None,
        export: Optional[str] = None,
    ):
        """Display organelles in the project.

        This method allows for flexible visualization of organelles in the project,
        supporting various overlays, filters, and rendering options.
        Many options only work as inteded when no other options are in use.

        Args:
            ids: Glob-style filter for organelle IDs to display. Defaults to "*"
                to show all organelles. Separate multiple selections with a comma.
            ids_highlight: Optional glob-style filter for organelles to highlight
                in a different color. Defaults to None.
            box: Optional bounding box to visualize as a wireframe. Specified as
                ((x_min, y_min, z_min), (x_max, y_max, z_max)) in micrometers.
            clipping_box: Whether to display the clipping box if clipping is set.
                Defaults to True.
            domain_box: Whether to display the domain box (full data extent).
                Defaults to True.
            curvature: Color the mesh by its curvature. Defaults to False.
            skeleton: Show the skeleton of organelles. Only shows pre-calculated
                skeleton data. Defaults to False.
            curv_log: Whether to apply logarithmic scaling to curvature coloring
                when curvature=True. Defaults to True.
            color_instances: Whether to color individual organelle instances
                differently. Defaults to False.
            mcs_min: Minimum distance threshold for membrane contact site (MCS)
                visualization. Requires mcs_max to be set. Defaults to None.
            mcs_max: Maximum distance threshold for MCS visualization.
                Triggers the calculation of this MCS analysis. The contact
                sites will be colored randomly. Defaults to None.
            axis: Whether to display coordinate axes. Defaults to True.
            rot_axis: Preview a rotation of the data around this axis.
                Display two lines indicating a rotation by a given angle.
                Yellow is the reference 0° line, orange is the rotated line.
                Can be `x`, `y`, or `z`. Defaults to None.
            rot_angle: Rotation angle in degrees for the rotation visualization.
                Defaults to None.
            volume: Volume threshold for previewing filtering organelles by size.
                Organelles with volume less than this threshold are colored organge.
                Bigger organelles are green. Open meshes are colored gray.
                Defaults to None.
            export: Optional path to export the visualization scene to a glb file.
                If provided, the scene will be exported to this location.
                Defaults to None.

        Returns:
            trimesh.Scene: The rendered scene containing all visualization elements.
        """
        orgs = self.get_organelles(ids=ids)
        if len(orgs) == 0:
            self.logger.warning(f"Selection {ids} does not match any organelles!")
            return
        source = orgs[0].source

        o_types = {o.id.split("_")[0] for o in orgs}

        transp = False
        if skeleton:
            transp = True

        to_show = []
        if mcs_max is not None:
            if mcs_min is None:
                mcs_min = 0.0
            mcs_label = self.search_mcs(max_distance=mcs_max, min_distance=mcs_min)
            mcs_orgs = self.mcs_queries[mcs_label]["organelles"]
            meshes = []
            non_mcs_meshes = []
            for org in orgs:
                if org.id in mcs_orgs:
                    meshes.append(org.get_mesh_mcs_colored(mcs_label))
                else:
                    non_mcs_meshes.append(org.mesh)
            meshes.append(merge_meshes(non_mcs_meshes, color=-1, transp=transp))
            mmesh = merge_meshes(meshes, color=0, transp=transp)
            to_show = [mmesh.compute()]

        elif curvature:
            org_per_source: dict[DataSource, list[Organelle]] = defaultdict(list)
            for o in orgs:
                org_per_source[o.source].append(o)
            meshes = []
            # calculate curvatures on all sources
            for source, orgs in org_per_source.items():
                labels = [o.label for o in orgs]
                source.calc_curvature(labels=labels)
            vmin, vmax = (0.0, 0.0)
            for source in org_per_source.keys():
                maxima = source.get_curvature_range()
                if (new_vmin := maxima["vmin"]) < vmin:
                    vmin = new_vmin
                if (new_vmax := maxima["vmax"]) > vmax:
                    vmax = new_vmax

            for s, o_s in org_per_source.items():
                labels = [o.label for o in o_s]
                meshes.append(
                    merge_meshes(
                        s.get_meshes_curvature_colored(
                            labels=labels, log=curv_log, vmin=vmin, vmax=vmax
                        ),
                        color=0,
                        transp=transp,
                    )
                )
            to_show.extend(compute(*meshes))

        elif volume:
            meshes = []
            for o in orgs:
                if not o.mesh_properties.water_tight:
                    meshes.append(
                        color_delayed_trimesh_vertices(
                            o.mesh, slice(None), [127, 127, 127, 255]
                        )
                    )
                    continue
                if o.mesh_properties.volume < volume:
                    meshes.append(color_delayed_trimesh(o.mesh, -2, False))
                else:
                    meshes.append(color_delayed_trimesh(o.mesh, -5, False))
            to_show.extend(compute(*meshes))

        else:  # not curvature or mcs
            if ids_highlight is not None:
                orgs_highlight = self.get_organelles(ids_highlight)
                to_merge = []
                # apply highlight coloring
                to_merge.append(
                    merge_meshes(
                        [o.mesh for o in orgs_highlight],
                        color=-2,
                        transp=transp,
                    )
                )
                # color all other organelles
                to_merge_2 = []
                for org in orgs:
                    if org not in orgs_highlight:
                        to_merge_2.append(org)
                if len(to_merge_2):
                    to_merge.append(
                        merge_meshes(
                            [o.mesh for o in to_merge_2],
                            color=-1,
                            transp=transp,
                        )
                    )
                mmesh = merge_meshes(to_merge, color=0)

            elif color_instances:
                mmesh = merge_meshes([o.mesh for o in orgs], color=1)

            else:  # No highlight
                if len(o_types) <= 1:
                    mmesh = merge_meshes([o.mesh for o in orgs], color=1, transp=transp)
                else:
                    meshes = []
                    for i, ot in enumerate(o_types):
                        ot_meshes = [o.mesh for o in orgs if ot in o.id]
                        meshes.append(
                            merge_meshes(ot_meshes, color=-(i + 1), transp=transp)
                        )
                    mmesh = merge_meshes(meshes, color=0)
            to_show = [mmesh.compute()]

        if skeleton:
            for o in orgs:
                if o.skeleton is not None:
                    to_show.append(o.skeleton.skeleton)

        if domain_box:
            size = np.array(source.metadata.size) * source.data_resolution
            domain_box = trimesh.path.creation.box_outline(
                extents=size,
                transform=trimesh.transformations.translation_matrix(size / 2),
            )
            domain_box.colors = ((0, 200, 0, 255),)
            to_show.append(domain_box)

        if clipping_box:
            if self.clipping is not None:
                # assert source._clip_low_corner_data is not None, (
                #     "source._clip_low_corner_data is missing"
                # )
                edges = corners_to_edges(*source.clipping_corners)
                trans = trimesh.transformations.translation_matrix(
                    source.clipping_corners[0] + (edges / 2)
                )

                clip_box = trimesh.path.creation.box_outline(
                    extents=edges, transform=trans
                )
                clip_box.colors = ((100, 0, 200, 255),)
                to_show.append(clip_box)

        if box:
            npbox = np.array(box)
            if np.all(npbox <= 1.0):
                npbox = npbox * source.metadata.size * source.data_resolution
            edges = corners_to_edges(*npbox)
            trans = trimesh.transformations.translation_matrix(npbox[0] + (edges / 2))

            box_outline = trimesh.path.creation.box_outline(
                extents=edges, transform=trans
            )
            box_outline.colors = ((200, 50, 50, 255),)
            to_show.append(box_outline)

        if axis:
            if self.clipping is not None and clipping_box:
                clip_size = corners_to_edges(*source.clipping_corners)
                clip_axis_mesh = self._create_axis_marker(clip_size)
                clip_axis_mesh.apply_transform(
                    trimesh.transformations.translation_matrix(
                        source.clipping_corners[0]
                    )
                )
                to_show.append(clip_axis_mesh)
            else:
                domain_size = np.array(source.metadata.size) * source.data_resolution
                axis_mesh = self._create_axis_marker(domain_size)
                to_show.append(axis_mesh)

        if rot_axis is not None:
            size = np.array(source.metadata.size) * source.data_resolution
            center = size / 2

            plane_axes = [0, 1, 2]
            axis_names = ["x", "y", "z"]
            plane_axes.remove(axis_names.index(rot_axis))
            radius = np.linalg.norm(size[plane_axes]) / 2

            u = np.zeros(3)
            v = np.zeros(3)
            u[plane_axes[0]] = 1.0
            v[plane_axes[1]] = 1.0

            def _line_in_plane(angle_deg):
                theta = np.deg2rad(angle_deg)
                direction = np.cos(theta) * u + np.sin(theta) * v
                end = center + direction * radius
                return trimesh.load_path(np.array([[center, end]]))

            ref_line = _line_in_plane(0.0)
            ref_line.colors = [(200, 200, 0, 255)]  # yellow = 0° reference
            to_show.append(ref_line)

            if rot_angle is not None and rot_angle % 360.0 != 0.0:
                rot_line = _line_in_plane(rot_angle)
                rot_line.colors = [(255, 100, 0, 255)]  # orange = current angle
                to_show.append(rot_line)

        scene = show(to_show)
        if export:
            exp_root = self.path / "exported_meshes"
            exp_root.mkdir(parents=True, exist_ok=True)
            scene.export(exp_root / Path(export).with_suffix(".glb"))

        return scene

    def export_mesh_all_fragments(self):
        exp_root = self.path / "exported_meshes"

        results = []
        for s in self.sources.values():
            exp_source = exp_root / s.org_name
            exp_source.mkdir(parents=True, exist_ok=True)

            tasks = []
            for i, mesh_d in np.ndenumerate(s.mesh_fragments):
                path = exp_source / f"{i}.glb"
                mm = merge_mesh_dict_values(mesh_d)
                tasks.append(export_delayed_trimesh(mm, path))
            self.logger.debug(f"Running {len(tasks)} export tasks")
            results.append(compute(*tasks, scheduler="synchronous"))

    def distance_filtering(
        self, ids_source="*", ids_target="*", filter_distance=0.01, attribute="labels"
    ):
        """Filter the organelles based on the distance between two filtered organelle lists.

        These can be from one or more types depending on the given filter.

        Args:
            ids_source: Filter string for the source ids, defaults to "*"
            ids_target: filter string for the target ids, defaults to "*"
            filter_distance: The distance in micro meter used for filtering,
                defaults to 0.01
            attribute: Show names, number of contacts (contacts) or return
                the organelle objects ("objects"), defaults to "names"
        Returns:
            Dictionary with the source organelle ids as keys and the target organelle ids or number of contact sites as values
        """

        if attribute not in ["labels", "contacts", "objects"]:
            raise ValueError(
                f"Attribute must be one of 'labels', 'contacts' or 'objects' but is {attribute}"
            )

        orgs_1 = self.get_organelle_ids(ids=ids_source)
        orgs_2 = self.get_organelle_ids(ids=ids_target)
        distance_matrix = self.distance_matrix.loc[orgs_1, orgs_2]
        # Filter the DataFrame by row values
        filtered_df = distance_matrix[
            distance_matrix.apply(
                lambda row: row.loc[row >= 0].min() < filter_distance, axis=1
            )
        ]

        # Convert the filtered DataFrame to a dictionary
        filtered_df_dict = filtered_df.to_dict("index")
        output_filtered_dict = defaultdict(list)
        # For each entry in the dictionary, replace the values with the column names that match the filter

        for col in filtered_df_dict:
            for row, value in filtered_df_dict[col].items():
                if (
                    0.0 <= value < filter_distance and col != row
                ):  # exclude self-contact
                    # if (
                    #     attribute in ["names", "objects"]
                    #     and row in output_filtered_dict.keys()
                    # ):
                    #     # this is not done when interested in the number of neighbors for each organelle.
                    #     continue  # skip entries that are already present as keys, this avoids doubling

                    output_filtered_dict[col].append(row)

        if attribute == "contacts":
            for key in output_filtered_dict.keys():
                output_filtered_dict[key] = len(output_filtered_dict[key])
                output_filtered_dict = dict(output_filtered_dict)

        elif attribute == "objects":
            obj_output_dict = defaultdict(list)
            for key in output_filtered_dict.keys():
                new_key = self.get_organelles(ids=key)[0]

                for key_target in output_filtered_dict[key]:
                    if key_target not in obj_output_dict.keys():
                        obj_output_dict[new_key].extend(
                            self.get_organelles(ids=key_target)
                        )
            return obj_output_dict

        return output_filtered_dict

    @property
    def max_distance(self):
        return self._max_compute_distance

    @max_distance.setter
    def max_distance(self, max_distance):
        self._max_compute_distance = max_distance

    def distance_analysis(self, ids_source="*", ids_target="*", attribute="dist"):
        """get more information about the distance between two filtered organelle lists.
           These can be from one or more types depending on the given filter.

        :param ids_source: Filter string for the source ids, defaults to "*"
        :type ids_source: str, optional
        :param ids_target: filter string for the target ids, defaults to "*"
        :type ids_target: str, optional
        :param attribute: The attribute to be used for the distance analysis.
        Possible values are "dist", "min", "mean".
        dist: the distance matrix between the organelles
        min: the minimum distance between all sources for any target and the corresponding target id
        mean: the mean distance between the organelles
        :type attribute: _type_
        """
        orgs_1 = self.get_organelle_ids(ids=ids_source)
        orgs_2 = self.get_organelle_ids(ids=ids_target)
        distance_matrix = self.distance_matrix.loc[orgs_1, orgs_2]

        if attribute == "dist":
            return distance_matrix

        elif attribute == "min":
            distance_matrix = distance_matrix.loc[orgs_1, orgs_2]
            df_min = pd.DataFrame(
                distance_matrix.idxmin(axis=1).items(),
                columns=["id_source", "id_target"],
            )
            df_min["distance"] = distance_matrix.min(axis=1).values
            return df_min

        elif attribute == "mean":
            df_mean = pd.DataFrame(
                distance_matrix.mean(axis=1).items(), columns=["id_source", "id_target"]
            )
            df_mean.rename(
                columns={"id_target": "mean distance to target"}, inplace=True
            )
            df_mean["std of target distance"] = distance_matrix.std(axis=1).values

            return df_mean

    def search_mcs(
        self,
        max_distance: float,
        min_distance: float = 0.0,
        ids_filter_1: str = "*",
        ids_filter_2: str = "*",
        overwrite_mcs_label=False,
    ):
        """
        This function is used to search for membrane contact sites within a project.
        Pairs will only be selected when their minimum mesh distance is between
        the requested min and max distance.

        Args:
            project (Project): The project object containing the distance
                matrix and organelles.
            mcs_label (str): The label for the MCS pairs.
            max_distance (float): The maximum distance for the MCS pairs.
            min_distance (float, optional): The minimum distance for the MCS
                pairs. Defaults to 0.
            ids_filter_1 (str, optional): Filter for the first set of
                organelles using glob-style patterns. Defaults to "*".
            ids_filter_2 (str, optional): Filter for the second set of
                organelles using glob-style patterns. Defaults to "*".
            overwrite_mcs_label (bool, optional): If the label exists, it
                should be overwritten. Defaults to False.

        Returns:
            str: The label of this mcs search
        """

        return generate_mcs(
            self,
            ids_filter_1=ids_filter_1,
            ids_filter_2=ids_filter_2,
            max_distance=max_distance,
            min_distance=min_distance,
            overwrite=overwrite_mcs_label,
        )

    @property
    def mcs_labels(self):
        return self._mcs_labels.keys()

    @property
    def mcs_queries(self):
        return self._mcs_labels

    @property
    def compression_level(self) -> str:
        """The compression level used for our computations."""

        return self._compression_level

    @compression_level.setter
    def compression_level(self, level: str):
        for s_name, s in self.sources.items():
            if level not in s.metadata.levels:
                raise ValueError(
                    f"Requested level {level} not available in source {s_name}!\n"
                    f"Levels in source: {s.metadata.levels}"
                )
        if getattr(self, "_compression_level", None) != level:
            self.clear_caches()

        self._compression_level = level

    def calculate_meshes(self):
        """Trigger the calculation of meshes for all organelles"""

        for source in self.sources.values():
            source.meshes

    @property
    def geometric_properties(self):
        """The geometry properties of the organelles
        Possible keywords are:
        "voxel_volume": for 3d this is the volume
        "voxel_bbox",
        "voxel_slice": the slice of the bounding box
        "voxel_centroid"
        "voxel_moments"
        "voxel_extent": how much volume of the bounding box is occupied by the object
        "voxel_solidity":ratio of pixels in the convex hull to those in the region
        "mesh_volume"
        "mesh_area"
        "mesh_centroid"
        "mesh_inertia"
        "water_tight"
        "sphericity": how spherical the mesh is (0-1)
        "flatness_ratio": how cubic the mesh is (0-1)

        """
        properties = {}
        for source in self.sources.values():
            source.basic_geometric_properties

        for organelle in self.get_organelles():
            # TODO: geometric data is expensive, and not parallel, necessary?
            properties[organelle.id] = (
                organelle.mesh_properties.to_dict() | organelle.geometric_data
            )

        df = pd.DataFrame(properties).T
        return df

    def n_organelles(self) -> dict[str, int]:
        counts = {}
        for s in self.sources.values():
            counts[s.org_name] = len(self.get_organelles(s.org_name + "*"))
        return counts

    @property
    def skeleton_info(self):
        skeleton_data = {}
        for org in self.organelles:
            if org.skeleton is not None:
                skeleton_data[org.id] = org.skeleton_info
        df = pd.DataFrame(skeleton_data).T
        if len(df) > 0:
            return df.sort_values(by="num_nodes", ascending=False)
        return pd.DataFrame()

    @property
    def curvature_map(self):
        """Get the curvature map for all organelles"""

        for source_key, source in self.sources.items():
            self._curvature_map[source_key] = source.curvature_map

        return self._curvature_map

    def set_curvature_radius(self, radius):
        """Set the radius for curvature calculations.
        Resets the cached curvature.
        """
        for source in self.sources.values():
            source.curvature_radius = radius

    @property
    def distance_matrix(self):
        """
        Returns the distance matrix of all organeles in the project in micro meters.
        """

        return generate_distance_matrix(self)

    def clear_blacklist(self):
        self.permanent_blacklist = []

    def blacklist_by_volume(self, max_vol: float):
        """Add organlles to the permanent blacklist based on their volume.

        Args:
            max_vol: Max volume of organlles allowed. Smaller organelles are
                added to the blacklist.
        """
        for o in self.organelles:
            if o.mesh_properties.volume < max_vol:
                if o.id not in self.permanent_blacklist:
                    self.permanent_blacklist.append(o.id)

        self.logger.info(
            "Excluded organelles based on volume.\n"
            f"Removed organelles: {len(self.permanent_blacklist)}\n"
            f"Remaining organelles: {len(self.organelles)}"
        )

    @property
    def clipping(
        self,
    ) -> None | np.ndarray:
        """The subcube of the original data to work with.
        All operations performed in this project must respect the clipping
        region set here.

        Attribute clipping:
            Tuple of two tuples of length three.
            Must be the lower corner and the upper corner.
        """

        if self._clipping is None:
            return None
        else:
            return self._clipping

    @clipping.setter
    def clipping(self, clipping: clipping_type | None):
        self.logger.info(f"Setting clipping to {clipping}")
        if (clipping is not None) and not all(a < b for a, b in zip(*clipping)):
            raise ValueError(
                "First clipping corner is lower left, second upper right. All "
                "coordinates of corner one must be smaller than of corner two"
            )
        if not np.all(np.array(clipping) == getattr(self, "clipping", None)):
            self.clear_caches()

        if clipping is None:
            self._clipping = None
            return
        _clipping = np.array(clipping)

        if not np.all(_clipping[0] >= 0) or not np.all(_clipping[1] <= 1):
            raise ValueError("Clipping must be in [0, 1]^3")

        if len(_clipping) != 2 or len(_clipping[0]) != 3 or len(_clipping[1]) != 3:
            raise ValueError("Clipping must be a tuple of two tuples of length 3")
        self._clipping = _clipping

    @property
    def clipping_corners(self):
        if len(self.sources) == 0:
            raise ValueError("No sources loaded! Can't compute clipping corners")
        return list(self.sources.values())[0].clipping_corners

    def _create_axis_marker(self, size):
        """Create an axis marker scaled to a given size.

        The axis length and radius are scaled based on the size.
        Axis is positioned at the origin (0,0,0) of the coordinate system.

        Args:
            size: Size of the domain or box to scale the axis to

        Returns:
            trimesh.Trimesh: The axis mesh
        """
        max_dim = max(size)
        scaled_length = max_dim / 10
        scaled_radius = scaled_length / 10

        axis = trimesh.creation.axis(
            axis_radius=scaled_radius,
            axis_length=scaled_length,
            origin_size=0.00001,
        )
        return axis

    @property
    def organelles(self) -> list[Organelle]:
        return self.get_organelles("*")

    @property
    def organelle_ids(self) -> list[str]:
        return self.get_organelle_ids("*")

    def get_organelles(
        self,
        ids: str | list[str] = "*",
    ) -> list[Organelle]:
        """Return a list of organelles found in the dataset

        This requires previous adding of data sources using add_source.
        The ids parameter is used to filter based on organelle ids.
        Filtering using the permanent_blacklist and permanent_whitelist is respected.

        Args:
            ids: The glob-style filtering expression for organelle ids to return.
                The default of "*" returns all organelles.
        """

        result = []
        if isinstance(ids, str):
            ids = [ids]

        for id_ in ids:
            for source in self.sources.values():
                if id_ in self.permanent_blacklist:
                    continue

                result.extend(
                    source.get_organelles(
                        id_,
                        self.permanent_whitelist,
                        self.permanent_blacklist,
                    )
                )

        return result

    def get_organelle_ids(
        self,
        ids: str | list[str] = "*",
    ) -> list[str]:
        """Return a list of organelle ids found in the dataset

        This requires previous adding of data sources using add_source.
        The ids parameter is used to filter based on organelle ids.

        Args:
            ids The glob-style filtering expression for organelle ids to return.
            The default of "*" returns all organelles.
        """

        result = []
        if isinstance(ids, str):
            ids = [ids]

        for id_ in ids:
            for source in self.sources.values():
                if id_ in self.permanent_blacklist:
                    continue

                result.extend(
                    source.get_organelle_ids(
                        id_,
                        self.permanent_whitelist,
                        self.permanent_blacklist,
                    )
                )

        return result

    def merged_meshes(
        self, ids: str | list[str] = "*", color=1, delayed=False
    ) -> Trimesh | Delayed:
        """Get a mesh composed of all or some organelles

        Set the compression level and clipping in the project.

        Args:
        ids
            Glob style filter for the organelles.
            The default "*" returns all organelles in the project.
        color
            0: don't color the mesh
            1: color per source or per id filter if multiple are supplied (default)
            2: color per face
        delayed
            If True, return a dask delayed object instead of the computed mesh.
            Default: False
        """

        # List of filters or list of sources
        if isinstance(ids, (list, tuple)):
            organelles = [self.get_organelles(id) for id in ids]
        else:
            organelles = [s.get_organelles(ids) for s in self.sources.values()]

        meshes = []
        for i, orgs in enumerate(organelles):
            c = 0
            if color == 1:
                # per filter or source
                c = -(i + 1)
            elif color == 2:
                # per instance
                c = 1
            elif color == 3:
                # per face
                c = 2
            meshes.append(merge_meshes([o.mesh for o in orgs], color=c))

        mesh = merge_meshes(meshes)
        if not delayed:
            mesh = mesh.compute()
        return mesh

    def get_caches(self, silent=False) -> list[Cache]:
        caches = []
        cs = self.cache_settings
        cache_dir = cs["cache_root"] / f"cache_{cs['project_name']}"

        # OS-safe tree drawing characters
        if platform == "win32":
            branch = "|-"
            pipe = "| "
        else:
            branch = "├─"
            pipe = "│ "

        messages = ["*** List of Caches: ***"]
        if cache_dir.exists():
            messages.append(str(cache_dir))
            for source in filter(lambda f: f.is_dir(), (cache_dir).iterdir()):
                messages.append(f"{branch} /{source.name}")
                for level in filter(lambda f: f.is_dir(), source.iterdir()):
                    messages.append(f"{pipe}  {branch} /{level.name}")
                    for clip_dir in filter(lambda f: f.is_dir, level.iterdir()):
                        messages.append(f"{pipe}  {pipe}  {branch} /{clip_dir.name}")
                        name = (
                            f"cache_{cs['project_name']}/{source.name}/"
                            f"{level.name}/{clip_dir.name}"
                        )
                        caches.append(
                            Cache(
                                cache_name=name,
                                disk=True,
                                cache_root=cs["cache_root"],
                            )
                        )
        else:
            messages.append(" No caches on disk!")

        cache_names = [c.cache_name for c in caches]
        for s in self.sources.values():
            if s._cache is not None:
                if s._cache.cache_name not in cache_names:
                    caches.append(s.cache)

        if not silent:
            self.logger.info("\n".join(messages))
            self.logger.info(f"Found {len(caches)} caches for project {self.path.name}")
        return caches

    @property
    def records(self):
        return self.registry.get_all()

    def get_record_stats(self):
        return self.registry.summary()

    def clear_memory_cache(self):
        """(Re)initialize project-level storage."""

        self._basic_geometric_properties = {}
        self._mesh_properties = {}
        self._geometric_properties = {}
        self._curvature_map = {}
        self._mcs_labels = {}  # {label: {max_distance: float, min_distance: float}}
        self._max_compute_distance = 0.0
        self._cache = None

    def clear_caches(self, clear_disk=False, silent=False):
        """Clear all caches related to this project, optionally also from disk"""

        for source in self.sources.values():
            source.clear_memory_cache()
        self.clear_memory_cache()
        if not silent:
            self.logger.debug("Cleared memory cache of all sources.")

        if clear_disk:
            # iterate over what is on disk rather than the currently loaded sources
            i = -1
            for i, cache in enumerate(self.get_caches(silent=silent)):
                cache.clear_disk_cache()
            if not silent:
                self.logger.info(f"Deleted {i + 1} caches from disk.")
