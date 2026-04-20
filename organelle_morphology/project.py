from dataclasses import dataclass
from typing import Optional

from organelle_morphology.statistics import Properties
from organelle_morphology.util import setup_logging
import logging
from dask.base import compute
from dask.delayed import Delayed
from trimesh import Trimesh
import trimesh
from organelle_morphology.organelle import Organelle
from organelle_morphology.source import DataSource
from organelle_morphology.util import (
    Cache,
    corners_to_edges,
    merge_meshes,
    show,
)
from organelle_morphology.distance_calculations import (
    generate_distance_matrix,
    generate_mcs,
)

from pathlib import Path
from dask.distributed import Client, LocalCluster

import numpy as np
import pandas as pd

from collections import defaultdict
import plotly.graph_objects as go


clipping_type = (
    tuple[tuple[float, float, float], tuple[float, float, float]] | list[list[float]]
)


@dataclass
class ProjectMeta(Properties):
    path: Path
    name: str
    clipping: Optional[np.ndarray]
    compression: str
    sources: list[str]
    blacklist: list[str]
    whitelist: list[str]


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
            "disk": True,
            "cache_root": lambda: self.path,
            "cache_meshes": True,
        }

        # debug help
        self.use_cache = True
        self.debug = False

        self.cluster = client.cluster if client else LocalCluster(n_workers=n_workers)
        self.client = client if client else Client(self.cluster)
        self.n_workers = n_workers

    def recreate_client(self):
        self.client = Client(self.cluster)

    def __str__(self):
        return f"Project at {self.path}"

    def set_loglevel(self, loglevel: Optional[str]):
        if loglevel:
            root_logger = logging.getLogger()
            root_logger.setLevel(getattr(logging, loglevel.upper()))
            self.logger.debug(f"Set logging level to: {loglevel}")

    @property
    def metadata(self):
        return ProjectMeta(
            path=self.path,
            name=self.path.name,
            clipping=self.clipping,
            compression=self.compression_level,
            sources=list(self.sources.keys()),
            blacklist=self.permanent_blacklist,
            whitelist=self.permanent_whitelist,
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
            name = f"cache_{cs['project_name']}/proj_{active_sources}/{cs['level']}/{cs['clipping']}"
            cache = Cache(cache_name=name, disk=cs["disk"], cache_root=cs["cache_root"])
            self._cache = cache
        return self._cache

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
        calculated_orgs = []
        for s, o_s in org_per_source.items():
            labels = [o.label for o in o_s]
            calculated_orgs.extend(
                s.generate_skeletons(
                    labels=labels,
                    skeletonization_type="wavefront",
                    theta=theta,
                    waves=waves,
                    step_size=step_size,
                    path_sample_dist=path_sample_dist,
                    recompute=recompute,
                )
            )
        self.logger.info("Skeletonization done!")
        return calculated_orgs

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
            f"Starting Skeleton wavefront generation for {len(orgs)} organelles. "
        )

        org_per_source: dict[DataSource, list[Organelle]] = defaultdict(list)
        for o in orgs:
            org_per_source[o.source].append(o)
        calculated_orgs = []
        for s, o_s in org_per_source.items():
            labels = [o.label for o in o_s]
            calculated_orgs.extend(
                s.generate_skeletons(
                    labels=labels,
                    skeletonization_type="vertex_clusters",
                    theta=theta,
                    epsilon=epsilon,
                    sampling_dist=sampling_dist,
                    path_sample_dist=path_sample_dist,
                    recompute=recompute,
                )
            )
        self.logger.info("Skeletonization done!")
        return calculated_orgs

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
    ):
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
            for org in orgs:
                if org.id in mcs_orgs:
                    meshes.append(org.get_mesh_mcs_colored(mcs_label))
                else:
                    meshes.append(org.mesh)
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
            self.logger.debug("About to compute `to_show`")
            to_show = [mmesh.compute()]
            self.logger.debug("Computed `to_show`")

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

        return show(to_show)

    def show_plotly(
        self,
        ids: str = "*",
        show_curvature: bool = False,
        show_skeleton: bool = False,
        mcs_label: Optional[str] = None,
        height: int = 800,
    ):
        orgs = self.get_organelles(ids=ids)

        if mcs_label and mcs_label not in self._mcs_labels:
            raise ValueError(
                f"MCS label {mcs_label} not found in project please "
                + "run search_mcs with the desired label first"
            )

        mcs_filter_ids = None
        if mcs_label:
            mcs_filter_ids = [org.id for org in orgs]

        # draw figure
        fig = go.Figure()
        for org in orgs:
            fig.add_traces(
                org.plotly_mesh(
                    show_curvature=show_curvature,
                    show_skeleton=show_skeleton,
                    mcs_label=mcs_label,
                    mcs_filter_ids=mcs_filter_ids,
                )
            )
            if show_skeleton and org.skeleton is not None:
                fig.add_traces(org.plotly_skeleton())
                sampled_path = org.sampled_skeleton[0]
                try:
                    sampled_scatter = go.Scatter3d(
                        x=sampled_path[:, 0],
                        y=sampled_path[:, 1],
                        z=sampled_path[:, 2],
                        mode="markers",
                        name=f"sampled_path_{org.id}",
                        marker=dict(size=1, color="red"),
                    )
                    fig.add_trace(sampled_scatter)
                except ValueError:
                    self.logger.warning(org.id, sampled_path, org.skeleton)
                    raise ValueError("sampled_path is not valid")

        fig.update_layout(
            scene=dict(
                xaxis=dict(title="", showticklabels=False, showgrid=False),
                yaxis=dict(title="", showticklabels=False, showgrid=False),
                zaxis=dict(title="", showticklabels=False, showgrid=False),
                aspectmode="cube",
            ),
            height=height,
        )

        return fig

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
            distance_matrix.apply(lambda row: row.min() < filter_distance, axis=1)
        ]

        # Convert the filtered DataFrame to a dictionary
        filtered_df_dict = filtered_df.to_dict("index")
        output_filtered_dict = defaultdict(list)
        # For each entry in the dictionary, replace the values with the column names that match the filter

        for col in filtered_df_dict:
            for row, value in filtered_df_dict[col].items():
                if value < filter_distance and col != row:  # exclude self-contact
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

    def hist_distance_matrix(
        self,
        ids_source="*",
        ids_target="*",
    ):
        orgs_1 = self.get_organelle_ids(ids=ids_source)
        orgs_2 = self.get_organelle_ids(ids=ids_target)
        distance_matrix = self.distance_matrix.loc[orgs_1, orgs_2]
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=distance_matrix.mean().values.flatten()))
        fig.update_layout(
            xaxis_title_text="Distance",
            yaxis_title_text="Count",
            title="Distance matrix histogram",
        )
        return fig

    def hist_skeletons(self, ids="*", attribute="num_nodes"):
        """Plot the histogram from the skeleton info.

        :param ids: Filter id, defaults to "*"
        :type ids: str, optional
        :param attribute: which attribute to plot, defaults to "num_nodes".
            can be:
            "num_nodes": number of nodes in the skeleton
            "num_branch_points": number of branch points in the skeleton
            "end points": number of end points in the skeleton
            "total_length": total length of the skeleton
            "mean_length": mean length of the skeleton
            "longest_path": longest path in the skeleton

        :type attribute: str, optional
        :return: _description_
        :rtype: _type_
        """
        orgs = self.get_organelles(ids=ids)
        # drop organelles without skeleton
        valid_orgs = []
        for org in orgs:
            if org.skeleton is not None:
                valid_orgs.append(org.id)

        skeleton_info = self.skeleton_info.loc[valid_orgs]
        data = skeleton_info[attribute].values
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=data))
        fig.update_layout(
            xaxis_title_text=attribute,
            yaxis_title_text="Count",
            title=f"{attribute} histogram",
        )
        return fig

    def calculate_profiles(
        self,
        method: str,
        ids: str = "*",
        axis: str | tuple = "z",
        num_slices: int = 20,
        num_planes: int = 20,
        seed: int = 42,
        sample_distance: float = 0.1,
    ):
        """
        API to calculate 2D profile metrics for organelles.

        :param method: The slicing method ("Fixed Axis", "Random Planes", "Skeleton Perpendicular").
        :param ids: Filter string for the organelle ids, defaults to "*".
        :param axis: The axis to slice along (for Fixed Axis).
        :param num_slices: The number of slices to make (for Fixed Axis).
        :param num_planes: The number of random planes to generate (for Random Planes).
        :param seed: Random seed (for Random Planes).
        :param sample_distance: Distance between sample points (for Skeleton Perpendicular).

        :return: A pandas DataFrame containing the profile statistics.
        """
        # Import locally to avoid circular dependencies if necessary
        from organelle_morphology.profile_calculations import ProfileCalculator

        calculator = ProfileCalculator(self)

        if method == "Fixed Axis":
            calculator.calculate_profile_lengths(
                ids=ids, axis=axis, num_slices=num_slices
            )
        elif method == "Random Planes":
            calculator.calculate_random_profiles(
                ids=ids, num_planes=num_planes, seed=seed
            )
        elif method == "Skeleton Perpendicular":
            calculator.calculate_skeleton_profiles(
                ids=ids, sample_distance=sample_distance
            )
        else:
            raise ValueError(f"Unknown profile calculation method: {method}")

        return calculator.get_dataframe()

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

    def n_oranelles(self) -> dict[str, int]:
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

    def filter_organelles_by_size(self, organelle_type, cutoff):
        """Take the largest entries of the specified organelle type
        until their combined volume reaches the cutoff value.
        Adds the remaining organelles to the permanent_blacklist.

        Args:
            organelle_type: The desired organelle type to perform the filter on.
                e.g "mito" or "er".
            cutoff: The cutoff value between 0 and 1.
        """
        geo_props = self.geometric_properties
        self.logger.info(
            f"Filtering organelles of type {organelle_type} to the largest "
            f"organelles that make up {cutoff * 100}% of the total volume."
        )

        df_sorted = geo_props.loc[
            geo_props.index.str.contains(organelle_type)
        ].sort_values("voxel_volume", ascending=False)
        df_sorted["cumulative_volume"] = df_sorted["voxel_volume"].cumsum()

        # get the n largest organelles that make up the cutoff volume of the total volume
        total_volume = df_sorted["voxel_volume"].sum()
        cutoff_volume = total_volume * cutoff
        df_filtered = df_sorted[df_sorted["cumulative_volume"] <= cutoff_volume]

        self.logger.info(f"Filtering {len(df_filtered)} organelles.")

        # now invert the filter to find the blacklisted organelles

        self.permanent_blacklist = df_sorted.index.difference(
            df_filtered.index
        ).tolist()

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

    def get_caches(self) -> list[Cache]:
        caches = []
        cs = self.cache_settings
        cache_dir = cs["cache_root"] / f"cache_{cs['project_name']}"
        messages = ["*** List of Caches: ***"]
        if cache_dir.exists():
            messages.append(str(cache_dir))
            for source in filter(lambda f: f.is_dir(), (cache_dir).iterdir()):
                messages.append(f"├─ /{source.name}")
                for level in filter(lambda f: f.is_dir(), source.iterdir()):
                    messages.append(f"│  ├─ /{level.name}")
                    for clip_dir in filter(lambda f: f.is_dir, level.iterdir()):
                        messages.append(f"│  │  ├─ /{clip_dir.name}")
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

        self.logger.info("\n".join(messages))
        self.logger.info(f"Found {len(caches)} caches for project {self.path.name}")
        return caches

    def add_stat(self, stat):
        self._stats.append(stat)

    @property
    def stats(self):
        return self._stats

    def get_stat_stats(self):
        """Generate some meta statistics
        about the already calculated statistics objects

        Returns:
            dict: Number of each statistics type available.
        """
        desc = "Collected statistics:\n"
        stat_count = defaultdict(int)
        desc += f"{'number of stats':<20}: {len(self.stats)}\n"

        for stat in self.stats:
            stat_count[stat.name] += 1
        for stat, count in stat_count.items():
            desc += f"{stat:<18}: {count}\n"

        self.logger.info(desc)
        return stat_count

    def clear_memory_cache(self):
        """(Re)initialize project-level storage."""

        self._basic_geometric_properties = {}
        self._mesh_properties = {}
        self._geometric_properties = {}
        self._curvature_map = {}
        self._mcs_labels = {}  # {label: {max_distance: float, min_distance: float}}
        self._max_compute_distance = 0.0
        self._cache = None
        self._stats = []

    def clear_caches(self, clear_disk=False):
        """Clear all caches related to this project, optionally also from disk"""

        for source in self.sources.values():
            source.clear_memory_cache()
        self.clear_memory_cache()
        self.logger.debug("Cleared memory cache of all sources.")

        if clear_disk:
            # iterate over what is on disk rather than the currently loaded sources
            i = -1
            for i, cache in enumerate(self.get_caches()):
                cache.clear_disk_cache()
            self.logger.info(f"Deleted {i + 1} caches from disk.")
