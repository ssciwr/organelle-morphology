from typing import Callable, Optional
from organelle_morphology.organelle import Organelle
from organelle_morphology.source import DataSource
from organelle_morphology.util import CACHE_DIR, Cache, disk_cache, get_logger
from organelle_morphology.distance_calculations import (
    generate_distance_matrix,
    _generate_mcs,
)

from pathlib import Path
from dask.distributed import Client

import numpy as np
import pandas as pd

from collections import defaultdict
import plotly.graph_objects as go


clipping_type = tuple[tuple[float, float, float], tuple[float, float, float]]


class Project:
    def __init__(
        self,
        project_path: Path | str,
        clipping: Optional[clipping_type] = None,
        compression_level: str = "s0",
        loglevel: Optional[str] = None,
    ):
        """Instantiate an EM project

        The given path is expected to contain either a CebraEM or Mobie
        project JSON file. This is a lazy operation. No data except metadata
        is loaded until it is required.

        :param project_path:
            The location of the project.

        :param clipping:
            If not None, the data is clipped with the given lower left and the given
            upper right corner as the bounding box of the clipping. Coordinates are
            expected to be in micrometer.

        :param compression_level:
            The compression level at which we operate. This is used to determine
            the resolution of the data that we work with. The default of 0
            corresponds to the highest resolution.
        """

        self._project_path = Path(project_path)

        if not self.path.exists():
            self.path.mkdir()

        self._clipping = None
        if clipping is not None:
            self.clipping = clipping

        # The dictionary of data sources that we have added
        self.sources: dict[str, DataSource] = {}

        self._basic_geometric_properties = {}
        self._mesh_properties = {}

        self._geometric_properties = {}

        self._meshes = {}
        self._distance_matrix = None
        self._morphology_map = {}

        # {label: {max_distance: float, min_distance: float}}
        self._mcs_labels = {}

        # this permanent filter can be used to soft drop organelles from the project
        # for example if they are too small to be considered
        self.permanent_whitelist = []
        self.permanent_blacklist = []

        # The compression level at which we operate
        self.compression_level = compression_level

        self.logger = get_logger(self.path.with_suffix(".log"))
        self.set_loglevel(loglevel)
        self.logger.info(f"\n ---- New Project {self.path} loaded ----\n")

        # callables will be updated on demand
        self._cache_settings = {
            "project_path": lambda: self.path,
            "clipping": lambda: str(self.clipping).replace("\n", ""),
            "level": lambda: str(self.compression_level),
            "disk": True,
        }

        # debug help
        self.use_cache = True
        self.debug = False

        self.client = Client()

    def set_loglevel(self, loglevel: Optional[str]):
        if loglevel:
            self.logger.handlers[0].setLevel(loglevel)
            self.logger.debug(f"Set logging level to: {loglevel}")

    @property
    def path(self) -> Path:
        return self._project_path

    @property
    def cache_settings(self):
        d = dict()
        for k, v in self._cache_settings.items():
            if callable(v):
                v = v()
            d[k] = v
        return d

    def add_source(
        self,
        xml_path: Path | str,
        organelle: Optional[str] = None,
        background_label: int = 0,
    ) -> DataSource:
        """Connect a data source in the project with an organelle type

        Args:
            source_path: The path to the xml source to add.
            organelle: The name of the organelle that is labelled in the data source.
                Must be on the strings returned by organelle_morphology.organelle_types
            background_label: The label in the data source that is used to encode the background.
                Assumed to be 0.

        Returns:
            DataSource: Also accessable as `project[xml_name]`

        Raises:
            ValueError: Source already loaded
            ValueError: Compression level of project not available
        """
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

        self.logger.info(f"Adding source {source_obj.metadata['name']}")

        # Double-check that it provides the current compression level
        if self.compression_level not in source_obj.metadata["levels"]:
            raise ValueError(
                f"Compression level {self.compression_level} is not available "
                f"for source {source_obj.metadata['name']}.\n Available levels:"
                f"{source_obj.metadata['levels']}"
            )
        self.sources[xml_path.stem] = source_obj
        return source_obj

    def skeletonize_wavefront(
        self,
        ids: str = "*",
        theta: float = 0.4,
        waves: int = 1,
        step_size: int = 2,
        skip_existing=False,
        path_sample_dist: float = 0.1,
    ):
        """Note that some meshes will be skipped if they are too small to be skeletonized.

        :param ids: _description_, defaults to "*"
        :type ids: str, optional
        :param theta: _description_, defaults to 0.4
        :type theta: float, optional
        :param waves: _description_, defaults to 1
        :type waves: int, optional
        :param step_size: _description_, defaults to 2
        :type step_size: int, optional
        :param skip_existing: _description_, defaults to False
        :type skip_existing: bool, optional
        :param path_sample_dist: _description_, defaults to 0.1
        :type path_sample_dist: float, optional
        """
        orgs = self.organelles(ids=ids)

        start_logger_str = (
            f"Starting Skeleton wavefront generation for {len(orgs)} organelles. "
        )
        if skip_existing:
            start_logger_str += "Skipping existing skeletons."
        self.logger.info(start_logger_str)

        for org in orgs:
            if skip_existing and org.skeleton is not None:
                continue

            org._generate_skeleton(
                skeletonization_type="wavefront",
                theta=theta,
                waves=waves,
                step_size=step_size,
                path_sample_dist=path_sample_dist,
            )

    def skeletonize_vertex_clusters(
        self,
        ids: str = "*",
        theta: float = 0.4,
        epsilon: float = 0.1,
        sampling_dist: float = 0.1,
        skip_existing=False,
        path_sample_dist: float = 0.1,
    ):
        orgs = self.organelles(ids=ids)

        start_logger_str = (
            f"Starting Skeleton wavefront generation for {len(orgs)} organelles. "
        )
        if skip_existing:
            start_logger_str += "Skipping existing skeletons."
        self.logger.info(start_logger_str)

        for org in orgs:
            if skip_existing and org.skeleton is not None:
                continue
            org._generate_skeleton(
                skeletonization_type="vertex_clusters",
                theta=theta,
                epsilon=epsilon,
                sampling_dist=sampling_dist,
                path_sample_dist=path_sample_dist,
            )

    def show(
        self,
        ids: str = "*",
        show_morphology: bool = False,
        show_skeleton: bool = False,
        mcs_label: Optional[str] = None,
        height: int = 800,
    ):
        orgs = self.organelles(ids=ids)

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
                    show_morphology=show_morphology,
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
        self, ids_source="*", ids_target="*", filter_distance=0.01, attribute="names"
    ):
        """Filter the organelles based on the distance between two filtered organelle lists.
              These can be from one or more types depending on the given filter.

        :param ids_source: Filter string for the source ids, defaults to "*"
        :type ids_source: str, optional
        :param ids_target: filter string for the target ids, defaults to "*"
        :type ids_target: str, optional
        :param filter_distance: The distance in micro meter used for filtering, defaults to 0.01
        :type filter_distance: float, optional
        :param attribute: Show names, number of contacts (contacts) or return the organelle objects ("objects"), defaults to "names"
        :type attribute: str, optional

        :return: Dictionary with the source organelle ids as keys and the target organelle ids or number of contact sites as values
        :rtype: dict
        """

        if attribute not in ["names", "contacts", "objects"]:
            raise ValueError(
                f"Attribute must be one of 'names', 'contacts' or 'objects' but is {attribute}"
            )

        orgs_1 = self.organelle_ids(ids=ids_source)
        orgs_2 = self.organelle_ids(ids=ids_target)
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
                new_key = self.organelles(ids=key)[0]

                for key_target in output_filtered_dict[key]:
                    if key_target not in obj_output_dict.keys():
                        obj_output_dict[new_key].extend(self.organelles(ids=key_target))
            return obj_output_dict

        return output_filtered_dict

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
        orgs_1 = self.organelle_ids(ids=ids_source)
        orgs_2 = self.organelle_ids(ids=ids_target)
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
        self, mcs_label, max_distance, min_distance=0, override_mcs_label=False
    ):
        """
        This function is used to search for MC within a project.
        Pairs will only be selected when their minimum mesh distance is between
        the requested min and max distance.



        Args:
            project (Project): The project object containing the distance matrix and organelles.
            mcs_label (str): The label for the MCS pairs.
            max_distance (float): The maximum distance for the MCS pairs.
            min_distance (float, optional): The minimum distance for the MCS pairs. Defaults to 0.

        Returns:
            None
        """

        if mcs_label in self._mcs_labels and not override_mcs_label:
            raise ValueError(f"MCS label {mcs_label} already exists in the project")

        _generate_mcs(self, mcs_label, max_distance, min_distance)

    @property
    def mcs_labels(self):
        mcs_str = ""
        for key, value in self._mcs_labels.items():
            mcs_str += (
                f"  {key}: {value['min_distance']}um - {value['max_distance']}um\n"
            )

        self.logger.info("Available MCS labels and their search radius: \n%s", mcs_str)
        return self._mcs_labels

    def hist_distance_matrix(
        self,
        ids_source="*",
        ids_target="*",
    ):
        orgs_1 = self.organelle_ids(ids=ids_source)
        orgs_2 = self.organelle_ids(ids=ids_target)
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
        orgs = self.organelles(ids=ids)
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

    @property
    def compression_level(self) -> str:
        """The compression level used for our computations."""

        return self._compression_level

    @compression_level.setter
    def compression_level(self, level: str):
        for s_name, s in self.sources.items():
            if level not in s.metadata["levels"]:
                raise ValueError(
                    f"Requested level {level} not available in source {s_name}!\n"
                    f"Levels in source: {s.metadata['levels']}"
                )
        if getattr(self, "_compression_level", None) != level:
            for source in self.sources.values():
                source.clear_memory_cache()

        self._compression_level = level

    def calculate_meshes(self):
        """Trigger the calculation of meshes for all organelles"""

        for source in self.sources.values():
            source.calculate_mesh()

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

        # Trigger the calculation of all meshes in a parallel pool

        properties = {}
        sources = list(self.sources.keys())
        cache_str = f"geometric_properties_{self.compression_level}_{sources}"
        with disk_cache(self, cache_str) as cache:
            if cache_str not in cache or not self.use_cache:
                self.calculate_meshes()

                for organelle in self.organelles():
                    properties[organelle.id] = (
                        organelle.geometric_data | organelle.mesh_properties
                    )

                cache[cache_str] = properties

            properties = cache[cache_str]

        # hide blacklisted organelles

        df = pd.DataFrame(properties).T

        valid_organelles = self.organelle_ids()
        df = df.loc[valid_organelles]

        return df

    def get_mcs_properties(self, ids="*", mcs_filter=None):
        """The properties of the MCS between organelles


        :param ids: Filter id, defaults to "*"
        :type ids: str, optional
        :return: _description_
        :rtype: _type_

        :param mcs_filter: Filter the MCS properties by the given mcs_filter, defaults to None
        :type mcs_filter: str|list, optional


        """

        orgs = self.organelles(ids=ids)

        mcs_properties = {}
        for org in orgs:
            org_mcs_data = org.mcs_dict
            for mcs_label, value in org_mcs_data.items():
                if mcs_filter and mcs_label not in mcs_filter:
                    continue
                mcs_properties[(mcs_label, org.id)] = value

        mcs_df = pd.DataFrame(mcs_properties).T
        mcs_df.sort_index(inplace=True)

        return mcs_df

    def get_mcs_overview(self, ids="*", mcs_filter=None):
        def _weighted_stats(x):
            # Calculate the weighted mean and standard deviation for 'mean_area' and 'mean_dist'
            mean_area = np.average(x["mean_area"], weights=x["n_contacts"])
            std_area = np.sqrt(
                np.average(
                    (x["std_area"] ** 2 + (x["mean_area"] - mean_area) ** 2),
                    weights=x["n_contacts"],
                )
            )
            mean_dist = np.average(x["mean_dist"], weights=x["n_contacts"])
            std_dist = np.sqrt(
                np.average(
                    (x["std_dist"] ** 2 + (x["mean_dist"] - mean_dist) ** 2),
                    weights=x["n_contacts"],
                )
            )

            # Calculate the sum, mean, and standard deviation for 'n_contacts' and 'total_area'
            total_contacts = x["n_contacts"].sum()
            mean_n_contacts = x["n_contacts"].mean()
            std_n_contacts = x["n_contacts"].std()
            total_area = x["total_area"].sum()
            mean_total_area = x["total_area"].mean()
            std_total_area = x["total_area"].std()

            new_columns = [
                ("overall", "total_contacts"),
                ("overall", "total_area"),
                ("per organelle", "mean_n_contacts"),
                ("per organelle", "std_n_contacts"),
                ("per organelle", "mean_total_area"),
                ("per organelle", "std_area"),
                ("per mcs", "mean_area"),
                ("per mcs", "std_area"),
                ("per mcs", "mean_dist"),
                ("per mcs", "std_dist"),
            ]
            new_index = pd.MultiIndex.from_tuples(new_columns)

            return pd.Series(
                [
                    total_contacts,
                    total_area,
                    mean_n_contacts,
                    std_n_contacts,
                    mean_total_area,
                    std_total_area,
                    mean_area,
                    std_area,
                    mean_dist,
                    std_dist,
                ],
                index=new_index,
            )

        mcs_df = self.get_mcs_properties(ids=ids, mcs_filter=mcs_filter)

        overview = mcs_df.groupby(level=0).apply(_weighted_stats)
        overview.sort_index(axis=1, inplace=True)
        return overview.T

    @property
    def skeleton_info(self):
        skeleton_data = {}
        for org in self.organelles():
            if org.skeleton is not None:
                skeleton_data[org.id] = org.skeleton_info
        return pd.DataFrame(skeleton_data).T.sort_values(
            by="num_nodes", ascending=False
        )

    @property
    def morphology_map(self):
        """Get the morphology map for all organelles"""

        # results should be saved on a source level
        for source_key, source in self.sources.items():
            self._morphology_map[source_key] = source.morphology_map

        return self._morphology_map

    @property
    def distance_matrix(self):
        """
        Returns the distance matrix of all organeles in the project in micro meters.

        :return: _description_
        :rtype: _type_
        """

        return generate_distance_matrix(self)

    def _add_permanent_whitelist(self, ids):
        """Add organelles to the permanent filter list

        :param ids: The organelle ids to be added to the permanent filter list
        :type ids: list
        """
        self.permanent_whitelist.extend(ids)

    def _add_permanend_blacklist(self, ids):
        """Add organelles to the permanent blacklist filter.
        For the time beeing only complete ids are supported.

        :param ids: The organelle ids to be added to the permanent filter list
        :type ids: list
        """
        self.permanent_blacklist.extend(ids)

    def filter_organelles_by_size(self, organelle_type, cutoff):
        """Take the largest entries of the specified organelle type
        until their combined volume reaches the cutoff value.

        :param organelle_type: The desired organelle type to perform the filter on. E.G "mito" or "er".
        :type organelle_type: str
        :param cutoff: The cutoff value between 0 and 1.
        :type cutoff: float

        """
        geo_props = self.geometric_properties
        self.logger.info(
            f"Filtering organelles of type {organelle_type} to the largest organelles that make up {cutoff * 100}% of the total volume."
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

        self._add_permanend_blacklist(
            df_sorted.index.difference(df_filtered.index).tolist()
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
        if (clipping is not None) and not all(a < b for a, b in zip(*clipping)):
            raise ValueError(
                "First clipping corner is lower left, second upper right. All "
                "coordinates of corner one must be smaller than of corner two"
            )
        if not np.all(np.array(clipping) == getattr(self, "clipping", None)):
            if getattr(self, "sources", False):
                for source in self.sources.values():
                    source.clear_memory_cache()

        if clipping is None:
            self._clipping = None
            return
        _clipping = np.array(clipping)

        if not np.all(_clipping[0] >= 0) or not np.all(_clipping[1] <= 1):
            raise ValueError("Clipping must be in [0, 1]^3")

        if len(_clipping) != 2 or len(_clipping[0]) != 3 or len(_clipping[1]) != 3:
            raise ValueError("Clipping must be a tuple of two tuples of length 3")
        self._clipping = _clipping

    def organelles(
        self,
        ids: str | list[str] = "*",
    ) -> list[Organelle]:
        """Return a list of organelles found in the dataset

        This requires previous adding of data sources using add_source.
        The ids parameter is used to filter based on organelle ids.

        :param ids:
            The filtering expression for organelle ids to return. The default
            of "*" returns all organelles. (What other syntax would we allow? fnmatch?)
        """

        result = []
        if isinstance(ids, str):
            ids = [ids]

        for id_ in ids:
            for source in self.sources.values():
                if id_ in self.permanent_blacklist:
                    continue

                result.extend(
                    source.organelles(
                        id_,
                        self.permanent_whitelist,
                        self.permanent_blacklist,
                    )
                )

        return result

    def organelle_ids(
        self,
        ids: str | list[str] = "*",
    ) -> list[str]:
        """Return a list of organelle ids found in the dataset

        This requires previous adding of data sources using add_source.
        The ids parameter is used to filter based on organelle ids.

        :param ids:
            The filtering expression for organelle ids to return. The default
            of "*" returns all organelles. (What other syntax would we allow? fnmatch?)
        """

        result = []
        if isinstance(ids, str):
            ids = [ids]

        for id_ in ids:
            for source in self.sources.values():
                if id_ in self.permanent_blacklist:
                    continue

                result.extend(
                    source.organelle_ids(
                        id_,
                        self.permanent_whitelist,
                        self.permanent_blacklist,
                    )
                )

        return result

    def get_caches(self):
        caches = []

        print(f"{CACHE_DIR / self.path.name}")
        for source in filter(
            lambda f: f.is_dir(), (CACHE_DIR / self.path.name).iterdir()
        ):
            print(f"├─ /{source.name}")

            for level in filter(lambda f: f.is_dir(), source.iterdir()):
                print(f"│  ├─ /{level.name}")
                for clip_dir in filter(lambda f: f.is_dir, level.iterdir()):
                    print(f"│  │  ├─ /{clip_dir.name}")
                    caches.append(
                        Cache(
                            project_path=self.path,
                            source=source.name,
                            level=level.name,
                            clipping=clip_dir.name,
                            disk=True,
                        )
                    )
        self.logger.info(f"Found {len(caches)} caches for project {self.path.name}")
        return caches
