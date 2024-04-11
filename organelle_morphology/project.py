from organelle_morphology.organelle import Organelle, organelle_types
from organelle_morphology.source import DataSource
from organelle_morphology.util import disk_cache, parallel_pool

import json
import os
import pathlib
import trimesh
import logging

import numpy as np
import pandas as pd

from collections import defaultdict
from scipy.spatial import distance
import plotly.graph_objects as go


def load_metadata(project_path: pathlib.Path) -> tuple[pathlib.Path, dict]:
    """Load the project metadata JSON file

    :param project_path:
        The path to the project directory. The project metadata JSON file
        is expected to be in this directory.
    """

    # This might be a CebraEM project
    if os.path.exists(project_path / "project.json"):
        with open(project_path / "project.json", "r") as f:
            data = json.load(f)
            if len(data["datasets"]) != 1:
                raise ValueError("Only single dataset projects are supported")

            return load_metadata(project_path / data["datasets"][0])

    # This might be a mobie project
    if os.path.exists(project_path / "dataset.json"):
        with open(project_path / "dataset.json", "r") as f:
            return project_path, json.load(f)

    raise FileNotFoundError(
        "Could not find project.json or dataset.json in the given directory"
    )


def _picklable_mesh_extractor(organelle):
    return organelle.mesh


def _picklable_distance_calculation(meshes, i):
    col_manager_test = trimesh.collision.CollisionManager()
    col_manager_test.add_object("mesh1", meshes[i])
    distances = []
    for j in np.arange(i + 1, len(meshes)):
        distances.append(col_manager_test.min_distance_single(meshes[j]))

    # prox_query = trimesh.proximity.ProximityQuery(meshes[i])
    # distance = prox_query.on_surface(meshes[j].vertices)[1]
    return distances


class Project:
    def __init__(
        self,
        project_path: pathlib.Path | str = os.getcwd(),
        clipping: tuple[tuple[float]] | None = None,
        compression_level: int = 0,
    ):
        """Instantiate an EM project

        The given path is expected to contain either a CebraEM or Mobie
        project JSON file. This is a lazy operation. No data except metadata
        is loaded until it is required.

        :param project_path:
            The location of the CebraEM/Mobie project

        :param clipping:
            If not None, the data is clipped with the given lower left and the given
            upper right corner as the bounding box of the clipping. Coordinates are
            expected to be in micrometer.

        :param compression_level:
            The compression level at which we operate. This is used to determine
            the resolution of the data that we work with. The default of 0
            corresponds to the highest resolution.
        """

        if isinstance(project_path, str):
            project_path = pathlib.Path(project_path)

        self._project_path = project_path

        # Identify the directory that contains project metadata JSON
        self._dataset_json_directory, self._project_metadata = load_metadata(
            project_path
        )

        if clipping is not None:
            clipping = np.array(clipping)
            if not np.all(clipping[0] < clipping[1]):
                raise ValueError("Clipping lower left must be smaller than upper right")

            if not np.all(clipping[0] > 0) or not np.all(clipping[1] < 1):
                raise ValueError("Clipping must be in [0, 1]^3")

            if len(clipping) != 2 or len(clipping[0]) != 3 or len(clipping[1]) != 3:
                raise ValueError("Clipping must be a tuple of two tuples of length 3")

        self._clipping = clipping

        # The dictionary of data sources that we have added
        self._sources = {}

        self._basic_geometric_properties = {}
        self._mesh_properties = {}

        self._geometric_properties = {}

        self._meshes = {}
        self._distance_matrix = None
        self._morphology_map = {}

        self._mcs_labels = {}

        # The compression level at which we operate
        if compression_level < 0:
            raise ValueError(f"Compression level must be >= 0, got {compression_level}")

        self._compression_level = compression_level

        self.logger = logging.getLogger(project_path.stem)
        self.logger.setLevel(logging.DEBUG)  # Set logger's level to INFO
        self.logger.propagate = False
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(f"{project_path.stem}.log")

        # Set levels - INFO for console, DEBUG for file
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.DEBUG)

        # Create formatters and add it to handlers
        c_format = logging.Formatter("%(levelname)s - %(message)s")
        f_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)
        self.logger.info(f"\n ---- New Project {project_path} loaded----\n")

    @property
    def path(self):
        return self._project_path

    def available_sources(self) -> list[str]:
        """List the data sources that are available in the project."""

        return list(self.metadata["sources"].keys())

    def add_source(
        self, source: str = None, organelle: str = None, background_label: int = 0
    ) -> None:
        """Connect a data source in the project with an organelle type

        :param source:
            The name of the data source in the original dataset. Must be
            one of the names returned by available_sources.

        :param organelle:
            The name of the organelle that is labelled in the data source.
            Must be on the strings returned by organelle_morphology.organelle_types
        :param background_label:
            The label in the data source that is used to encode the background.
            Assumed to be 0.
        """

        if source not in self.available_sources():
            raise ValueError(f"Unknown data source {source}")

        if organelle not in organelle_types():
            raise ValueError(f"Unknown organelle type {organelle}")

        # Instantiate the new source
        source_path = self.metadata["sources"][source]["image"]["imageData"]["bdv.n5"][
            "relativePath"
        ]
        source_obj = DataSource(
            self,
            self._dataset_json_directory / source_path,
            organelle,
            background_label,
        )

        self.logger.info("Adding source %s", source)

        # Double-check that it provides the current compression level
        if self.compression_level >= len(source_obj.metadata["downsampling"]):
            raise ValueError(
                f"Compression level {self.compression_level} is not available for source {source}"
            )

        self._sources[source] = source_obj

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
        orgs = self.organelles(ids=ids, return_ids=False)

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
        orgs = self.organelles(ids=ids, return_ids=False)

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
        mcs_label: str = None,
        height: int = 800,
    ):
        orgs = self.organelles(ids=ids, return_ids=False)

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
                except:
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
        :param attribute: Show names, number of contact sites ("n_mcs") or return the organelle objects ("objects"), defaults to "names"
        :type attribute: str, optional

        :return: Dictionary with the source organelle ids as keys and the target organelle ids or number of contact sites as values
        :rtype: dict
        """
        orgs_1 = self.organelles(ids=ids_source, return_ids=True)
        orgs_2 = self.organelles(ids=ids_target, return_ids=True)
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
                    if (
                        attribute in ["names", "objects"]
                        and row in output_filtered_dict.keys()
                    ):
                        # this is not done when interested in the number of neighbors for each organelle.
                        continue  # skip entries that are already present as keys, this avoids doubling

                    output_filtered_dict[col].append(row)

        if attribute == "n_mcs":
            for key in output_filtered_dict.keys():
                output_filtered_dict[key] = len(output_filtered_dict[key])

        elif attribute == "objects":
            obj_output_dict = defaultdict(list)
            for key in output_filtered_dict.keys():
                new_key = self.organelles(ids=key, return_ids=False)[0]

                for key_target in output_filtered_dict[key]:
                    if key_target not in obj_output_dict.keys():
                        obj_output_dict[new_key].extend(
                            self.organelles(ids=key_target, return_ids=False)
                        )
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
        orgs_1 = self.organelles(ids=ids_source, return_ids=True)
        orgs_2 = self.organelles(ids=ids_target, return_ids=True)
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

    def _calc_mcs(self, org_source, org_target, max_distance, min_distance=0):
        # extracted this function to use parallel pool

        cache_name = f"mcs_{org_source.id}_{org_target.id}_{max_distance}_{min_distance}_{self.compression_level}"
        with disk_cache(self, cache_name) as cache:
            if cache_name not in cache:
                mesh_source = org_source.mesh
                mesh_target = org_target.mesh

                for mesh in [mesh_source, mesh_target]:
                    trimesh.repair.fix_inversion(mesh)
                    trimesh.repair.fix_winding(mesh)
                    trimesh.repair.fix_normals(mesh)

                (
                    distance_target_source,
                    index_target_source,
                ) = mesh_source.nearest.vertex(org_target.mesh.vertices)
                (
                    distance_source_target,
                    index_source_target,
                ) = mesh_target.nearest.vertex(org_source.mesh.vertices)

                # if a minimum threshold is given we want to exclude any mcs that have a part that is closer than the minimum distance.
                # this avoids having overlapping contact sites when filtering multiple distances.
                if (
                    np.min(distance_target_source) < min_distance
                    or np.min(distance_source_target) < min_distance
                ):
                    return None

                # if we want to now which vertices of mesh 1 are close to mesh 2
                # we need to run the search on org2 with the points of org1
                # this will give us a distance and an index for each vertex of org1

                mask_1 = (distance_source_target > min_distance) & (
                    distance_source_target < max_distance
                )
                mask_2 = (distance_target_source > min_distance) & (
                    distance_target_source < max_distance
                )

                ind_source = np.nonzero(mask_1)
                ind_target = np.nonzero(mask_2)

                # now we now which vertex indeces of mesh 1 and 2 are suitable.

                # figure out if the surface point is facing the other mesh
                # get the normals of the vertices

                normals_source = mesh_source.vertex_normals[ind_source]
                normals_target = mesh_target.vertex_normals[ind_target]

                # get the direction of the vector between the two points
                vector_source = (
                    mesh_target.vertices[index_source_target[ind_source]]
                    - mesh_source.vertices[ind_source]
                )
                vector_target = (
                    mesh_source.vertices[index_target_source[ind_target]]
                    - mesh_target.vertices[ind_target]
                )

                # calculate the dot product of the normals and the vectors
                dot_source = np.sum(normals_source * vector_source, axis=1)
                dot_target = np.sum(normals_target * vector_target, axis=1)

                # calculate the dot product for the opposite orientation of the normals
                dot_source_opposite = np.sum(-normals_source * vector_source, axis=1)
                dot_target_opposite = np.sum(-normals_target * vector_target, axis=1)

                # if the dot product is positive the surface is facing the other mesh
                # choose the orientation that gives the most vertices facing the other mesh

                ind_source_normal = ind_source[0][dot_source > 0]
                ind_source_inverse = ind_source[0][dot_source_opposite > 0]

                ind_target_normal = ind_target[0][dot_target > 0]
                ind_target_inverse = ind_target[0][dot_target_opposite > 0]

                # for some reason some of the meshes behave differently when comparing the dot product
                # of the normals and the vectors.
                # This is a quick fix to choose the orientation by comparing the mean distance
                # of the vertices selected by the dot product.

                if ind_source_normal.size > 0 and ind_source_inverse.size > 0:
                    if np.mean(distance_source_target[ind_source_normal]) < np.mean(
                        distance_source_target[ind_source_inverse]
                    ):
                        ind_source = ind_source_normal
                    else:
                        ind_source = ind_source_inverse
                else:
                    ind_source = (
                        ind_source_normal
                        if ind_source_normal.size > 0
                        else ind_source_inverse
                    )

                if ind_target_normal.size > 0 and ind_target_inverse.size > 0:
                    if np.mean(distance_target_source[ind_target_normal]) < np.mean(
                        distance_target_source[ind_target_inverse]
                    ):
                        ind_target = ind_target_normal
                    else:
                        ind_target = ind_target_inverse
                else:
                    ind_target = (
                        ind_target_normal
                        if ind_target_normal.size > 0
                        else ind_target_inverse
                    )

                result_dist_s_t = distance_source_target[ind_source]
                result_dist_t_s = distance_target_source[ind_target]

                cache[cache_name] = (
                    ind_source,
                    ind_target,
                    result_dist_s_t,
                    result_dist_t_s,
                )

            else:
                ind_source, ind_target, result_dist_s_t, result_dist_t_s = cache[
                    cache_name
                ]

        return ind_source, ind_target, result_dist_s_t, result_dist_t_s

    def search_mcs(
        self,
        mcs_label,
        ids_source="*",
        ids_target="*",
        max_distance=0.5,
        min_distance=0,
    ):
        if mcs_label in self._mcs_labels:
            raise ValueError(
                f"MCS label {mcs_label} already exists in project please use a different label."
            )

        filtered_distance_dict = self.distance_filtering(
            ids_source=ids_source,
            ids_target=ids_target,
            filter_distance=max_distance,
            attribute="objects",
        )
        n_pairs = sum(
            [len(filtered_distance_dict[key]) for key in filtered_distance_dict.keys()]
        )

        self.logger.info(
            "Found %s organelle pairs closer than %s um.",
            n_pairs,
            max_distance,
        )

        with parallel_pool(n_pairs) as (pool, pbar):
            for org_source in filtered_distance_dict.keys():
                for org_target in filtered_distance_dict[org_source]:
                    results = pool.apply_async(
                        self._calc_mcs,
                        (org_source, org_target, max_distance, min_distance),
                        callback=lambda _: pbar.update(),
                    ).get()

                    if results is None:
                        continue

                    (
                        org_source_close_vertices,
                        org_target_close_vertices,
                        result_dist_s_t,
                        result_dist_t_s,
                    ) = results

                    org_source.add_mcs(
                        mcs_label,
                        org_target.id,
                        org_source_close_vertices,
                        result_dist_s_t,
                    )
                    org_target.add_mcs(
                        mcs_label,
                        org_source.id,
                        org_target_close_vertices,
                        result_dist_t_s,
                    )
                    self.logger.debug(
                        "Found MCS between %s and %s", org_source.id, org_target.id
                    )
            self._mcs_labels[mcs_label] = {
                "max_distance": max_distance,
                "min_distance": min_distance,
            }

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
        orgs_1 = self.organelles(ids=ids_source, return_ids=True)
        orgs_2 = self.organelles(ids=ids_target, return_ids=True)
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
        orgs = self.organelles(ids=ids, return_ids=False)
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
    def compression_level(self):
        """The compression level used for our computations."""

        return self._compression_level

    def calculate_meshes(self):
        """Trigger the calculation of meshes for all organelles"""

        with parallel_pool(len(self.organelles())) as (pool, pbar):
            for organelle in self.organelles():
                pool.apply_async(
                    _picklable_mesh_extractor,
                    (organelle,),
                    callback=lambda _: pbar.update(),
                ).get()

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
        self.calculate_meshes()

        properties = {}

        with disk_cache(
            self, f"geometric_properties_{self.compression_level}"
        ) as cache:
            if f"geometric_properties_{self.compression_level}" not in cache:
                for organelle in self.organelles():
                    properties[organelle.id] = organelle.geometric_data | organelle.mesh_properties

                cache[f"geometric_properties_{self.compression_level}"] = properties

            properties = cache[f"geometric_properties_{self.compression_level}"]

        return pd.DataFrame(properties).T

    def get_mcs_properties(self, ids="*", mcs_filter=None):
        """The properties of the MCS between organelles


        :param ids: Filter id, defaults to "*"
        :type ids: str, optional
        :return: _description_
        :rtype: _type_

        :param mcs_filter: Filter the MCS properties by the given mcs_filter, defaults to None
        :type mcs_filter: str|list, optional


        """

        orgs = self.organelles(ids=ids, return_ids=False)

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
        for source_key, source in self._sources.items():
            self._morphology_map[source_key] = source.morphology_map

        return self._morphology_map

    @property
    def distance_matrix(self):
        """
        Returns the distance matrix of all organeles in the project in micro meters.

        :return: _description_
        :rtype: _type_
        """
        self.logger.info("Calculating distance matrix")
        # Trigger the calculation of all meshes in a parallel pool
        active_sources = list(self._sources.keys())
        with disk_cache(
            self, f"distance_matrix_{active_sources}_{self.compression_level}"
        ) as cache:
            if (
                f"distance_matrix_{active_sources}_{self.compression_level}"
                not in cache
            ):
                self.calculate_meshes()

                meshes = []
                organelles = []
                for organelle in self.organelles():
                    organelles.append(organelle.id)
                    meshes.append(organelle.mesh)

                num_rows = len(meshes)
                distance_matrix = np.zeros((num_rows, num_rows))
                with parallel_pool(num_rows) as (pool, pbar):
                    for i in np.arange(num_rows):
                        # for j in np.arange(i + 1, num_rows):
                        result = pool.apply_async(
                            _picklable_distance_calculation,
                            (
                                meshes,
                                i,
                            ),
                            callback=lambda _: pbar.update(),
                        )
                        distances = result.get()
                        distance_matrix[i, i] = 0
                        distance_matrix[i, i + 1 :] = distances
                        distance_matrix[i + 1 :, i] = distances

                distance_df = pd.DataFrame(
                    distance_matrix,
                    index=organelles,
                    columns=organelles,
                )
                cache[
                    f"distance_matrix_{active_sources}_{self.compression_level}"
                ] = distance_df

            return cache[f"distance_matrix_{active_sources}_{self.compression_level}"]

    @property
    def metadata(self):
        """The project metadata stored in the project JSON file"""

        return self._project_metadata

    @property
    def clipping(self):
        """The subcube of the original data that we work with"""

        if self._clipping is None:
            return None
        else:
            return self._clipping

    def organelles(
        self, ids: list[str] = "*", return_ids: bool = False
    ) -> list[Organelle] | list[str]:
        """Return a list of organelles found in the dataset

        This requires previous adding of data sources using add_source.
        Depending on the return_ids argument, either the organelles are
        returned as objects that can further inspected and used for analysis
        or the list of organelle ids are returned. The ids parameter
        is used to filter based on organelle ids.

        :param ids:
            The filtering expression for organelle ids to return. The default
            of "*" returns all organelles. (What other syntax would we allow? fnmatch?)

        :param return_ids:
            Whether to only return ids or the actual organelle objects.
        """

        result = []
        if isinstance(ids, str):
            ids = [ids]

        for id in ids:
            for source in self._sources.values():
                result.extend(source.organelles(id, return_ids))

        return result
