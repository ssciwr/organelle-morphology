from organelle_morphology.organelle import Organelle, organelle_types
from organelle_morphology.source import DataSource
from organelle_morphology.util import disk_cache, parallel_pool

import json
import os
import pathlib
import numpy as np

import numpy as np
import pandas as pd
import trimesh
from functools import reduce
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

        # The compression level at which we operate
        if compression_level < 0:
            raise ValueError(f"Compression level must be >= 0, got {compression_level}")

        self._compression_level = compression_level

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
        path_samplple_dist: float = 0.1,
    ):
        orgs = self.organelles(ids=ids, return_ids=False)
        for org in orgs:
            if skip_existing and org.skeleton is not None:
                continue

            org._generate_skeleton(
                skeletonization_type="wavefront",
                theta=theta,
                waves=waves,
                step_size=step_size,
                path_samplple_dist=path_samplple_dist,
            )

    def skeletonize_vertex_clusters(
        self,
        ids: str = "*",
        theta: float = 0.4,
        epsilon: float = 0.1,
        sampling_dist: float = 0.1,
        skip_existing=False,
        path_samplple_dist: float = 0.1,
    ):
        orgs = self.organelles(ids=ids, return_ids=False)
        for org in orgs:
            if skip_existing and org.skeleton is not None:
                continue
            org._generate_skeleton(
                skeletonization_type="vertex_clusters",
                theta=theta,
                epsilon=epsilon,
                sampling_dist=sampling_dist,
                path_samplple_dist=path_samplple_dist,
            )

    def show(
        self,
        ids: str = "*",
        show_morphology: bool = False,
        show_skeleton: bool = False,
        height: int = 800,
    ):
        orgs = self.organelles(ids=ids, return_ids=False)

        # draw figure
        fig = go.Figure()
        for org in orgs:
            fig.add_traces(
                org.plotly_mesh(
                    show_morphology=show_morphology, show_skeleton=show_skeleton
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
                    print(org.id, sampled_path, org.skeleton)
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

    def distance_analysis(self, ids_source="*", ids_target="*", attribute="dist"):
        """get more information about the distance between two filtered organelle lists.
           These can be from one or more types depending on the given filter.

        :param ids_1: _description_
        :type ids_1: _type_
        :param ids_2: _description_
        :type ids_2: _type_
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
            df_mean["std of target distance"] = distance_matrix.std(axis=1).values
            df_mean.rename(
                columns={0: "id_source", 1: "mean distance to target"}, inplace=True
            )
            return df_mean

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
        for organelle in self.organelles():
            properties[organelle.id] = (
                organelle.geometric_data | organelle.mesh_properties
            )

        return pd.DataFrame(properties).T

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
                # num_distances = (num_rows * (num_rows - 1)) / 2
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
