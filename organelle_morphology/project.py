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

    @property
    def compression_level(self):
        """The compression level used for our computations."""

        return self._compression_level

    def calculate_meshes(self):
        """Trigger the calculation of meshes for all organelles"""

        with parallel_pool() as pool:
            for organelle in self.organelles():
                pool.apply_async(_picklable_mesh_extractor, (organelle,)).get()

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
    def morphology_map(self):
        """Get the morphology map for all organelles"""

        # results should be saved on a source level
        for source_key, source in self._sources.items():
            self._morphology_map[source_key] = source.morphology_map

        return self._morphology_map

    @property
    def distance_matrix(self):
        # Trigger the calculation of all meshes in a parallel pool
        self.calculate_meshes()

        with disk_cache(self, "distance_matrix") as cache:
            if f"distance_matrix_{self.compression_level}" not in cache:
                meshes = []
                organelles = []
                for organelle in self.organelles():
                    organelles.append(organelle.id)
                    meshes.append(organelle.mesh)

                num_rows = len(meshes)
                distance_matrix = np.zeros((num_rows, num_rows))

                for i in np.arange(num_rows):
                    for j in np.arange(i + 1, num_rows):
                        col_manager_test = trimesh.collision.CollisionManager()
                        col_manager_test.add_object("mesh1", meshes[i])
                        col_manager_test.add_object("mesh2", meshes[j])
                        distance = col_manager_test.min_distance_internal()

                        distance_matrix[i, j] = distance
                        distance_matrix[j, i] = distance

                distance_df = pd.DataFrame(
                    distance_matrix,
                    index=organelles,
                    columns=organelles,
                )
                cache[f"distance_matrix_{self.compression_level}"] = distance_df

            return cache[f"distance_matrix_{self.compression_level}"]

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
