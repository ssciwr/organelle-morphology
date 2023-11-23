from organelle_morphology.organelle import Organelle, organelle_types
from organelle_morphology.source import DataSource

import json
import os
import pathlib
import numpy as np


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


class Project:
    def __init__(
        self,
        project_path: pathlib.Path | str = os.getcwd(),
        clipping: tuple[tuple[float]] | None = None,
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
        """

        if isinstance(project_path, str):
            project_path = pathlib.Path(project_path)

        # Identify the directory that containes project metadata JSON
        self._dataset_json_directory, self._project_metadata = load_metadata(
            project_path
        )

        if clipping is not None:
            clipping = np.array(clipping)
            if np.any(clipping[0] > clipping[1]):
                raise ValueError("Clipping lower left must be smaller than upper right")

            if np.any(clipping[0] < 0) or np.any(clipping[1] > 1):
                raise ValueError("Clipping must be in [0, 1]^3")

            if len(clipping) != 2 or len(clipping[0]) != 3 or len(clipping[1]) != 3:
                raise ValueError("Clipping must be a tuple of two tuples of length 3")

        self._clipping = clipping

        # The dictionary of data sources that we have added
        self._sources = {}

        # The compression level at which we operate
        self._compression_level = 0

    def available_sources(self) -> list[str]:
        """List the data sources that are available in the project."""

        return list(self.metadata["sources"].keys())

    def add_source(self, source: str = None, organelle: str = None) -> None:
        """Connect a data source in the project with an organelle type

        :param source:
            The name of the data source in the original dataset. Must be
            one of the names returned by available_sources.

        :param organelle:
            The name of the organelle that is labelled in the data source.
            Must be on the strings returned by organelle_morphology.organelle_types
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
            self, self._dataset_json_directory / source_path, organelle
        )

        # Double-check that it provides the current compression level
        if self.compression_level >= len(source_obj.metadata["downsampling"]):
            raise ValueError(
                f"Compression level {self.compression_level} is not available for source {source}"
            )

        self._sources[source] = source_obj

    @property
    def compression_level(self):
        """The compression level used for our computations."""

        return self._compression_level

    @compression_level.setter
    def compression_level(self, compression_level):
        # Validate against the compression levels being in the already registered
        # data sources. For sources added later, add_sources checks whether they
        # provide the current compression level.

        if compression_level < 0:
            raise ValueError(f"Compression level must be >= 0, got {compression_level}")

        for source in self._sources.values():
            if compression_level >= len(source.metadata["downsampling"]):
                raise ValueError(
                    f"Compression level {compression_level} is not available for source {source.metadata['data_root']}"
                )

        self._compression_level = compression_level

    @property
    def metadata(self):
        """The project metadata stored in the project JSON file"""

        return self._project_metadata

    @property
    def clipping(self):
        """The subcube of the original data that we work with"""

        if self._clipping is None:
            print("No clipping was selected for this project.")
            return None
        else:
            return self._clipping

    def organelles(
        self, ids: str = "*", return_ids: bool = False
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

        for source in self._sources.values():
            result.extend(source.organelles(ids, return_ids))

        return result
