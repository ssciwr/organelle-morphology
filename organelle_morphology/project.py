from organelle_morphology.organelle import Organelle

import os
import pathlib


class Project:
    def __init__(
        self,
        project_path: pathlib.Path = os.getcwd(),
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
        # Identify the directory that containes project metadata JSON
        self._project_json_directory = None

        # The project metadata JSON
        self._project_metadata = {}

        # The compression level at which we operate
        self._compression_level = 0

    def available_sources(self) -> list[str]:
        """List the data sources that are available in the project."""

        raise NotImplementedError

    def add_source(self, source: str = None, organelle: str = None) -> None:
        """Connect a data source in the project with an organelle type

        :param source:
            The name of the data source in the original dataset. Must be
            one of the names returned by available_sources.

        :param organelle:
            The name of the organelle that is labelled in the data source.
            Must be on the strings returned by organelle_morphology.organelle_types
        """

        raise NotImplementedError

    @property
    def compression_level(self):
        """The compression level used for our computations."""

        return self._compression_level

    @compression_level.setter
    def compression_level(self, compression_level):
        # Validate against the compression levels being in the already registered
        # data sources. For sources added later, add_sources checks whether they
        # provide the current compression level.

        raise NotImplementedError

    @property
    def metadata(self):
        """The project metadata stored in the project JSON file"""

        raise NotImplementedError

    @property
    def clipping(self):
        """The subcube of the original data that we work with"""

        raise NotImplementedError

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

        raise NotImplementedError
