from organelle_morphology.organelle import Organelle

import numpy as np


class DataSource:
    def __init__(self, project):
        """Initialize a data source.

        This typically does not happen under user control, but it is implicitly
        done by organelle_morphology.

        :param project:
            The CebraEM/Mobie project this is linked to.
        :type project: organelle_morphology.Project
        """

        self.project = project

        # The data will be loaded lazily
        self._data = None
        self._coarse_data = None

    @property
    def data(self) -> np.ndarray:
        """Load the raw data into memory and return it."""

        if self._data is None:
            raise NotImplementedError

        return self._data

    @property
    def coarse_data(self) -> np.ndarray:
        """Load the coarsest version of the dataset.

        This can be used for algorithms that do not critically depend
        on the data resolution and should be fast. The user's choice of
        resolution for the analysis should always be respected.
        """

        if self._coarse_data is None:
            raise NotImplementedError

        return self._coarse_data

    @property
    def data_resolution(self) -> tuple[float]:
        """Return the resolution at which the data is stored."""

        raise NotImplementedError

    @property
    def resolution(self) -> tuple[float]:
        """Return the resolution of our data at the chosen compression level."""

        raise NotImplementedError

    @property
    def labels(self) -> tuple[int]:
        """Return the list of labels present in the data source."""

        raise NotImplementedError

    def organelles(
        self, ids: str = "*", return_ids: bool = False
    ) -> list[Organelle] | list[str]:
        """Return a list of organelles found in the data source

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
