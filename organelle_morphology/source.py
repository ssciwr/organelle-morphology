from organelle_morphology.organelle import Organelle, organelle_registry

from elf.io import open_file

import fnmatch
import json
import numpy as np
import pathlib
import xml.etree.ElementTree as ET


class DataSource:
    def __init__(self, project, xml_path: pathlib.Path, organelle: str):
        """Initialize a data source.

        This method initializes a data source for organelle morphology analysis.
        The data source is linked to a CebraEM/Mobie project and contains information
        about a single organelle data source.

        :param project:
            The CebraEM/Mobie project this data source is linked to.
        :type project: organelle_morphology.Project
        :param xml_path:
            The path to the XML file describing the source data.
        :type source_path: pathlib.Path
        :param organelle:
            The name of the organelle being analyzed.
        :type organelle: str
        """

        self._project = project
        self._xml_path = xml_path
        self._organelle = organelle

        # The data will be loaded lazily
        self._data = None
        self._coarse_data = None
        self._metadata = None

        # The computed organelles are stored
        self._organelles = None

    @property
    def metadata(self):
        """Load the metadata for the given source and return it."""

        if self._metadata is None:
            # Initialize the metadata dictionary
            self._metadata = {}

            # Read the XML file
            tree = ET.parse(self._xml_path)
            xmldata = tree.getroot()
            filename = (
                xmldata.find("SequenceDescription").find("ImageLoader").find("n5").text
            )
            num_setups = len(
                xmldata.find("SequenceDescription")
                .find("ViewSetups")
                .findall("ViewSetup")
            )
            size = (
                xmldata.find("SequenceDescription")
                .find("ViewSetups")
                .find("ViewSetup")
                .find("size")
                .text
            )
            resolution = (
                xmldata.find("SequenceDescription")
                .find("ViewSetups")
                .find("ViewSetup")
                .find("voxelSize")
                .find("size")
                .text
            )

            # TODO: Extract more relevant metadata as we need it

            # Make assertions on the given data
            assert num_setups == 1

            # Load the attributes JSON file for setup0
            attributes_file = (
                self._project._dataset_json_directory
                / "images"
                / "bdv-n5"
                / filename
                / "setup0"
                / "attributes.json"
            )
            with open(attributes_file, "r") as f:
                attributes = json.load(f)

            # Store the metadata
            self._metadata["data_root"] = (
                self._project._dataset_json_directory / "images" / "bdv-n5" / filename
            )
            self._metadata["downsampling"] = attributes["downsamplingFactors"]
            self._metadata["size"] = tuple((int(i) for i in size.split(" ")))
            self._metadata["resolution"] = tuple(
                (float(i) for i in resolution.split(" "))
            )

        return self._metadata

    @property
    def data(self) -> np.ndarray:
        """Load the raw data."""

        if self._data is None:
            with open_file(str(self.metadata["data_root"]), "r") as f:
                self._data = f[f"setup0/timepoint0/s{self._project.compression_level}"]

        if self._project.clipping is not None:
            lower_corner, upper_corner = self._project.clipping
            data_shape = self._data.shape
            clipped_low_corner = np.floor(lower_corner * data_shape).astype(int)
            clipped_high_corner = np.ceil(upper_corner * data_shape).astype(int)
            cube_slice = tuple(
                slice(clip_low, clip_high, 1)
                for clip_low, clip_high in zip(clipped_low_corner, clipped_high_corner)
            )
            self._data = f[f"setup0/timepoint0/s{self._project.compression_level}"][
                cube_slice
            ]

        return self._data

    @property
    def coarse_data(self) -> np.ndarray:
        """Load the coarsest version of the dataset.

        This can be used for algorithms that do not critically depend
        on the data resolution and should be fast. The user's choice of
        resolution for the analysis should always be respected.
        """

        if self._coarse_data is None:
            with open_file(str(self.metadata["data_root"]), "r") as f:
                self._coarse_data = f[
                    f"setup0/timepoint0/s{len(self.metadata['downsampling']) - 1}"
                ]

        return self._coarse_data

    @property
    def data_resolution(self) -> tuple[float]:
        """Return the resolution in micrometers at which the data is stored."""

        return self.metadata["resolution"]

    @property
    def resolution(self) -> tuple[float]:
        """Return the resolution of our data at the chosen compression level."""

        return tuple(
            (
                r * d
                for r, d in zip(
                    self.data_resolution,
                    self.metadata["downsampling"][self._project.compression_level],
                )
            )
        )

    @property
    def labels(self) -> tuple[int]:
        """Return the list of labels present in the data source."""

        return tuple(np.unique(self.coarse_data))

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

        # Ensure that all organelles are computed
        if self._organelles is None:
            self._organelles = {}

            # Iterate available organelle classes and construct organelles
            for orgclass in organelle_registry.values():
                for organelle in orgclass.construct(self, self.labels):
                    self._organelles[organelle.id] = organelle

        # Filter the organelles with the given ids pattern
        filtered_ids = fnmatch.filter(self._organelles.keys(), ids)

        # Return ids or organelle objects
        if return_ids:
            return filtered_ids
        else:
            return [self._organelles[id] for id in filtered_ids]
