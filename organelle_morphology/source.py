from typing import Optional
from pathlib import Path
from organelle_morphology.organelle import Organelle, organelle_registry
from organelle_morphology.util import disk_cache, parallel_pool

import dask.array as da

import fnmatch
import numpy as np
import xml.etree.ElementTree as ET
from skimage.measure import regionprops
from dataclasses import dataclass, field
import z5py

import warnings

warnings.filterwarnings("ignore", category=UserWarning, append=True)


@dataclass
class Timepoint:
    name: str
    file: Path
    resolution: list[int]
    n5: str
    downsamplingFactors: list = field(default_factory=list)
    levels: list[str] = field(default_factory=list)


@dataclass
class Data_level:
    level: str
    chunks: tuple
    chunks_per_dimension: list
    downsamplingFactor: list[int]
    data: z5py.dataset.Dataset


def load_n5(n5: Path) -> dict[str, Timepoint]:
    f = z5py.File(n5, "r")
    timepoints = {}

    def _meta_finder(name: str, obj):
        print(name, end="\n\t")
        print(str(obj), end="\n\t")
        print(list(obj.attrs.items()))

        name_parts = name.split("/")

        if len(name_parts) == 2:
            # new timepoint
            timepoints[name_parts[1]] = Timepoint(
                name=name_parts[1],
                file=n5,
                resolution=obj.attrs["resolution"],
                n5=obj.attrs["n5"],
            )
        if len(name_parts) == 3:
            # new sampling level
            assert isinstance(obj, z5py.dataset.Dataset), "Unkown data structure"
            tp = timepoints[name_parts[1]]

            dl = Data_level(
                level=name_parts[2],
                chunks=obj.chunks,
                chunks_per_dimension=obj.chunks_per_dimension,
                downsamplingFactor=obj.attrs["downsamplingFactors"],
                data=obj,
            )

            tp.downsamplingFactors.append(obj.attrs["downsamplingFactors"])
            tp.levels.append(name_parts[2])

            setattr(tp, name_parts[2], dl)

    f.visititems(_meta_finder)
    return timepoints


class DataSource:
    def __init__(
        self,
        project,
        xml_path: Path,
        organelle: Optional[str],
        background_label: int = 0,
    ):
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

        self.project = project
        self.xml_path = xml_path
        self.org_name = organelle
        self.background_label = background_label

        if not organelle_registry.get(self.org_name):
            raise ValueError(f"Unknown organelle class {self.org_name}")

        # The data will be loaded lazily
        self._metadata: Optional[dict] = None
        self._basic_geometric_properties = {}
        self._mesh_properties = {}
        self._meshes = {}
        self._morphology_map = {}

        # The computed organelles are stored
        self._organelles = None

        self.logger = self.project.logger

    def load_metadata(self):
        self._metadata = {}

        # Read the XML file
        tree = ET.parse(self.xml_path)
        xmldata = tree.getroot()
        try:
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
            name = (
                xmldata.find("SequenceDescription")
                .find("ViewSetups")
                .find("ViewSetup")
                .find("name")
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
            resolution = list((float(i) for i in resolution.split(" ")))
            first_timepoint = int(
                xmldata.find("SequenceDescription")
                .find("Timepoints")
                .find("first")
                .text
            )
            last_timepoint = int(
                xmldata.find("SequenceDescription").find("Timepoints").find("last").text
            )
            num_timepoints = last_timepoint - first_timepoint + 1

        except AttributeError:
            raise ValueError("Could not parse XML!")

        # Make assertions on the given data
        assert num_setups == 1, "Only one ViewSetup supported"
        assert num_timepoints == 1, "Only one timepoint supported"
        assert filename is not None, "n5 Filename could not be parsed from xml!"

        timepoints = load_n5(self.xml_path.parent / filename)
        self.timepoint = list(timepoints.values())[0]

        assert resolution == self.timepoint.resolution[::-1]

        coarse_level = self.timepoint.levels[0]
        for level in self.timepoint.levels:
            if int(level[-1]) > int(coarse_level[-1]):
                coarse_level = level

        # Store the metadata
        self._metadata["data_root"] = self.xml_path / filename
        self._metadata["downsampling"] = self.timepoint.downsamplingFactors
        self._metadata["levels"] = self.timepoint.levels
        self._metadata["size"] = tuple((int(i) for i in size.split(" ")))
        self._metadata["resolution"] = resolution
        self._metadata["name"] = name
        self._metadata["coarse_level"] = coarse_level

    @property
    def metadata(self) -> dict:
        """Return the metadata of this source. Loads the metadata, if necessary"""

        if self._metadata is None:
            self.load_metadata()

        return self._metadata

    @property
    def basic_geometric_properties(self):
        """get basic properties from scikit-image"""
        comp_level = self.project.compression_level

        self.logger.debug("get basic properties from scikit-image")
        if comp_level not in self._basic_geometric_properties:
            geometric_properties = regionprops(self.data, spacing=self.resolution)

            # filter region props for useful properties
            # https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
            filtered_region_props = {
                "voxel_volume": "area",  # for 3d this is the volume
                "voxel_bbox": "bbox",
                "voxel_slice": "slice",  # the slice of the bounding box
                "voxel_centroid": "centroid",
                "voxel_extent": "extent",  # how much volume of the bounding box is occupied by the object
                "voxel_solidity": "solidity",  # ratio of pixels in the convex hull to those in the region
            }

            self._basic_geometric_properties[comp_level] = {
                f"{self.org_name}_{str(region['label']).zfill(4)}": {
                    prop_name: region[prop]
                    for prop_name, prop in filtered_region_props.items()
                }
                for region in geometric_properties
            }

        return self._basic_geometric_properties[comp_level]

    @property
    def morphology_map(self):
        """Get the morphology map for all organelles"""
        self.logger.debug("get morphology map for all organelles")

        with parallel_pool(len(self.organelles())) as (pool, pbar):
            for organelle in self.organelles():
                result = pool.apply_async(
                    organelle.morphology_map,
                    callback=lambda _: pbar.update(),
                ).get()

                self._morphology_map[organelle.id] = result
        return self._morphology_map

    @property
    def data(self) -> da.Array:
        return self.get_data(None)

    def get_data(self, compression_level: Optional[str]) -> da.Array:
        """Get data of this source as array.

        Args:
            compression_level: Override the compression level specified in
            the project. Default: None, get level from project.

        Returns:
            Dask array of the data. Respects the in the project set clipping.
        """
        if compression_level is None:
            compression_level = self.project.compression_level
        data_at_level = getattr(self.timepoint, f"s{compression_level}").data

        # chunk factor for efficieny, needs tuning
        data = da.from_array(data_at_level, chunks="auto")

        if self.project.clipping is not None:
            lower_corner, upper_corner = self.project.clipping
            data_shape = data_at_level.shape
            clipped_low_corner = np.floor(lower_corner * data_shape).astype(int)
            clipped_high_corner = np.ceil(upper_corner * data_shape).astype(int)
            cube_slice = tuple(
                slice(clip_low, clip_high, 1)
                for clip_low, clip_high in zip(clipped_low_corner, clipped_high_corner)
            )
            return data[cube_slice]
        return data

    @property
    def coarse_data(self) -> da.Array:
        """Return the coarsest version of the dataset.

        This can be used for algorithms that do not critically depend
        on the data resolution and should be fast.
        """

        return self.get_data(self.metadata["coarse_level"])

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
                    getattr(
                        self.timepoint, f"s{self.project.compression_level}"
                    ).downsamplingFactor,
                )
            )
        )

    @property
    def labels(self) -> tuple[int]:
        """Return the list of labels present in the data source."""
        # TODO: avoid compute calls
        labels = da.where(da.unique(self.data).compute() != self.background_label)[
            0
        ].compute()
        return labels

    def organelles(
        self,
        ids: str = "*",
        permanent_whitelist=None,
        permanent_blacklist=None,
    ) -> list[Organelle]:
        """Return a list of organelle objects.

        Args:
            ids: glob pattern to match the ids. Default "*" includes all.
            permanent_whitelist: ids to always include
            permanent_blacklist: ids to always exclude

        Returns:
            List of organelle objects
        """
        filtered_ids = self.organelle_ids(
            ids=ids,
            permanent_whitelist=permanent_whitelist,
            permanent_blacklist=permanent_blacklist,
        )
        return [self._organelles[id] for id in filtered_ids]

    def organelle_ids(
        self,
        ids: str = "*",
        permanent_whitelist=None,
        permanent_blacklist=None,
    ) -> list[Organelle] | list[str]:
        """Return a list of organelle ids.

        Args:
            ids: glob pattern to match the ids. Default "*" includes all.
            permanent_whitelist: ids to always include
            permanent_blacklist: ids to always exclude

        Returns:
            List of organelle ids
        """

        # Ensure that all organelles are computed
        if self._organelles is None:
            self._organelles = {}

            # Iterate available organelle classes and construct organelles
            if not (orgclass := organelle_registry.get(self.org_name)):
                raise ValueError(f"Unknown organelle class {self.org_name}")
            for organelle in orgclass.construct(self, self.labels):
                self._organelles[organelle.id] = organelle

        # Filter the organelles with the given ids pattern
        filtered_ids = fnmatch.filter(self._organelles.keys(), ids)

        if permanent_blacklist is not None:
            filtered_ids = [
                org_id for org_id in filtered_ids if org_id not in permanent_blacklist
            ]

        if permanent_whitelist is not None:
            filtered_ids.extend(permanent_whitelist)

        return filtered_ids
