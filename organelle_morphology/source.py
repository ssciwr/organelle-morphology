from collections import defaultdict
from itertools import combinations
from typing import Optional
from pathlib import Path

from organelle_morphology.organelle import Organelle, organelle_registry

from dask import persist, compute
import dask.array as da
from dask.array.core import Array
from dask.delayed import Delayed, delayed
from zmesh import Mesh, Mesher

import fnmatch
import numpy as np
import xml.etree.ElementTree as ET
from skimage.measure import regionprops
from dataclasses import dataclass, field
import z5py

import warnings


warnings.filterwarnings("ignore", category=UserWarning, append=True)


@delayed(nout=2)
def _block_mesher(block, space_offset: tuple[int, ...]):
    """Delayed function to calculate meshes from label data

    Args:
        block: 3D array containing labels
        space_offset: 3D tuple containing the offset of the current block
            in relation to the whole data. The vertices are therefore *not*
            local to the block, but can directly be merged with other blocks.

    Returns:
        meshes: Dict with labels as keys and meshes as values.
        labels: list of labels present in this block.

    """

    mesher = Mesher((1, 1, 1))
    mesher.mesh(block, close=False)
    meshes = {}
    for id in mesher.ids():
        assert meshes.get(id) is None, f"{id} was in mesh already!"
        mesh = mesher.get(
            id,
            normals=False,
            reduction_factor=0,
            voxel_centered=False,
            max_error=None,  # None: max 1 voxel, otherwise unit of data
        )
        mesh.vertices += space_offset
        meshes[id] = mesh
    return meshes, list(meshes.keys())


@delayed
def _block_ids(block):
    labels = np.unique(block)
    return list(filter(lambda i: i != 0, labels))


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
        self._meshes_chunked = None
        self._ids_to_chunks: Optional[dict] = None
        self._labels = None

        # to avoid recomputation, persisted delayed objects need to be held
        self._storage = {}

        # The computed organelles are stored
        self._organelles = None

        self.logger = self.project.logger

    def load_n5(self, n5: Path) -> dict[str, Timepoint]:
        f = z5py.File(n5, "r")
        timepoints = {}
        self.project.logger.debug(f"Start loading {n5}, metadata structure:")

        def _meta_finder(name: str, obj):
            self.project.logger.debug(
                name + ": " + str(obj.__class__) + str(list(obj.attrs.items()))
            )

            name_parts = Path(name).parts

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

        timepoints = self.load_n5(self.xml_path.parent / filename)
        if len(timepoints) != 1:
            self.project.logger.warning(
                "Only single timepoints supported, ignoring all but the first!"
            )

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

        for organelle in self.organelles():
            self._morphology_map[organelle.id] = organelle.morphology_map
        return self._morphology_map

    @property
    def data(self) -> Array:
        return self.get_data(None)

    def get_data(self, compression_level: Optional[str]) -> Array:
        """Get data of this source as array.

        Args:
            compression_level: Override the compression level specified in
            the project. Default: None, get level from project.

        Returns:
            Dask array of the data. Respects the in the project set clipping.
        """
        if compression_level is None:
            compression_level = self.project.compression_level
        data_at_level: z5py.dataset.Dataset = getattr(
            self.timepoint, compression_level
        ).data

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
                        self.timepoint, self.project.compression_level
                    ).downsamplingFactor,
                )
            )
        )

    @property
    def labels(self) -> list[int]:
        """Return the list of labels present in the data source."""
        return list(self.ids_to_chunks.keys())

    @property
    def meshes_chunked(self) -> np.ndarray:
        """Get an array over all chunks containing dask delayed objects."""
        if self._meshes_chunked is None:
            self.calculate_mesh()
        return self._meshes_chunked

    def get_meshes_from_id(self, id: int) -> Optional[list[Delayed]]:
        """Given the label id get the associated meshes."""

        if (chunks := self.ids_to_chunks.get(id)) is None:
            return None

        return [self.meshes_chunked[chunk][id] for chunk in chunks]

    def merge_meshes(self, meshes: list[Delayed]) -> Mesh:
        """Merges two delayed meshes into one concrete new Mesh object"""

        @delayed(nout=2)  # pyright: ignore
        def get_faces_verts(meshes):
            faces = [np.array(m.faces) for m in meshes]
            offsets = [0] + [m.vertices.shape[0] for m in meshes][:-1]
            for f, offset in zip(faces, offsets):
                f += offset
            faces = np.concatenate(faces)
            verts = np.concatenate([m.vertices for m in meshes])
            return verts, faces

        verts, faces = compute(*get_faces_verts(meshes))
        return Mesh(
            vertices=verts,
            faces=faces,
            normals=None,
        )

    @delayed
    def _mesh_mover(
        self,
        mesh: Mesh,
        space_offset: Optional[tuple[int, int, int]] = None,
        id_offset: Optional[int] = None,
    ):
        if space_offset is not None:
            mesh.vertices += space_offset
        if id_offset is not None:
            mesh.faces += id_offset
        return mesh

    @property
    def ids_to_chunks(self) -> dict:
        if self._ids_to_chunks is None:
            self.calculate_mesh()
        assert self._ids_to_chunks is not None
        return self._ids_to_chunks

    def calculate_mesh(self, smooth=True):
        @delayed
        def build_chunk_lookup(ids_chunks, indices) -> dict:
            res = defaultdict(list)
            for ids, index in zip(ids_chunks, indices):
                for id in ids:
                    res[id].append(index)

            return dict(res)

        d_data = self.data.to_delayed()
        self._meshes_chunked = np.empty_like(d_data)
        ids_chunked = np.empty_like(d_data)
        size_offset_cumsum = [
            np.cumsum(np.array([0] + list(ch))) for ch in self.data.chunks
        ]

        for index, d_block in np.ndenumerate(d_data):
            space_offset = tuple(
                int(size_offset_cumsum[i][c]) for i, c in enumerate(index)
            )
            meshes_chunk, ids_chunk = _block_mesher(d_block, space_offset)
            self._meshes_chunked[index] = meshes_chunk
            ids_chunked[index] = ids_chunk

        # flatten the ids and delayed meshes
        d_ids = []
        indices = []
        d_meshes = []
        for index, ids_chunk in np.ndenumerate(ids_chunked):
            mesh = self._meshes_chunked[index]
            d_ids.append(ids_chunk)
            indices.append(index)
            d_meshes.append(mesh)

        self._storage["d_meshes"] = persist(*d_meshes)

        self._ids_to_chunks = build_chunk_lookup(d_ids, indices).compute()
        assert self._ids_to_chunks is not None  # for linter

        # get some statistics
        amounts, freqs = np.unique(
            [len(idxs) for idxs in self._ids_to_chunks.values()], return_counts=True
        )
        for amount, frq in zip(amounts, freqs):
            self.project.logger.debug(f"{frq} labels in {amount} chunks")

        # mesh = Trimesh(verts, faces, process=False)
        # mesh.fix_normals()
        # if smooth:
        #     trimesh.smoothing.filter_humphrey(mesh)
        #
        # self.logger.debug("Generated mesh for %s", self.id)
        return

    def fix_meshes_across_chunks(self):
        # Util functions
        def get_adjacent_blocks(blocks):
            for a, b in combinations(blocks):
                pass

        # main body
        labels_to_fix = {k: v for k, v in self.ids_to_chunks.items() if len(v) > 1}

        for label, chunks in labels_to_fix.items():
            pass

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
