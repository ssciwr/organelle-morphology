from collections import defaultdict
from time import time
from typing import Optional
from pathlib import Path

import matplotlib as mpl
from trimesh import Trimesh
import trimesh

import organelle_morphology
from organelle_morphology.organelle import Organelle, organelle_registry

from dask import persist, compute
import dask.array as da
from dask.array.core import Array
from dask.delayed import Delayed, delayed
from dask.distributed import print as dprint

from zmesh import Mesh, Mesher

import fnmatch
import numpy as np
import xml.etree.ElementTree as ET
from skimage.measure import regionprops
from dataclasses import dataclass, field
import z5py

import warnings

from organelle_morphology.util import Cache


warnings.filterwarnings("ignore", category=UserWarning, append=True)


@delayed(nout=2)
def _block_mesher(
    block,
    space_offset: tuple[int, ...],
    reduction_factor=0,
    debug_color=0,
):
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

    def plane(origin, normal):
        """Plane for debug view"""
        normal_i = np.array(normal, dtype=bool)
        planeverts = np.stack([origin] * 4)
        # axis along which to strech from origin to plane
        for i, axis in enumerate((~normal_i).nonzero()[0]):
            if i == 0:
                planeverts[0, axis] += 50
                planeverts[1, axis] += 50
                planeverts[2, axis] -= 50
                planeverts[3, axis] -= 50

            else:
                planeverts[0, axis] += 50
                planeverts[2, axis] += 50
                planeverts[1, axis] -= 50
                planeverts[3, axis] -= 50

        plane = Trimesh(planeverts, [[0, 1, 2], [3, 2, 1]])
        plane += Trimesh(planeverts, [[2, 1, 0], [1, 2, 3]])

        plane.visual.vertex_colors = (0, 0, 0, 40)
        plane.visual.vertex_colors[:, [normal_i.nonzero()[0][0]]] = 200

        return plane

    # calculate slice planes, assuming overlap of 2 on all sides
    bs = np.array(block.shape)
    overlap = 2
    slice_points = []
    slice_normals = []

    for i in range(3):
        lower = bs.copy() / 2
        lower[i] = overlap
        slice_points.append(lower)
        dir = np.zeros((3,), dtype=int)
        dir[i] = 1
        slice_normals.append(dir)

        upper = bs.copy() / 2
        upper[i] = bs[i] - overlap
        slice_points.append(upper)
        dir = np.zeros((3,), dtype=int)
        dir[i] = -1
        slice_normals.append(dir)

    mesher = Mesher((1, 1, 1))
    mesher.mesh(block, close=False)
    meshes = {}
    assert len(mesher.ids()) == np.unique(block).shape[0] - 1  # zero is no label

    for id in mesher.ids():
        assert meshes.get(id) is None, f"{id} was in mesh already!"
        mesh = mesher.get(
            id,
            normals=False,
            reduction_factor=reduction_factor,
            voxel_centered=True,
            max_error=None,  # None: max 1 voxel, otherwise unit of data
        )
        mesh = Trimesh(mesh.vertices, mesh.faces, process=False)

        parts = []
        masks = []
        if debug_color:
            mesh.visual.vertex_colors[:, :] = trimesh.visual.random_color()
            mesh.visual.vertex_colors[:, 3] = 120

        planes = []
        for origin, normal in zip(slice_points, slice_normals):
            mesh_slice = mesh.slice_plane(origin, normal)
            assert mesh_slice is not None
            if mesh_slice.vertices.shape[0] == 0 and not debug_color:
                mesh = mesh_slice
                continue

            if debug_color:
                planes.append(plane(origin, normal))
                if (
                    mesh_slice.vertices.shape[0] != mesh.vertices.shape[0]
                    or mesh_slice.faces.shape[0] != mesh.faces.shape[0]
                ):
                    normal_i = np.array(normal, dtype=bool)
                    mask = np.logical_and(
                        mesh.vertices[:, normal_i] < (origin[normal_i] + 1),
                        mesh.vertices[:, normal_i] > (origin[normal_i] - 1),
                    ).nonzero()[0]
                    masks.append(mask)

                    # Add removed slice in pink
                    mesh_outer = mesh.slice_plane(origin, normal * -1)
                    mesh_outer.visual.vertex_colors = (255, 0, 255, 100)
                    parts.append(mesh_outer)

                    if debug_color >= 3:
                        mesh.visual.vertex_colors[:, :3] = (
                            mesh.vertices / (mesh.vertices.max(axis=0)) * 200
                        )
                        mesh.visual.vertex_colors[:, 3] = 200
            else:
                # Debug views contain the non-sliced mesh!
                mesh = mesh_slice
        if masks:
            mask = np.concatenate(masks)
            mesh.visual.vertex_colors[mask] = (0, 255, 0, 180)
        if parts:
            mesh += sum(parts)

        if mesh.vertices.shape[0] > 0:
            mesh.vertices += space_offset
            meshes[id] = mesh

    if debug_color >= 1:
        try:
            i, m = meshes.popitem()

            verts = np.array(
                [
                    [0, 0, 0],
                    [bs[0], 0, 0],
                    [0, bs[1], 0],
                    [0, 0, bs[2]],
                    [bs[0], bs[1], 0],
                    [0, bs[1], bs[2]],
                    [bs[0], 0, bs[2]],
                    [bs[0], bs[1], bs[2]],
                ]
            )
            verts += space_offset
            for v in verts:
                m += trimesh.primitives.Sphere(3, v, subdivisions=1)
            if debug_color >= 2:
                pl = sum(planes)
                pl.vertices += space_offset
                m += pl
            meshes[i] = m
        except KeyError:
            pass

    return meshes, list(meshes.keys())


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
        project: "organelle_morphology.Project",
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

        self.project: "organelle_morphology.Project" = project
        self.xml_path = xml_path
        self.org_name = organelle
        self.background_label = background_label
        self.logger = self.project.logger

        if not organelle_registry.get(self.org_name):
            raise ValueError(f"Unknown organelle class {self.org_name}")

        self._metadata = None

        self._default_values = {
            "_basic_geometric_properties": {},
            "_mesh_properties": {},
            "_meshes": None,
            "_computed_compression": None,
            "_morphology_map": {},
            "_meshes_chunked": None,
            "_ids_to_chunks": None,
            "_labels": None,
            "_storage": {},
            "_organelles": None,
        }
        self.clear_memory_cache()

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
            size: str = (
                xmldata.find("SequenceDescription")
                .find("ViewSetups")
                .find("ViewSetup")
                .find("size")
                .text
            )
            name: str = (
                xmldata.find("SequenceDescription")
                .find("ViewSetups")
                .find("ViewSetup")
                .find("name")
                .text
            )
            resolution: str = (
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

        assert self._metadata is not None
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
            # data_shape = data_at_level.shape
            clipped_low_corner = np.floor(lower_corner * data.shape).astype(int)
            clipped_high_corner = np.ceil(upper_corner * data.shape).astype(int)
            cube_slice = tuple(
                slice(clip_low, clip_high, 1)
                for clip_low, clip_high in zip(clipped_low_corner, clipped_high_corner)
            )
            return data[cube_slice]
        return data

    @property
    def coarse_data(self) -> Array:
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
        return list(self.meshes.keys())

    @property
    def meshes(self) -> dict[int, Delayed]:
        @delayed
        def _get_from_cache(key, project_path, source, level, clipping, disk):
            cache = Cache(project_path, source, level, clipping, disk)
            verts, faces = cache[key]

            return Trimesh(verts, faces)

        @delayed
        def _write_to_cache(key, mesh, project_path, source, level, clipping, disk):
            cache = Cache(project_path, source, level, clipping, disk)
            cache[key] = (mesh.vertices, mesh.faces)

        self.logger.debug("Requested meshes")
        if self._meshes is None or (
            self._computed_compression != self.project.compression_level
        ):
            self.logger.debug("Meshes not loaded")
            self._meshes = {}
            cache_settings = self.project.cache_settings
            cache_settings["source"] = self.xml_path.stem
            cache = Cache(**cache_settings)
            labels = None

            if "labels" in cache:
                self.logger.debug("Meshes in cache, reading labels..")
                labels = cache["labels"]
                self.logger.debug(f"{len(labels)} labels found in cache")

                for label in labels:
                    self._meshes[label] = _get_from_cache(label, **cache_settings)

                # keep meshes in distributed memory, gc previously computed meshes
                self._storage["ref_meshes"] = persist(*self._meshes.values())

                self._computed_compression = self.project.compression_level
                self.logger.debug("Meshes loaded from cache")

            else:
                self.logger.info("Meshes not in cache, calculating..")
                self.calculate_mesh()
                self.logger.debug("Saving meshes to cache..")

                cache["labels"] = list(self._meshes.keys())
                meshes_d = list(self._meshes.values())
                delayed_saves = []
                for label, mesh_d in self._meshes.items():
                    delayed_saves.append(
                        _write_to_cache(label, mesh_d, **cache_settings)
                    )
                compute(*delayed_saves)

                self.logger.debug("Meshes saved in cache")

        return self._meshes

    def get_bounding_box_of_mesh(self, mesh: Trimesh | Delayed):
        """Calculate the corner with the smallest and the one
        with the biggest corrdinates.

        Returns:
        -------
            min: np.ndarray
                First corner
            max: np.ndarray
                Second corner
        """

        if isinstance(mesh, Delayed):
            mesh = mesh.compute()
        assert isinstance(mesh, Trimesh)

        min = np.min(mesh.vertices, axis=0)
        max = np.max(mesh.vertices, axis=0)

        return min, max

    def merge_meshes(self, meshes: list[Delayed], color=None) -> Delayed:
        """Merges delayed meshes into one concrete new Mesh object

        Needs overlapping meshes, otherwise the intersections will not be
        connected.
        """

        @delayed
        def color_mesh(tmesh, color):
            if color:
                if color == 1:
                    tmesh.visual.vertex_colors = trimesh.visual.random_color()
                elif color == 2:
                    viridis = mpl.colormaps.get("viridis")
                    tmesh.visual.face_colors = viridis.resampled(
                        len(tmesh.faces)
                    ).colors
            return tmesh

        @delayed
        def merge_two_trimesh(tmeshes: list[Trimesh]):
            tmesh: Trimesh = Trimesh()
            for mesh in tmeshes:
                tmesh += mesh
            tmesh.merge_vertices(
                merge_tex=True,
                merge_norm=True,
                digits_vertex=2,
                digits_norm=2,
                digits_uv=2,
            )
            return tmesh

        if color is None:
            color = 0
        if color:
            meshes = [color_mesh(m, color) for m in meshes]

        while (length := len(meshes)) > 1:
            merged = []
            for i, _ in enumerate(meshes[::2]):
                j = length - (i + 1)
                # odd-length: indices meet in the middle
                if i == j:
                    merged.append(meshes[i])
                    break
                merged.append(merge_two_trimesh([meshes[i], meshes[j]]))
            meshes = merged

        return meshes[0]

    @property
    def ids_to_chunks(self) -> dict:
        if self._ids_to_chunks == {}:
            self.calculate_mesh()
        return self._ids_to_chunks

    def calculate_mesh(
        self,
        reduction_factor=0,
        overlap=True,
        simplify: Optional[float] = None,
        debug_color: Optional[int] = None,
    ):
        """Compute meshes for this source.

        Meshes crossing chunks are merged.
        Populates following fields:
         * self._meshes
         * self._meshes_chunked
         * self._ids_to_chunks

        Respects the set compression and clipping of the project.
        Computation is done in parallel using the dask client of the project.
        """
        if debug_color is None:
            if self.project.debug:
                debug_color = 1
            else:
                debug_color = 0

        @delayed
        def build_chunk_lookup(ids_chunks, indices) -> dict:
            res = defaultdict(list)
            for ids, index in zip(ids_chunks, indices):
                for id in ids:
                    res[id].append(index)

            return dict(res)

        @delayed
        def simplify_mesh(mesh, factor=0.1):
            """Simplify a trimesh object"""
            mesh = mesh.simplify_quadric_decimation(factor, aggression=1)
            return mesh

        if overlap:
            d_data = da.overlap.overlap(
                self.data, depth={0: 2, 1: 2, 2: 2}, boundary={0: 0, 1: 0, 2: 0}
            ).to_delayed()
        else:
            d_data = self.data.to_delayed()
        self._meshes_chunked = np.empty_like(d_data)
        ids_chunked = np.empty_like(d_data)
        size_offset_cumsum = [
            np.cumsum(np.array([0] + list(ch))) for ch in self.data.chunks
        ]

        for index, d_block in np.ndenumerate(d_data):
            space_offset = tuple(
                int(size_offset_cumsum[i][c]) - (2 if overlap else 0)
                for i, c in enumerate(index)
            )
            meshes_chunk, ids_chunk = _block_mesher(
                d_block, space_offset, reduction_factor, debug_color
            )
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

        # calculate meshes and keep references to make mesh persistend on workers
        self._storage["ref_meshes"] = persist(*(d_meshes + d_ids))

        self._ids_to_chunks = build_chunk_lookup(d_ids, indices).compute()
        assert self._ids_to_chunks is not None  # for linter

        # get some statistics
        all_chunks = list(self._ids_to_chunks.values())
        all_ids = np.array(list(self._ids_to_chunks.keys()))
        id_amounts = [len(idxs) for idxs in all_chunks]
        amounts, inverse, freqs = np.unique(
            id_amounts, return_counts=True, return_inverse=True
        )
        for amount, frq in zip(amounts, freqs):
            self.logger.info(f"{frq} labels in {amount} chunks")

        # Cleanup: Merge meshes crossing chunks
        self._meshes = {}
        duplicate_ids = all_ids[np.nonzero(inverse != 0)]
        for ind in duplicate_ids:
            chunk_idxs = self._ids_to_chunks[ind]
            meshes = [self._meshes_chunked[idx][ind] for idx in chunk_idxs]
            merged_mesh = self.merge_meshes(meshes, color=0)
            if simplify:
                merged_mesh = simplify_mesh(merged_mesh, simplify)
            self._meshes[ind] = merged_mesh

        unique_ids = all_ids[np.nonzero(inverse == 0)]
        for ind in unique_ids:
            mesh = self._meshes_chunked[self._ids_to_chunks[ind][0]][ind]
            if simplify:
                mesh = simplify_mesh(mesh, simplify)
            self._meshes[ind] = mesh

        # keep references to make simplified meshes persistent, gc raw meshes
        self._storage["ref_meshes"] = persist(*self._meshes.values())

        self._computed_compression = self.project.compression_level

    def clear_memory_cache(self):
        """Invalidates memory caches.
        Necessary after change of compression level or clipping.
        """
        self.logger.debug(f"Resetting source {self.org_name}: {self.xml_path.name}")
        for k, v in self._default_values.items():
            setattr(self, k, v)

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
