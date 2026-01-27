from collections import defaultdict
from typing import Optional
from pathlib import Path

from trimesh import Trimesh
import trimesh

import organelle_morphology
from organelle_morphology.organelle import Organelle, organelle_registry

from dask.base import persist, compute
import dask.array as da
from dask.array.core import Array
from dask.delayed import Delayed, delayed

from zmesh import Mesher
import skeletor as sk

import fnmatch
import numpy as np
import xml.etree.ElementTree as ET
from skimage.measure import regionprops
from dataclasses import dataclass, field
import z5py

import warnings

from organelle_morphology.util import (
    Cache,
    color_delayed_trimesh_rgba,
    get_skeleton_info,
    merge_meshes,
    mesure_gaussian_curvature_delayed,
    sample_skeleton,
)


warnings.filterwarnings("ignore", category=UserWarning, append=True)


@delayed(nout=2)
def _block_mesher(
    block,
    space_offset: tuple[int, ...],
    reduction_factor=0,
    debug_color=0,
    scaling_factors=[1, 1, 1],
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

    for id in mesher.ids():
        assert meshes.get(id) is None, f"{id} was in mesh already!"
        mesh = mesher.get(
            id,
            normals=False,
            reduction_factor=reduction_factor,
            voxel_centered=False,
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
                m += trimesh.primitives.Sphere(3, v, subdivisions=1, mutable=True)
            if debug_color >= 2:
                pl = sum(planes)
                pl.vertices += space_offset
                m += pl
            meshes[i] = m
        except KeyError:
            pass

    for label in meshes:
        meshes[label].vertices *= scaling_factors

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
        organelle: str,
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

        # if not organelle_registry.get(self.org_name):
        #     raise ValueError(f"Unknown organelle class {self.org_name}")

        self._metadata = None

        # initializes hidden fields for properies
        self.clear_memory_cache()

    def __repr__(self):
        return f"<{type(self).__module__}> {self.xml_path.stem}"

    def __str__(self):
        return f"DataSource of xml: {self.xml_path.stem}"

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
        self._metadata["size"] = tuple([int(i) for i in size.split(" ")][::-1])
        self._metadata["resolution"] = resolution
        self._metadata["name"] = name
        self._metadata["coarse_level"] = coarse_level

    @property
    def metadata(self) -> dict:
        """Return the metadata of this source. Loads the metadata, if necessary

        coarse_level: coarsest level available
        data_root: path to n5 data directory
        downsampling: list of factors by which the resolution can be decreased
        levels: names of downsampling levels, aligned with downsampling list
        name: name
        resolution: size of one voxel at highest resolution
        size: number of voxels at highest resolution
        """

        if self._metadata is None:
            self.load_metadata()

        assert self._metadata is not None
        return self._metadata

    @property
    def basic_geometric_properties(self):
        """get basic properties from scikit-image"""

        self.logger.debug("get basic properties from scikit-image")

        if "basic_geo_props" not in self.cache:
            geometric_properties = regionprops(
                self.data, spacing=self.resolution, cache=False
            )
            geometric_properties = compute(geometric_properties)[0]

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

            self.cache["basic_geo_props"] = {
                f"{self.org_name}_{str(region['label']).zfill(4)}": {
                    prop_name: region[prop]
                    for prop_name, prop in filtered_region_props.items()
                }
                for region in geometric_properties
            }

        return self.cache["basic_geo_props"]

    @property
    def curvature_map(self):
        """Get the curvature map for all organelles"""
        self.logger.debug("get curvature map for all organelles")

        self.get_curvature(labels=None, color=False)
        return self._curvature_map

    @property
    def data(self) -> Array:
        return self.get_data(None)

    @property
    def curvature_radius(self) -> float:
        return self._curv_radius

    @curvature_radius.setter
    def curvature_radius(self, radius):
        """Set the radius for curvature calculations.
        Resets the cached curvature.
        """
        if radius != self._curv_radius:
            self._curv_radius = radius
            self._curvature_map = {}

    @property
    def clipping_corners(self):
        """Lower and upper clipping corner after scaling"""
        if self.project.clipping is not None:
            if self._clip_low_corner is None or self._clip_high_corner is None:
                self.get_data(None)
        return self._clip_low_corner, self._clip_high_corner

    @clipping_corners.setter
    def clipping_corners(self, corners):
        self._clip_low_corner, self._clip_high_corner = corners

    @property
    def clipping_corners_data(self):
        """Lower and upper clipping corner matching the raw data"""
        if self._clip_low_corner_data is None or self._clip_high_corner_data is None:
            self.get_data(None)
        return self._clip_low_corner_data, self._clip_high_corner_data

    @clipping_corners_data.setter
    def clipping_corners_data(self, corners):
        self._clip_low_corner_data, self._clip_high_corner_data = corners

    def get_data(self, compression_level: Optional[str], clipping=None) -> Array:
        """Get data of this source as array.

        Sets self.clipping_corners and self.clipping_corners_data, if
        project clipping is not overwritten

        Args:
            compression_level: Override the compression level specified in
                the project. Default: None, get level from project.
            clipping: Override the clipping specified in the project.
                Default: None, get clipping from project

        Returns:
            Dask array of the data. Respects the in the project set clipping.
        """
        level = compression_level
        if level is None:
            level = self.project.compression_level

        data_at_level: z5py.dataset.Dataset = getattr(self.timepoint, str(level)).data

        # chunk factor for efficieny, needs tuning
        data = da.from_array(data_at_level, chunks="auto")

        _idx = np.nonzero(np.array(self.metadata["levels"]) == level)[0][0]

        cube_slice = (slice(None), slice(None), slice(None))

        if clipping is not None:
            lower_corner = np.array(clipping[0])
            upper_corner = np.array(clipping[1])
        elif self.project.clipping is not None:
            lower_corner, upper_corner = self.project.clipping
        else:
            lower_corner = np.zeros((3,))
            upper_corner = np.ones((3,))

        c_low_d = np.floor(lower_corner * data.shape).astype(int)
        c_high_d = np.ceil(upper_corner * data.shape).astype(int)
        cube_slice = tuple(slice(low, high, 1) for low, high in zip(c_low_d, c_high_d))

        # self._scaling_factors = self.metadata["downsampling"][_idx]
        self._scaling_factors = self.resolution
        self.clipping_corners_data = (c_low_d, c_high_d)
        self.clipping_corners = (
            c_low_d * self._scaling_factors,
            c_high_d * self._scaling_factors,
        )

        return data[cube_slice]

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
    def cache(self):
        """Get the cache for this source.

        Returns the same cache object on consecutive calls.
        On clipping or level change, this cache must be invalidated by calling
        `source.clear_memory_cache()`
        """

        if self._cache is None:
            cs = self.project.cache_settings
            cs["source"] = self.xml_path.stem
            name = f"cache_{cs['project_name']}/{cs['source']}/{cs['level']}/{cs['clipping']}"
            cache = Cache(cache_name=name, disk=cs["disk"], cache_root=cs["cache_root"])
            self._cache = cache
        return self._cache

    @property
    def labels(self) -> list[int]:
        """Return the list of labels present in the data source."""
        return list(self.meshes.keys())

    @property
    def meshes(self) -> dict[int, Delayed]:
        @delayed(pure=False)
        def _get_from_cache(key, cache):
            verts, faces = cache[key]
            return Trimesh(verts, faces)

        @delayed(pure=False)
        def _write_to_cache(key, mesh, cs):
            name = f"cache_{cs['project_name']}/{cs['source']}/{cs['level']}/{cs['clipping']}"
            cache = Cache(cache_name=name, disk=cs["disk"], cache_root=cs["cache_root"])
            cache[key] = (mesh.vertices, mesh.faces)

        self.logger.debug("Requested meshes")
        if self._meshes is None or (
            self._computed_compression != self.project.compression_level
        ):
            self.logger.debug("Meshes not loaded")
            self._meshes = {}
            labels = None

            if "labels" in self.cache:
                self.logger.debug("Meshes in cache, reading labels..")
                labels = self.cache["labels"]
                self.logger.debug(f"{len(labels)} labels found in cache")

                for label in labels:
                    self._meshes[label] = _get_from_cache(label, self.cache)

                # keep meshes in distributed memory, gc previously computed meshes
                self._storage["ref_meshes"] = persist(*self._meshes.values())

                if self.project.clipping is not None:
                    assert "clipping_data" in self.cache, (
                        "clipping_data missing in cache!"
                    )
                    assert "clipping_scaled" in self.cache, (
                        "clipping_scaled missing in cache!"
                    )
                    self.clipping_corners = self.cache["clipping_scaled"]
                    self.clipping_corners_data = self.cache["clipping_data"]
                    self._scaling_factors = self.cache["scaling"]

                self._computed_compression = self.project.compression_level
                self.logger.debug("Meshes loaded from cache")

            else:
                self.logger.info("Meshes not in cache, calculating..")
                self.calculate_mesh()
                self.logger.debug("Saving meshes to cache..")

                self.cache["labels"] = list(self._meshes.keys())

                if self.project.clipping is not None:
                    self.cache["clipping_scaled"] = self.clipping_corners
                    self.cache["clipping_data"] = self.clipping_corners_data
                    self.cache["scaling"] = self._scaling_factors

                cs = self.project.cache_settings.copy()
                cs["source"] = self.xml_path.stem

                delayed_saves = []
                for label, mesh_d in self._meshes.items():
                    delayed_saves.append(_write_to_cache(label, mesh_d, cs))
                compute(*delayed_saves)

                self.logger.debug("Meshes saved in cache")

        return self._meshes

    def get_curvature(self, labels: Optional[int | list[int]], color=True, log=True):
        """Calculate the curvature on vertices.
        If no label is supplied, all meshes are calculated.
        To color the meshes they are all moved to a singel worker, avoid for
        large meshes.

        Parameters
        ----------
        labels
            Optional label or list of labels of meshes to compute the curvature for.
        color
            Whether to color the original mesh according to the curvature.
            Defaults to True
        log
            Use a log color scale

        Returns
        -------
        curvature
            List of arrays of the curvature at each vertex. List over all labels.
        mesh
            Only returned if color is requested. List of computed meshes, colored
            by curvature.
        """
        if labels is None:
            labels = self.labels
        if not isinstance(labels, (list, tuple)):
            labels = [labels]

        tasks = []
        for label in labels:
            dmesh = self.meshes[label]
            if label in self._curvature_map:
                curvature = self._curvature_map[label]
                tasks.append(curvature)
            else:
                curvature = mesure_gaussian_curvature_delayed(
                    dmesh, radius=self._curv_radius
                )
                tasks.append(curvature)
            if color:
                tasks.append(color_delayed_trimesh_rgba(dmesh, curvature, log=log))
        step = 2 if color else 1
        result = compute(*tasks)
        self._curvature_map = {
            label: result[::step][i] for i, label in enumerate(labels)
        }
        if color:
            return result[::2], result[1::2]
        return result

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
        _meshes_chunked = np.empty_like(d_data)
        ids_chunked = np.empty_like(d_data)

        assert self._clip_low_corner_data is not None
        # cumsum of chunksizes over xyz, starting from lower clipping bound
        size_offset_cumsum = [
            np.cumsum(np.array([self._clip_low_corner_data[dim]] + list(ch)))
            for dim, ch in enumerate(self.data.chunks)
        ]

        for index, d_block in np.ndenumerate(d_data):
            space_offset = tuple(
                int(size_offset_cumsum[i][c]) - (2 if overlap else 0)
                for i, c in enumerate(index)
            )
            meshes_chunk, ids_chunk = _block_mesher(
                block=d_block,
                space_offset=space_offset,
                reduction_factor=reduction_factor,
                debug_color=debug_color,
                scaling_factors=self._scaling_factors,
            )
            _meshes_chunked[index] = meshes_chunk
            ids_chunked[index] = ids_chunk

        # flatten the ids and delayed meshes
        d_ids = []
        indices = []
        d_meshes = []
        for index, ids_chunk in np.ndenumerate(ids_chunked):
            mesh = _meshes_chunked[index]
            d_ids.append(ids_chunk)
            indices.append(index)
            d_meshes.append(mesh)

        # calculate meshes and keep references to make mesh persistend on workers
        self._storage["ref_meshes"] = persist(*(d_meshes + d_ids))

        _ids_to_chunks = build_chunk_lookup(d_ids, indices).compute()
        assert _ids_to_chunks is not None  # for linter

        # get some statistics
        all_chunks = list(_ids_to_chunks.values())
        all_ids = np.array(list(_ids_to_chunks.keys()))
        id_amounts = np.array([len(idxs) for idxs in all_chunks])
        amounts, inverse, freqs = np.unique(
            id_amounts, return_counts=True, return_inverse=True
        )
        for amount, frq in zip(amounts, freqs):
            self.logger.info(f"{frq} labels in {amount} chunks")

        # Cleanup: Merge meshes crossing chunks
        self._meshes = {}
        duplicate_ids = all_ids[np.nonzero(id_amounts > 1)]
        for ind in duplicate_ids:
            chunk_idxs = _ids_to_chunks[ind]
            meshes = [_meshes_chunked[idx][ind] for idx in chunk_idxs]
            merged_mesh = merge_meshes(meshes, color=0)
            if simplify:
                merged_mesh = simplify_mesh(merged_mesh, simplify)
            self._meshes[ind] = merged_mesh

        unique_ids = all_ids[np.nonzero(id_amounts == 1)]
        for ind in unique_ids:
            mesh = _meshes_chunked[_ids_to_chunks[ind][0]][ind]
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
        self._basic_geometric_properties = {}
        self._meshes = None
        self._computed_compression = None
        self._curvature_map = {}
        self._storage = {}
        self._cache = None
        self._organelles = None
        self._clip_low_corner = None
        self._clip_high_corner = None
        self._clip_low_corner_data = None
        self._clip_high_corner_data = None
        self._scaling_factors = None
        self._curv_radius = 4.0

    def instantiate_organelles(self):
        if self._organelles is None:
            self._organelles = {}
            self.logger.debug(f"Initializing Organelles {self.org_name}")

            # Iterate available organelle classes and construct organelles
            if not (orgclass := organelle_registry[self.org_name]):
                raise ValueError(f"Unknown organelle class {self.org_name}")
            for organelle in orgclass.construct(self, self.labels):
                self._organelles[organelle.id] = organelle

    @property
    def organelles(self) -> list[Organelle]:
        self.instantiate_organelles()
        return self.get_organelles(
            ids="*",
            permanent_whitelist=self.project.permanent_whitelist,
            permanent_blacklist=self.project.permanent_blacklist,
        )

    @property
    def organelle_ids(self) -> list[str]:
        self.instantiate_organelles()
        return self.get_organelle_ids(
            ids="*",
            permanent_whitelist=self.project.permanent_whitelist,
            permanent_blacklist=self.project.permanent_blacklist,
        )

    def get_organelles(
        self,
        ids: str | list[str] = "*",
        permanent_whitelist=None,
        permanent_blacklist=None,
    ) -> list[Organelle]:
        """Return a list of organelle objects.
        Apply filters by id glob pattern and permanent filter lists.

        Args:
            ids: glob pattern to match the ids. Default "*" includes all.
            permanent_whitelist: ids to always include,
                does *not* respect the project setting.
            permanent_blacklist: ids to always exclude,
                does *not* respect the project setting.

        Returns:
            List of organelle objects
        """
        filtered_ids = self.get_organelle_ids(
            ids=ids,
            permanent_whitelist=permanent_whitelist,
            permanent_blacklist=permanent_blacklist,
        )
        assert isinstance(self._organelles, dict), (
            "RuntimeError: source._organelles did not get set!"
        )
        return [self._organelles[id] for id in filtered_ids]

    def get_organelle_ids(
        self,
        ids: str | list[str] = "*",
        permanent_whitelist=None,
        permanent_blacklist=None,
    ) -> list[str]:
        """Return a list of organelle ids.
        Computes self._organelles

        Args:
            ids: glob pattern to match the ids. Default "*" includes all.
            permanent_whitelist: ids to always include,
                does *not* respect the project setting.
            permanent_blacklist: ids to always exclude,
                does *not* respect the project setting.

        Returns:
            List of organelle ids
        """
        self.instantiate_organelles()
        assert isinstance(self._organelles, dict)

        result = []
        if isinstance(ids, str):
            ids = [ids]

        for id_ in ids:
            # Filter the organelles with the given ids pattern
            filtered_ids = fnmatch.filter(self._organelles.keys(), id_)

            if permanent_blacklist is not None:
                filtered_ids = [
                    org_id
                    for org_id in filtered_ids
                    if org_id not in permanent_blacklist
                ]

            if permanent_whitelist is not None:
                filtered_ids.extend(permanent_whitelist)
            result.extend(filtered_ids)

        return result

    def generate_skeletons(
        self,
        labels: Optional[int | list[int]],
        skeletonization_type: str = "wavefront",
        theta: float = 0.4,
        waves: int = 1,
        step_size: int = 2,
        epsilon: float = 0.1,
        sampling_dist: float = 0.1,
        path_sample_dist: float = 0.1,
        recompute: bool = False,
    ):
        """
        Generates a skeleton for the organelle.
        The skeleton is generated and cleaned using the skeletor library.

        The last generated skeleton will be kept in memory and can be accessed
        using the skeleton or sampled_skeleton property.

        Args:
            skeletonization_type: The type of skeletonization to use. Can be either
                "wavefront" or "vertex_clusters".

            waves: The number of waves to use for the wavefront skeletonization.
                The higher the number of waves, the more branches are removed.
            step_size: The step size for the wavefront skeletonization. The higher
                the step size, the less vertices are used for the skeleton.
            theta: The threshold for the clean_up function. The higher the threshold,
                the more branches are removed.
            epsilon: The epsilon value for the contract function. The higher the epsilon,
                the more the mesh is contracted.
            sampling_dist: The sampling distance for the skeletonize function.
                The higher the sampling distance, the less vertices are used
                for the skeleton.
            path_sample_dist: The distance between the sample points on the skeleton
                arms. The higher the distance, the less sample points are used.
        Returns:
            List of organelles which have a skeleton.

        """

        @delayed
        def _delayed_skeletonize(
            mesh,
            label,
            skeletonization_type,
            theta,
            waves,
            step_size,
            epsilon,
            sampling_dist,
            path_sample_dist,
        ):
            result = [None, ""]
            try:
                fixed_mesh = sk.pre.fix_mesh(mesh)
            except IndexError as e:
                result[1] += (
                    f"Could not fix mesh in skeleton generation for {label} with error {e}"
                )
                fixed_mesh = mesh

            if len(fixed_mesh.vertices) < 10:
                result[1] += f"Not enough vertices for skeleton! {label}"
                return result

            try:
                if skeletonization_type == "wavefront":
                    skel = sk.skeletonize.by_wavefront(
                        fixed_mesh, waves=waves, progress=False, step_size=step_size
                    )
                elif skeletonization_type == "vertex_clusters":
                    try:
                        cont = sk.pre.contract(
                            fixed_mesh, epsilon=epsilon, progress=False
                        )
                    except IndexError:
                        result[1] += (
                            f"couldn't contract mesh using normal mesh for {label}"
                        )

                        cont = fixed_mesh
                    skel = sk.skeletonize.by_vertex_clusters(
                        cont, sampling_dist=sampling_dist, progress=False
                    )
                    skel.mesh = fixed_mesh
                else:
                    result[1] += f"Unknown skeletonize_type: {skeletonization_type}"
                    return result

                sk.post.clean_up(skel, inplace=True, theta=theta)
                sk.post.radii(skel, method="knn")

                # if no skeleton can be created just skip this
                if len(skel.vertices) <= 1:
                    result[1] += f"No vertices for {label}"
                    return result

                sampled_skeleton = sample_skeleton(
                    skel, path_sample_dist=path_sample_dist
                )

                # reset skeleton info for new calculation
                skeleton_info = get_skeleton_info(skel)
                result[1] += f"Generated skeleton for {label}"
                result[0] = (skel, skeleton_info, sampled_skeleton, label)

            except Exception as e:
                result[1] += f"Could not generate skeleton for {label} with error {e}"

            return result

        if skeletonization_type not in ["wavefront", "vertex_clusters"]:
            raise ValueError(
                "Skeletonization type must be either 'wavefront' or 'vertex_clusters'."
            )

        if labels is None:
            labels = self.labels
        if not isinstance(labels, (list, tuple)):
            labels = [labels]

        organelles_labeled = {o.label: o for o in self.organelles}
        tasks = []
        for label in labels:
            o = organelles_labeled[label]
            if o.skeleton is not None and not recompute:
                tasks.append(
                    ((o.skeleton, o.skeleton_info, o.sampled_skeleton, label), "")
                )
                continue

            dmesh = self.meshes[label]
            tasks.append(
                _delayed_skeletonize(
                    dmesh,
                    label,
                    skeletonization_type,
                    theta,
                    waves,
                    step_size,
                    epsilon,
                    sampling_dist,
                    path_sample_dist,
                )
            )

        results = compute(*tasks)

        orgs = []
        for result in results:
            if result[0] is None:
                self.logger.debug(result[1])
                continue

            skel, skeleton_info, sampled_skeleton, label = result[0]
            organelles_labeled[label].skeleton = skel
            organelles_labeled[label].skeleton_info = skeleton_info
            organelles_labeled[label].sampled_skeleton = sampled_skeleton
            orgs.append(organelles_labeled[label])
        return orgs
