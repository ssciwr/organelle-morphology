from collections import defaultdict
import logging
from typing import Optional
from pathlib import Path
import organelle_morphology
from organelle_morphology.organelle import Organelle, organelle_registry

from dask.base import compute, persist
import dask.array as da
from dask.array.core import Array
from dask.delayed import Delayed, delayed

import skeletor as sk

import fnmatch
import numpy as np
import xml.etree.ElementTree as ET
from skimage.measure import regionprops
from dataclasses import dataclass, field
import z5py
from organelle_morphology.block_mesher import block_mesher


from organelle_morphology.records import PropertyBlock
from organelle_morphology.util import (
    Cache,
    color_delayed_trimesh_rgba,
    get_skeleton_info,
    merge_meshes,
    measure_gaussian_curvature_delayed,
    sample_skeleton,
)


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


@dataclass
class SourceMetadata(PropertyBlock):
    data_root: Path
    downsampling: list[list[int]]
    levels: list[str]
    size: tuple[int, ...]
    resolution: list[float]
    name: str
    coarse_level: str


class DataSource:
    def __init__(
        self,
        project: "organelle_morphology.Project",
        xml_path: Path,
        organelle: str,
        background_label: int = 0,
    ):
        """Initialize a data source.

        Initialize a data source for organelle morphology analysis.
        The data source is linked to a CebraEM/Mobie project and contains information
        about a single organelle data source.

        Args:
            project:
                The CebraEM/Mobie project this data source is linked to.
            xml_path:
                The path to the XML file describing the source data.
            organelle:
                The name of the organelle being analyzed.
            background_label:
                The label used for background pixels.
        """

        self.logger = logging.getLogger(__name__)
        self.project: "organelle_morphology.Project" = project
        self.xml_path = xml_path
        self.org_name = organelle
        self.background_label = background_label

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
        self.logger.debug(f"Start loading {n5}, metadata structure:")

        def _meta_finder(name: str, obj):
            self.logger.debug(
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
            _resolution: str = (
                xmldata.find("SequenceDescription")
                .find("ViewSetups")
                .find("ViewSetup")
                .find("voxelSize")
                .find("size")
                .text
            )
            resolution = list((float(i) for i in _resolution.split(" ")))
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
            self.logger.warning(
                "Only single timepoints supported, ignoring all but the first!"
            )

        self.timepoint = list(timepoints.values())[0]

        assert resolution == self.timepoint.resolution[::-1]

        coarse_level = self.timepoint.levels[0]
        for level in self.timepoint.levels:
            if int(level[-1]) > int(coarse_level[-1]):
                coarse_level = level

        self._metadata = SourceMetadata(
            data_root=self.xml_path / filename,
            downsampling=self.timepoint.downsamplingFactors,
            levels=self.timepoint.levels,
            size=tuple([int(i) for i in size.split(" ")][::-1]),
            resolution=resolution,
            name=name,
            coarse_level=coarse_level,
        )

    @property
    def metadata(self) -> SourceMetadata:
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

        self.calc_curvature(labels=None)
        return self._curvature_map

    @property
    def data(self) -> Array:
        return self.get_data(None)

    @property
    def curvature_radius(self) -> float:
        if self._curv_radius is None:
            self._curv_radius = self.resolution[0] * 2
        return self._curv_radius

    @curvature_radius.setter
    def curvature_radius(self, radius: float):
        """Set the radius for curvature calculations.
        Resets the cached curvature.
        """
        if radius != self._curv_radius:
            self._curv_radius = radius
            self._curvature_map = {}

    @property
    def clipping_corners(self) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper clipping corner in units of the resolution"""
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

    def get_data(self, compression_level: Optional[str] = None, clipping=None) -> Array:
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

        _idx = np.nonzero(np.array(self.metadata.levels) == level)[0][0]

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

        if clipping is None and compression_level is None:
            self._scaling_factors = self.resolution
            self.clipping_corners_data = (c_low_d, c_high_d)
            self.clipping_corners = (
                c_low_d * self._scaling_factors,
                c_high_d * self._scaling_factors,
            )
        else:
            self.logger.debug(
                "Getting non-default data,"
                "clipping corners and scaling_factor are not updated!"
            )

        return data[cube_slice]

    @property
    def coarse_data(self) -> Array:
        """Return the coarsest version of the dataset.

        This can be used for algorithms that do not critically depend
        on the data resolution and should be fast.
        """

        return self.get_data(self.metadata.coarse_level)

    @property
    def data_resolution(self) -> list[float]:
        """Return the resolution in micrometers at which the data is stored."""

        return self.metadata.resolution

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
    def coarse_resolution(self) -> tuple[float]:
        """Return the resolution of our data at the coarsest compression level."""

        return tuple(
            (
                r * d
                for r, d in zip(
                    self.data_resolution,
                    getattr(
                        self.timepoint, self.metadata.coarse_level
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
    def centroid(self) -> np.ndarray:
        if self._centroid is None:
            self._centroid = (
                np.mean(self.data.nonzero(), axis=1) * self.resolution
            ) + self.clipping_corners[0]

        return self._centroid

    @property
    def global_coarse_centroid(self):
        if self._global_coarse_centroid is None:
            data = self.get_data(
                compression_level=self.metadata.coarse_level,
                clipping=[[0, 0, 0], [1, 1, 1]],
            )
            self._global_coarse_centroid = (
                np.mean(data.nonzero(), axis=1) * self.coarse_resolution
            )
        return self._global_coarse_centroid

    @property
    def labels(self) -> list[int]:
        """Return the list of labels present in the data source."""
        return list(self.ids_to_chunks.keys())

    @property
    def ids_to_chunks(self):
        try:
            ids_to_chunks = self.cache["ids_to_chunks"]
            return ids_to_chunks
        except KeyError:
            self.mesh_fragments
            ids_to_chunks = self.cache["ids_to_chunks"]
            return ids_to_chunks

    @property
    def mesh_fragments(self) -> tuple[dict[int, list[tuple[int]]], np.ndarray]:
        """Get the mesh fragments from cache, or calculate them.

        The mesh fragments are delayed read from the source.cache, so reads
        are cached in memory.

        Returns:
            dict[int, list[tuple[int]]]: dict to translate mesh ids to a list of chunk
                indices to find all chunks containing parts of this mesh.
            np.ndarray: Array containing the delayed chunks.
                Shape matches storage on disk, after applying the clipping.
        """
        return self._get_mesh_fragments()

    def _get_mesh_fragments(self, recursed=False):
        """Private implementation of `mesh_fragments`.
        This is a separate function because of the recursion fail-safe.
        """

        @delayed(pure=False)
        def _write_frag_cache_batch(key_values, cs):
            """Write multiple meshes to cache in a single operation"""
            name = f"cache_{cs['project_name']}/{cs['source']}/{cs['level']}/{cs['clipping']}"
            cache = Cache(cache_name=name, disk=cs["disk"], cache_root=cs["cache_root"])
            for key, value in key_values:
                cache[key] = value

        @delayed(pure=True)
        def _get_fragment_cache(key, cs):
            name = f"cache_{cs['project_name']}/{cs['source']}/{cs['level']}/{cs['clipping']}"
            cache = Cache(cache_name=name, disk=cs["disk"], cache_root=cs["cache_root"])
            return cache[key]

        if self._fragments_chunked is None:
            if "ids_to_chunks" not in self.cache:
                ids_to_chunks, meshes_chunked_d = self.calculate_mesh()

                self.cache["ids_to_chunks"] = ids_to_chunks
                self.cache["chunks_shape"] = meshes_chunked_d.shape

                cs = self.project.cache_settings.copy()
                cs["source"] = self.xml_path.stem

                delayed_saves = []
                batch_size = 1  # TODO: check optimal batch size over chunks

                tasks = []
                for idx, meshes_in_chunk in np.ndenumerate(meshes_chunked_d):
                    tasks.append((f"fragment_{idx}", meshes_in_chunk))

                for i in range(0, len(tasks), batch_size):
                    to_save = tasks[i : i + batch_size]
                    delayed_save = _write_frag_cache_batch(to_save, cs)
                    delayed_saves.append(delayed_save)

                self.logger.debug(
                    f"Saving {len(delayed_saves)} batches of mesh fragments"
                )
                compute(*delayed_saves, traverse=False)
                self.logger.debug("Mesh fragments saved to cache")

            else:
                try:
                    meshes_chunked_d = np.empty(
                        self.cache["chunks_shape"], dtype=object
                    )
                    cs = self.project.cache_settings.copy()
                    cs["source"] = self.xml_path.stem
                    for idx, _ in np.ndenumerate(meshes_chunked_d):
                        meshes_chunked_d[idx] = _get_fragment_cache(
                            f"fragment_{idx}", cs
                        )

                except KeyError as e:
                    self.logger.warning(
                        f"Could not load mesh fragment from cache, recomputing..\n{e}"
                    )
                    del self.cache["ids_to_chunks"]
                    if not recursed:
                        return self._get_mesh_fragments(recursed=True)
                    raise RuntimeError(
                        "Error: Could not compute mesh fragments after two attempts!"
                    )
            self._fragments_chunked = meshes_chunked_d

        return self.cache["ids_to_chunks"], self._fragments_chunked

    @property
    def meshes(self):
        @delayed(pure=False)
        def _write_mesh_cache_batch(key_values, cs):
            """Write multiple meshes to cache in a single operation"""
            name = f"cache_{cs['project_name']}/{cs['source']}/{cs['level']}/{cs['clipping']}"
            cache = Cache(cache_name=name, disk=cs["disk"], cache_root=cs["cache_root"])
            for key, value in key_values:
                cache[key] = value

        def _write_mesh_cache(key_value, cs):
            """Write multiple meshes to cache in a single operation"""
            name = f"cache_{cs['project_name']}/{cs['source']}/{cs['level']}/{cs['clipping']}"
            cache = Cache(cache_name=name, disk=cs["disk"], cache_root=cs["cache_root"])
            cache[key_value[0]] = compute(key_value[1])[0]

        @delayed(pure=True)
        def _get_mesh_cache(key, cs):
            name = f"cache_{cs['project_name']}/{cs['source']}/{cs['level']}/{cs['clipping']}"
            cache = Cache(cache_name=name, disk=cs["disk"], cache_root=cs["cache_root"])
            return cache[key]

        if self.project._cache_settings["cache_meshes"]:
            if "mesh_ids" not in self.cache:
                cs = self.project.cache_settings.copy()
                cs["source"] = self.xml_path.stem

                # # None batched variant
                # tasks = []
                # ids = []
                #
                # for idx, mesh in self.merge_fragments_into_meshes(
                #     *self.mesh_fragments
                # ).items():
                #     ids.append(idx)
                #     tasks.append((f"mesh_{idx}", mesh))
                #
                # self.logger.debug(f"Saving {len(tasks)} meshes")
                # self.project.client.gather(
                #     self.project.client.map(_write_mesh_cache, tasks, cs=cs)
                # )
                #
                # self.cache["mesh_ids"] = ids
                # self.logger.debug("Meshes saved to cache")

                delayed_saves = []
                batch_size = 10  # TODO: check optimal batch size over chunks

                tasks = []
                ids = []
                for idx, mesh in self.merge_fragments_into_meshes(
                    *self.mesh_fragments
                ).items():
                    ids.append(idx)
                    tasks.append((f"mesh_{idx}", mesh))

                for i in range(0, len(tasks), batch_size):
                    to_save = tasks[i : i + batch_size]
                    delayed_save = _write_mesh_cache_batch(to_save, cs)
                    delayed_saves.append(delayed_save)

                self.cache["mesh_ids"] = ids
                self.logger.debug(f"Saving {len(delayed_saves)} batches of meshes")
                compute(*delayed_saves, traverse=False)
                self.logger.debug("Meshes saved to cache")

                # # batched by chunk
                # delayed_saves = []
                #
                # meshes = self.merge_fragments_into_meshes(*self.mesh_fragments)
                #
                # tasks = defaultdict(list)
                #
                # for ind, mesh in meshes.items():
                #     chunk = self.ids_to_chunks[ind][0]
                #     tasks[chunk].append((f"mesh_{ind}", mesh))
                #
                # stats = defaultdict(int)
                # for batch in tasks.values():
                #     stats[len(batch)] += 1
                # self.logger.debug(f"Record about saving mesh batches:\n{stats}")
                #
                # for to_save in tasks.values():
                #     delayed_save = _write_mesh_cache_batch(to_save, cs)
                #     delayed_saves.append(delayed_save)
                #
                # self.logger.debug(f"Saving {len(delayed_saves)} batches of meshes")
                #
                # compute(*delayed_saves, traverse=False)
                # self.cache["mesh_ids"] = list(meshes.keys())
                # self.logger.debug("Meshes saved to cache")

            if self._meshes is None:
                cs = self.project.cache_settings.copy()
                cs["source"] = self.xml_path.stem
                self._meshes = {}
                for idx in self.cache["mesh_ids"]:
                    self._meshes[idx] = persist(_get_mesh_cache(f"mesh_{idx}", cs))[0]
            return self._meshes

        return self.merge_fragments_into_meshes(*self.mesh_fragments)

    def calc_curvature(self, labels: Optional[int | list[int]] = None) -> dict:
        """Calculate the curvature on vertices.
        If no label is supplied, all meshes are calculated.

        Args:
            labels: Optional label or list of labels of meshes to compute
            the curvature for.
        Returns:
            curvature_map dict for all labels
        """
        if labels is None:
            labels = self.labels
        if not isinstance(labels, (list, tuple)):
            labels = [labels]

        if all([la in self._curvature_map for la in labels]):
            self.logger.debug("All curvatures already calculated.")
            return self._curvature_map

        tasks = []
        self.logger.debug("Starting curvature calculation")
        for label in labels:
            dmesh = self.meshes[label]
            curvature = measure_gaussian_curvature_delayed(
                dmesh, radius=self.curvature_radius
            )
            tasks.append(curvature)

        result = compute(*tasks)
        self._curvature_map.update({label: result[i] for i, label in enumerate(labels)})
        return self._curvature_map

    def get_curvature_range(self):
        # mean_mean = np.mean([c.mean() for c in self._curvature_map.values()])
        mean_std = np.mean([c.std() for c in self._curvature_map.values()])

        return {
            "vmin": -15 * mean_std,
            "vmax": 15 * mean_std,
        }

    def get_meshes_curvature_colored(
        self,
        labels: Optional[int | list[int]] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        log=True,
    ) -> list[Delayed]:
        if labels is None:
            labels = self.labels
        if not isinstance(labels, (list, tuple)):
            labels = [labels]

        self.calc_curvature(labels)
        if vmin is None or vmax is None:
            new_maxima = self.get_curvature_range()
        if vmin is None:
            vmin = new_maxima["vmin"]
        if vmax is None:
            vmax = new_maxima["vmax"]

        tasks = []
        for label in labels:
            dmesh = self.meshes[label]
            curvature = self._curvature_map[label]
            tasks.append(
                color_delayed_trimesh_rgba(
                    dmesh, curvature, log=log, vmin=vmin, vmax=vmax
                )
            )
        return tasks

    def calculate_mesh(
        self,
        reduction_factor=0,
        overlap=True,
        debug_color: Optional[int] = None,
    ) -> tuple[dict[int, list[tuple[int]]], np.ndarray]:
        """Compute meshes for this source.

        Meshes crossing chunks are merged.

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

        if overlap:
            overlapped = da.overlap.overlap(
                self.data, depth={0: 2, 1: 2, 2: 2}, boundary="reflect"
            )
            chunks = overlapped.chunks
            d_data = overlapped.to_delayed()
        else:
            chunks = self.data.chunks
            d_data = self.data.to_delayed()
        meshes_chunked = np.empty_like(d_data)
        ids_chunked = np.empty_like(d_data)
        assert self._clip_low_corner_data is not None

        # cumsum of chunksizes over xyz, starting from lower clipping bound
        size_offset_cumsum = []
        for dim, ch in enumerate(chunks):
            if overlap:
                # remove overlap to get correct size
                ch = [c - 4 for c in ch]
            else:
                ch = list(ch)
            size_offset_cumsum.append(
                np.cumsum(np.array([self._clip_low_corner_data[dim]] + ch))
            )

        for index, d_block in np.ndenumerate(d_data):
            space_offset = tuple(
                int(size_offset_cumsum[i][c]) - (2 if overlap else 0)
                for i, c in enumerate(index)
            )
            meshes_chunk, ids_chunk = block_mesher(
                block=d_block,
                space_offset=space_offset,
                reduction_factor=reduction_factor,
                debug_color=debug_color,
                scaling_factors=self._scaling_factors,
            )
            meshes_chunked[index] = meshes_chunk
            ids_chunked[index] = ids_chunk

        # flatten the ids and delayed meshes
        d_ids = []
        indices = []
        d_meshes = []
        for index, ids_chunk in np.ndenumerate(ids_chunked):
            mesh = meshes_chunked[index]
            d_ids.append(ids_chunk)
            indices.append(index)
            d_meshes.append(mesh)

        # calculate meshes and keep references to make mesh persistend on workers
        # self._storage["ref_meshes"] = persist(*(d_meshes + d_ids))

        ids_to_chunks = build_chunk_lookup(d_ids, indices).compute()
        assert ids_to_chunks is not None  # for linter

        return ids_to_chunks, meshes_chunked

    def merge_fragments_into_meshes(
        self,
        ids_to_chunks: dict,
        meshes_chunked: np.ndarray,
        simplify: Optional[float] = None,
    ) -> dict[int, Delayed]:
        @delayed
        def simplify_mesh(mesh, factor=0.1):
            """Simplify a trimesh object"""
            mesh = mesh.simplify_quadric_decimation(factor, aggression=1)
            return mesh

        # get some statistics
        all_chunks = list(ids_to_chunks.values())
        all_ids = np.array(list(ids_to_chunks.keys()))
        id_amounts = np.array([len(idxs) for idxs in all_chunks])
        amounts, inverse, freqs = np.unique(
            id_amounts, return_counts=True, return_inverse=True
        )
        for amount, frq in zip(amounts, freqs):
            self.logger.info(f"{frq} labels in {amount} chunks")

        # Cleanup: Merge meshes crossing chunks
        meshes = {}
        duplicate_ids = all_ids[np.nonzero(id_amounts > 1)]
        for ind in duplicate_ids:
            chunk_idxs = ids_to_chunks[ind]
            loc_meshes = [meshes_chunked[idx][ind] for idx in chunk_idxs]
            merged_mesh = merge_meshes(loc_meshes, color=0)
            if simplify:
                merged_mesh = simplify_mesh(merged_mesh, simplify)
            meshes[ind] = merged_mesh

        unique_ids = all_ids[np.nonzero(id_amounts == 1)]
        for ind in unique_ids:
            mesh = meshes_chunked[ids_to_chunks[ind][0]][ind]
            if simplify:
                mesh = simplify_mesh(mesh, simplify)
            meshes[ind] = mesh

        self._computed_compression = self.project.compression_level
        self.logger.debug(
            f"finished merging fragments (delayed) for source {self.org_name}"
        )
        return meshes

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
        self._meshes = None
        self._fragments_chunked = None
        self._organelles = None
        self._clip_low_corner = None
        self._clip_high_corner = None
        self._clip_low_corner_data = None
        self._clip_high_corner_data = None
        self._scaling_factors = None
        self._curv_radius = None
        self._mcs_dicts = {}
        self._centroid = None
        self._global_coarse_centroid = None

    @property
    def mcs_dicts(self) -> dict:
        for mcs_label in self.project.mcs_labels:
            if mcs_label in self._mcs_dicts:
                continue
            labeled_dict = defaultdict(dict)
            self._mcs_dicts[mcs_label] = labeled_dict
            for org in self.organelles:
                if mcs := org.mcs[mcs_label]:
                    labeled_dict[org.id] = mcs

        return self._mcs_dicts

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
