import logging
from collections import defaultdict
from dataclasses import InitVar, dataclass, fields
from pathlib import Path
from typing import Optional

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
from dask.delayed import Delayed, delayed
from trimesh import Trimesh

import organelle_morphology
from organelle_morphology.records import PropertyBlock, Record
from organelle_morphology.util import (
    bounding_box_delayed,
    color_delayed_trimesh_faces,
    numpy_to_python,
    reset_color_delayed,
)

# Version of the on-disk MCS record schema. Bump whenever the meaning of the
# stored aggregates changes; records with any other version are rejected on
# load instead of being mixed with freshly calculated values.
CURRENT_MCS_SCHEMA_VERSION = 2


def organelle_types() -> list[str]:
    """The list of organelles currently implemented.

    The strings used here to encode the organelles are expected in
    various APIs when referring to a specific organelle.
    """
    return list(organelle_registry.keys())


@dataclass(frozen=True)
class OrganelleMetadata(PropertyBlock):
    source: Path
    label: int
    id: str


# unsafe hash instead of frozen due to __post_init__
@dataclass(unsafe_hash=True)
class McsMetadata(PropertyBlock):
    mcs_label: str
    max_dist: float
    schema_version: int
    min_dist: float = 0.0
    filter_1: str = "*"
    filter_2: str = "*"
    organelle_source: Optional[Path] = None
    organelle_label: Optional[int] = None
    organelle_id: Optional[str] = None
    organelle_meta: InitVar[Optional[OrganelleMetadata]] = None

    def __post_init__(self, organelle_meta: Optional[OrganelleMetadata]):
        if self.schema_version != CURRENT_MCS_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported MCS schema_version {self.schema_version}, "
                f"expected {CURRENT_MCS_SCHEMA_VERSION}: pre-fix MCS records "
                "are invalid and must be recalculated."
            )
        if organelle_meta is not None:
            for f in fields(organelle_meta):
                setattr(
                    self,
                    "organelle_" + f.name,
                    getattr(organelle_meta, f.name),
                )


@dataclass(frozen=True)
class MeshData(PropertyBlock):
    volume: float
    area: float
    centroid: tuple[float]
    inertia: tuple[float]
    water_tight: bool
    sphericity: Optional[float]
    flatness_ratio: Optional[float]


@dataclass
class McsData(PropertyBlock):
    n_contacts: int
    n_partners: int
    total_area: float
    mean_area: float
    std_area: float
    mean_dist: float
    std_dist: float
    min_dist: float
    max_dist: float
    partners: tuple[str]
    min_dist_per_org: tuple[float]
    max_dist_per_org: tuple[float]
    n_contacts_per_area: float
    n_contacts_per_volume: float
    area_per_area: float
    area_per_volume: float


class Organelle:
    _name = "organelle_name"
    logger = logging.getLogger(__name__)

    def __init__(self, source: "organelle_morphology.DataSource", label: int):
        """The organelle base class

        Holds references to its mesh and label. Also holds analysis results.

        Note that instances of Organelle typically are not instantiated directly,
        but through the corresponding subclass of OrganelleFactory.

        Args:
            source: The source object containing this organelle
            label: label used in the original data for this organelle.
        """
        self.source = source
        self.project = source.project
        self.label = label
        self._organelle_id = f"{self._name}_{str(label).zfill(4)}"
        self._mesh_properties = None
        self._skeleton = None
        self._sampled_skeleton = None
        self._skeleton_info = {}
        # label -> partner id -> list of own MCS patch entries
        self._mcs = defaultdict(lambda: defaultdict(list))
        self._chunks = None

    @classmethod
    def construct(cls, source, labels: list[int]):
        """A trivial factory method for organelle instances.

        It constructs an instance per label. The construction process for each
        organelle is independent of all others. Other organelles can subclass
        this to implement a construction process that e.g. takes into account
        all organelle instances.
        """
        for label in labels:
            yield organelle_registry[cls._name](
                source=source,
                label=label,
            )

    def __repr__(self):
        return f"{self.__class__.__name__}({self._organelle_id})"

    @property
    def metadata(self) -> OrganelleMetadata:
        return OrganelleMetadata(
            source=self.source.xml_path, label=self.label, id=self._organelle_id
        )

    @property
    def chunks(self):
        if self._chunks is None:
            self._chunks = self.source.ids_to_chunks[self.label]
        return self._chunks

    @property
    def skeleton(self):
        """Get the skeleton for this organelle"""

        return self._skeleton

    @skeleton.setter
    def skeleton(self, value):
        self._skeleton = value

    @property
    def skeleton_info(self):
        # calculate some basic skeleton properties from the skeleton graph
        if not self._skeleton:
            return None

        if self._skeleton_info:
            return self._skeleton_info

    @skeleton_info.setter
    def skeleton_info(
        self, value
    ) -> "organelle_morphology.skeleton_analysis.SkeletonData":
        self._skeleton_info = value

    @property
    def sampled_skeleton(self):
        """Get the sampled skeleton for this organelle.
        This included the sampled points,
        as well as the corresponding reference point to later get the correct plane normal vector
        """
        if self._skeleton is None:
            self.logger.warning(
                f"Skeleton has not been generated for {self.id} yet. Please run project.generate_skeletons() first."
            )
            return None

        return self._sampled_skeleton

    @sampled_skeleton.setter
    def sampled_skeleton(self, value):
        self._sampled_skeleton = value

    @property
    def mesh(self) -> Delayed:
        """Get the mesh for this organelle"""
        return self.source.meshes[self.label]

    def get_mesh_mcs_colored(self, mcs_label=None) -> Delayed:
        """Get mcs colored delayed meshes"""
        cm = plt.get_cmap("tab20")
        colored = self.mesh
        colored = reset_color_delayed(colored)
        if mcs_label is None:
            labels = list(self.source.project.mcs_labels)
            if len(labels) == 0:
                self.logger.warning("No mcs data, first run project.search_mcs")
                return colored
            mcs_label = labels[0]

        if mcs_partners_dict := self.mcs.get(mcs_label):
            for i, patches in enumerate(mcs_partners_dict.values()):
                if not patches:
                    continue
                # Colour per partner for now (ticket 04 moves to per patch).
                faces = np.unique(
                    np.concatenate(
                        [
                            np.asarray(patch["faces_index"], dtype=np.int64)
                            for patch in patches
                        ]
                    )
                )
                colored = color_delayed_trimesh_faces(
                    colored,
                    faces,
                    [c * 255 for c in cm(i % 20)],
                )
        return colored

    @property
    def id(self):
        """Get the organelle ID of this organelle"""

        return self._organelle_id

    @property
    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        return bounding_box_delayed(self.mesh).compute()

    @property
    def geometric_data(self):
        """Get the geometric data for this organelle
        Possible keywords are:
        "voxel_volume": for 3d this is the volume
        "voxel_bbox",
        "voxel_slice": the slice of the bounding box
        "voxel_centroid"
        "voxel_moments"
        "voxel_extent": how much volume of the bounding box is occupied by the object
        "voxel_solidity":ratio of pixels in the convex hull to those in the region

        """
        return self.source.basic_geometric_properties[self.id]

    @property
    def mesh_properties(self) -> MeshData:
        """Get properties of the mesh for this organelle
        Causes a compute call, if you need this for many organelles,
        use `get_mesh_properties_delayed`
        """
        return self.source.mesh_properties[self.label]

    @property
    def curvature_map(self) -> np.ndarray:
        """Get the mesh data for this organelle"""
        return self.source.calc_curvature(self.label)[self.label]

    @property
    def curvature_mesh(self) -> Delayed:
        return self.source.get_meshes_curvature_colored(labels=self.label)[0]

    def add_mcs(self, mcs_dict, mcs_meta):
        """Append the own-side MCS patches of one partner to this organelle.

        A pair can form several appositions, so each patch is stored as its own
        entry. Repeated calls for the same ``(label, partner)`` append rather
        than overwrite.

        Args:
            mcs_dict (dict): per-side MCS result from the geometry core,
                containing a ``patches`` list.
            mcs_meta (dict): label parameters (band, id filters) for the run.
        """
        mcs_target = mcs_dict["partner_id"]
        mcs_label = mcs_dict["mcs_label"]
        patches = mcs_dict.get("patches", ())
        if not patches:
            return

        entries = self._mcs[mcs_label][mcs_target]
        for patch in patches:
            entries.append(
                {
                    "patch_id": patch["patch_id"],
                    "faces_index": patch["faces_index"],
                    "distances": patch["distances"],
                    "area": patch["area"],
                    "truncated": patch["truncated"],
                    "partner_patch_ids": patch["partner_patch_ids"],
                    "meta": mcs_meta,
                }
            )

    def calc_mcs_dict_entry(self, mcs_label, mcs_meta=None):
        """Aggregate this organelle's patches for ``mcs_label`` into a record.

        ``n_contacts`` counts own patches (across partners), ``n_partners``
        counts partners. Areas are aggregated over patches; distance means and
        standard deviations are weighted by patch area. Per-partner distances
        stay aligned with the ``partners`` tuple.

        Args:
            mcs_label (str): label of the MCS run.
            mcs_meta (dict, optional): label parameters for the run
                (``min_distance``, ``max_distance``, ``ids_filter_1/2``). Falls
                back to the project's label entry or the stored patch metadata.
        """
        mcs_dict = {}
        volume = self.mesh_properties.volume
        surface = self.mesh_properties.area

        if (volume <= 0) or (surface <= 0):
            self.logger.debug(
                f"Missing volume/surface in Organelle {self}. "
                f"Volume: {volume}, Surface {surface}"
            )

        # Only partners with at least one usable patch contribute.
        partner_patches = []
        for partner, patches in self.mcs[mcs_label].items():
            usable = [
                patch
                for patch in patches
                if len(np.asarray(patch["distances"])) > 0
            ]
            if usable:
                partner_patches.append((partner, usable))

        if not partner_patches:
            self.logger.debug(
                "No distributions found for mcs %s in organelle %s", mcs_label, self
            )
            return

        partners = [partner for partner, _ in partner_patches]
        patches = [patch for _, own in partner_patches for patch in own]

        patch_areas = np.array(
            [float(patch["area"]) for patch in patches], dtype=np.float64
        )
        patch_means = np.array(
            [np.mean(patch["distances"]) for patch in patches], dtype=np.float64
        )
        patch_stds = np.array(
            [np.std(patch["distances"]) for patch in patches], dtype=np.float64
        )
        patch_min = np.array(
            [np.min(patch["distances"]) for patch in patches], dtype=np.float64
        )
        patch_max = np.array(
            [np.max(patch["distances"]) for patch in patches], dtype=np.float64
        )

        total_area = float(np.sum(patch_areas))
        mcs_dict["n_contacts"] = len(patches)
        mcs_dict["n_partners"] = len(partners)
        mcs_dict["partners"] = partners
        mcs_dict["total_area"] = total_area
        mcs_dict["mean_area"] = float(np.mean(patch_areas))
        mcs_dict["std_area"] = (
            float(np.std(patch_areas)) if len(patch_areas) > 1 else 0.0
        )
        mcs_dict["min_dist"] = float(np.min(patch_min))
        mcs_dict["max_dist"] = float(np.max(patch_max))

        # Area-weighted mean and variance across patches. This is equivalent
        # to weighting every distance sample by its patch's area, without
        # concatenating the per-patch distance arrays.
        total_weight = float(np.sum(patch_areas))
        if total_weight > 0:
            overall_mean = float(
                np.average(patch_means, weights=patch_areas)
            )
            overall_var = float(
                np.average(
                    patch_stds**2 + (patch_means - overall_mean) ** 2,
                    weights=patch_areas,
                )
            )
        else:
            overall_mean = float(np.mean(patch_means))
            overall_var = float(
                np.mean(patch_stds**2 + (patch_means - overall_mean) ** 2)
            )
        mcs_dict["mean_dist"] = overall_mean
        mcs_dict["std_dist"] = float(np.sqrt(max(overall_var, 0.0)))

        min_dist_per_org = []
        max_dist_per_org = []
        offset = 0
        for _, own in partner_patches:
            stop = offset + len(own)
            min_dist_per_org.append(float(np.min(patch_min[offset:stop])))
            max_dist_per_org.append(float(np.max(patch_max[offset:stop])))
            offset = stop
        mcs_dict["min_dist_per_org"] = min_dist_per_org
        mcs_dict["max_dist_per_org"] = max_dist_per_org

        if surface > 0:
            mcs_dict["n_contacts_per_area"] = mcs_dict["n_contacts"] / surface
            mcs_dict["area_per_area"] = total_area / surface
        else:
            mcs_dict["n_contacts_per_area"] = np.nan
            mcs_dict["area_per_area"] = np.nan

        if volume > 0:
            mcs_dict["area_per_volume"] = total_area / volume
            mcs_dict["n_contacts_per_volume"] = mcs_dict["n_contacts"] / volume
        else:
            mcs_dict["area_per_volume"] = np.nan
            mcs_dict["n_contacts_per_volume"] = np.nan

        label_meta = self._mcs_label_meta(mcs_label, mcs_meta)
        mcs_props = McsData(**numpy_to_python(mcs_dict))
        meta = McsMetadata(
            organelle_meta=self.metadata,
            mcs_label=mcs_label,
            min_dist=float(label_meta.get("min_distance", 0.0)),
            max_dist=float(label_meta.get("max_distance", 0.0)),
            filter_1=label_meta.get("ids_filter_1", "*"),
            filter_2=label_meta.get("ids_filter_2", "*"),
            schema_version=CURRENT_MCS_SCHEMA_VERSION,
        )
        stat = Record(mcs_props, meta, project=self.project)
        self.project.registry.add(stat)

    def _mcs_label_meta(self, mcs_label, mcs_meta=None) -> dict:
        """Resolve the parameters of an MCS run from the most precise source."""
        if mcs_meta is not None:
            return mcs_meta
        label_meta = self.project._mcs_labels.get(mcs_label)
        if label_meta is not None:
            return label_meta
        for patches in self.mcs[mcs_label].values():
            if patches:
                return patches[0].get("meta") or {}
        return {}

    @property
    def mcs(self) -> dict:
        return self._mcs

    @property
    def data(self) -> da.Array:
        """Get the raw data for this organelle
        by filtering the data of the source object"""
        source_ds = self.source.data[:]
        return da.where(source_ds == self.label, source_ds, 0)


class Mitochondrium(Organelle):
    _name = "mito"


class EndoplasmicReticulum(Organelle):
    _name = "er"


class AutoFillDict(dict):
    """Dictionary that populates itself"""

    def __missing__(self, key: str):
        new = type(key.capitalize(), (Organelle,), {"_name": key})
        self[key] = new
        return new


# The dictionary of registered organelle subclasses, mapping names
# to classes
organelle_registry: dict[str, "Organelle"] = AutoFillDict()
organelle_registry["mito"] = Mitochondrium
organelle_registry["er"] = EndoplasmicReticulum


@delayed
def get_mesh_properties_delayed(mesh: Trimesh) -> MeshData:
    """Extract mesh properties from a trimesh object and return as dict.

    This function is designed to be called on delayed mesh objects to
    compute mesh properties on the worker nodes rather than transferring
    the mesh to the main node.

    Args:
        mesh: A trimesh object (delayed)

    Returns:
        dict: Dictionary containing mesh properties that can be used to
            construct a mesh_properties dataclass
    """
    sph = None
    flat_ratio = None
    if mesh.vertices.shape[0] > 4:
        sph = ((36 * np.pi * mesh.volume**2) ** (1 / 3) / mesh.area).item()
        flat_ratio = (
            min(mesh.bounding_box_oriented.extents)
            / max(mesh.bounding_box_oriented.extents)
        ).item()
    return MeshData(
        volume=abs((mesh.volume).item()) if mesh.is_watertight else np.nan,
        area=(mesh.area).item(),
        centroid=tuple((mesh.centroid).tolist()),
        inertia=tuple((mesh.moment_inertia).tolist()),
        water_tight=mesh.is_watertight,
        sphericity=sph,
        flatness_ratio=flat_ratio,
    )
