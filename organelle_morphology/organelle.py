import numpy as np
import trimesh
from skimage import measure
import logging

# The dictionary of registered organelle subclasses, mapping names
# to classes
organelle_registry = {}


def organelle_types() -> list[str]:
    """The list of organelles currently implemented.

    The strings used here to encode the organelles are expected in
    various APIs when refering to a specific organelle.
    """
    return organelle_registry.keys()


class Organelle:
    def __init__(self, source, source_label: int, organelle_id: str):
        """The organelle base class implementing generic geometric properties.

        Note that instances of Organelle typically are not instantiated directly,
        but through the corresponding subclass of OrganelleFactory.

        :param source:
            The data source instance holding the data for this organelle.
        :type source: organelle_morphology.source.DataSource

        :param source_label:
            The label used in the original data to identify this organelle.

        :param organelle_id:
            The string ID that is used to refer to this organelle.
        """
        self._source = source
        self._source_label = source_label
        self._organelle_id = organelle_id
        self._mesh_properties = {}
        self._mesh = None

    def __init_subclass__(cls, name=None):
        """Register a given subclass in the global dictionary 'organelles'"""
        if name is not None:
            organelle_registry[name] = cls
            cls._name = name

    @classmethod
    def construct(cls, source: str, labels: tuple[int] = ()):
        """A trivial factory method for organelle instances.

        It constructs an instance per label. The construction process for each
        organelle is independent of all others. Other organelles can subclass
        this to implement a construction process that e.g. takes into account
        all organelle instances.
        """
        for label in labels:
            yield organelle_registry[cls._name](
                source=source,
                source_label=label,
                organelle_id=f"{cls._name}_{str(label).zfill(4)}",
            )

    def _generate_mesh(self, smooth=True):
        try:
            verts, faces, _, _ = measure.marching_cubes(
                self.data, spacing=self._source.resolution
            )
        except RuntimeError:
            logging.warning("Could not generate mesh for label %s", self.id)
            return None
        mesh = trimesh.Trimesh(verts, faces, process=False)
        mesh.fix_normals()
        if smooth:
            trimesh.smoothing.filter_humphrey(mesh)
        self._mesh = mesh

    @property
    def mesh(self):
        """Get the mesh for this organelle"""
        if self._mesh is None:
            self._generate_mesh()
        return self._mesh

    @property
    def id(self):
        """Get the organelle ID of this organelle"""

        return self._organelle_id

    @property
    def geometric_data(self):
        """Get the geometric data for this organelle
        Possible keywords are:
        "area",  # for 3d this is the volume
        "bbox",
        "slice",  # the slice of the bounding box
        "centroid",
        "moments",
        "extent",  # how much volume of the bounding box is occupied by the object
        "solidity",  # ratio of pixels in the convex hull to those in the region

        """
        return self._source.basic_geometric_properties[self.id]

    @property
    def mesh_properties(self):
        """Get the mesh data for this organelle"""

        if self._mesh is None:
            self._generate_mesh()

        if self._mesh_properties == {}:
            self._mesh_properties["mesh_volume"] = self.mesh.volume
            self._mesh_properties["mesh_area"] = self.mesh.area
            self._mesh_properties["mesh_centroid"] = self.mesh.centroid
            self._mesh_properties["mesh_inertia"] = self.mesh.moment_inertia

            self._mesh_properties["water_tight"] = self.mesh.is_watertight
            self._mesh_properties["sphericity"] = (
                36 * np.pi * self.mesh.volume**2
            ) ** (1 / 3) / self.mesh.area
            dimensions = self.mesh.bounding_box_oriented.extents
            self._mesh_properties["flatness_ratio"] = min(dimensions) / max(dimensions)

        return self._mesh_properties

    @property
    def data(self):
        """Get the raw data for this organelle
        by filtering the data of the source object"""
        source_ds = self._source.data[:]
        org_ds = np.where(source_ds == self._source_label, source_ds, 0)

        return org_ds


class Mitochondrium(Organelle, name="mito"):
    pass


class EndoplasmicReticulum(Organelle, name="er"):
    pass
