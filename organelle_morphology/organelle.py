from dask.delayed import Delayed
import numpy as np
import plotly.graph_objects as go
import dask.array as da
from collections import defaultdict

from trimesh import Trimesh

import organelle_morphology
from organelle_morphology.util import bounding_box_delayed


def organelle_types() -> list[str]:
    """The list of organelles currently implemented.

    The strings used here to encode the organelles are expected in
    various APIs when referring to a specific organelle.
    """
    return list(organelle_registry.keys())


class Organelle:
    _name = "organelle_name"

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
        self.label = label
        self._organelle_id = f"{self._name}_{str(label).zfill(4)}"
        self._mesh_properties = {}
        self._skeleton = None
        self._sampled_skeleton = None
        self._skeleton_info = {}
        self._mcs = defaultdict(dict)
        self._mcs_dict = defaultdict(dict)

        self.logger = self.source.project.logger

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

    def plotly_skeleton(self):
        if self._skeleton is None:
            return None

        nodes = self.skeleton.vertices
        edges = self.skeleton.edges

        line_width = 10

        # Create a 3D line plot for the edges
        x_values = []
        y_values = []
        z_values = []

        for edge in edges:
            x_values.extend(
                [nodes[edge[0]][0], nodes[edge[1]][0], None]
            )  # add None to separate lines
            y_values.extend(
                [nodes[edge[0]][1], nodes[edge[1]][1], None]
            )  # add None to separate lines
            z_values.extend(
                [nodes[edge[0]][2], nodes[edge[1]][2], None]
            )  # add None to separate lines

        skeleton_trace = go.Scatter3d(
            x=x_values,
            y=y_values,
            z=z_values,
            mode="lines",
            line=dict(width=line_width),  # Set line width
            name=f"Skeleton_{self.id}",  # Set label
        )
        return skeleton_trace

    def plotly_mesh(
        self,
        show_curvature: bool = False,
        show_skeleton: bool = False,
        mcs_label=False,
        mcs_filter_ids=None,
    ):
        # prepare the plotly mesh object for visualization

        verts = self.mesh.compute().vertices
        faces = self.mesh.compute().faces

        # prepare data for plotly
        vertsT = np.transpose(verts)
        facesT = np.transpose(faces)

        # initialize basic drawing settings
        intensity = None
        colorscale = None
        opacity = 1

        # override settings if special visualization is requested
        if show_curvature:
            curvature_vertices = self.curvature_map
            intensity = curvature_vertices
            colorscale = "Viridis"
            opacity = 1

        if show_skeleton:
            opacity = 0.7

        # add coloration for the close regions
        if mcs_label:
            intensity = np.zeros(len(verts))  # Default intensity is 0.5

            for mcs_key, mcs in self.mcs.get(mcs_label, {}).items():
                if mcs_filter_ids is not None:
                    if mcs_key not in mcs_filter_ids:
                        continue
                t_close_vertices = np.transpose(mcs["vertices_index"])
                intensity[t_close_vertices] = 1  # Close vertices have intensity 1
            colorscale = [
                [0, "rgb(110,150,220)"],
                [1, "rgb(255,0,0)"],
            ]  # Map intensity to color

        go_mesh = go.Mesh3d(
            x=vertsT[0],
            y=vertsT[1],
            z=vertsT[2],
            i=facesT[0],
            j=facesT[1],
            k=facesT[2],
            name=self.id,
            opacity=opacity,
            intensity=intensity,
            colorscale=colorscale,
            showscale=False,
        )
        return go_mesh

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
    def skeleton_info(self, value):
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
    def mesh(self):
        """Get the mesh for this organelle"""
        return self.source.meshes[self.label]

    @property
    def id(self):
        """Get the organelle ID of this organelle"""

        return self._organelle_id

    @property
    def bounding_box(self) -> Delayed:
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
    def mesh_properties(self):
        """Get the mesh data for this organelle"""
        comp_level = self.source.project.compression_level

        if comp_level not in self._mesh_properties:
            mesh = self.mesh.compute()
            self._mesh_properties[comp_level] = {}

            self._mesh_properties[comp_level]["mesh_volume"] = mesh.volume
            self._mesh_properties[comp_level]["mesh_area"] = mesh.area
            self._mesh_properties[comp_level]["mesh_centroid"] = mesh.centroid
            self._mesh_properties[comp_level]["mesh_inertia"] = mesh.moment_inertia

            self._mesh_properties[comp_level]["water_tight"] = mesh.is_watertight
            self._mesh_properties[comp_level]["sphericity"] = (
                36 * np.pi * mesh.volume**2
            ) ** (1 / 3) / mesh.area
            dimensions = mesh.bounding_box_oriented.extents
            self._mesh_properties[comp_level]["flatness_ratio"] = min(dimensions) / max(
                dimensions
            )

        return self._mesh_properties[comp_level]

    @property
    def curvature_map(self) -> np.ndarray:
        """Get the mesh data for this organelle"""
        return self.source.get_curvature(self.label, color=False)[0]

    @property
    def curvature_mesh(self) -> Trimesh:
        return self.source.get_curvature(self.label, color=True)[1][0]

    def add_mcs(self, mcs_dict):
        mcs_target = mcs_dict["partner_id"]
        mcs_label = mcs_dict["mcs_label"]

        mcs_entry = {
            "vertices_index": mcs_dict["vertices_index"],
            "distances": mcs_dict["distances"],
            "area": mcs_dict["area"],
        }

        self._mcs[mcs_label][mcs_target] = mcs_entry

    def get_mcs_dict_entry(self, mcs_label):
        """
        Calculate the properties of the mcs partners for the given mcs label

        """

        _mcs_dict = self.mcs_dict

        len_dist_list = []
        mean_dist_list = []
        std_dist_list = []
        area_list = []

        for entries in self.mcs[mcs_label].values():
            mean_dist_list.append(np.mean(entries["distances"]))
            std_dist_list.append(np.std(entries["distances"]))
            len_dist_list.append(len(entries["distances"]))

            area_list.append(entries["area"])

        mean_dist_list = np.array(mean_dist_list)
        std_dist_list = np.array(std_dist_list)
        len_dist_list = np.array(len_dist_list)
        if len(len_dist_list) == 0 or 0 in len_dist_list:
            self.logger.debug(
                "No distributions found for mcs %s in organelle %s", mcs_label, self
            )
            return

        _mcs_dict[(mcs_label)]["n_contacts"] = len(len_dist_list)

        _mcs_dict[(mcs_label)]["total_area"] = np.sum(entries["area"])
        _mcs_dict[(mcs_label)]["mean_area"] = np.mean(area_list)

        if len(area_list) == 1:
            _mcs_dict[(mcs_label)]["std_area"] = 0
        else:
            _mcs_dict[(mcs_label)]["std_area"] = np.std(area_list)

        # calculate the mean and std from the sub_mean and std values for each mcs partner
        try:
            overall_mean = np.average(mean_dist_list, weights=mean_dist_list)
        except ZeroDivisionError:
            overall_mean = 0

        try:
            overall_var = np.average(
                (std_dist_list**2 + (mean_dist_list - overall_mean) ** 2),
                weights=len_dist_list,
            )
        except ZeroDivisionError:
            overall_var = 0
        overall_std = np.sqrt(overall_var)

        _mcs_dict[(mcs_label)]["mean_dist"] = overall_mean
        _mcs_dict[(mcs_label)]["std_dist"] = overall_std

        self._mcs_dict = _mcs_dict

    @property
    def mcs(self):
        return self._mcs

    @property
    def mcs_dict(self):
        return self._mcs_dict

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
