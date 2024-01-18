from organelle_morphology.util import disk_cache

import numpy as np
import trimesh
from skimage import measure
import logging
import plotly.graph_objects as go
import skeletor as sk

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
        self._mesh = {}
        self._morphology_map = {}
        self._skeleton = None
        self._sampled_skeleton = None

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
        print("mesh for", self.id)
        try:
            verts, faces, _, _ = measure.marching_cubes(
                self.data, spacing=self._source.resolution
            )
        except RuntimeError:
            logging.warning("Could not generate mesh for label %s", self.id)
            self._mesh[self._source._project.compression_level] = None
            return None
        mesh = trimesh.Trimesh(verts, faces, process=False)
        mesh.fix_normals()
        if smooth:
            trimesh.smoothing.filter_humphrey(mesh)
        return mesh

    def generate_skeleton(
        self, theta: float = 0.4, epsilon: float = 0.05, sampling_dist: float = 2.0
    ):
        """
        Generates a skeleton for the organelle. The skeleton is generated and cleaned using the skeletor library.

        The last generated skeleton will be kept in memory and can be accessed using the skeleton or sampled_skeleton property.

        :param theta: The threshold for the clean_up function. The higher the threshold, the more branches are removed.
        :param epsilon: The epsilon value for the contract function. The higher the epsilon, the more the mesh is contracted.
        :param sampling_dist: The sampling distance for the skeletonize function. The higher the sampling distance, the less vertices are used for the skeleton.

        """

        fixed_mesh = sk.pre.fix_mesh(self.mesh)
        # cont = sk.pre.contract(fixed_mesh, epsilon=epsilon, progress=False)
        # skel = sk.skeletonize.by_vertex_clusters(
        #     cont, sampling_dist=sampling_dist, progress=False
        # )
        print(self.id)

        try:
            skel = sk.skeletonize.by_wavefront(
                fixed_mesh, waves=1, progress=False, step_size=2
            )

            skel.mesh = fixed_mesh
            sk.post.clean_up(skel, inplace=True, theta=theta)

            self._skeleton = skel
        except IndexError:
            print(f"Can't generate skeleton for {self.id}")
            self._skeleton = None
        # sample the skeleton

    def sample_skeleton(self):
        # the sample points are points along the skeleton arms
        # and the reference points are the vertices of the skeleton from which these samples have been generated.
        # we need these to later calculate the normal vector for the plane which will intersect our mesh
        sampled_path = []
        reference_point = []

        for edge in self.skeleton.edges:
            edge_len = np.linalg.norm(
                np.array(self.skeleton.vertices[edge[0]])
                - np.array(self.skeleton.vertices[edge[1]])
            )
            # set the distance between points for the path sampling
            distance_between_points = 0.1

            if edge_len > distance_between_points:
                p1 = np.array(self.skeleton.vertices[edge[0]])
                p2 = np.array(self.skeleton.vertices[edge[1]])

                # find number of points to add bewteen the two vertices
                n_points = np.ceil(edge_len / distance_between_points).astype(int)
                factors = np.linspace(0, 1, n_points)

                # Compute the interpolated points
                interpolated_points = (1 - factors[:, np.newaxis]) * p1 + factors[
                    :, np.newaxis
                ] * p2
                sampled_path.extend(interpolated_points)
                reference_point.append(p1)

        sampled_path = np.asarray(sampled_path)
        reference_point = np.asarray(reference_point)
        self._sampled_skeleton = sampled_path, reference_point

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

    def plotly_mesh(self, show_morphology: bool = False, show_skeleton: bool = False):
        # prepare the plotly mesh object for visualization

        verts = self.mesh.vertices
        faces = self.mesh.faces

        # prepare data for plotly
        vertsT = np.transpose(verts)
        facesT = np.transpose(faces)

        # initilize basic drawing settings
        intensity = None
        colorscale = None
        opacity = 1

        # override settings if special visualization is requested
        if show_morphology:
            curvature_vertices = self.morphology_map
            intensity = curvature_vertices
            colorscale = "Viridis"
            opacity = 1

        if show_skeleton:
            opacity = 0.7

        go_mesh = go.Mesh3d(
            x=vertsT[0],
            y=vertsT[1],
            z=vertsT[2],
            i=facesT[0],
            j=facesT[1],
            k=facesT[2],
            name=self.id,
            opacity=opacity,
            # note: opacity below 1 seems to be an ongoing issue with plotly in 3d.
            # shapes might not be drawn in the correct order and overlap wierdly when moving the camera,
            intensity=intensity,
            colorscale=colorscale,
            showscale=False,
        )
        return go_mesh

    @property
    def skeleton(self):
        """Get the skeleton for this organelle"""
        # if self._skeleton is None:
        #     raise ValueError(
        #         f"Skeleton has not been generated for {self.id} yet. Please run project.generate_skeletons() first."
        #     )

        return self._skeleton

    @property
    def sampled_skeleton(self):
        """Get the sampled skeleton for this organelle.
        This included the sampled points,
        as well as the corresponding reference point to later get the correct plane normal vector
        """
        if self._skeleton is None:
            raise ValueError(
                f"Skeleton has not been generated for {self.id} yet. Please run project.generate_skeletons() first."
            )
        self.sample_skeleton()
        return self._sampled_skeleton

    @property
    def mesh(self):
        """Get the mesh for this organelle"""
        with disk_cache(self._source._project, f"mesh_{self._organelle_id}") as cache:
            if self._source._project.compression_level not in cache:
                cache[self._source._project.compression_level] = self._generate_mesh()
            print("found in cache", self.id)

            return cache[self._source._project.compression_level]

    @property
    def id(self):
        """Get the organelle ID of this organelle"""

        return self._organelle_id

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
        return self._source.basic_geometric_properties[self.id]

    @property
    def mesh_properties(self):
        """Get the mesh data for this organelle"""
        comp_level = self._source._project.compression_level

        if comp_level not in self._mesh_properties:
            self._mesh_properties[comp_level] = {}

            self._mesh_properties[comp_level]["mesh_volume"] = self.mesh.volume
            self._mesh_properties[comp_level]["mesh_area"] = self.mesh.area
            self._mesh_properties[comp_level]["mesh_centroid"] = self.mesh.centroid
            self._mesh_properties[comp_level]["mesh_inertia"] = self.mesh.moment_inertia

            self._mesh_properties[comp_level]["water_tight"] = self.mesh.is_watertight
            self._mesh_properties[comp_level]["sphericity"] = (
                36 * np.pi * self.mesh.volume**2
            ) ** (1 / 3) / self.mesh.area
            dimensions = self.mesh.bounding_box_oriented.extents
            self._mesh_properties[comp_level]["flatness_ratio"] = min(dimensions) / max(
                dimensions
            )

        return self._mesh_properties[comp_level]

    @property
    def morphology_map(self):
        """Get the mesh data for this organelle"""
        comp_level = self._source._project.compression_level

        # morph radius can be 0 if vertices are used as sample points.
        morph_radius = 0.0

        if comp_level not in self._morphology_map:
            mesh = self.mesh
            if mesh is None:
                self._morphology_map[comp_level] = None
                return self._morphology_map[comp_level]

            sample_points = mesh.vertices
            curvature_vertices = trimesh.curvature.discrete_gaussian_curvature_measure(
                mesh, sample_points, radius=morph_radius
            )

            self._morphology_map[comp_level] = curvature_vertices
        return self._morphology_map[comp_level]

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
