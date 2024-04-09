from organelle_morphology.util import disk_cache

import numpy as np
import trimesh
from skimage import measure
import logging
import plotly.graph_objects as go
import skeletor as sk
import networkx

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
        self._skeleton_info = {}
        self.logger = self._source._project.logger

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

    def __repr__(self):
        return f"{self.__class__.__name__}({self._organelle_id})"

    def _generate_mesh(self, smooth=True):
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

        self.logger.debug("Generated mesh for %s", self.id)
        return mesh

    def _generate_skeleton(
        self,
        skeletonization_type: str = "wavefront",
        theta: float = 0.4,
        waves: int = 1,
        step_size: int = 2,
        epsilon: float = 0.1,
        sampling_dist: float = 0.1,
        path_sample_dist: float = 0.1,
    ):
        """
        Generates a skeleton for the organelle. The skeleton is generated and cleaned using the skeletor library.

        The last generated skeleton will be kept in memory and can be accessed using the skeleton or sampled_skeleton property.
        :param skeletonization_type: The type of skeletonization to use. Can be either "wavefront" or "vertex_clusters".

        :param waves: The number of waves to use for the wavefront skeletonization. The higher the number of waves, the more branches are removed.
        :param step_size: The step size for the wavefront skeletonization. The higher the step size, the less vertices are used for the skeleton.

        :param theta: The threshold for the clean_up function. The higher the threshold, the more branches are removed.
        :param epsilon: The epsilon value for the contract function. The higher the epsilon, the more the mesh is contracted.
        :param sampling_dist: The sampling distance for the skeletonize function. The higher the sampling distance, the less vertices are used for the skeleton.
        :param path_sample_dist: The distance between the sample points on the skeleton arms. The higher the distance, the less sample points are used.
        """

        try:
            fixed_mesh = sk.pre.fix_mesh(self.mesh)
        except IndexError as e:
            self.logger.debug(
                "Could not fix mesh in skeleton generation for %s with error %s"
                % (self.id, e)
            )

            fixed_mesh = self.mesh

        if skeletonization_type not in ["wavefront", "vertex_clusters"]:
            raise ValueError(
                "Skeletonization type must be either 'wavefront' or 'vertex_clusters'."
            )
        try:
            if skeletonization_type == "wavefront":
                skel = sk.skeletonize.by_wavefront(
                    fixed_mesh, waves=waves, progress=False, step_size=step_size
                )
            elif skeletonization_type == "vertex_clusters":
                try:
                    cont = sk.pre.contract(fixed_mesh, epsilon=epsilon, progress=False)
                except IndexError:
                    self.logger.debug(
                        "couldnt contract mesh using normal mesh for %s" % self.id
                    )
                    cont = fixed_mesh
                skel = sk.skeletonize.by_vertex_clusters(
                    cont, sampling_dist=sampling_dist, progress=False
                )
                skel.mesh = fixed_mesh

            sk.post.clean_up(skel, inplace=True, theta=theta)
            sk.post.radii(skel, method="knn")

            # if no skeleton can be created just skip this
            if len(skel.vertices) <= 1:
                return None

            self._skeleton = skel

            self.sampled_skeleton = self._sample_skeleton(
                path_sample_dist=path_sample_dist
            )

            # reset skeleton info for new calculation
            self._skeleton_info = self._get_skeleton_info()
            self.logger.debug("Generated skeleton for %s", self.id)

        except Exception as e:
            self.logger.debug(
                "Could not generate skeleton for %s with error %s" % (self.id, e),
                exc_info=True,
            )
            self._skeleton = None

    def _sample_skeleton(self, path_sample_dist: float = 0.1):
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

            if edge_len > path_sample_dist:
                p1 = np.array(self.skeleton.vertices[edge[0]])
                p2 = np.array(self.skeleton.vertices[edge[1]])

                # find number of points to add bewteen the two vertices
                n_points = np.ceil(edge_len / path_sample_dist).astype(int)
                factors = np.linspace(0, 1, n_points)

                # Compute the interpolated points
                interpolated_points = (1 - factors[:, np.newaxis]) * p1 + factors[
                    :, np.newaxis
                ] * p2
                sampled_path.extend(interpolated_points)
                reference_point.append(p1)

        if len(sampled_path) == 0:
            sampled_path.append(np.array(self.skeleton.vertices[0]))
            reference_point.append(np.array(self.skeleton.vertices[0]))

        sampled_path = np.asarray(sampled_path)
        reference_point = np.asarray(reference_point)
        self._sampled_skeleton = sampled_path, reference_point
        return self._sampled_skeleton

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

    def _get_skeleton_info(self):
        if self._skeleton is None:
            return None

        skeleton_info = {}
        graph = self.skeleton.get_graph()
        if len(graph.nodes) == 0:
            return 0

        skeleton_info["num_nodes"] = len(graph.nodes)
        skeleton_info["num_branch_points"] = len(
            [node for node, degree in graph.degree() if degree > 2]
        )
        skeleton_info["end_points"] = len(
            [node for node, degree in graph.degree() if degree == 1]
        )
        skeleton_info["total_length"] = sum(
            d["weight"] for u, v, d in graph.edges(data=True)
        )
        skeleton_info["longest_path"] = networkx.dag_longest_path_length(graph)

        lengths = [d["weight"] for u, v, d in graph.edges(data=True)]
        skeleton_info["mean_length"] = np.mean(lengths)
        skeleton_info["std_length"] = np.std(lengths)

        skeleton_info["mean_radius"] = np.mean(self.skeleton.radius[0])
        skeleton_info["std_radius"] = np.std(self.skeleton.radius[0])
        return skeleton_info

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
        else:
            self._skeleton_info = self._get_skeleton_info()

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
        with disk_cache(self._source._project, f"mesh_{self._organelle_id}") as cache:
            if self._source._project.compression_level not in cache:
                cache[self._source._project.compression_level] = self._generate_mesh()

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
