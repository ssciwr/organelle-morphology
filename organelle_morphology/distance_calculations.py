from scipy.spatial import KDTree
import trimesh

import numpy as np
import pandas as pd

from tqdm import tqdm

import organelle_morphology
from dask.base import compute
import multiprocessing

from organelle_morphology.util import bounding_box_delayed, boxes_overlap


def get_min_dist(args):
    id_1, id_2, mesh_1, mesh_2 = args
    mcs_calculator = MembraneContactSiteCalculator()
    mcs_calculator.search_mcs(id_1, id_2, mesh_1, mesh_2)
    return (id_1, id_2), mcs_calculator.min_distance


def _check_overlap(args):
    box, bounding_boxes = args
    is_in = np.empty((len(bounding_boxes),), dtype=bool)
    for i, org_bb in enumerate(bounding_boxes):
        is_in[i] = boxes_overlap(box, org_bb)

    return is_in


class MembraneContactSiteCalculator:
    def search_mcs(self, id_1, id_2, mesh_1, mesh_2):
        """
        first search the entire membrane surface that is facing the other organelle
        we do this by first performing a kd-tree search to find the nearest vertices.
        Then we filter the vertices based on the dot product of the normal vectors
        of the two meshes.
        This will give us the vertices that are facing the other organelle.


        Args:
            id_1 (int): The ID of the first organelle.
            id_2 (int): The ID of the second organelle.
            mesh_1 (Mesh, optional): The mesh of the first organelle.
            mesh_2 (Mesh, optional): The mesh of the second organelle.
            If not provided, it will be retrieved from the project.

        """
        self._repair_meshes(mesh_1, mesh_2)

        watertight = True
        if len(mesh_1.vertices) < len(mesh_2.vertices):
            if mesh_2.is_watertight:
                ordering = 1
            elif mesh_1.is_watertight:
                ordering = 0
            else:
                # both not watertight -> smaller as target
                ordering = 1
                watertight = False
        elif mesh_1.is_watertight:
            ordering = 0
        elif mesh_2.is_watertight:
            ordering = 1
        else:
            # both not watertight -> smaller as target
            ordering = 0
            watertight = False

        if ordering:
            mesh_source = mesh_2
            mesh_target = mesh_1
            id_source = id_2
            id_target = id_1
        else:
            mesh_source = mesh_1
            mesh_target = mesh_2
            id_source = id_1
            id_target = id_2

        if watertight:
            distance, index_source = mesh_source.nearest.vertex(mesh_target.vertices)
        else:
            source_tree = KDTree(mesh_source.vertices)
            distance, index_source = source_tree.query(mesh_target.vertices, k=1)

        self.distances = distance
        self.index_source: np.ndarray = index_source
        self.index_target: np.ndarray = np.asarray((range(len(mesh_target.vertices))))

        self.normals_source = mesh_source.vertex_normals[index_source]
        self.normals_target = mesh_target.vertex_normals

        self.dot_products = np.einsum(
            "ij,ij->i", self.normals_source, self.normals_target
        )

        # some of the meshes have inverted normal vectors,
        # i think this can happen when the meshes are not watertight at the cell borders
        # this means that sometimes we have to search for a negative dot product
        # and other times for a positive.
        # To determine which one to search for, we calculate the mean distance for both cases
        # and choose the one with the smaller mean distance.

        _normal_dists = [
            self.distances[i]
            for i in range(len(self.distances))
            if self.dot_products[i] < 0
        ]
        mean_distance_normal = np.nan
        if _normal_dists:
            mean_distance_normal = np.mean(_normal_dists)

        _inverse_dists = [
            self.distances[i]
            for i in range(len(self.distances))
            if self.dot_products[i] > 0
        ]
        mean_distance_inverse = np.nan
        if _inverse_dists:
            mean_distance_inverse = np.mean(_inverse_dists)

        if mean_distance_normal < mean_distance_inverse:
            self._filter_distances(lambda i: self.dot_products[i] < 0)
        else:
            self._filter_distances(lambda i: self.dot_products[i] > 0)

        self.id_source = id_source
        self.id_target = id_target
        self.mesh_source = mesh_source
        self.mesh_target = mesh_target

    def analyze_mcs(self, max_distance, min_distance=0.0):
        """After searching the entire surface we now filter only the area we are interested in

        :param max_distance:  Maximum distance to the other organelle
        :type max_distance: float
        :param min_distance: minimum distance, defaults to 0
        :type min_distance: int, optional
        :raises ValueError: search_mcs must be performed first.
        """
        if self.distances is None:
            raise ValueError("No MCS found. Please run search_mcs first.")

        self.mcs_label = f"{min_distance}-{max_distance}"

        filtered_indices = [
            i
            for i in range(len(self.distances))
            if min_distance <= self.distances[i] <= max_distance
        ]

        self._vertices_index_source = self.index_source[filtered_indices]
        self._vertices_index_target = self.index_target[filtered_indices]

        self._vertices_source = self.mesh_source.vertices[self._vertices_index_source]
        self._vertices_target = self.mesh_target.vertices[self._vertices_index_target]

        faces_source = np.nonzero(
            np.any(np.isin(self.mesh_source.faces, self._vertices_index_source), axis=1)
        )
        faces_target = np.nonzero(
            np.any(np.isin(self.mesh_target.faces, self._vertices_index_target), axis=1)
        )

        area_source = self.mesh_source.area_faces[faces_source].sum()
        area_target = self.mesh_target.area_faces[faces_target].sum()

        self._distances = self.distances[filtered_indices]
        self._area_source = area_source
        self._area_target = area_target

    def _filter_distances(self, filter_func):
        # depending on the normal orientation, we need to filter the distances differently
        filtered_indices = [i for i in range(len(self.distances)) if filter_func(i)]
        self.distances = self.distances[filtered_indices]
        self.index_source = self.index_source[filtered_indices]
        self.index_target = self.index_target[filtered_indices]
        self.normals_source = self.normals_source[filtered_indices]
        self.normals_target = self.normals_target[filtered_indices]
        self.dot_products = self.dot_products[filtered_indices]

    @property
    def min_distance(self):
        if len(self.distances) == 0:
            # this can sometimes happen, not sure why
            return np.nan

        return np.min(self.distances)

    @property
    def mcs_target(self):
        if self.mcs_label is None:
            raise ValueError(
                "No MCS found. Please run 'search_mcs' and 'analyze_mcs' first."
            )

        return {
            "self_id": self.id_target,
            "partner_id": self.id_source,
            "mcs_label": self.mcs_label,
            # "vertices": self._vertices_target, # TODO: Remove unnecessary this copy
            "vertices_index": self._vertices_index_target,
            "distances": self._distances,
            "area": self._area_target,
        }

    @property
    def mcs_source(self):
        if self.mcs_label is None:
            raise ValueError(
                "No MCS found. Please run 'search_mcs' and 'analyze_mcs' first."
            )

        return {
            "self_id": self.id_source,
            "partner_id": self.id_target,
            "mcs_label": self.mcs_label,
            # "vertices": self._vertices_source, # TODO: Remove unnecessary this copy
            "vertices_index": self._vertices_index_source,
            "distances": self._distances,
            "area": self._area_source,
        }

    def _repair_meshes(self, mesh_source, mesh_target):
        for mesh in [mesh_source, mesh_target]:
            trimesh.repair.fix_inversion(mesh)
            trimesh.repair.fix_winding(mesh)
            trimesh.repair.fix_normals(mesh)


def generate_distance_matrix(
    project: "organelle_morphology.Project",
    domain_decomposition=True,
) -> pd.DataFrame:
    cache = project.cache
    max_dist = project.max_distance
    max_cached = 0
    if sources := list(project.sources.values()):
        source = sources[0]
    else:
        raise ValueError("No sources loaded! Can't calculate distances.")
    if max_dist == 0:
        max_dist = source.resolution[0] * 10
        project.logger.warning(
            f"No max distance set, calculating distance matrix up to {max_dist}"
        )

    if "max_distance_computed" in cache:
        max_cached = cache["max_distance_computed"]

    if (
        "distance_matrix" not in cache or not project.use_cache
    ) or max_dist > max_cached:
        project.logger.info("Initilizing distance matrix")

        project.logger.info("Loading meshes")
        project.calculate_meshes()

        organelles = project.organelles
        organelles_ids = project.organelle_ids
        meshes = []
        bounding_boxes = []
        for organelle in organelles:
            meshes.append(organelle.mesh)
            bounding_boxes.append(bounding_box_delayed(organelle.mesh))
        meshes, bounding_boxes = compute(meshes, bounding_boxes)

        project.logger.info("Calculating distance matrix")
        num_rows = len(meshes)

        distance_matrix = np.ones((num_rows, num_rows)) * -1
        distance_df = pd.DataFrame(
            distance_matrix,
            index=organelles_ids,
            columns=organelles_ids,
        )

        if domain_decomposition:
            # domain decomposition into chunks of
            # size = size of clipped data in units of the resolution
            size = np.array(source.resolution) * source.data.shape
            cube_size = max_dist * 10
            stride = max_dist * 9
            clp_of = source.clipping_corners[0]
            n_x_cubes = min(max(int(size[0] // cube_size), 1), 50)
            n_y_cubes = min(max(int(size[1] // cube_size), 1), 50)
            n_z_cubes = min(max(int(size[2] // cube_size), 1), 50)

            xs = np.linspace(
                clp_of[0],
                size[0] + clp_of[0],
                n_x_cubes,
                endpoint=False,
            )
            ys = np.linspace(
                clp_of[1],
                size[1] + clp_of[1],
                n_y_cubes,
                endpoint=False,
            )
            zs = np.linspace(
                clp_of[2],
                size[2] + clp_of[2],
                n_z_cubes,
                endpoint=False,
            )

            xg, yg, zg = np.meshgrid(xs, ys, zs)
            xg, yg, zg = list(map(lambda a: a.flatten(), (xg, yg, zg)))

            project.logger.debug(f"Cube size: {cube_size}")
            project.logger.debug(
                f"n cubes: {n_x_cubes}, {n_y_cubes}, {n_z_cubes} => {len(xg)}"
            )
            project.logger.debug(f"n organelles: {len(organelles)}")

            tasks = []
            for x, y, z in zip(xg, yg, zg):
                start = np.array((x, y, z))
                end = start + stride
                box = (start, end)
                tasks.append((box, bounding_boxes))
            with multiprocessing.Pool(project.n_workers) as pool:
                masks = pool.imap_unordered(_check_overlap, tasks, chunksize=100)
                pool.close()
                pool.join()
        else:
            # no domain decomposition -> all to all
            masks = [np.ones((num_rows,))]

        tasks = []
        empty_cubes = 0
        for mask in masks:
            if not np.any(mask):
                empty_cubes += 1
                continue

            indices = np.nonzero(mask)[0]
            for i, idx1 in enumerate(indices):
                mesh_1 = meshes[idx1]
                id_1 = organelles_ids[idx1]
                for idx2 in indices[i + 1 :]:
                    mesh_2 = meshes[idx2]
                    id_2 = organelles_ids[idx2]
                    tasks.append((id_1, id_2, mesh_1, mesh_2))
        project.logger.debug(f"n empty cube: {empty_cubes}")

        with multiprocessing.Pool(project.n_workers) as pool:
            results = pool.imap_unordered(get_min_dist, tasks, chunksize=500)
            pool.close()
            pool.join()
        # results = map(get_min_dist, tasks)

        for res in tqdm(results, "gathering distances", total=len(tasks)):
            distance_df.loc[res[0]] = res[1]
            distance_df.loc[res[0][::-1]] = res[1]

        cache["distance_matrix"] = distance_df
        cache["max_distance_computed"] = max_dist

    else:
        project.logger.info("Retrieving distance matrix from cache")

    return cache["distance_matrix"]


def generate_mcs(
    project, max_distance: float, min_distance: float = 0, overwrite=False
) -> str:
    """
    Generates the MCS (Membrane Contact Site) pairs for a given project.


    Args:
        project (Project): The project object containing the distance matrix and organelles.
        max_distance (float): The maximum distance for the MCS pairs.
        min_distance (float, optional): The minimum distance for the MCS pairs. Defaults to 0.

    Returns:
        str: the label of this mcs search
    """
    mcs_label = f"{min_distance}-{max_distance}"

    if project._mcs_labels.get(mcs_label) and not overwrite:
        return mcs_label

    distance_matrix = project.distance_matrix

    # filter dataframe and get row, column where the distance is between min_distance and max_distance

    rows, columns = np.where(
        (distance_matrix >= min_distance) & (distance_matrix <= max_distance)
    )

    # generate the pairs of organelles
    pairs = set()
    for i, j in zip(rows, columns):
        org_1_label = distance_matrix.index[i]
        org_2_label = distance_matrix.columns[j]

        if org_1_label == org_2_label:
            continue

        pairs.add(tuple(sorted([org_1_label, org_2_label])))

    project.logger.info(f"Found {len(pairs)} pairs of organelles to calculate MCS")

    meshes = {}
    for labels in pairs:
        for label in labels:
            if label not in meshes:
                meshes[label] = project.get_organelles(label)[0].mesh
    meshes = compute(meshes)[0]

    for org_1_label, org_2_label in tqdm(pairs, "mcs calculation"):
        org1 = project.get_organelles(org_1_label)[0]
        org2 = project.get_organelles(org_2_label)[0]
        mcs_calculator = MembraneContactSiteCalculator()
        mcs_calculator.search_mcs(
            org_1_label, org_2_label, meshes[org_1_label], meshes[org_2_label]
        )
        mcs_calculator.analyze_mcs(max_distance, min_distance)

        mcs_source = mcs_calculator.mcs_source
        mcs_target = mcs_calculator.mcs_target

        # add mcs to organelle
        if org1.id == mcs_source["self_id"]:
            org1.add_mcs(mcs_source)
            org2.add_mcs(mcs_target)
        else:
            org1.add_mcs(mcs_target)
            org2.add_mcs(mcs_source)

    org_ids = {distance_matrix.index[i] for i in rows}
    for org_id in org_ids:
        org = project.get_organelles(org_id)[0]
        org.calc_mcs_dict_entry(mcs_label)

    project._mcs_labels[mcs_label] = {
        "max_distance": max_distance,
        "min_distance": min_distance,
        "organelles": org_ids,
    }
    return mcs_label
