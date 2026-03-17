import logging
from dask.distributed import span, delayed
from scipy.spatial import KDTree
import trimesh

import numpy as np
from multiprocessing import Pool
import pandas as pd

from tqdm import tqdm

import organelle_morphology
from dask.base import compute, persist

from organelle_morphology.util import bounding_box_delayed, boxes_overlap

logger = logging.getLogger(__name__)


def get_min_dist(args):
    id_1, id_2, mesh_1, mesh_2 = args
    mcs_calculator = MembraneContactSiteCalculator()
    mcs_calculator.search_mcs(id_1, id_2, mesh_1, mesh_2)
    return (id_1, id_2), mcs_calculator.min_distance


def _check_overlap(box, bounding_boxes):
    is_in = np.empty((len(bounding_boxes),), dtype=bool)
    for i, org_bb in enumerate(bounding_boxes):
        is_in[i] = boxes_overlap(box, org_bb)

    return is_in


@delayed
def delayed_domain_min_dists(args):
    local_meshes, local_ids = args
    tasks = []

    for i in range(len(local_meshes)):
        mesh_1 = local_meshes[i]
        id_1 = local_ids[i]
        for j in range(i + 1, len(local_meshes)):
            mesh_2 = local_meshes[j]
            id_2 = local_ids[j]

            tasks.append(get_min_dist((id_1, id_2, mesh_1, mesh_2)))
    return tasks


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

        if mean_distance_normal < mean_distance_inverse or np.isnan(
            mean_distance_inverse
        ):
            self._filter_distances(lambda i: self.dot_products[i] < 0)
        else:
            self._filter_distances(lambda i: self.dot_products[i] > 0)

        self.id_source = id_source
        self.id_target = id_target
        self.mesh_source = mesh_source
        self.mesh_target = mesh_target

    def analyze_mcs(self, mcs_label: str, max_distance: float, min_distance=0.0):
        """After searching the entire surface we now filter only the area we are interested in

        Args:
            mcs_label (str): label of this mcs search
            max_distance (float): Maximum distance to the other organelle
            min_distance (float, optional): minimum distance. Defaults to 0.

        Raises:
            ValueError: search_mcs must be performed first.
        """

        self.mcs_label = mcs_label

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
        logger.warning(
            f"No max distance set, calculating distance matrix up to {max_dist}"
        )

    if "max_distance_computed" in cache:
        max_cached = cache["max_distance_computed"]

    if (
        "distance_matrix" not in cache or not project.use_cache
    ) or max_dist > max_cached:
        project.logger.info("Initializing distance matrix")

        organelles = np.array(project.organelles)
        organelles_ids = np.array(project.organelle_ids)
        meshes = []
        bounding_boxes = []
        for organelle in organelles:
            meshes.append(organelle.mesh)
            bounding_boxes.append(bounding_box_delayed(organelle.mesh))
        meshes = persist(*meshes)
        # WHY is this single threaded?? maybe bad distribution between workers
        with span("dist_matrix_bounding_boxes"):
            # bounding_boxes = compute(bounding_boxes)[0]
            # with many meshes (20k) ~30% faster then computing directly:
            bounding_boxes = project.client.gather(
                project.client.map(lambda m: m.compute().bounding_box.bounds, meshes)
            )
        print(f"bounding_boxes {len(bounding_boxes)}")

        project.logger.info("Calculating distance matrix")
        num_rows = len(meshes)

        distance_matrix = np.ones((num_rows, num_rows)) * -1
        distance_df = pd.DataFrame(
            distance_matrix,
            index=organelles_ids,
            columns=organelles_ids,
        )

        # TODO: Better use data chunks instead of our own boxes
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
            # masks = list(map(lambda b: _check_overlap(b, bounding_boxes), tasks))
            with Pool(processes=project.n_workers) as pool:
                masks = pool.starmap(_check_overlap, tasks, chunksize=100)

        else:
            # no domain decomposition -> all to all
            masks = [np.ones((num_rows,))]

        project.logger.debug(f"Masks created: {len(masks)}")
        results = []
        empty_cubes = 0
        tasks = []
        for mask in masks:
            indices = np.nonzero(mask)[0]
            local_meshes_d = [o.mesh for o in organelles[indices]]
            local_ids = organelles_ids[indices]

            if len(indices) < 2:
                empty_cubes += 1
                continue

            tasks.append((local_meshes_d, local_ids))

        project.logger.debug(f"n empty cube: {empty_cubes}")
        project.logger.debug(f"n tasks get_min_dist: {len(tasks)}")

        with span("dist_matrix_min_dists"):
            results = map(delayed_domain_min_dists, tasks)
            results = compute(results)[0]
            results = [r for res in results for r in res]

        for res in tqdm(results, "gathering distances"):
            distance_df.loc[res[0]] = res[1]
            distance_df.loc[res[0][::-1]] = res[1]

        cache["distance_matrix"] = distance_df
        cache["max_distance_computed"] = max_dist

    else:
        project.logger.info("Retrieving distance matrix from cache")

    project.max_distance = max_dist
    return cache["distance_matrix"]


def generate_mcs(
    project,
    ids_filter_1: str,
    ids_filter_2: str,
    max_distance: float,
    min_distance: float = 0,
    overwrite=False,
) -> str:
    """Generates the MCS (Membrane Contact Site) pairs for a given project.
    The MCSs are calculated between two sets of organelles, defined by the
    two filter strings provided.


    Args:
        project (Project): The project object containing the distance
            matrix and organelles.
        max_distance (float): The maximum distance for the MCS pairs.
        min_distance (float, optional): The minimum distance for the MCS pairs.
            Defaults to 0.
        ids_filter_1 (str, optional): Filter for the first set of
            organelles using glob-style patterns. Defaults to "*".
        ids_filter_2 (str, optional): Filter for the second set of
            organelles using glob-style patterns. Defaults to "*".

        overwrite (bool, optional): Whether to overwrite existing MCS data.
            Defaults to False.

    Returns:
        str: The label of this MCS search
    """

    if max_distance > project.max_distance:
        project.max_distance = max_distance
    label_1 = ids_filter_1.replace("*", "")
    label_2 = ids_filter_2.replace("*", "")
    mcs_label = f"{min_distance}-{max_distance},{label_1}-{label_2}"
    if project._mcs_labels.get(mcs_label) and not overwrite:
        return mcs_label

    ids_1 = project.get_organelle_ids(ids_filter_1)
    ids_2 = project.get_organelle_ids(ids_filter_2)
    distance_matrix = project.distance_matrix

    # filter dataframe and get row, column where the distance is between min_distance and max_distance
    rows, columns = np.where(
        (distance_matrix >= min_distance) & (distance_matrix <= max_distance)
    )

    # generate the pairs of organelles
    pairs = set()
    for i, j in zip(rows, columns):
        org_1_id = distance_matrix.index[i]
        org_2_id = distance_matrix.columns[j]

        if org_1_id == org_2_id:
            continue

        if org_1_id not in ids_1 or org_2_id not in ids_2:
            continue

        pairs.add(tuple(sorted([org_1_id, org_2_id])))

    project.logger.info(f"Found {len(pairs)} pairs of organelles to calculate MCS")

    meshes = {}
    for ids in pairs:
        for ind in ids:
            if ind not in meshes:
                meshes[ind] = project.get_organelles(ind)[0].mesh

    tasks = {}
    for id_1, id_2 in tqdm(pairs, "mcs calculation"):
        mcs_calc_delayed = calc_mcs_delayed(
            mcs_label,
            id_1,
            id_2,
            meshes[id_1],
            meshes[id_2],
            max_distance,
            min_distance,
        )
        tasks[id_1 + id_2] = mcs_calc_delayed
    results = compute(tasks)[0]

    for id_1, id_2 in tqdm(pairs, "mcs calculation"):
        org1 = project.get_organelles(id_1)[0]
        org2 = project.get_organelles(id_2)[0]

        mcs_source, mcs_target = results[id_1 + id_2]

        # add mcs to organelle
        if org1.id == mcs_source["self_id"]:
            org1.add_mcs(mcs_source)
            org2.add_mcs(mcs_target)
        else:
            org1.add_mcs(mcs_target)
            org2.add_mcs(mcs_source)

    org_ids = set()
    for o1, o2 in pairs:
        org_ids.add(o1)
        org_ids.add(o2)
    for org_id in org_ids:
        org = project.get_organelles(org_id)[0]
        org.calc_mcs_dict_entry(mcs_label)

    project._mcs_labels[mcs_label] = {
        "max_distance": max_distance,
        "min_distance": min_distance,
        "organelles": org_ids,
    }
    return mcs_label


@delayed
def calc_mcs_delayed(
    mcs_label,
    id_1,
    id_2,
    mesh_1,
    mesh_2,
    max_dist,
    min_dist,
):
    mcs_calculator = MembraneContactSiteCalculator()
    mcs_calculator.search_mcs(id_1, id_2, mesh_1, mesh_2)
    mcs_calculator.analyze_mcs(mcs_label, max_distance=max_dist, min_distance=min_dist)

    mcs_source = mcs_calculator.mcs_source
    mcs_target = mcs_calculator.mcs_target
    return mcs_source, mcs_target
