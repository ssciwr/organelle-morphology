from organelle_morphology.organelle import Organelle, organelle_types
from organelle_morphology.source import DataSource
from organelle_morphology.util import disk_cache, parallel_pool


import trimesh
import logging

import numpy as np
import pandas as pd

from collections import defaultdict
import plotly.graph_objects as go
import multiprocessing as mp
from tqdm import tqdm


class MembraneContactSiteCalculator:
    def __init__(self, project):
        self.use_cache = project.use_cache
        self.project = project

        self.distances = []
        self.index_source = []
        self.index_target = []
        self.normals_source = []
        self.normals_target = []
        self.dot_products = []

        self.id_source = None
        self.id_target = None
        self.mesh_source = None
        self.mesh_target = None

        self.mcs_label = None
        self._distances = None
        self._area_source = None
        self._area_target = None
        self._vertices_target = None
        self._vertices_source = None
        self._vertices_index_source = None
        self._vertices_index_target = None

    def search_mcs(self, id_1, id_2, mesh_1=None, mesh_2=None):
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
            If not provided, it will be retrieved from the project.
            mesh_2 (Mesh, optional): The mesh of the second organelle.
            If not provided, it will be retrieved from the project.

        """
        if mesh_1 is None:
            mesh_1 = self.project.organelles(id_1)[0].mesh
        if mesh_2 is None:
            mesh_2 = self.project.organelles(id_2)[0].mesh

        self._repair_meshes(mesh_1, mesh_2)

        if len(mesh_1.vertices) < len(mesh_2.vertices):
            mesh_source = mesh_2
            mesh_target = mesh_1
            id_source = id_2
            id_target = id_1
        else:
            mesh_source = mesh_1
            mesh_target = mesh_2
            id_source = id_1
            id_target = id_2

        distance, index_source = mesh_source.nearest.vertex(mesh_target.vertices)

        self.distances = distance
        self.index_source = index_source
        self.index_target = np.asarray((range(len(mesh_target.vertices))))

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

        mean_distance_normal = np.mean(
            [
                self.distances[i]
                for i in range(len(self.distances))
                if self.dot_products[i] < 0
            ]
        )
        mean_distance_inverse = np.mean(
            [
                self.distances[i]
                for i in range(len(self.distances))
                if self.dot_products[i] > 0
            ]
        )

        if mean_distance_normal < mean_distance_inverse:
            self._filter_distances(lambda i: self.dot_products[i] < 0)
        else:
            self._filter_distances(lambda i: self.dot_products[i] > 0)

        self.id_source = id_source
        self.id_target = id_target
        self.mesh_source = mesh_source
        self.mesh_target = mesh_target

    def analyze_mcs(self, mcs_label, max_distance, min_distance=0):
        """After searching the entire surface we now filter only the area we are interested in

        :param mcs_label: The label that will be assigned to this search. Used for later access.
        :type mcs_label: str
        :param max_distance:  Maximum distance to the other organelle
        :type max_distance: float
        :param min_distance: minimum distance, defaults to 0
        :type min_distance: int, optional
        :raises ValueError: search_mcs must be performed first.
        """
        if self.distances is None:
            raise ValueError("No MCS found. Please run search_mcs first.")

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

        faces_source = self.mesh_source.faces[
            np.any(np.isin(self.mesh_source.faces, self._vertices_index_source), axis=1)
        ]
        faces_target = self.mesh_target.faces[
            np.any(np.isin(self.mesh_target.faces, self._vertices_index_target), axis=1)
        ]

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
            "vertices": self._vertices_target,
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
            "vertices": self._vertices_source,
            "vertices_index": self._vertices_index_source,
            "distances": self._distances,
            "area": self._area_source,
        }

    def _repair_meshes(self, mesh_source, mesh_target):
        for mesh in [mesh_source, mesh_target]:
            trimesh.repair.fix_inversion(mesh)
            trimesh.repair.fix_winding(mesh)
            trimesh.repair.fix_normals(mesh)


def generate_distance_matrix(project) -> pd.DataFrame:
    active_sources = list(project._sources.keys())
    with disk_cache(
        project, f"distance_matrix_{active_sources}_{project.compression_level}"
    ) as cache:
        if (
            f"distance_matrix_{active_sources}_{project.compression_level}" not in cache
            or not project.use_cache
        ):
            project.logger.info("Initilizing distance matrix")

            project.logger.info("Loading meshes")

            project.calculate_meshes()

            meshes = []
            organelles_ids = []
            for organelle in project.organelles():
                organelles_ids.append(organelle.id)
                meshes.append(organelle.mesh)

            project.logger.info("Calculating distance matrix")
            num_rows = len(meshes)

            distance_matrix = np.zeros((num_rows, num_rows))
            distance_df = pd.DataFrame(
                distance_matrix,
                index=organelles_ids,
                columns=organelles_ids,
            )

            for i in tqdm(np.arange(num_rows)):
                mesh_1 = meshes[i]
                id_1 = organelles_ids[i]

                for j in np.arange(i, num_rows):
                    if i == j:
                        continue

                    mesh_2 = meshes[j]
                    id_2 = organelles_ids[j]

                    mcs_calculator = MembraneContactSiteCalculator(project)
                    mcs_calculator.search_mcs(id_1, id_2, mesh_1, mesh_2)
                    min_distance = mcs_calculator.min_distance

                    distance_df.loc[id_1, id_2] = min_distance
                    distance_df.loc[id_2, id_1] = min_distance

            cache[
                f"distance_matrix_{active_sources}_{project.compression_level}"
            ] = distance_df

        else:
            project.logger.info("Retrieving distance matrix from cache")

        return cache[f"distance_matrix_{active_sources}_{project.compression_level}"]


def _generate_mcs(project, mcs_label, max_distance, min_distance=0):
    """
    Generates the MCS (Membrane Contact Site) pairs for a given project.


    Args:
        project (Project): The project object containing the distance matrix and organelles.
        mcs_label (str): The label for the MCS pairs.
        max_distance (float): The maximum distance for the MCS pairs.
        min_distance (float, optional): The minimum distance for the MCS pairs. Defaults to 0.

    Returns:
        None
    """

    distance_matrix = project.distance_matrix

    # filter dataframe and get row, column where the distance is between min_distance and max_distance

    rows, columns = np.where(
        (distance_matrix >= min_distance) & (distance_matrix <= max_distance)
    )

    # generate the pair dictionary
    pair_dict = {}
    for i, j in zip(rows, columns):
        org_1_label = distance_matrix.index[i]
        org_2_label = distance_matrix.columns[j]

        if org_1_label == org_2_label:
            continue

        if org_1_label not in pair_dict and org_2_label not in pair_dict:
            pair_dict[org_1_label] = [org_2_label]

        elif org_1_label in pair_dict and org_2_label not in pair_dict:
            if org_2_label in pair_dict[org_1_label]:
                continue
            pair_dict[org_1_label].append(org_2_label)

        elif org_1_label not in pair_dict and org_2_label in pair_dict:
            if org_1_label in pair_dict[org_2_label]:
                continue
            pair_dict[org_2_label].append(org_1_label)

        else:
            pair_dict[org_1_label].append(org_2_label)

    project.logger.info(
        f"Found {len(pair_dict)} pairs of organelles to calculate MCS for {mcs_label}"
    )
    # generate the mcs dictionary

    meshes = {}

    for organelle in project.organelles(distance_matrix.index.tolist()):
        meshes[organelle.id] = organelle.mesh

    for org_1_label, org_2_labels in tqdm(pair_dict.items()):
        org1 = project.organelles(org_1_label)[0]
        for org_2_label in org_2_labels:
            mcs_calculator = MembraneContactSiteCalculator(project)
            mcs_calculator.search_mcs(
                org_1_label, org_2_label, meshes[org_1_label], meshes[org_2_label]
            )
            mcs_calculator.analyze_mcs(mcs_label, max_distance, min_distance)

            mcs_source = mcs_calculator.mcs_source
            mcs_target = mcs_calculator.mcs_target

            # add mcs to organelle
            org2 = project.organelles(org_2_label)[0]

            if org1.id == mcs_source["self_id"]:
                org1.add_mcs(mcs_source)
                org2.add_mcs(mcs_target)
            else:
                org1.add_mcs(mcs_target)
                org2.add_mcs(mcs_source)

    for i in rows:
        org_id = distance_matrix.index[i]
        org = project.organelles(org_id)[0]
        org.get_mcs_dict_entry(mcs_label)

    project._mcs_labels[mcs_label] = {
        "max_distance": max_distance,
        "min_distance": min_distance,
    }
