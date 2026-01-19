from itertools import combinations
from dask.delayed import delayed
import numpy as np
import trimesh
from dask.base import compute

from organelle_morphology import Organelle


@delayed
def delayed_calculate_mcs(mesh1, mesh2, max_distance):
    # dist for each vert in mesh1, and nearest index in mesh2
    distances1, indices1 = mesh2.nearest.vertex(mesh1.vertices)

    if distances1.min() > max_distance:
        return None

    _repair_meshe(mesh1)
    _repair_meshe(mesh2)

    distances2, indices2 = mesh1.nearest.vertex(mesh2.vertices)

    dot_product1 = np.einsum(
        "ij,ij->i", mesh1.vertex_normals, mesh2.vertex_normals[indices1]
    )
    dot_product2 = np.einsum(
        "ij,ij->i", mesh2.vertex_normals, mesh1.vertex_normals[indices2]
    )

    # some of the meshes have inverted normal vectors,
    # i think this can happen when the meshes are not watertight at the cell borders
    # this means that sometimes we have to search for a negative dot product
    # and other times for a positive.
    # To determine which one to search for, we calculate the mean distance for both cases
    # and choose the one with the smaller mean distance.

    mean_dist_1a = np.mean(
        [
            distances1[i]
            for i in range(len(distances1))
            if dot_product1[i] < 0 and distances1[i] <= max_distance
        ]
    )
    mean_dist_1b = np.mean(
        [
            distances1[i]
            for i in range(len(distances1))
            if dot_product1[i] > 0 and distances1[i] <= max_distance
        ]
    )
    mean_dist_2a = np.mean(
        [
            distances2[i]
            for i in range(len(distances2))
            if dot_product2[i] < 0 and distances2[i] <= max_distance
        ]
    )
    mean_dist_2b = np.mean(
        [
            distances2[i]
            for i in range(len(distances2))
            if dot_product2[i] > 0 and distances2[i] <= max_distance
        ]
    )

    if mean_dist_1a < mean_dist_1b:
        filter1 = (dot_product1 < 0) & (distances1 <= max_distance)

    elif mean_dist_1a > mean_dist_1b:
        filter1 = (dot_product1 > 0) & (distances1 <= max_distance)
    else:
        filter1 = distances1 <= max_distance
    stats1 = {}
    stats1["indices"] = np.nonzero(filter1)
    stats1["distances"] = distances1[filter1]

    if mean_dist_2a < mean_dist_2b:
        filter2 = (dot_product2 < 0) & (distances2 <= max_distance)

    elif mean_dist_2a > mean_dist_2b:
        filter2 = (dot_product2 > 0) & (distances2 <= max_distance)
    else:
        filter2 = distances2 <= max_distance
    stats2 = {}
    stats2["indices"] = np.nonzero(filter2)
    stats2["distances"] = distances2[filter2]

    # area calculation
    faces1 = np.nonzero(np.any(np.isin(mesh1.faces, stats1["indices"]), axis=1))[0]
    stats1["area"] = mesh1.area_faces[faces1].sum()
    faces2 = np.nonzero(np.any(np.isin(mesh2.faces, stats2["indices"]), axis=1))[0]
    stats2["area"] = mesh2.area_faces[faces2].sum()

    return stats1, stats2


def _repair_meshe(mesh):
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)


def calculate_mcs(organelles: list[Organelle], max_distance: float):
    """Calculate all mcs between everything in the project"""

    org_data = {}
    for org1, org2 in combinations(organelles, 2):
        name = "@".join(sorted([org1.id, org2.id]))
        org_data[name] = delayed_calculate_mcs(org1.mesh, org2.mesh, max_distance)
    org_data = compute(org_data)[0]
    return dict((k, v) for k, v in org_data.items() if v)
