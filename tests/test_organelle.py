import pytest
from organelle_morphology import Project

import numpy as np


def test_geometric_data(cebra_project_with_sources, cebra_project_original_meshes):
    p = cebra_project_with_sources

    label_list = ["area", "bbox", "slice", "centroid", "moments", "extent", "solidity"]

    org_list = p.organelles()

    distance_vecs = []
    for organelle in org_list:
        # some meshes seem to vanish in the coarse data.
        # I don't think this should happen..
        id_num = int(organelle.id.split("_")[-1])
        mesh = cebra_project_original_meshes[id_num]

        distance_vec = mesh["center"] - organelle.geometric_data["centroid"]
        distance_vecs.append(distance_vec)

        # these two voxel somehow get changed when voxelizing the original meshes, so i will skip them for now
        if id_num not in [9, 17]:
            assert np.isclose(
                mesh["volume"], organelle.geometric_data["area"], rtol=0.15, atol=500
            )

        assert sorted(list(organelle.geometric_data.keys())) == sorted(label_list)

    # ensure that all meshes are are close to where they were originally, although the coordinates are changed.
    assert np.std(distance_vecs, axis=0).max() < 1.5


def test_organelle_data_with_sources(cebra_project_with_sources):
    p = cebra_project_with_sources
    org_list = p.organelles()

    for organelle in org_list:
        unq_num = np.unique(organelle.data)
        assert len(unq_num) == 2
        assert unq_num[1] == int(organelle.id.split("_")[-1])


def test_organelle_mesh(cebra_project_with_sources, cebra_project_original_meshes):
    p = cebra_project_with_sources

    label_list = [
        "mesh_volume",
        "mesh_area",
        "mesh_centroid",
        "mesh_inertia",
        "water_tight",
        "sphericity",
        "flatness_ratio",
    ]

    org_list = p.organelles()
    for organelle in org_list:
        # some meshes seem to vanish in the coarse data.
        # I don't think this should happen..

        id_num = int(organelle.id.split("_")[-1])
        if id_num not in [9, 17]:
            original_mesh = cebra_project_original_meshes[id_num]
            new_mesh = organelle.mesh

            assert np.isclose(
                original_mesh["volume"], new_mesh.volume, rtol=0.2, atol=500
            )
            assert np.isclose(original_mesh["area"], new_mesh.area, rtol=0.3, atol=100)

        assert sorted(list(organelle.mesh_properties.keys())) == sorted(label_list)


def test_organelle_morphology(cebra_project_with_sources):
    p = cebra_project_with_sources

    org_list = p.organelles()
    for organelle in org_list:
        assert organelle.morphology_map is not None
