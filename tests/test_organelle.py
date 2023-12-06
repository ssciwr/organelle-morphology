import pytest
from organelle_morphology import Project

import numpy as np


def test_geometric_data(cebra_project_with_sources, cebra_project_original_meshes):
    p = cebra_project_with_sources

    label_list = ["area", "bbox", "slice", "centroid", "moments", "extent", "solidity"]

    org_list = p.organelles()
    for organelle in org_list:
        # some meshes seem to vanish in the coarse data.
        # I don't think this should happen..
        id_num = int(organelle.id.split("_")[-1])
        print(id_num)
        mesh = cebra_project_original_meshes[id_num]

        assert np.isclose(
            mesh["volume"], organelle.geometric_data["area"], rtol=0.25, atol=500
        )
        assert sorted(list(organelle.geometric_data.keys())) == sorted(label_list)


def test_organelle_data_with_sources(cebra_project_with_sources):
    p = cebra_project_with_sources
    org_list = p.organelles()

    for organelle in org_list:
        unq_num = np.unique(organelle.data)
        assert len(unq_num) == 2
        assert unq_num[1] == int(organelle.id.split("_")[-1])


def test_organelle_mesh(cebra_project_with_sources, cebra_project_original_meshes):
    p = cebra_project_with_sources

    org_list = p.organelles()
    for organelle in org_list:
        # some meshes seem to vanish in the coarse data.
        # I don't think this should happen..
        id_num = int(organelle.id.split("_")[-1])
        original_mesh = cebra_project_original_meshes[id_num]
        new_mesh = organelle.mesh

        assert np.isclose(original_mesh["volume"], new_mesh.volume, rtol=0.25, atol=500)
        assert np.isclose(original_mesh["area"], new_mesh.area, rtol=0.40, atol=100)
