from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from dask import delayed

from organelle_morphology.mcs_geometry import pair_patches
from organelle_morphology.organelle import (
    CURRENT_MCS_SCHEMA_VERSION,
    Mitochondrium,
)

from .test_mcs_geometry import _deep_u_mesh, _icosphere


class _StubSource:
    """Minimal source stand in for record aggregation tests."""

    def __init__(self, project, label, volume=10.0, surface=20.0, mesh=None):
        self.project = project
        self.xml_path = project.path / "stub_source.xml"
        self.mesh_properties = {
            label: SimpleNamespace(volume=volume, area=surface)
        }
        self.meshes = {label: delayed(mesh)} if mesh is not None else {}


def _make_organelle(project, label=1, volume=10.0, surface=20.0):
    source = _StubSource(project, label, volume=volume, surface=surface)
    return Mitochondrium(source=source, label=label)


def _mcs_meta(
    mcs_label="0.0-0.1,-",
    min_distance=0.0,
    max_distance=0.1,
    ids_filter_1="*",
    ids_filter_2="*",
):
    return {
        "ids_filter_1": ids_filter_1,
        "ids_filter_2": ids_filter_2,
        "min_distance": min_distance,
        "max_distance": max_distance,
    }


def _patch(
    self_id,
    partner_id,
    index,
    distances,
    area,
    truncated=False,
    partner_indices=(),
    sample_areas=None,
):
    distances = np.asarray(distances, dtype=np.float64)
    if sample_areas is None:
        sample_areas = np.full(len(distances), float(area) / len(distances))
    return {
        "patch_id": f"{self_id}~{partner_id}~{index}",
        "faces_index": np.arange(len(distances), dtype=np.int64),
        "distances": distances,
        "sample_areas": np.asarray(sample_areas, dtype=np.float64),
        "area": float(area),
        "truncated": truncated,
        "partner_patch_ids": tuple(
            f"{partner_id}~{self_id}~{i}" for i in partner_indices
        ),
    }


def _mcs_dict(mcs_label, self_id, partner_id, patches):
    return {
        "self_id": self_id,
        "partner_id": partner_id,
        "mcs_label": mcs_label,
        "patches": patches,
    }


def _mcs_record(project, organelle_id, mcs_label):
    records = [
        rec
        for rec in project.registry.get_by_type("McsData")
        if rec.meta.organelle_id == organelle_id
        and rec.meta.mcs_label == mcs_label
    ]
    assert len(records) == 1
    return records[0]


def test_organelle_init(project_with_sources):
    p = project_with_sources

    organelles = p.organelles
    assert len(p.sources["synth_data"].labels) == 19
    assert len(organelles) == 19
    for org in organelles:
        assert isinstance(org, Mitochondrium)


def test_organelle_curvature(project_with_sources, mocker):
    s = project_with_sources.sources["synth_data"]
    mock_calc = mocker.patch.object(s, "calc_curvature")
    for org in project_with_sources.organelles:
        mock_calc.return_value = {org.label: np.ones((5,))}
        curv = org.curvature_map
        assert isinstance(curv, np.ndarray)
        mock_calc.assert_called_with(org.label)
        mock_calc.reset_mock()


def test_curvature_map(project_with_sources):
    org = project_with_sources.get_organelles("mito_0007")[0]
    cmap = org.curvature_map
    assert cmap.shape == (451,)


def test_curvature_mesh(project_with_sources):
    org = project_with_sources.get_organelles("mito_0007")[0]
    mesh = org.curvature_mesh.compute()
    assert len(mesh.vertices) == 451


def test_get_mesh_mcs_colored(project_with_sources):
    project_with_sources.search_mcs(10)
    o = project_with_sources.get_organelles("mito_0015")[0]
    label = next(iter(o.mcs))
    n_partners = len(o.mcs[label])
    n_patches = sum(len(patches) for patches in o.mcs[label].values())
    mesh = o.get_mesh_mcs_colored(label).compute()
    n_colors = np.unique(mesh.visual.vertex_colors, axis=0).shape[0]
    # At least the background plus one colour per partner; at most one colour
    # per patch. Shared boundary vertices can overwrite a patch colour, so the
    # exact count is bounded rather than fixed. The per-patch behaviour itself
    # is pinned by test_get_mesh_mcs_colored_uses_one_color_per_patch.
    assert n_partners + 1 <= n_colors <= n_patches + 1
    assert n_colors > n_partners


def test_get_mesh_mcs_colored_missing(project_with_sources):
    o = project_with_sources.get_organelles("mito_0015")[0]
    mesh = o.get_mesh_mcs_colored().compute()
    assert np.unique(mesh.visual.vertex_colors, axis=0).shape == (1, 4)


def test_get_mesh_mcs_colored_uses_one_color_per_patch(project):
    """Two patches of the same partner must not share a colour."""
    mesh = trimesh.Trimesh(
        vertices=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [10.0, 1.0, 0.0],
        ],
        faces=[[0, 1, 2], [3, 4, 5]],
        process=False,
    )
    source = _StubSource(project, label=7, mesh=mesh)
    org = Mitochondrium(source=source, label=7)
    label = "0.0-0.1,-"
    partner = "er_0001"
    meta = _mcs_meta(label)

    org.add_mcs(
        _mcs_dict(
            label,
            org.id,
            partner,
            [
                _patch(org.id, partner, 0, [0.01], 1.0),
                _patch(org.id, partner, 1, [0.02], 1.0),
            ],
        ),
        meta,
    )

    colored = org.get_mesh_mcs_colored(label).compute()
    unique = np.unique(colored.visual.vertex_colors, axis=0)
    # one colour per patch; every vertex here belongs to a patch
    assert unique.shape == (2, 4)
    assert not np.array_equal(
        colored.visual.vertex_colors[0], colored.visual.vertex_colors[3]
    )


def test_get_metadata(project_with_sources):
    o = project_with_sources.get_organelles("mito_0015")[0]
    meta = o.metadata
    assert getattr(meta, "source")
    assert getattr(meta, "label")
    assert getattr(meta, "id")


def test_add_mcs_appends_patch_entries(project):
    """A second call for the same pair appends instead of overwriting."""
    org = _make_organelle(project)
    label = "0.0-0.1,-"
    partner = "mito_0002"
    meta = _mcs_meta(label)

    org.add_mcs(
        _mcs_dict(label, org.id, partner, [_patch(org.id, partner, 0, [0.01], 1.0)]),
        meta,
    )
    org.add_mcs(
        _mcs_dict(label, org.id, partner, [_patch(org.id, partner, 1, [0.02], 2.0)]),
        meta,
    )

    entries = org.mcs[label][partner]
    assert isinstance(entries, list)
    assert [entry["patch_id"] for entry in entries] == [
        f"{org.id}~{partner}~0",
        f"{org.id}~{partner}~1",
    ]
    assert all(entry["meta"] is meta for entry in entries)
    assert [entry["area"] for entry in entries] == [1.0, 2.0]


def test_clear_mcs_drops_only_the_requested_label(project):
    """Overwrite support: clearing a label removes its patches on this organelle."""
    org = _make_organelle(project)
    label = "0.0-0.1,-"
    other_label = "0.1-0.2,-"
    partner = "mito_0002"
    meta = _mcs_meta(label)
    org.add_mcs(
        _mcs_dict(label, org.id, partner, [_patch(org.id, partner, 0, [0.01], 1.0)]),
        meta,
    )
    org.add_mcs(
        _mcs_dict(
            other_label,
            org.id,
            partner,
            [_patch(org.id, partner, 0, [0.15], 1.0)],
        ),
        _mcs_meta(other_label),
    )

    org.clear_mcs(label)

    assert label not in org.mcs
    assert org.mcs[other_label][partner][0]["patch_id"] == f"{org.id}~{partner}~0"
    # clearing an unknown label is a no-op
    org.clear_mcs("missing")
    assert other_label in org.mcs


def test_calc_mcs_dict_entry_counts_patches_for_one_partner(project):
    """One pair with two appositions gives two own patches, one partner."""
    org = _make_organelle(project, volume=10.0, surface=20.0)
    label = "0.0-0.1,-"
    partner = "mito_0002"
    meta = _mcs_meta(label)

    distances_1 = np.array([0.02, 0.04])
    distances_2 = np.array([0.06, 0.08, 0.10])
    org.add_mcs(
        _mcs_dict(
            label,
            org.id,
            partner,
            [
                _patch(org.id, partner, 0, distances_1, 1.0),
                _patch(org.id, partner, 1, distances_2, 3.0),
            ],
        ),
        meta,
    )
    org.calc_mcs_dict_entry(label, meta)

    data = _mcs_record(project, org.id, label).data
    assert data.n_contacts == 2
    assert data.n_partners == 1
    assert data.partners == (partner,)
    assert data.total_area == pytest.approx(4.0)
    assert data.mean_area == pytest.approx(2.0)
    assert data.std_area == pytest.approx(1.0)
    assert data.min_dist == pytest.approx(0.02)
    assert data.max_dist == pytest.approx(0.10)
    assert data.min_dist_per_org == (pytest.approx(0.02),)
    assert data.max_dist_per_org == (pytest.approx(0.10),)
    assert data.n_contacts_per_area == pytest.approx(2 / 20.0)
    assert data.area_per_area == pytest.approx(4.0 / 20.0)
    assert data.n_contacts_per_volume == pytest.approx(2 / 10.0)
    assert data.area_per_volume == pytest.approx(4.0 / 10.0)

    # distances are averaged with the patch area as weight
    mean_1, mean_2 = distances_1.mean(), distances_2.mean()
    expected_mean = (1.0 * mean_1 + 3.0 * mean_2) / 4.0
    expected_var = (
        1.0 * (distances_1.var() + (mean_1 - expected_mean) ** 2)
        + 3.0 * (distances_2.var() + (mean_2 - expected_mean) ** 2)
    ) / 4.0
    assert data.mean_dist == pytest.approx(expected_mean)
    assert data.std_dist == pytest.approx(np.sqrt(expected_var))


def test_calc_mcs_dict_entry_area_weights_within_patch(project):
    """Samples inside one patch are weighted by their own area, not equally.

    Whole faces contribute a single sample carrying the whole face area while
    subdivided faces contribute equal-area leaves, so an unweighted mean is
    biased toward finely refined parts of the same patch.
    """
    org = _make_organelle(project)
    label = "0.0-0.1,-"
    partner = "mito_0002"
    meta = _mcs_meta(label)
    distances = np.array([0.02, 0.08])
    sample_areas = np.array([9.0, 1.0])

    org.add_mcs(
        _mcs_dict(
            label,
            org.id,
            partner,
            [
                _patch(
                    org.id,
                    partner,
                    0,
                    distances,
                    area=10.0,
                    sample_areas=sample_areas,
                )
            ],
        ),
        meta,
    )
    org.calc_mcs_dict_entry(label, meta)

    data = _mcs_record(project, org.id, label).data
    expected_mean = float(np.average(distances, weights=sample_areas))
    expected_var = float(np.average((distances - expected_mean) ** 2, weights=sample_areas))
    assert expected_mean == pytest.approx(0.026)
    assert data.mean_dist == pytest.approx(expected_mean)
    assert data.std_dist == pytest.approx(np.sqrt(expected_var))
    # unweighted mean would be 0.05, so this rejects the old behaviour
    assert data.mean_dist != pytest.approx(0.05)


def test_calc_mcs_dict_entry_multi_partner_metadata(project):
    """Metadata filters and band come from the label, not the last patch."""
    org = _make_organelle(project)
    label = "0.01-0.1,mito-er"
    partner_a = "er_0001"
    partner_b = "er_0002"
    meta = _mcs_meta(
        label,
        min_distance=0.01,
        max_distance=0.1,
        ids_filter_1="mito*",
        ids_filter_2="er*",
    )

    org.add_mcs(
        _mcs_dict(
            label,
            org.id,
            partner_a,
            [
                _patch(org.id, partner_a, 0, [0.01, 0.02], 1.0),
                _patch(org.id, partner_a, 1, [0.05], 1.0),
            ],
        ),
        meta,
    )
    org.add_mcs(
        _mcs_dict(
            label,
            org.id,
            partner_b,
            [_patch(org.id, partner_b, 0, [0.09], 2.0)],
        ),
        meta,
    )
    org.calc_mcs_dict_entry(label, meta)

    record = _mcs_record(project, org.id, label)
    data = record.data
    assert data.n_contacts == 3
    assert data.n_partners == 2
    assert data.partners == (partner_a, partner_b)
    assert data.min_dist_per_org == (
        pytest.approx(0.01),
        pytest.approx(0.09),
    )
    assert data.max_dist_per_org == (
        pytest.approx(0.05),
        pytest.approx(0.09),
    )

    assert record.meta.schema_version == CURRENT_MCS_SCHEMA_VERSION
    assert record.meta.filter_1 == "mito*"
    assert record.meta.filter_2 == "er*"
    assert record.meta.min_dist == pytest.approx(0.01)
    assert record.meta.max_dist == pytest.approx(0.1)


def test_calc_mcs_dict_entry_skips_when_no_patches(project):
    org = _make_organelle(project)
    org.calc_mcs_dict_entry("0.0-0.1,-")
    assert project.registry.get_by_type("McsData") == []


def test_u_case_patch_counts_per_side(project):
    """Sphere has one MCS, the deep U has two, for the same label."""
    sphere = _icosphere(radius=1.0, center=(0, 0, 0))
    u_mesh = _deep_u_mesh()

    sphere_patches, u_patches = pair_patches(
        "sphere", "u", sphere, u_mesh, 0.0, 0.1
    )
    assert len(sphere_patches) == 1
    assert len(u_patches) == 2

    label = "0.0-0.1,-"
    meta = _mcs_meta(label)
    sphere_org = _make_organelle(project, label=1)
    u_org = _make_organelle(project, label=2)

    sphere_org.add_mcs(
        _mcs_dict(label, sphere_org.id, u_org.id, sphere_patches), meta
    )
    u_org.add_mcs(_mcs_dict(label, u_org.id, sphere_org.id, u_patches), meta)
    sphere_org.calc_mcs_dict_entry(label, meta)
    u_org.calc_mcs_dict_entry(label, meta)

    sphere_data = _mcs_record(project, sphere_org.id, label).data
    u_data = _mcs_record(project, u_org.id, label).data
    assert sphere_data.n_contacts == 1
    assert u_data.n_contacts == 2
    assert sphere_data.n_partners == 1
    assert u_data.n_partners == 1
    assert sphere_data.total_area == pytest.approx(
        sum(p["area"] for p in sphere_patches)
    )
    assert u_data.total_area == pytest.approx(
        sum(p["area"] for p in u_patches)
    )
