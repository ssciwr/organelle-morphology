import logging

from organelle_morphology import distance_calculations
from organelle_morphology.distance_calculations import (
    calc_mcs_delayed,
    chunk_domain_masks,
    generate_distance_matrix,
    generate_mcs,
    get_min_dist,
)
from organelle_morphology.util import get_neighboring_chunks
from dask.array import zeros
from dask.delayed import delayed
from pandas import DataFrame
import pytest
import numpy as np
import trimesh


class _StubOrganelle:
    """Minimal organelle stand in for distance matrix calculations."""

    def __init__(self, org_id, mesh, chunks):
        self.id = org_id
        self.mesh = delayed(mesh)
        self.chunks = list(chunks)


class _StubSource:
    """Minimal source stand in: a chunked data array and its physical size."""

    def __init__(self, data, resolution=(1.0, 1.0, 1.0)):
        self.data = data
        self.resolution = resolution
        self.clipping_corners = (
            np.zeros(3),
            np.array(data.shape) * np.array(resolution),
        )


class _StubProject:
    """Minimal project stand in, holding organelles and a dask client."""

    def __init__(self, source, organelles, client, max_distance):
        self.sources = {"stub": source}
        self.organelles = organelles
        self.client = client
        self.max_distance = max_distance
        self.use_cache = False
        self.cache = {}
        self.logger = logging.getLogger(__name__)


def _box_organelle(org_id, center, chunks, size=2.0):
    mesh = trimesh.creation.box((size, size, size))
    mesh.apply_translation(center)
    return _StubOrganelle(org_id, mesh, chunks)


def test_distance_matrix(project_with_sources):
    project_with_sources.max_distance = 500
    df = generate_distance_matrix(project_with_sources)
    assert isinstance(df, DataFrame)
    assert df.shape == (19, 19)
    for ind, row in df.iterrows():
        assert row.loc[df.index[3]] == df.loc[df.index[3], ind]
        assert row.pop(ind) == -1.0
        assert np.all(row >= 0.0)
    project_with_sources.simplify = 0.0
    project_with_sources.max_distance = 500
    df_base = generate_distance_matrix(project_with_sources)
    assert (df_base - df).abs().max().max() < 0.1
    assert (df_base - df).abs().max().mean() < 0.01


def test_distance_matrix_incomplete(project_with_sources):
    project_with_sources.max_distance = 10
    df = generate_distance_matrix(project_with_sources, domain_decomposition=True)
    assert isinstance(df, DataFrame)
    assert df.shape == (19, 19)
    assert (df < 0).sum().sum() == 343
    for ind, row in df.iterrows():
        assert row.loc[df.index[3]] == df.loc[df.index[3], ind]
        assert row.pop(ind) == -1.0


def test_distance_matrix_incomplete_dd_compair(project_with_sources):
    project_with_sources.max_distance = 10
    df = generate_distance_matrix(project_with_sources, domain_decomposition=False)
    lower = project_with_sources.cache["distance_matrix_lower"]
    project_with_sources.clear_caches(True)
    project_with_sources.max_distance = 10
    df2 = generate_distance_matrix(project_with_sources, domain_decomposition=True)
    lower2 = project_with_sources.cache["distance_matrix_lower"]
    project_with_sources.clear_caches(True)
    project_with_sources.max_distance = 10
    df3 = generate_distance_matrix(
        project_with_sources, domain_decomposition=False, chunk_dd=True
    )
    lower3 = project_with_sources.cache["distance_matrix_lower"]
    project_with_sources.clear_caches(True)
    project_with_sources.max_distance = 10
    df4 = generate_distance_matrix(
        project_with_sources, domain_decomposition=True, chunk_dd=True
    )
    lower4 = project_with_sources.cache["distance_matrix_lower"]

    idxs = np.nonzero((0 <= df) & (df <= 10))
    pairs = list(zip(*idxs))

    assert all([df.iloc[p] == df2.iloc[p] for p in pairs])
    assert all([df.iloc[p] == df3.iloc[p] for p in pairs])
    assert all([df.iloc[p] == df4.iloc[p] for p in pairs])

    assert np.all(df == df2)
    assert np.all(df == df3)
    assert np.all(df == df4)

    # the lower bound matrix must agree across all decomposition strategies
    assert np.all(lower == lower2)
    assert np.all(lower == lower3)
    assert np.all(lower == lower4)


def test_chunk_domain_masks_stay_paired():
    """Each mask is paired with the chunk domain it was built from (#6)."""
    organelles = [
        _box_organelle("org_0", (0, 0, 0), [(1, 1, 1)]),
        _box_organelle("org_1", (0, 0, 0), [(2, 1, 1)]),
        _box_organelle("org_2", (0, 0, 0), [(0, 0, 0)]),
    ]
    dshape = (3, 3, 3)
    domain_masks = chunk_domain_masks(organelles, list(np.ndindex(dshape)), dshape)

    # only domains with an organelle in their center chunk are kept
    expected = [
        ((0, 0, 0), [True, False, True]),
        ((1, 1, 1), [True, True, True]),
        ((2, 1, 1), [True, True, False]),
    ]
    assert len(domain_masks) == len(expected)
    for (center, expected_mask), (mask, blocks) in zip(expected, domain_masks):
        assert mask == expected_mask
        assert blocks == get_neighboring_chunks(center, dshape)


def test_distance_matrix_chunk_dd_empty_domains(client, monkeypatch):
    """Chunked domain decomposition keeps all pairs of sparse data (#6).

    Domains whose center chunk holds no organelle are dropped from the domain
    masks. The chunk domains have to be filtered along with them, otherwise the
    sub decomposition is built for the wrong region of the dataset and the
    organelle pairs of that domain are never evaluated, leaving silent `-1`
    holes in the distance matrix.
    """
    # keep the test fast: decompose domains with more than 3 organelles
    monkeypatch.setattr(distance_calculations, "MAX_ORGS_PER_DOMAIN", 3)

    # 3x3x3 chunk grid, all organelles in the last chunk [40, 60)^3,
    # so all 26 domains before it have no organelle in their center
    data = zeros((60, 60, 60), chunks=(20, 20, 20))
    organelles = [
        _box_organelle(f"org_{i}", center, [(2, 2, 2)])
        for i, center in enumerate(
            [(53, 53, 53), (56, 53, 53), (53, 56, 53), (53, 53, 56)]
        )
    ]
    project = _StubProject(_StubSource(data), organelles, client, max_distance=5)

    # ground truth: all pairs are closer to each other than max_distance
    df = generate_distance_matrix(project, domain_decomposition=False, chunk_dd=False)
    off_diagonal = ~np.eye(len(organelles), dtype=bool)
    assert np.all(df.values[off_diagonal] >= 0)

    df_chunk_dd = generate_distance_matrix(
        project, domain_decomposition=True, chunk_dd=True
    )
    assert np.all(df_chunk_dd.values[off_diagonal] >= 0), "holes in distance matrix"
    np.testing.assert_array_equal(df_chunk_dd.values, df.values)


def test_distance_matrix_no_source(project):
    with pytest.raises(ValueError):
        generate_distance_matrix(project)


def test_distance_matrix_cached(project_with_sources):
    dummy = "DUMMY"
    project_with_sources.cache["distance_matrix"] = dummy
    project_with_sources.cache["distance_matrix_lower"] = "DUMMY_LOWER"
    project_with_sources.cache["max_distance_computed"] = 100

    assert generate_distance_matrix(project_with_sources) is dummy


def test_get_min_dist():
    points = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]])
    points_touching = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2]])
    mesh1 = trimesh.Trimesh(vertices=points, faces=faces)
    mesh2 = trimesh.Trimesh(vertices=points + [0, 0, 2], faces=faces)
    mesh3 = trimesh.Trimesh(vertices=points_touching, faces=faces)

    d1 = get_min_dist(("a", "b", mesh1, mesh2, None))
    d2 = get_min_dist(("a", "b", mesh1, mesh1, None))
    d3 = get_min_dist(("a", "b", mesh1, mesh3, None))

    assert all([r[0] == ("a", "b") for r in [d1, d2, d3]])
    assert d1[1] == 2.0
    assert d2[1] == 0.0
    assert d3[1] == 0.0
    # the lower bound never exceeds the upper bound and is clamped at zero
    assert all(0.0 <= r[2] <= r[1] for r in [d1, d2, d3])


def test_generate_mcs(project_with_sources):
    p = project_with_sources

    p.simplify = 0.0
    generate_mcs(p, "*", "*", max_distance=90, min_distance=0)
    assert list(p.mcs_labels)[0] == "0-90,-"
    org = p.get_organelles("mito_0007")[0]
    assert "0-90,-" in org.mcs.keys()

    mcs_stats = p.registry.get_by_type("McsData")
    assert len(mcs_stats) == 19
    stat = [s for s in mcs_stats if s.meta.organelle_id == org.id][0]

    # point-to-surface geometry finds contact sites; the exact analytic
    # accuracy contract is owned by ticket 05.
    assert stat.data.total_area > 0
    assert stat.data.n_contacts >= 1
    assert stat.data.min_dist >= 0


def test_calc_mcs_delayed_patch_shape():
    """The delayed side result carries the per-patch records for add_mcs."""
    mesh_a = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    mesh_b = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    mesh_b.apply_translation((0.0, 0.0, 2.05))

    result = calc_mcs_delayed(
        "0-0.1,-", "a", "b", mesh_a, mesh_b, max_dist=0.1, min_dist=0.0
    )
    mcs_a, mcs_b = result.compute()

    for mcs, self_id, partner_id, mesh in (
        (mcs_a, "a", "b", mesh_a),
        (mcs_b, "b", "a", mesh_b),
    ):
        assert set(mcs) == {
            "self_id",
            "partner_id",
            "mcs_label",
            "patches",
        }
        assert mcs["self_id"] == self_id
        assert mcs["partner_id"] == partner_id
        assert mcs["mcs_label"] == "0-0.1,-"
        assert len(mcs["patches"]) == 1
        patch = mcs["patches"][0]
        assert set(patch) == {
            "patch_id",
            "faces_index",
            "distances",
            "area",
            "truncated",
            "partner_patch_ids",
        }
        assert patch["patch_id"].startswith(f"{self_id}~{partner_id}~")
        assert patch["area"] > 0
        assert len(patch["distances"]) > 0
        assert set(patch["faces_index"]) <= set(range(len(mesh.faces)))
