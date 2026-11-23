"""Pair gate and distance matrix bounds (ticket 02).

The distance matrix stores an upper bound on the pairwise surface distance;
a second cached matrix stores a rigorous lower bound. The pair gate keeps a
pair iff its lower bound is within ``max_distance`` -- ``min_distance`` only
filters surface points later, never pairs.
"""

import fnmatch
import logging

import numpy as np
import pandas as pd
import pytest
import trimesh
from dask.array import zeros
from dask.delayed import delayed

from organelle_morphology.distance_calculations import (
    generate_distance_matrix,
    generate_mcs,
    get_min_dist,
)


def _icosphere_pair(gap, radius=1.0, subdivisions=2):
    mesh_a = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    mesh_b = mesh_a.copy()
    mesh_b.apply_translation((0.0, 0.0, 2.0 * radius + gap))
    return mesh_a, mesh_b


def _grid_plane(extent=1.0, cells=20, z=0.0):
    """Flat triangulated square in the xy plane at height ``z``."""
    coordinates = np.linspace(-extent, extent, cells + 1)
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="ij")
    vertices = np.column_stack(
        [xx.reshape(-1), yy.reshape(-1), np.full(xx.size, z)]
    )
    faces = []
    stride = cells + 1
    for i in range(cells):
        for j in range(cells):
            lower_left = i * stride + j
            faces.append([lower_left, lower_left + stride, lower_left + stride + 1])
            faces.append([lower_left, lower_left + stride + 1, lower_left + 1])
    return trimesh.Trimesh(
        vertices=vertices, faces=np.asarray(faces), process=False
    )


class _StubSource:
    """Just enough source for ``generate_distance_matrix``."""

    def __init__(self, shape=(10, 10, 10), resolution=(1.0, 1.0, 1.0)):
        self.data = zeros(shape, chunks=shape)
        self.resolution = resolution
        self.clipping_corners = (
            np.zeros(3),
            np.array(shape) * np.array(resolution),
        )


class _DelayedOrganelle:
    def __init__(self, org_id, mesh):
        self.id = org_id
        self.mesh = delayed(mesh)
        self.chunks = [(0, 0, 0)]


class _MatrixProject:
    def __init__(self, organelles, max_distance):
        self.sources = {"stub": _StubSource()}
        self.organelles = organelles
        self.client = None
        self.max_distance = max_distance
        self.use_cache = True
        self.cache = {}
        self.logger = logging.getLogger(__name__)


class _GateOrganelle:
    def __init__(self, org_id, mesh):
        self.id = org_id
        self.mesh = mesh
        self.mcs = {}
        self.mcs_dict_entries = {}

    def add_mcs(self, mcs, mcs_meta):
        entries = self.mcs.setdefault(mcs["mcs_label"], {}).setdefault(
            mcs["partner_id"], []
        )
        entries.extend(mcs["patches"])

    def calc_mcs_dict_entry(self, mcs_label, mcs_meta=None):
        self.mcs_dict_entries[mcs_label] = True


class _GateProject:
    """Project stub exposing matrices directly, so no meshing is involved."""

    def __init__(self, organelles, upper, lower, max_distance):
        self.organelles = list(organelles)
        self.max_distance = max_distance
        self.use_cache = True
        self.cache = {
            "distance_matrix": upper,
            "distance_matrix_lower": lower,
            "max_distance_computed": max_distance,
        }
        self._mcs_labels = {}
        self.logger = logging.getLogger(__name__)

    @property
    def distance_matrix(self):
        return self.cache["distance_matrix"]

    def get_organelle_ids(self, ids):
        result = []
        for pattern in [ids] if isinstance(ids, str) else ids:
            result.extend(
                organelle.id
                for organelle in self.organelles
                if fnmatch.fnmatch(organelle.id, pattern)
            )
        return result

    def get_organelles(self, ids):
        result = []
        for pattern in [ids] if isinstance(ids, str) else ids:
            result.extend(
                organelle
                for organelle in self.organelles
                if fnmatch.fnmatch(organelle.id, pattern)
            )
        return result


def _pair_matrices(meshes, max_distance):
    """Build the upper/lower matrices for a small dict of meshes."""
    ids = list(meshes)
    upper = np.full((len(ids), len(ids)), -1.0)
    lower = np.full((len(ids), len(ids)), -1.0)
    for i, id_1 in enumerate(ids):
        for j in range(i + 1, len(ids)):
            id_2 = ids[j]
            (_, _), upper_value, lower_value = get_min_dist(
                (id_1, id_2, meshes[id_1], meshes[id_2], max_distance)
            )
            upper[i, j] = upper[j, i] = upper_value
            lower[i, j] = lower[j, i] = lower_value
    return (
        pd.DataFrame(upper, index=ids, columns=ids),
        pd.DataFrame(lower, index=ids, columns=ids),
    )


def test_distance_matrix_exposes_upper_and_lower_bounds():
    """The matrix keeps the upper bound; the lower bound is cached beside it."""
    mesh_a, mesh_b = _icosphere_pair(gap=0.05)
    project = _MatrixProject(
        [_DelayedOrganelle("a", mesh_a), _DelayedOrganelle("b", mesh_b)],
        max_distance=0.1,
    )

    matrix = generate_distance_matrix(project, domain_decomposition=False)
    lower = project.cache["distance_matrix_lower"]

    assert isinstance(matrix, pd.DataFrame)
    assert isinstance(lower, pd.DataFrame)
    assert list(matrix.index) == ["a", "b"]
    assert list(lower.index) == ["a", "b"]
    # 0.05 gap icospheres: the vertex-to-surface upper bound is in the band
    assert 0.04 <= matrix.loc["a", "b"] < 0.1
    assert matrix.loc["a", "b"] == matrix.loc["b", "a"]
    assert lower.loc["a", "b"] == lower.loc["b", "a"]
    assert 0.0 <= lower.loc["a", "b"] <= matrix.loc["a", "b"]
    # diagonal and uncomputed entries are never distances
    assert matrix.loc["a", "a"] == -1
    assert lower.loc["a", "a"] == -1

    # cached recompute is identical
    cached = generate_distance_matrix(project, domain_decomposition=False)
    np.testing.assert_array_equal(cached.values, matrix.values)
    np.testing.assert_array_equal(
        project.cache["distance_matrix_lower"].values, lower.values
    )

    # a cache from before the gate change (no lower matrix) is recomputed
    del project.cache["distance_matrix_lower"]
    recomputed = generate_distance_matrix(project, domain_decomposition=False)
    np.testing.assert_array_equal(recomputed.values, matrix.values)
    np.testing.assert_array_equal(
        project.cache["distance_matrix_lower"].values, lower.values
    )


def test_gate_selects_small_gap_icospheres():
    """0.05-gap icospheres at max_distance 0.1 are not missed by the gate."""
    mesh_a, mesh_b = _icosphere_pair(gap=0.05)
    upper, lower = _pair_matrices({"a": mesh_a, "b": mesh_b}, max_distance=0.1)
    org_a = _GateOrganelle("a", mesh_a)
    org_b = _GateOrganelle("b", mesh_b)
    project = _GateProject([org_a, org_b], upper, lower, max_distance=0.1)

    label = generate_mcs(project, "a", "b", max_distance=0.1)

    assert label == "0-0.1,a-b"
    assert list(project._mcs_labels) == [label]
    for organelle, partner_id in ((org_a, "b"), (org_b, "a")):
        assert label in organelle.mcs
        patches = organelle.mcs[label][partner_id]
        assert patches
        assert all(patch["area"] > 0 for patch in patches)
        distances = np.concatenate(
            [patch["distances"] for patch in patches]
        )
        assert np.all(distances <= 0.1 + 1e-9)


def _flipped_plane(extent=2.0, cells=10, z=0.0):
    """A grid plane whose normals point away from a partner above it."""
    plane = _grid_plane(extent=extent, cells=cells, z=z)
    plane.faces = np.asarray(plane.faces)[:, ::-1]
    return plane


def test_gate_keeps_one_sided_contact():
    """A side without its own MCS must not suppress the other side's record.

    A sphere hovers above a plane whose winding points away from it, so the
    plane has no facing (hence no member) surface points while the sphere
    still has a facing bottom cap inside the band. The apposition is
    one-sided; the sphere's side must still be recorded.
    """
    plane = _flipped_plane()
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    sphere.apply_translation((0.0, 0.0, 1.002))
    upper, lower = _pair_matrices(
        {"plane": plane, "sphere": sphere}, max_distance=0.1
    )
    org_plane = _GateOrganelle("plane", plane)
    org_sphere = _GateOrganelle("sphere", sphere)
    project = _GateProject(
        [org_plane, org_sphere], upper, lower, max_distance=0.1
    )

    label = generate_mcs(project, "plane", "sphere", max_distance=0.1)

    assert label in org_sphere.mcs
    assert org_sphere.mcs[label]["plane"]
    assert all(patch["area"] > 0 for patch in org_sphere.mcs[label]["plane"])
    # the plane contributes no patch, so it gets no aggregate entry
    assert label not in org_plane.mcs
    assert org_plane.mcs_dict_entries == {}
    assert org_sphere.mcs_dict_entries == {label: True}
    assert project._mcs_labels[label]["organelles"] == {"sphere"}


def test_gate_ignores_min_distance_pair_filter():
    """A pair whose global minimum is below min_distance is kept.

    The sphere sits 0.002 above a plane; the band [0.01, 0.1] is applied to
    surface points, so the annulus around the closest point is an MCS and the
    region at distance 0.05 must be reported.
    """
    plane = _grid_plane(extent=2.0, cells=10, z=0.0)
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    sphere.apply_translation((0.0, 0.0, 1.002))
    upper, lower = _pair_matrices(
        {"plane": plane, "sphere": sphere}, max_distance=0.1
    )
    assert upper.loc["plane", "sphere"] < 0.01  # global minimum below the band
    assert lower.loc["plane", "sphere"] <= 0.1

    org_plane = _GateOrganelle("plane", plane)
    org_sphere = _GateOrganelle("sphere", sphere)
    project = _GateProject(
        [org_plane, org_sphere], upper, lower, max_distance=0.1
    )

    label = generate_mcs(
        project, "plane", "sphere", max_distance=0.1, min_distance=0.01
    )

    assert label == "0.01-0.1,plane-sphere"
    assert label in org_sphere.mcs
    patches = org_sphere.mcs[label]["plane"]
    assert patches
    distances = np.concatenate(
        [patch["distances"] for patch in patches]
    )
    assert np.all(distances >= 0.01 - 1e-9)
    assert np.all(distances <= 0.1 + 1e-9)
    # a region around distance 0.05 is part of the reported MCS
    assert distances.max() > 0.05


def test_lower_bound_brackets_the_true_minimum_on_analytic_fixtures():
    """``get_min_dist``'s bounds bracket independently known minima.

    The previous version recomputed the lower-bound formula from
    ``min_dist_upper`` itself, which passes by construction. Here the true
    minima are known from the fixture geometry and the public ``get_min_dist``
    result must satisfy ``lower <= true <= upper`` for both argument orders.
    """
    gap = 0.2
    tol = 1e-12

    # two equal spheres: the closest surface points lie on the center line
    sphere_a, sphere_b = _icosphere_pair(gap=gap, radius=1.0, subdivisions=3)
    for id_1, id_2, mesh_1, mesh_2 in (
        ("a", "b", sphere_a, sphere_b),
        ("b", "a", sphere_b, sphere_a),
    ):
        pair, upper, lower = get_min_dist((id_1, id_2, mesh_1, mesh_2, None))
        assert pair == (id_1, id_2)
        assert -tol <= lower <= gap + tol
        assert gap - tol <= upper
        assert lower <= upper + tol

    # a sphere hovering above a finite plane: the true minimum is the vertical
    # gap at the plane centre, which lies well inside the triangulated square
    plane = _grid_plane(extent=2.0, cells=10, z=0.0)
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    sphere.apply_translation((0.0, 0.0, 1.0 + gap))
    for id_1, id_2, mesh_1, mesh_2 in (
        ("s", "p", sphere, plane),
        ("p", "s", plane, sphere),
    ):
        pair, upper, lower = get_min_dist((id_1, id_2, mesh_1, mesh_2, None))
        assert pair == (id_1, id_2)
        assert -tol <= lower <= gap + tol
        assert gap - tol <= upper
        assert lower <= upper + tol


def test_get_min_dist_is_symmetric_and_returns_lower_bound():
    mesh_a, mesh_b = _icosphere_pair(gap=0.05)
    forward = get_min_dist(("a", "b", mesh_a, mesh_b, 0.1))
    backward = get_min_dist(("b", "a", mesh_b, mesh_a, 0.1))

    assert forward[0] == ("a", "b")
    assert backward[0] == ("b", "a")
    assert forward[1] == pytest.approx(backward[1])
    assert forward[2] == pytest.approx(backward[2])
    assert 0.0 <= forward[2] <= forward[1]

    # out-of-band pairs keep an uncomputed (-1) upper bound, but the lower
    # bound still carries the gate information.
    far = get_min_dist(("a", "b", mesh_a, mesh_b, 0.01))
    assert np.isnan(far[1])
    assert np.isfinite(far[2])
