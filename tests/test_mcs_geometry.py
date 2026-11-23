"""Tests for the symmetric per-side MCS geometry core.

These tests cover the public interface of ``organelle_morphology.mcs_geometry``
and the analytic expectations of ADR 0002: point-to-surface distance in both
directions, local facing tests, adaptive area refinement, connected patches,
and open-mesh ``truncated`` flags.
"""

import numpy as np
import pytest
import trimesh

from organelle_morphology.mcs_geometry import (
    clear_geometry_cache,
    get_mesh_geometry,
    pair_patches,
    sampling_radius,
    vertex_surface_distances,
)

from .mesh_fixtures import deep_u_mesh, icosphere


# ---------------------------------------------------------------------------
# sampling radius
# ---------------------------------------------------------------------------


def test_sampling_radius_acute_triangle_uses_circumradius():
    side = 2.0
    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [side, 0.0, 0.0],
            [side / 2, side * np.sqrt(3) / 2, 0.0],
        ]
    )
    F = np.array([[0, 1, 2]])
    np.testing.assert_allclose(sampling_radius(V, F), side / np.sqrt(3))


def test_sampling_radius_obtuse_triangle_uses_half_longest_edge():
    # the circumcentre of an obtuse triangle lies outside the face
    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [2.0, 0.5, 0.0],
        ]
    )
    F = np.array([[0, 1, 2]])
    np.testing.assert_allclose(sampling_radius(V, F), 2.0)


def test_sampling_radius_degenerate_and_empty():
    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    F = np.array([[0, 1, 2]])
    np.testing.assert_allclose(sampling_radius(V, F), 1.5)
    assert sampling_radius(np.empty((0, 3)), np.empty((0, 3), dtype=int)) == 0.0
    assert sampling_radius(V, np.empty((0, 3), dtype=int)) == 0.0


# ---------------------------------------------------------------------------
# per-mesh cache
# ---------------------------------------------------------------------------


def _plane_mesh(z=0.0, size=10.0):
    V = np.array(
        [
            [-size, -size, z],
            [size, -size, z],
            [size, size, z],
            [-size, size, z],
        ]
    )
    F = np.array([[0, 1, 2], [0, 2, 3]])
    return trimesh.Trimesh(vertices=V, faces=F, process=False)


def test_geometry_cache_reuses_and_clears():
    mesh = _plane_mesh()
    first = get_mesh_geometry(mesh)
    assert get_mesh_geometry(mesh) is first
    assert first.face_normals.shape == (2, 3)
    np.testing.assert_allclose(first.face_areas, [200.0, 200.0])
    assert first.tree is not None

    clear_geometry_cache()
    second = get_mesh_geometry(mesh)
    assert second is not first
    np.testing.assert_allclose(second.face_areas, first.face_areas)


def test_geometry_boundary_edges_for_open_and_closed_mesh():
    plane = get_mesh_geometry(_plane_mesh())
    assert plane.boundary_edges.shape == (4, 2)

    box = get_mesh_geometry(trimesh.creation.box((2.0, 2.0, 2.0)))
    assert box.boundary_edges.shape == (0, 2)


# ---------------------------------------------------------------------------
# vertex -> partner surface distance
# ---------------------------------------------------------------------------


def test_vertex_surface_distances_are_exact_point_to_triangle():
    plane = _plane_mesh(z=0.0)
    probe = trimesh.Trimesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.5],
                [1.0, 1.0, 2.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        faces=np.array([[0, 1, 2]]),
        process=False,
    )
    geom_plane = get_mesh_geometry(plane)
    geom_probe = get_mesh_geometry(probe)

    distances, face_index, closest = vertex_surface_distances(geom_probe, geom_plane)

    np.testing.assert_allclose(distances, [0.5, 2.0, 1.0])
    # closest points lie on the z=0 plane below the probes
    np.testing.assert_allclose(closest[:, 2], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(closest[:, :2], probe.vertices[:, :2])
    assert face_index.shape == (3,)
    assert np.all(face_index >= 0)


def test_vertex_surface_distances_against_empty_partner():
    probe = _plane_mesh()
    empty = trimesh.Trimesh(
        vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=int), process=False
    )
    distances, face_index, closest = vertex_surface_distances(
        get_mesh_geometry(probe), get_mesh_geometry(empty)
    )
    assert distances.shape == (4,)
    assert np.all(np.isinf(distances))
    assert np.all(face_index == -1)
    assert np.all(np.isnan(closest))


# ---------------------------------------------------------------------------
# helpers for patch tests
# ---------------------------------------------------------------------------


def _zone_area(radius_self, radius_partner, center_distance, max_distance):
    """Analytic area of the in-band spherical zone on ``radius_self``."""
    cos_theta = (
        center_distance**2
        + radius_self**2
        - (max_distance + radius_partner) ** 2
    ) / (2 * center_distance * radius_self)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return 2 * np.pi * radius_self**2 * (1 - cos_theta)


# ---------------------------------------------------------------------------
# pair patches: analytic sphere/sphere contact
# ---------------------------------------------------------------------------

SPHERE_RADIUS = 1.0
SPHERE_GAP = 0.05
BAND_MAX = 0.1
CENTER_DISTANCE = 2 * SPHERE_RADIUS + SPHERE_GAP


def test_pair_patches_sphere_area_matches_analytic_zone():
    mesh_a = icosphere(radius=SPHERE_RADIUS, center=(0, 0, 0))
    mesh_b = icosphere(
        radius=SPHERE_RADIUS, center=(0, 0, CENTER_DISTANCE)
    )
    assert len(mesh_a.vertices) > 10000

    patches_a, patches_b = pair_patches(
        "a", "b", mesh_a, mesh_b, 0.0, BAND_MAX
    )

    expected = _zone_area(
        SPHERE_RADIUS, SPHERE_RADIUS, CENTER_DISTANCE, BAND_MAX
    )
    for patches, self_id in ((patches_a, "a"), (patches_b, "b")):
        assert len(patches) == 1
        patch = patches[0]
        assert set(patch) == {
            "patch_id",
            "faces_index",
            "distances",
            "sample_areas",
            "area",
            "truncated",
            "partner_patch_ids",
        }
        # one sample area per distance sample; together they are the area
        assert patch["sample_areas"].shape == patch["distances"].shape
        assert patch["sample_areas"].sum() == pytest.approx(patch["area"])
        assert patch["patch_id"].startswith(f"{self_id}~")
        assert patch["truncated"] is False
        assert len(patch["faces_index"]) > 0
        assert np.all(patch["distances"] >= 0.0)
        assert np.all(patch["distances"] <= BAND_MAX)
        assert patch["area"] == pytest.approx(expected, rel=0.02)


def test_subdivision_depth_trades_area_precision():
    """A larger refinement depth moves the area closer to the analytic zone."""
    mesh_a = icosphere(radius=SPHERE_RADIUS, subdivisions=3, center=(0, 0, 0))
    mesh_b = icosphere(
        radius=SPHERE_RADIUS, subdivisions=3, center=(0, 0, CENTER_DISTANCE)
    )
    expected = _zone_area(
        SPHERE_RADIUS, SPHERE_RADIUS, CENTER_DISTANCE, BAND_MAX
    )

    coarse, _ = pair_patches(
        "a", "b", mesh_a, mesh_b, 0.0, BAND_MAX, subdivision_depth=0
    )
    fine, _ = pair_patches(
        "a", "b", mesh_a, mesh_b, 0.0, BAND_MAX, subdivision_depth=3
    )

    assert len(coarse) == len(fine) == 1
    assert coarse[0]["area"] != pytest.approx(fine[0]["area"], rel=1e-9)
    assert abs(fine[0]["area"] - expected) < abs(coarse[0]["area"] - expected)
    # leaves refine into samples: depth 3 has more of them than depth 0
    assert len(fine[0]["distances"]) > len(coarse[0]["distances"])


def test_pair_patches_swap_symmetry():
    mesh_a = icosphere(radius=SPHERE_RADIUS, center=(0, 0, 0))
    mesh_b = icosphere(
        radius=SPHERE_RADIUS, center=(0, 0, CENTER_DISTANCE)
    )

    patches_a, patches_b = pair_patches(
        "a", "b", mesh_a, mesh_b, 0.0, BAND_MAX
    )
    swapped_b, swapped_a = pair_patches(
        "b", "a", mesh_b, mesh_a, 0.0, BAND_MAX
    )

    for forward, backward in ((patches_a, swapped_a), (patches_b, swapped_b)):
        assert len(forward) == len(backward)
        for p_fwd, p_bwd in zip(forward, backward):
            np.testing.assert_array_equal(
                p_fwd["faces_index"], p_bwd["faces_index"]
            )
            np.testing.assert_allclose(p_fwd["area"], p_bwd["area"])
            np.testing.assert_allclose(
                np.sort(p_fwd["distances"]), np.sort(p_bwd["distances"])
            )
            assert p_fwd["truncated"] == p_bwd["truncated"]
        assert [p["patch_id"].split("~", 2)[2] for p in forward] == [
            p["patch_id"].split("~", 2)[2] for p in backward
        ]


# ---------------------------------------------------------------------------
# pair patches: U-case and clipping
# ---------------------------------------------------------------------------


def test_pair_patches_u_case_sphere_one_patch_u_two_patches():
    sphere = icosphere(radius=SPHERE_RADIUS, center=(0, 0, 0))
    u_mesh = deep_u_mesh()

    patches_sphere, patches_u = pair_patches(
        "sphere", "u", sphere, u_mesh, 0.0, BAND_MAX
    )

    assert len(patches_sphere) == 1
    assert len(patches_u) == 2


def _clipped_sphere(radius=1.0):
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=radius)
    clipped = trimesh.intersections.slice_mesh_plane(
        sphere,
        plane_normal=(0.0, 0.0, 1.0),
        plane_origin=(0.0, 0.0, 0.0),
        cap=False,
    )
    return clipped


def test_truncated_flags_faces_touching_the_rim():
    """A huge face lying flush on the rim is truncated even at its centre.

    The two-triangle analytic planes have their face centroids far from the
    open boundary, so the centroid-only test reported ``truncated=False``.
    Every face vertex lies exactly on the rim, which must count as distance 0.
    """
    lower = _plane_mesh(z=0.0, size=10.0)
    upper = _plane_mesh(z=0.05, size=10.0)
    upper.faces = np.asarray(upper.faces)[:, ::-1]  # face down toward lower

    patches_lower, patches_upper = pair_patches(
        "lower", "upper", lower, upper, 0.0, 0.1
    )

    for patches in (patches_lower, patches_upper):
        assert len(patches) == 1
        assert patches[0]["truncated"] is True

    # closed control: no rim, so a closed contact is never truncated
    closed_a = trimesh.creation.box((1.0, 1.0, 1.0))
    closed_b = closed_a.copy()
    closed_b.apply_translation((0.0, 0.0, 1.05))
    closed_patches, _ = pair_patches(
        "a", "b", closed_a, closed_b, 0.0, 0.1
    )
    assert all(not patch["truncated"] for patch in closed_patches)


def test_clipped_rim_contact_is_included_and_truncated():
    hemisphere = _clipped_sphere(radius=1.0)
    assert hemisphere.is_watertight is False

    partner = icosphere(radius=0.05, subdivisions=2, center=(1.12, 0.0, 0.0))

    patches, _ = pair_patches(
        "clipped", "partner", hemisphere, partner, 0.0, BAND_MAX
    )
    assert len(patches) >= 1
    assert patches[0]["truncated"] is True

    # closed control: same partner against the full sphere is not truncated
    closed = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    control_patches, _ = pair_patches(
        "closed", "partner", closed, partner, 0.0, BAND_MAX
    )
    assert len(control_patches) >= 1
    assert all(not p["truncated"] for p in control_patches)


# ---------------------------------------------------------------------------
# no mesh repair in the MCS path
# ---------------------------------------------------------------------------


def test_pair_patches_does_not_call_mesh_repair(monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError("mesh repair must not run in the MCS path")

    monkeypatch.setattr(trimesh.repair, "fix_inversion", _fail)
    monkeypatch.setattr(trimesh.repair, "fix_winding", _fail)
    monkeypatch.setattr(trimesh.repair, "fix_normals", _fail)

    mesh_a = icosphere(radius=SPHERE_RADIUS, subdivisions=2, center=(0, 0, 0))
    mesh_b = icosphere(
        radius=SPHERE_RADIUS, subdivisions=2, center=(0, 0, CENTER_DISTANCE)
    )
    patches_a, patches_b = pair_patches(
        "a", "b", mesh_a, mesh_b, 0.0, BAND_MAX
    )
    assert len(patches_a) == 1
    assert len(patches_b) == 1
