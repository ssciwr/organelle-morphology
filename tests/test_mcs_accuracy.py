"""Analytic accuracy contract for the MCS geometry core (ticket 05).

These tests replace the old algorithm's hardcoded outputs with closed-form
ground truth. The contract is deliberately stated at the public seams:

- ``pair_patches`` returns one patch list per side; every area assertion is
  compared against an independent analytic value (not recomputed by the
  implementation).
- ``get_min_dist`` provides the pair gate bounds; the missed-pair regression
  checks that a true 0.05 surface gap is found at ``max_distance=0.1``.
- The real-data orientation check inspects the exported meshes and, when the
  raw segmentation is present, verifies locally that face normals point into
  empty space.

Tolerances
----------
``PLANE_RTOL``: parallel planes have no band boundary inside the mesh, so the
area is exact; only floating-point summation error remains.

``AREA_RTOL = 0.02``: interior faces are counted whole and faces straddling a
band/facing boundary are midpoint subdivided to depth 3. On a subdivision-5
icosphere (~10k vertices) the residual boundary strip is roughly 1/8 of a face
wide; measured relative errors are 0.4-0.7%, so 2% leaves a factor >2 headroom
for platform/libigl variation while still rejecting the old algorithm (which
overestimated by tens of percent).
"""

import glob
import itertools
from pathlib import Path

import numpy as np
import pytest
import trimesh

from organelle_morphology.distance_calculations import get_min_dist
from organelle_morphology.mcs_geometry import (
    clear_geometry_cache,
    pair_patches,
)

PLANE_RTOL = 1e-9
AREA_RTOL = 0.02
SUBDIVISION_DEPTH = 3
# subdivision 5 -> 10242 vertices per sphere ("~10k-vertex icospheres")
REFERENCE_SUBDIVISIONS = 5


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def _icosphere(radius, center, subdivisions):
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    mesh.apply_translation(center)
    return mesh


def _plane(size, z, flip=False):
    """A triangulated square of side ``size`` in the xy plane at height ``z``."""
    half = size / 2.0
    vertices = np.array(
        [
            [-half, -half, z],
            [half, -half, z],
            [half, half, z],
            [-half, half, z],
        ]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    if flip:
        faces = faces[:, ::-1]
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _zone_area(radius_self, radius_partner, center_distance, min_distance, max_distance):
    """In-band spherical zone area on ``radius_self`` in closed form.

    A point on the sphere at polar angle ``theta`` (measured from the line to
    the partner centre) has surface distance

        ``sqrt(D^2 + r_self^2 - 2 D r_self cos(theta)) - r_partner``.

    Solving each band edge for ``cos(theta)`` gives the zone between two polar
    angles; its area is ``2 pi r_self^2 (cos(theta_min) - cos(theta_max))``.
    """

    def cos_at(distance):
        value = (
            center_distance**2
            + radius_self**2
            - (distance + radius_partner) ** 2
        ) / (2.0 * center_distance * radius_self)
        return float(np.clip(value, -1.0, 1.0))

    return (
        2.0
        * np.pi
        * radius_self**2
        * (cos_at(min_distance) - cos_at(max_distance))
    )


def _total_area(patches):
    return float(sum(patch["area"] for patch in patches))


def _assert_area_contract(patches, expected, side):
    assert len(patches) == 1, f"{side}: expected one patch, got {len(patches)}"
    patch = patches[0]
    assert patch["truncated"] is False
    assert np.all(patch["distances"] >= -1e-12)
    assert _total_area(patches) == pytest.approx(
        expected, rel=AREA_RTOL
    ), f"{side}: area {_total_area(patches)} vs analytic {expected}"


def _sphere_pair(radius_a, radius_b, gap, subdivisions=REFERENCE_SUBDIVISIONS):
    distance = radius_a + radius_b + gap
    mesh_a = _icosphere(radius_a, (0.0, 0.0, 0.0), subdivisions)
    mesh_b = _icosphere(radius_b, (0.0, 0.0, distance), subdivisions)
    return mesh_a, mesh_b, distance


# ---------------------------------------------------------------------------
# parallel triangulated planes: exact area, no boundary effects
# ---------------------------------------------------------------------------


def test_parallel_planes_contact_area_is_exactly_l_squared():
    size = 2.0
    gap = 0.05
    band_max = 0.1
    # lower plane faces +z, upper plane is flipped to face -z (toward lower)
    lower = _plane(size, z=0.0)
    upper = _plane(size, z=gap, flip=True)

    patches_lower, patches_upper = pair_patches(
        "lower", "upper", lower, upper, 0.0, band_max
    )

    for patches, side in ((patches_lower, "lower"), (patches_upper, "upper")):
        assert len(patches) == 1
        # the analytic planes are open and every triangle touches the rim
        assert patches[0]["truncated"] is True
        assert np.allclose(patches[0]["distances"], gap)
        assert patches[0]["area"] == pytest.approx(
            size * size, rel=PLANE_RTOL
        )


# ---------------------------------------------------------------------------
# spheres: closed-form in-band zone area on each side
# ---------------------------------------------------------------------------


def test_sphere_equal_radii_matches_analytic_zone_on_both_sides():
    radius = 1.0
    mesh_a, mesh_b, distance = _sphere_pair(radius, radius, gap=0.05)

    patches_a, patches_b = pair_patches(
        "a", "b", mesh_a, mesh_b, 0.0, 0.1
    )

    expected = _zone_area(radius, radius, distance, 0.0, 0.1)
    _assert_area_contract(patches_a, expected, "a")
    _assert_area_contract(patches_b, expected, "b")


def test_sphere_unequal_radii_matches_analytic_zone_per_side():
    radius_a, radius_b = 1.0, 0.4
    mesh_a, mesh_b, distance = _sphere_pair(radius_a, radius_b, gap=0.05)

    patches_a, patches_b = pair_patches(
        "a", "b", mesh_a, mesh_b, 0.0, 0.1
    )

    expected_a = _zone_area(radius_a, radius_b, distance, 0.0, 0.1)
    expected_b = _zone_area(radius_b, radius_a, distance, 0.0, 0.1)

    _assert_area_contract(patches_a, expected_a, "a")
    _assert_area_contract(patches_b, expected_b, "b")
    # the sides are independent: a bump's cap is not forced to match the sheet
    assert _total_area(patches_a) != pytest.approx(_total_area(patches_b))


def test_sphere_annular_band_matches_analytic_zone():
    radius = 1.0
    mesh_a, mesh_b, distance = _sphere_pair(radius, radius, gap=0.05)

    # gap 0.05 lies below min_distance 0.06, so the closest cap is excluded
    # and the in-band region is an annulus.
    patches_a, patches_b = pair_patches(
        "a", "b", mesh_a, mesh_b, 0.06, 0.1
    )

    expected = _zone_area(radius, radius, distance, 0.06, 0.1)
    _assert_area_contract(patches_a, expected, "a")
    _assert_area_contract(patches_b, expected, "b")

    distances = np.concatenate([p["distances"] for p in patches_a])
    assert np.all(distances >= 0.06 - 1e-9)
    assert np.all(distances <= 0.1 + 1e-9)


# ---------------------------------------------------------------------------
# missed-pair regression: the old vertex-to-vertex gate skipped this pair
# ---------------------------------------------------------------------------


def test_small_gap_icospheres_are_found_by_gate_and_areas_are_analytic():
    radius = 1.0
    gap = 0.05
    mesh_a, mesh_b, distance = _sphere_pair(radius, radius, gap)

    pair, upper, lower = get_min_dist(("a", "b", mesh_a, mesh_b, 0.1))
    assert pair == ("a", "b")
    # the rigorous lower bound gates the pair in; the upper bound is the
    # (in-band) vertex-to-surface minimum
    assert 0.0 <= lower <= 0.1
    assert lower <= upper <= 0.1

    patches_a, patches_b = pair_patches(
        "a", "b", mesh_a, mesh_b, 0.0, 0.1
    )
    expected = _zone_area(radius, radius, distance, 0.0, 0.1)
    _assert_area_contract(patches_a, expected, "a")
    _assert_area_contract(patches_b, expected, "b")


# ---------------------------------------------------------------------------
# refinement convergence: no systematic drift away from the closed form
# ---------------------------------------------------------------------------


def test_area_converges_under_input_mesh_refinement():
    radius = 1.0
    distance = 2.0 * radius + 0.05
    expected = _zone_area(radius, radius, distance, 0.0, 0.1)

    errors = []
    signed_errors = []
    try:
        for subdivisions in (2, 3, 4, REFERENCE_SUBDIVISIONS):
            mesh_a = _icosphere(radius, (0.0, 0.0, 0.0), subdivisions)
            mesh_b = _icosphere(radius, (0.0, 0.0, distance), subdivisions)
            patches_a, patches_b = pair_patches(
                "a", "b", mesh_a, mesh_b, 0.0, 0.1
            )
            # both sides must refine together; the equal spheres make their
            # error identical up to floating-point summation order
            signed = (_total_area(patches_a) - expected) / expected
            signed_errors.append(signed)
            errors.append(abs(signed))
            assert _total_area(patches_b) == pytest.approx(
                _total_area(patches_a), rel=1e-12
            )
            clear_geometry_cache()
    finally:
        clear_geometry_cache()

    # every refinement level reduces the discretisation error
    assert all(later < earlier for earlier, later in itertools.pairwise(errors))
    assert errors[-1] <= AREA_RTOL
    # the residual bias is a small, shrinking underestimate, not a drift
    assert all(value < 0.0 for value in signed_errors)
    assert -AREA_RTOL <= signed_errors[-1] <= 0.0


# ---------------------------------------------------------------------------
# swap symmetry at the accuracy level
# ---------------------------------------------------------------------------


def test_swap_symmetry_preserves_per_side_analytic_areas():
    radius_a, radius_b = 1.0, 0.4
    mesh_a, mesh_b, distance = _sphere_pair(radius_a, radius_b, gap=0.05)
    expected_a = _zone_area(radius_a, radius_b, distance, 0.0, 0.1)
    expected_b = _zone_area(radius_b, radius_a, distance, 0.0, 0.1)

    forward_a, forward_b = pair_patches(
        "a", "b", mesh_a, mesh_b, 0.0, 0.1
    )
    swapped_b, swapped_a = pair_patches(
        "b", "a", mesh_b, mesh_a, 0.0, 0.1
    )

    for forward, swapped, expected, side in (
        (forward_a, swapped_a, expected_a, "a"),
        (forward_b, swapped_b, expected_b, "b"),
    ):
        assert len(forward) == len(swapped) == 1
        assert forward[0]["area"] == pytest.approx(swapped[0]["area"], rel=1e-12)
        assert forward[0]["area"] == pytest.approx(expected, rel=AREA_RTOL)
        assert forward[0]["truncated"] == swapped[0]["truncated"]
        assert side in forward[0]["patch_id"]


# ---------------------------------------------------------------------------
# real-data orientation: meshes exported from the example project
# ---------------------------------------------------------------------------

#: candidate raw segmentations for the exported chunk meshes, in preference
#: order; the exported chunks of ``example_analysis`` stem from the
#: interphase ``old_cebraEM`` data (same volumes as ``data/interphase_4T``)
_RAW_SEGMENTATION_CANDIDATES = {
    "mito": (
        "data/old_cebraEM/Interphase_4T/mito_it00_b0_7_stitched.n5",
        "data/interphase_4T/mito_it00_b0_7_stitched.n5",
        "data/old_cebraEM/Metaphase_dT/mito_it03_b0_7_stitched.n5",
        "data/mitosis_dT/mito_it03_b0_7_stitched.n5",
    ),
    "er": (
        "data/old_cebraEM/Interphase_4T/er_it00_b0_7_stitched.n5",
        "data/interphase_4T/er_it00_b0_7_stitched.n5",
        "data/old_cebraEM/Metaphase_dT/er_it02_b0_7_stitched.n5",
        "data/mitosis_dT/er_it02_b0_7_stitched.n5",
    ),
}


def _find_upwards(*parts):
    for parent in Path(__file__).resolve().parents:
        candidate = parent.joinpath(*parts)
        if candidate.exists():
            return candidate
    return None


def _exported_mesh_files():
    directory = _find_upwards("example_analysis", "exported_meshes")
    if directory is None:
        return []
    return sorted(glob.glob(str(directory / "*" / "*.glb")))


def _non_empty_meshes():
    meshes = []
    for path in _exported_mesh_files():
        mesh = trimesh.load(path, force="mesh", process=False)
        if len(mesh.faces) > 0:
            meshes.append((Path(path), mesh))
    return meshes


def _signed_volume(mesh):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = vertices[np.asarray(mesh.faces, dtype=np.int64)]
    # translation invariant for closed meshes; harmless for the comparison
    triangles = triangles - vertices.mean(axis=0)
    return float(
        np.einsum(
            "ij,ij->i",
            triangles[:, 0],
            np.cross(triangles[:, 1], triangles[:, 2]),
        ).sum()
        / 6.0
    )


def test_exported_real_meshes_have_consistent_outward_winding():
    meshes = _non_empty_meshes()
    if not meshes:
        pytest.skip("example_analysis/exported_meshes not available")

    inconsistent = [path for path, mesh in meshes if not mesh.is_winding_consistent]
    assert not inconsistent, f"inconsistent winding: {inconsistent}"

    watertight = [(path, mesh) for path, mesh in meshes if mesh.is_watertight]
    inverted = [
        path
        for path, mesh in watertight
        if _signed_volume(mesh) <= 0.0
    ]
    assert not inverted, f"negative signed volume (inward): {inverted}"


def _aligned_dataset(mesh, raw_path):
    """Open the raw level whose voxel grid the exported mesh sits on."""
    import z5py

    handle = z5py.File(str(raw_path), "r")
    timepoint = handle["setup0/timepoint0"]
    base_resolution = np.asarray(timepoint.attrs["resolution"], dtype=np.float64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    best = None
    for name in sorted(timepoint.keys()):
        dataset = timepoint[name]
        factors = np.asarray(dataset.attrs["downsamplingFactors"], dtype=np.float64)
        resolution = base_resolution * factors
        coordinates = vertices / resolution
        aligned = np.mean(
            np.all(np.abs(coordinates - np.round(coordinates)) < 0.05, axis=1)
        )
        upper = np.floor(coordinates.max(axis=0)).astype(np.int64) + 1
        if aligned < 0.99 or np.any(upper > np.asarray(dataset.shape)):
            continue
        # a mesh built on a finer level also aligns with the coarser grid;
        # take the coarsest aligned level, which is the one that was meshed
        voxel_size = float(np.mean(resolution))
        if best is None or voxel_size > best[0]:
            best = (voxel_size, dataset, resolution)
    return None if best is None else best[1:]


def _local_outward_fraction(mesh, dataset, resolution, sample=1000, delta=1.0):
    """Fraction of sampled faces whose normal points from filled to empty.

    The raw segmentation is the independent ground truth: the surface bounds
    the labelled volume, so a correct outward normal leaves the mask when
    stepped along the normal and stays inside the mask behind the face.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    shape = np.asarray(dataset.shape, dtype=np.int64)
    lower = np.maximum(np.floor(vertices.min(axis=0) / resolution).astype(int) - 4, 0)
    upper = np.minimum(np.ceil(vertices.max(axis=0) / resolution).astype(int) + 5, shape)
    occupancy = np.asarray(
        dataset[lower[0] : upper[0], lower[1] : upper[1], lower[2] : upper[2]]
    ) > 0

    rng = np.random.default_rng(0)
    chosen = rng.choice(
        len(faces), size=min(sample, len(faces)), replace=False
    )
    triangles = vertices[faces[chosen]]
    centroids = triangles.mean(axis=1)
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 0.0
    normals[valid] /= lengths[valid, None]
    centroids, normals = centroids[valid], normals[valid]

    def at(points):
        indices = np.round(points / resolution).astype(np.int64) - lower
        inside = np.all((indices >= 0) & (indices < occupancy.shape), axis=1)
        result = np.zeros(len(points), dtype=bool)
        good = indices[inside]
        result[inside] = occupancy[good[:, 0], good[:, 1], good[:, 2]]
        return result

    ahead = at(centroids + delta * resolution * normals)
    behind = at(centroids - delta * resolution * normals)
    decisive = ahead != behind
    if not np.any(decisive):
        return 0.0
    return float(np.mean(behind[decisive] & ~ahead[decisive]))


def test_exported_real_mesh_normals_point_into_empty_space():
    meshes = _non_empty_meshes()
    if not meshes:
        pytest.skip("example_analysis/exported_meshes not available")

    workspace = _find_upwards("data")
    if workspace is None:
        pytest.skip("raw segmentation data not available")

    fractions = []
    for path, mesh in meshes:
        kind = path.parent.name
        dataset = None
        for candidate in _RAW_SEGMENTATION_CANDIDATES.get(kind, ()):
            raw_path = _find_upwards(*Path(candidate).parts)
            if raw_path is None:
                continue
            dataset = _aligned_dataset(mesh, raw_path)
            if dataset is not None:
                break
        if dataset is None:
            pytest.skip(f"no aligned raw segmentation for {path}")
        level, resolution = dataset
        fractions.append(_local_outward_fraction(mesh, level, resolution))

    assert fractions
    assert min(fractions) >= 0.9, (
        f"inward normals on some real-data meshes: {sorted(fractions)}"
    )
