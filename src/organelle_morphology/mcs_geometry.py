"""Symmetric per-side geometry for membrane contact sites (MCSs).

This module contains pure geometry only: no dask, no I/O and no mesh repair.
An MCS is an edge-connected patch on *one* organelle's own membrane, whose
surface points lie within a distance band to a partner organelle's surface.
Both sides of an apposition are computed by the same code path, so argument
order does not change the result.

The public interface is intentionally small; tickets 02-05 build on it:

``MeshGeometry``
    Immutable-ish bundle of the per-mesh data needed by the MCS calculation:
    vertices, faces, cached ``igl.AABB`` tree, face normals, face areas, the
    sampling radius and open boundary (rim) edges.

``get_mesh_geometry(mesh)``
    Lazily builds and caches ``MeshGeometry`` for a ``trimesh.Trimesh``.
    The cache is keyed by mesh identity and can be dropped with
    ``clear_geometry_cache()``.

``sampling_radius(V, F)``
    Maximum covering radius of a face over the mesh. Used by the pair gate
    (ticket 02) to turn a vertex-to-surface minimum into a rigorous lower
    bound.

``vertex_surface_distances(geom_self, geom_partner)``
    Exact point-to-surface distance from every vertex of ``geom_self`` to the
    partner surface, using the partner's cached AABB tree.

``pair_patches(...)``
    Classify both sides' surface points against the partner, refine faces that
    straddle a band/facing boundary, group member (sub-)faces into connected
    patches and return one patch list per side.

Design notes (see docs/adr/0002-mcs-per-organelle-surface-distance.md):

- Face normals are used as meshed (winding is trusted). The facing test for a
  sample point ``p`` with incident face normal ``n`` is
  ``dot(n, C - p) > 0`` where ``C`` is the closest point on the partner
  surface. No global normal-sign heuristic and no mesh repair is applied.
- A face is counted whole when all three vertices and the centroid classify as
  member. Faces that straddle a band edge or a facing sign are midpoint
  subdivided (4 children) down to ``subdivision_depth``; each leaf is
  classified by its centroid and contributes its centroid distance as a
  sample. Interior faces are never subdivided.
- Components are edge-connected sets of member (sub-)faces. Edges shared by
  children of one parent and edges shared by adjacent parent faces are both
  included.
- ``truncated`` is set when a patch has a member (sub-)face within
  ``max_distance`` of an open boundary edge (clipping rim).
"""

from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass

import igl
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass(eq=False)
class MeshGeometry:
    """Per-mesh data shared by all MCS calculations involving this mesh."""

    vertices: np.ndarray
    faces: np.ndarray
    tree: object | None
    face_normals: np.ndarray
    face_areas: np.ndarray
    sampling_radius: float
    boundary_edges: np.ndarray

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        return len(self.faces)


_GEOMETRY_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_CACHE_LOCK = threading.Lock()


def clear_geometry_cache() -> None:
    """Drop all cached per-mesh geometry (trees, normals, rim edges)."""
    with _CACHE_LOCK:
        _GEOMETRY_CACHE.clear()


def sampling_radius(V: np.ndarray, F: np.ndarray) -> float:
    """Maximum covering radius of a face over the mesh.

    For every face the covering radius is the circumradius when the
    circumcentre lies inside the face (acute triangle), otherwise half of the
    longest edge. Degenerate (zero-area) and empty inputs are guarded and
    yield half the longest edge or ``0.0`` respectively.
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    if V.size == 0 or F.size == 0:
        return 0.0
    if F.ndim != 2:
        F = F.reshape(-1, 3)
    tri = V[F]
    p0, p1, p2 = tri[:, 0], tri[:, 1], tri[:, 2]
    # squared edge lengths, each opposite the same-index vertex
    a2 = np.sum((p2 - p1) ** 2, axis=1)
    b2 = np.sum((p0 - p2) ** 2, axis=1)
    c2 = np.sum((p1 - p0) ** 2, axis=1)
    cross = np.cross(p1 - p0, p2 - p0)
    area2 = np.linalg.norm(cross, axis=1)

    longest = np.sqrt(np.maximum(np.maximum(a2, b2), c2))
    # circumcentre inside iff the triangle is acute (right -> longest/2)
    acute = (
        (a2 < b2 + c2)
        & (b2 < a2 + c2)
        & (c2 < a2 + b2)
        & (area2 > 0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        circumradius = np.sqrt(a2 * b2 * c2) / (2.0 * area2)
    radii = np.where(acute, circumradius, longest / 2.0)
    radii = np.where(np.isfinite(radii), radii, longest / 2.0)
    if radii.size == 0:
        return 0.0
    return float(np.max(radii))


def build_mesh_geometry(mesh) -> MeshGeometry:
    """Build a :class:`MeshGeometry` for ``mesh`` (no caching)."""
    vertices = np.ascontiguousarray(np.asarray(mesh.vertices, dtype=np.float64))
    faces = np.ascontiguousarray(np.asarray(mesh.faces, dtype=np.int64))
    if faces.ndim != 2:
        faces = faces.reshape(-1, 3)
    n_faces = len(faces)

    if n_faces == 0 or len(vertices) == 0:
        return MeshGeometry(
            vertices=vertices,
            faces=faces,
            tree=None,
            face_normals=np.zeros((n_faces, 3), dtype=np.float64),
            face_areas=np.zeros(n_faces, dtype=np.float64),
            sampling_radius=0.0,
            boundary_edges=np.empty((0, 2), dtype=np.int64),
        )

    normals = np.ascontiguousarray(igl.per_face_normals(vertices, faces))
    tri = vertices[faces]
    face_areas = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    boundary_edges = np.ascontiguousarray(igl.exterior_edges(faces))

    tree = igl.AABB()
    tree.init(vertices, faces)

    return MeshGeometry(
        vertices=vertices,
        faces=faces,
        tree=tree,
        face_normals=normals,
        face_areas=face_areas,
        sampling_radius=sampling_radius(vertices, faces),
        boundary_edges=boundary_edges,
    )


def get_mesh_geometry(mesh) -> MeshGeometry:
    """Return the cached geometry for ``mesh``, building it on first use."""
    geometry = _GEOMETRY_CACHE.get(mesh)
    if geometry is not None:
        return geometry

    geometry = build_mesh_geometry(mesh)
    with _CACHE_LOCK:
        cached = _GEOMETRY_CACHE.get(mesh)
        if cached is not None:
            return cached
        _GEOMETRY_CACHE[mesh] = geometry
    return geometry


def _query_closest(geom_partner: MeshGeometry, points: np.ndarray):
    """Point-to-surface query against a partner's cached AABB tree."""
    points = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    n_points = len(points)
    if geom_partner.tree is None or geom_partner.n_faces == 0 or n_points == 0:
        return (
            np.full(n_points, np.inf, dtype=np.float64),
            np.full(n_points, -1, dtype=np.int64),
            np.full((n_points, 3), np.nan, dtype=np.float64),
        )
    sqr_d, face_index, closest = geom_partner.tree.squared_distance(
        geom_partner.vertices, geom_partner.faces, points
    )
    distances = np.sqrt(np.maximum(np.asarray(sqr_d, dtype=np.float64), 0.0))
    return (
        distances,
        np.asarray(face_index, dtype=np.int64),
        np.asarray(closest, dtype=np.float64),
    )


def vertex_surface_distances(geom_self: MeshGeometry, geom_partner: MeshGeometry):
    """Exact distance from all own vertices to the partner surface.

    Returns:
        tuple: ``(distances, partner_face_index, closest_points)``, one entry
        per own vertex. Infeasible partners (no faces) yield ``inf``
        distances, ``-1`` face indices and ``nan`` closest points.
    """
    return _query_closest(geom_partner, geom_self.vertices)


def min_dist_upper(geom_a: MeshGeometry, geom_b: MeshGeometry):
    """Bidirectional vertex-to-surface minima between two meshes.

    Each value is an upper bound on the true surface-to-surface minimum
    (error bounded by the sampling radius of the queried mesh). Both
    directions are returned so the pair gate (ticket 02) can build a
    conservative lower bound from them.
    """
    distances_ab, _, _ = vertex_surface_distances(geom_a, geom_b)
    distances_ba, _, _ = vertex_surface_distances(geom_b, geom_a)
    d_ab = float(np.min(distances_ab)) if distances_ab.size else np.inf
    d_ba = float(np.min(distances_ba)) if distances_ba.size else np.inf
    return d_ab, d_ba


# ---------------------------------------------------------------------------
# adaptive subdivision
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class _SubdivisionGrid:
    """Parent-agnostic midpoint grid used for adaptive face refinement."""

    centroid_weights: np.ndarray  # (n_leaf, 3) barycentric leaf centroids
    touches_edge: np.ndarray  # (n_leaf, 3) leaf has an edge on parent edge e
    edge_pairs: np.ndarray  # (n_pairs, 2) sibling leaf adjacency


_GRID_CACHE: dict[int, _SubdivisionGrid] = {}


def _unique_rows_inverse(rows: np.ndarray) -> np.ndarray:
    _, inverse = np.unique(rows, axis=0, return_inverse=True)
    return inverse.reshape(-1)


def _subdivision_grid(depth: int) -> _SubdivisionGrid:
    """Return (and cache) the 4-way midpoint grid for ``depth``."""
    depth = max(int(depth), 0)
    cached = _GRID_CACHE.get(depth)
    if cached is not None:
        return cached

    # corner barycentric coordinates are integer triples summing to 2**depth
    corners = np.array(
        [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=np.int64
    )
    for _ in range(depth):
        a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
        ab, ac, bc = a + b, a + c, b + c
        corners = np.concatenate(
            [
                np.stack([2 * a, ab, ac], axis=1),
                np.stack([ab, 2 * b, bc], axis=1),
                np.stack([ac, bc, 2 * c], axis=1),
                np.stack([ab, bc, ac], axis=1),
            ],
            axis=0,
        )

    scale = 1 << depth
    centroid_weights = corners.mean(axis=1) / float(scale)

    # a leaf lies on parent edge e when two of its vertices have coordinate 0
    touches_edge = (corners == 0).sum(axis=1) >= 2

    # sibling adjacency: leaves sharing an edge of the refined grid
    local_vertices = _unique_rows_inverse(corners.reshape(-1, 3))
    local_vertices = local_vertices.reshape(-1, 3)
    n_local_vertices = int(local_vertices.max()) + 1
    leaf_edges = np.concatenate(
        [
            local_vertices[:, [0, 1]],
            local_vertices[:, [1, 2]],
            local_vertices[:, [2, 0]],
        ],
        axis=0,
    )
    leaf_edges.sort(axis=1)
    leaf_of_edge = np.tile(np.arange(len(corners), dtype=np.int64), 3)
    keys = leaf_edges[:, 0] * n_local_vertices + leaf_edges[:, 1]
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    leaves = leaf_of_edge[order]
    same = keys[1:] == keys[:-1]
    edge_pairs = np.stack([leaves[:-1][same], leaves[1:][same]], axis=1)

    grid = _SubdivisionGrid(
        centroid_weights=centroid_weights,
        touches_edge=touches_edge,
        edge_pairs=edge_pairs,
    )
    _GRID_CACHE[depth] = grid
    return grid


# ---------------------------------------------------------------------------
# point-to-surface classification and connected patches
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class _PatchData:
    faces_index: np.ndarray
    distances: np.ndarray
    area: float
    truncated: bool
    partner_faces: np.ndarray
    min_parent: int
    min_node: int


@dataclass(eq=False)
class _SideResult:
    patches: list[_PatchData]
    face_to_patch: np.ndarray


def _points_near_segments(
    points: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    max_distance: float,
) -> np.ndarray:
    """Boolean mask: which points lie within ``max_distance`` of a segment."""
    n_points = len(points)
    near = np.zeros(n_points, dtype=bool)
    if n_points == 0 or len(seg_a) == 0:
        return near

    ab = seg_b - seg_a
    ab_len2 = np.maximum(np.sum(ab * ab, axis=1), np.finfo(np.float64).tiny)

    # keep the broadcast arrays bounded for very large rims / patches
    max_pairs = 2_000_000
    chunk = max(1, max_pairs // len(seg_a))
    threshold2 = float(max_distance) ** 2
    for start in range(0, n_points, chunk):
        stop = min(start + chunk, n_points)
        ap = points[start:stop, None, :] - seg_a[None, :, :]
        t = np.einsum("kej,ej->ke", ap, ab) / ab_len2[None, :]
        t = np.clip(t, 0.0, 1.0)
        closest = seg_a[None, :, :] + t[:, :, None] * ab[None, :, :]
        delta = points[start:stop, None, :] - closest
        dist2 = np.einsum("kej,kej->ke", delta, delta)
        near[start:stop] = np.min(dist2, axis=1) <= threshold2
    return near


def _side_components(
    geom_self: MeshGeometry,
    geom_partner: MeshGeometry,
    min_distance: float,
    max_distance: float,
    depth: int,
) -> _SideResult:
    """Classify one side's faces against the partner and build patches."""
    faces = geom_self.faces
    vertices = geom_self.vertices
    n_faces = len(faces)
    face_to_patch = np.full(n_faces, -1, dtype=np.int64)
    if n_faces == 0 or geom_partner.n_faces == 0:
        return _SideResult(patches=[], face_to_patch=face_to_patch)

    # --- classify own vertex samples against the partner surface ----------
    vertex_distances, _, vertex_closest = vertex_surface_distances(
        geom_self, geom_partner
    )
    tri = vertices[faces]
    centroids = tri.mean(axis=1)
    centroid_distances, centroid_partner_faces, centroid_closest = (
        _query_closest(geom_partner, centroids)
    )

    normals = geom_self.face_normals
    band_vertices = (vertex_distances >= min_distance) & (
        vertex_distances <= max_distance
    )
    facing_vertices = (
        np.einsum(
            "fj,fkj->fk",
            normals,
            vertex_closest[faces] - vertices[faces],
        )
        > 0.0
    )
    member_vertices = band_vertices[faces] & facing_vertices

    facing_centroids = (
        np.einsum("fj,fj->f", normals, centroid_closest - centroids) > 0.0
    )
    member_centroids = (
        (centroid_distances >= min_distance)
        & (centroid_distances <= max_distance)
        & facing_centroids
    )

    all_member = member_vertices.all(axis=1) & member_centroids
    any_member = member_vertices.any(axis=1) | member_centroids
    straddling = any_member & ~all_member

    whole_faces = np.flatnonzero(all_member)
    subdivided_faces = np.flatnonzero(straddling)
    grid = _subdivision_grid(depth)
    leaves_per_face = len(grid.centroid_weights)

    # --- assemble leaves (one per whole face, 4**depth per split face) -----
    n_whole = len(whole_faces)
    parent_blocks = []
    member_blocks = []
    distance_blocks = []
    area_blocks = []
    partner_face_blocks = []
    point_blocks = []

    if n_whole:
        parent_blocks.append(whole_faces)
        member_blocks.append(np.ones(n_whole, dtype=bool))
        distance_blocks.append(centroid_distances[whole_faces])
        area_blocks.append(geom_self.face_areas[whole_faces])
        partner_face_blocks.append(centroid_partner_faces[whole_faces])
        point_blocks.append(centroids[whole_faces])

    max_points = 1_000_000
    chunk_faces = max(1, max_points // leaves_per_face)
    sub_member_blocks = []
    for start in range(0, len(subdivided_faces), chunk_faces):
        idx = subdivided_faces[start : start + chunk_faces]
        points = np.einsum(
            "lk,fkj->flj", grid.centroid_weights, tri[idx]
        )
        distances, partner_faces, closest = _query_closest(
            geom_partner, points.reshape(-1, 3)
        )
        distances = distances.reshape(len(idx), leaves_per_face)
        partner_faces = partner_faces.reshape(len(idx), leaves_per_face)
        closest = closest.reshape(len(idx), leaves_per_face, 3)
        facing = (
            np.einsum("fj,flj->fl", normals[idx], closest - points) > 0.0
        )
        member = (
            (distances >= min_distance)
            & (distances <= max_distance)
            & facing
        )
        parent_blocks.append(np.repeat(idx, leaves_per_face))
        member_blocks.append(member.reshape(-1))
        sub_member_blocks.append(member.reshape(-1))
        distance_blocks.append(distances.reshape(-1))
        area_blocks.append(
            np.repeat(
                geom_self.face_areas[idx] / leaves_per_face, leaves_per_face
            )
        )
        partner_face_blocks.append(partner_faces.reshape(-1))
        point_blocks.append(points.reshape(-1, 3))

    parent = np.concatenate(parent_blocks)
    member = np.concatenate(member_blocks)
    distance = np.concatenate(distance_blocks)
    area = np.concatenate(area_blocks)
    partner_face = np.concatenate(partner_face_blocks)
    sample_points = np.concatenate(point_blocks)
    n_nodes = len(parent)

    # --- edges between sibling leaves of a subdivided parent --------------
    src_blocks = []
    dst_blocks = []
    if len(subdivided_faces) and len(grid.edge_pairs):
        base = n_whole + np.arange(len(subdivided_faces)) * leaves_per_face
        src = base[:, None, None] + grid.edge_pairs[None, :, 0]
        dst = base[:, None, None] + grid.edge_pairs[None, :, 1]
        src = src.reshape(-1)
        dst = dst.reshape(-1)
        keep = member[src] & member[dst]
        src_blocks.append(src[keep])
        dst_blocks.append(dst[keep])

    # --- edges shared by adjacent parent faces ----------------------------
    # column e is the parent edge opposite vertex e, matching touches_edge[:, e]
    edge_next = faces[:, [1, 2, 0]]
    edge_prev = faces[:, [2, 0, 1]]
    edge_lo = np.minimum(edge_next, edge_prev)
    edge_hi = np.maximum(edge_next, edge_prev)
    edge_keys = edge_lo.astype(np.int64) * len(vertices) + edge_hi

    incidence_keys = []
    incidence_nodes = []
    if n_whole:
        incidence_keys.append(edge_keys[whole_faces].reshape(-1))
        incidence_nodes.append(
            np.repeat(np.arange(n_whole, dtype=np.int64), 3)
        )
    if len(subdivided_faces) and leaves_per_face:
        sub_member = np.concatenate(sub_member_blocks)
        touches = (
            np.repeat(
                grid.touches_edge[None, :, :],
                len(subdivided_faces),
                axis=0,
            )
            & sub_member.reshape(-1, leaves_per_face, 1)
        )
        face_pos, leaf_pos, edge_pos = np.nonzero(touches)
        incidence_keys.append(
            edge_keys[subdivided_faces][face_pos, edge_pos]
        )
        incidence_nodes.append(
            n_whole + face_pos * leaves_per_face + leaf_pos
        )

    if incidence_keys:
        keys = np.concatenate(incidence_keys)
        nodes = np.concatenate(incidence_nodes)
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        nodes = nodes[order]
        same = keys[1:] == keys[:-1]
        src_blocks.append(nodes[:-1][same])
        dst_blocks.append(nodes[1:][same])

    if src_blocks:
        src = np.concatenate(src_blocks)
        dst = np.concatenate(dst_blocks)
        graph = coo_matrix(
            (np.ones(len(src), dtype=np.int8), (src, dst)),
            shape=(n_nodes, n_nodes),
        )
        _, labels = connected_components(graph, directed=False)
    else:
        labels = np.arange(n_nodes, dtype=np.int64)

    # --- patches ----------------------------------------------------------
    member_nodes = np.flatnonzero(member)
    if member_nodes.size == 0:
        return _SideResult(patches=[], face_to_patch=face_to_patch)

    member_labels = labels[member_nodes]
    order = np.argsort(member_labels, kind="stable")
    member_labels = member_labels[order]
    member_nodes = member_nodes[order]
    splits = np.flatnonzero(np.diff(member_labels)) + 1
    component_nodes = np.split(member_nodes, splits)

    near_boundary = _points_near_segments(
        sample_points[member_nodes],
        geom_self.vertices[geom_self.boundary_edges[:, 0]]
        if len(geom_self.boundary_edges)
        else np.empty((0, 3)),
        geom_self.vertices[geom_self.boundary_edges[:, 1]]
        if len(geom_self.boundary_edges)
        else np.empty((0, 3)),
        max_distance,
    )
    near_all_nodes = np.zeros(n_nodes, dtype=bool)
    near_all_nodes[member_nodes] = near_boundary

    patches = []
    for nodes in component_nodes:
        parents = parent[nodes]
        patches.append(
            _PatchData(
                faces_index=np.unique(parents),
                distances=distance[nodes],
                area=float(np.sum(area[nodes])),
                truncated=bool(np.any(near_all_nodes[nodes])),
                partner_faces=np.unique(partner_face[nodes]),
                min_parent=int(np.min(parents)),
                min_node=int(np.min(nodes)),
            )
        )
    patches.sort(key=lambda p: (p.min_parent, p.min_node))

    for patch_index, patch in enumerate(patches):
        free = face_to_patch[patch.faces_index] == -1
        face_to_patch[patch.faces_index[free]] = patch_index

    return _SideResult(patches=patches, face_to_patch=face_to_patch)


def _finish_patches(
    self_id: str,
    partner_id: str,
    side: _SideResult,
    partner: _SideResult,
) -> list[dict]:
    """Attach deterministic ids and cross-side links to a side's patches."""
    finished = []
    for index, patch in enumerate(side.patches):
        partner_indices = partner.face_to_patch[patch.partner_faces]
        partner_indices = np.unique(
            partner_indices[partner_indices >= 0]
        )
        finished.append(
            {
                "patch_id": f"{self_id}~{partner_id}~{index}",
                "faces_index": patch.faces_index,
                "distances": patch.distances,
                "area": patch.area,
                "truncated": patch.truncated,
                "partner_patch_ids": tuple(
                    f"{partner_id}~{self_id}~{int(i)}"
                    for i in partner_indices
                ),
            }
        )
    return finished


def pair_patches(
    self_id: str,
    partner_id: str,
    mesh_self,
    mesh_partner,
    min_distance: float,
    max_distance: float,
    subdivision_depth: int = 3,
):
    """Compute the MCS patches of both sides of an apposition.

    Both sides are classified independently by the same code path: every own
    sample point is tested against the partner's exact surface distance and
    the incident face normal. Faces that straddle a band or facing boundary
    are midpoint subdivided to ``subdivision_depth``.

    Args:
        self_id: id of the first organelle (used in its ``patch_id``).
        partner_id: id of the second organelle.
        mesh_self: first organelle's ``trimesh.Trimesh``.
        mesh_partner: second organelle's ``trimesh.Trimesh``.
        min_distance: lower bound of the distance band (inclusive).
        max_distance: upper bound of the distance band (inclusive).
        subdivision_depth: maximum midpoint refinement depth (default 3).

    Returns:
        tuple[list[dict], list[dict]]: patches of the first and the second
        side. Every patch has the keys ``patch_id``, ``faces_index``,
        ``distances``, ``area``, ``truncated`` and ``partner_patch_ids``.
    """
    depth = max(int(subdivision_depth), 0)
    geom_self = get_mesh_geometry(mesh_self)
    geom_partner = get_mesh_geometry(mesh_partner)

    side_self = _side_components(
        geom_self, geom_partner, min_distance, max_distance, depth
    )
    side_partner = _side_components(
        geom_partner, geom_self, min_distance, max_distance, depth
    )

    return (
        _finish_patches(self_id, partner_id, side_self, side_partner),
        _finish_patches(partner_id, self_id, side_partner, side_self),
    )
