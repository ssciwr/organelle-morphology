"""Mesh topology checks shared by the regression tests (issue #20)."""

import numpy as np


def count_degenerate_faces(mesh) -> int:
    """Number of faces with a repeated vertex index (zero-area faces)."""
    f = np.asarray(mesh.faces)
    return int(
        ((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])).sum()
    )


def count_non_manifold_edges(mesh) -> int:
    """Number of edges shared by more than two faces."""
    edges = np.sort(np.asarray(mesh.edges), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int((counts > 2).sum())


def assert_watertight(mesh):
    """Assert that a mesh is closed and manifold.

    Welding mesh fragments must not create degenerate faces or non-manifold
    edges, otherwise trimesh reports the mesh as not watertight.
    """
    assert count_degenerate_faces(mesh) == 0
    assert count_non_manifold_edges(mesh) == 0
    assert mesh.is_watertight
