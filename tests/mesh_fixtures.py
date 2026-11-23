"""Reusable mesh fixtures for the MCS geometry and record tests."""

import numpy as np
import trimesh
from shapely.geometry import Polygon


def icosphere(radius=1.0, subdivisions=5, center=(0.0, 0.0, 0.0)):
    """An icosphere translated to ``center``.

    Args:
        radius: Sphere radius.
        subdivisions: Icosphere subdivision level.
        center: Translation applied after construction.

    Returns:
        trimesh.Trimesh: Closed triangulated sphere.
    """
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    mesh.apply_translation(center)
    return mesh


def deep_u_mesh():
    """A connected U-shaped mesh with two prongs pointing at a partner.

    The prong tips are close enough that their contact zones merge on a
    facing sphere, but the notch between them runs deep, so the U's own MCS
    regions stay separate. Unlike concatenated boxes this is a single
    connected, watertight surface, so the two U patches are separated by the
    distance band and not by disconnected mesh components.

    Returns:
        trimesh.Trimesh: Watertight U-shaped prism.
    """
    gap = 0.1
    width = 0.4
    z_top = -1.03
    z_notch = -2.0
    z_bottom = -3.0
    depth = 0.2
    half_width = width / 2
    half_gap = gap / 2
    outline = [
        (-half_width, z_bottom),
        (-half_width, z_top),
        (-half_gap, z_top),
        (-half_gap, z_notch),
        (half_gap, z_notch),
        (half_gap, z_top),
        (half_width, z_top),
        (half_width, z_bottom),
    ]
    u_mesh = trimesh.creation.extrude_polygon(Polygon(outline), height=depth)
    # local (x, y, z) -> world (x, z, y): the outline's y is world z
    u_mesh.apply_transform(
        np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
            dtype=float,
        )
    )
    u_mesh.apply_translation((0.0, depth / 2, 0.0))
    assert u_mesh.is_watertight
    return u_mesh
