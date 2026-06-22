from trimesh import Trimesh
import trimesh

from dask.delayed import delayed
import numpy as np
from zmesh import Mesher


@delayed(nout=2)
def block_mesher(
    block,
    space_offset: tuple[int, ...],
    reduction_factor=0,
    debug_color=0,
    scaling_factors=(1, 1, 1),
):
    """Delayed function to calculate meshes from label data

    Args:
        block: 3D array containing labels
        space_offset: 3D tuple containing the offset of the current block
            in relation to the whole data. The vertices are therefore *not*
            local to the block, but can directly be merged with other blocks.

    Returns:
        meshes: Dict with labels as keys and meshes as values.
        labels: list of labels present in this block.

    """

    def plane(origin, normal):
        """Plane for debug view"""
        normal_i = np.array(normal, dtype=bool)
        planeverts = np.stack([origin] * 4)
        # axis along which to strech from origin to plane
        for i, axis in enumerate((~normal_i).nonzero()[0]):
            if i == 0:
                planeverts[0, axis] += 50
                planeverts[1, axis] += 50
                planeverts[2, axis] -= 50
                planeverts[3, axis] -= 50

            else:
                planeverts[0, axis] += 50
                planeverts[2, axis] += 50
                planeverts[1, axis] -= 50
                planeverts[3, axis] -= 50

        plane = Trimesh(planeverts, [[0, 1, 2], [3, 2, 1]])
        plane += Trimesh(planeverts, [[2, 1, 0], [1, 2, 3]])

        plane.visual.vertex_colors = (0, 0, 0, 40)
        plane.visual.vertex_colors[:, [normal_i.nonzero()[0][0]]] = 200

        return plane

    # calculate slice planes, assuming overlap of 2 on all sides
    bs = np.array(block.shape)
    overlap = 2
    slice_points = []
    slice_normals = []

    for i in range(3):
        lower = bs.copy() / 2
        lower[i] = overlap
        slice_points.append(lower)
        dir = np.zeros((3,), dtype=int)
        dir[i] = 1
        slice_normals.append(dir)

        upper = bs.copy() / 2
        upper[i] = bs[i] - overlap
        slice_points.append(upper)
        dir = np.zeros((3,), dtype=int)
        dir[i] = -1
        slice_normals.append(dir)

    mesher = Mesher((1, 1, 1))
    mesher.mesh(block, close=False)
    meshes = {}

    for id in mesher.ids():
        assert meshes.get(id) is None, f"{id} was in mesh already!"
        mesh = mesher.get(
            id,
            normals=False,
            reduction_factor=reduction_factor,
            voxel_centered=False,
            max_error=None,  # None: max 1 voxel, otherwise unit of data
        )
        mesh = Trimesh(mesh.vertices, mesh.faces, process=False)

        parts = []
        masks = []
        if debug_color:
            mesh.visual.vertex_colors[:, :] = trimesh.visual.random_color()
            mesh.visual.vertex_colors[:, 3] = 120

        planes = []
        for origin, normal in zip(slice_points, slice_normals):
            mesh_slice = mesh.slice_plane(origin, normal)
            assert mesh_slice is not None
            if mesh_slice.vertices.shape[0] == 0 and not debug_color:
                mesh = mesh_slice
                continue

            if debug_color:
                planes.append(plane(origin, normal))
                if (
                    mesh_slice.vertices.shape[0] != mesh.vertices.shape[0]
                    or mesh_slice.faces.shape[0] != mesh.faces.shape[0]
                ):
                    normal_i = np.array(normal, dtype=bool)
                    mask = np.logical_and(
                        mesh.vertices[:, normal_i] < (origin[normal_i] + 1),
                        mesh.vertices[:, normal_i] > (origin[normal_i] - 1),
                    ).nonzero()[0]
                    masks.append(mask)

                    # Add removed slice in pink
                    mesh_outer = mesh.slice_plane(origin, normal * -1)
                    mesh_outer.visual.vertex_colors = (255, 0, 255, 100)
                    parts.append(mesh_outer)

                    if debug_color >= 3:
                        mesh.visual.vertex_colors[:, :3] = (
                            mesh.vertices / (mesh.vertices.max(axis=0)) * 200
                        )
                        mesh.visual.vertex_colors[:, 3] = 200
            else:
                # Debug views contain the non-sliced mesh!
                mesh = mesh_slice
        if masks:
            mask = np.concatenate(masks)
            mesh.visual.vertex_colors[mask] = (0, 255, 0, 180)
        if parts:
            mesh += sum(parts)

        if mesh.vertices.shape[0] > 0:
            mesh.vertices += space_offset
            meshes[id] = mesh

    if debug_color >= 1:
        try:
            i, m = meshes.popitem()

            verts = np.array(
                [
                    [0, 0, 0],
                    [bs[0], 0, 0],
                    [0, bs[1], 0],
                    [0, 0, bs[2]],
                    [bs[0], bs[1], 0],
                    [0, bs[1], bs[2]],
                    [bs[0], 0, bs[2]],
                    [bs[0], bs[1], bs[2]],
                ]
            )
            verts += space_offset
            for v in verts:
                m += trimesh.primitives.Sphere(3, v, subdivisions=1, mutable=True)
            if debug_color >= 2:
                pl = sum(planes)
                pl.vertices += space_offset
                m += pl
            meshes[i] = m
        except KeyError:
            pass

    for label in meshes:
        meshes[label].vertices *= scaling_factors

    return meshes, list(meshes.keys())
