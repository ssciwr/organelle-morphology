import logging
from multiprocessing import Pool
from time import time

import numpy as np
import pandas as pd
from dask.base import compute
from dask.delayed import delayed
from dask.distributed import span
import os

import organelle_morphology
from organelle_morphology.mcs_geometry import (
    get_mesh_geometry,
    min_dist_upper,
    pair_patches,
)
from organelle_morphology.util import (
    block_to_coords,
    boxes_overlap,
    get_neighboring_chunks,
)

logger = logging.getLogger(__name__)

# chunk domains containing more organelles than this are decomposed further
# in `generate_distance_matrix` (chunk_dd branch)
MAX_ORGS_PER_DOMAIN = 500


def get_min_dist(args):
    id_1, id_2, mesh_1, mesh_2, max_dist = args
    geometry_1 = get_mesh_geometry(mesh_1)
    geometry_2 = get_mesh_geometry(mesh_2)
    d_1, d_2 = min_dist_upper(geometry_1, geometry_2)
    value = min(d_1, d_2)
    # Keep the historical matrix semantics: pairs beyond max_dist stay
    # uncomputed (-1) instead of storing an out-of-band distance.
    if max_dist is not None and value > max_dist:
        value = np.nan
    return (id_1, id_2), value


def _check_overlap(box, bounding_boxes):
    is_in = np.empty((len(bounding_boxes),), dtype=bool)
    for i, org_bb in enumerate(bounding_boxes):
        is_in[i] = boxes_overlap(box, org_bb)

    return is_in


@delayed
def delayed_domain_min_dists(args):
    local_meshes, local_ids, max_dist = args
    tasks = []

    for i in range(len(local_meshes)):
        mesh_1 = local_meshes[i]
        id_1 = local_ids[i]
        for j in range(i + 1, len(local_meshes)):
            mesh_2 = local_meshes[j]
            id_2 = local_ids[j]

            tasks.append(get_min_dist((id_1, id_2, mesh_1, mesh_2, max_dist)))
    return tasks


def chunk_domain_masks(organelles, domain_idxs, dshape):
    """Assign organelles to chunk domains.

    A domain covers one chunk of the data and its direct neighbors. Domains
    without an organelle in their center chunk are dropped.

    Args:
        organelles: organelles to distribute over the domains.
        domain_idxs: chunk indices of all chunks of the data.
        dshape: shape of the chunk grid.

    Returns:
        list[tuple[list[bool], list[tuple[int, ...]]]]: for each kept domain,
            the organelle mask and the chunk indices the domain covers. Mask
            and chunks are paired, so the mask index always refers to the
            domain it was built from, even if domains are dropped or reordered.
    """
    domain_masks = []
    for idx in domain_idxs:
        neighbor_chunks = get_neighboring_chunks(idx, dshape)
        has_org_in_center = False
        mask = []
        for org in organelles:
            # in central block of domain:
            if idx in org.chunks:
                mask.append(True)
                has_org_in_center = True
            # in neighbor block of domain:
            elif any([nb in org.chunks for nb in neighbor_chunks]):
                mask.append(True)
            # not in domain:
            else:
                mask.append(False)
        if has_org_in_center:
            domain_masks.append((mask, neighbor_chunks))
    return domain_masks


def make_domains(
    size: np.ndarray,
    max_dist: float,
    start_corner: np.ndarray,
    meshes: np.ndarray,
    client,
):
    """Decompose space into domains

    Args:
        size: max size of space to decompose.
        max_dist: maximal contact distance which can be computed reliably.
        start_corner: lowest corner of axis aligned bounding box of space.
        meshes: list of meshes within the space to decompose.

    Returns:
        list[list[bool]]: binary masks
    """

    stride = max_dist * 100
    n_x_cubes = min(max(int(size[0] // stride), 1), 20)
    n_y_cubes = min(max(int(size[1] // stride), 1), 20)
    n_z_cubes = min(max(int(size[2] // stride), 1), 20)
    cube_size = size / np.array([n_x_cubes, n_y_cubes, n_z_cubes])
    cube_size += 2 * max_dist

    xs = np.linspace(
        start_corner[0],
        size[0] + start_corner[0],
        n_x_cubes,
        endpoint=False,
    )
    ys = np.linspace(
        start_corner[1],
        size[1] + start_corner[1],
        n_y_cubes,
        endpoint=False,
    )
    zs = np.linspace(
        start_corner[2],
        size[2] + start_corner[2],
        n_z_cubes,
        endpoint=False,
    )

    xg, yg, zg = np.meshgrid(xs, ys, zs)
    xg, yg, zg = list(map(lambda a: a.flatten(), (xg, yg, zg)))

    logger.debug(f"Cube size: {cube_size}")
    logger.debug(f"n cubes: {n_x_cubes}, {n_y_cubes}, {n_z_cubes} => {len(xg)}")

    logger.debug("Calculating bounding boxes")
    bounding_boxes = []
    with span("dist_matrix_bounding_boxes"):
        # Persist meshes into distributed cluster memory because Windows can't fork
        persisted_meshes = client.persist(list(meshes))
        # with many meshes (20k) ~30% faster then computing directly:
        bounding_boxes = client.gather(
            client.map(lambda m: m.compute().bounding_box.bounds, persisted_meshes)
        )
    logger.debug(f"bounding_boxes {len(bounding_boxes)}")
    logger.debug("bounding boxes finished, stating overlap calculations")

    tasks = []
    for x, y, z in zip(xg, yg, zg):
        start = np.array((x, y, z))
        end = start + cube_size
        box = (start, end)
        tasks.append((box, bounding_boxes))
    # masks = list(map(lambda b: _check_overlap(b, bounding_boxes), tasks))
    t0 = time()
    n_procs = 100  # manually adjusted
    if os.name == "nt":
        n_procs = 60  # 63 is max on windows
    with Pool(processes=n_procs) as pool:
        masks = pool.starmap(_check_overlap, tasks, chunksize=100)
    logger.debug(f"Organelle to domain attribution: {time() - t0}s")

    # remove masks with only one organelle
    filtered_masks = []
    for m in masks:
        if np.count_nonzero(m) > 1:
            filtered_masks.append(m)
    return filtered_masks


def generate_distance_matrix(
    project: "organelle_morphology.Project",
    domain_decomposition=True,
    chunk_dd=False,
) -> pd.DataFrame:
    cache = project.cache
    max_dist = project.max_distance
    max_cached = 0
    if sources := list(project.sources.values()):
        source = sources[0]
    else:
        raise ValueError("No sources loaded! Can't calculate distances.")
    if max_dist == 0:
        max_dist = source.resolution[0] * 10
        logger.warning(
            f"No max distance set, calculating distance matrix up to {max_dist}"
        )

    if "max_distance_computed" in cache:
        max_cached = cache["max_distance_computed"]

    if (
        "distance_matrix" not in cache or not project.use_cache
    ) or max_dist > max_cached:
        project.logger.info("Initializing distance matrix")

        organelles = np.array(project.organelles)
        organelles_ids = np.array([o.id for o in organelles])
        meshes = []
        for organelle in organelles:
            meshes.append(organelle.mesh)
        meshes = np.array(meshes)

        project.logger.info("Calculating distance matrix")
        num_rows = len(meshes)

        distance_matrix = np.ones((num_rows, num_rows)) * -1
        distance_df = pd.DataFrame(
            distance_matrix,
            index=organelles_ids,
            columns=organelles_ids,
        )

        if domain_decomposition and not chunk_dd:
            # domain decomposition into chunks of
            # size = size of clipped data in units of the resolution [um]
            size = np.array(source.resolution) * source.data.shape
            clp_of = source.clipping_corners[0]  # lower corner clipping offset
            masks = make_domains(
                size=size,
                max_dist=max_dist,
                start_corner=clp_of,
                meshes=meshes,
                client=project.client,
            )

        # See #82 : usually slower than the regular domain decomposition
        elif chunk_dd:
            # only do domain decomposition if max dist is within one chunk
            chunksize = np.array(source.data.chunksize) * source.resolution
            if any([max_dist > r for r in chunksize]):
                masks = [np.ones((num_rows,))]
                project.logger.debug("No domain decomposition as max_dist > chunksize")
            else:
                dshape = source.data.blocks.shape
                domain_idxs = list(np.ndindex(dshape))
                # (mask, domain chunks) pairs: a mask and the chunk
                # group it was built from have to stay aligned when domains
                # are dropped or reordered
                domain_masks = chunk_domain_masks(organelles, domain_idxs, dshape)
                project.logger.debug(f"Masks unfiltered: {len(domain_idxs)}")

                too_large = np.nonzero(
                    [
                        len(np.nonzero(m)[0]) > MAX_ORGS_PER_DOMAIN
                        for m, _ in domain_masks
                    ]
                )[0]
                project.logger.debug(f"masks > {MAX_ORGS_PER_DOMAIN}: {len(too_large)}")

                if domain_decomposition:
                    t0 = time()
                    clp_of = source.clipping_corners[0]  # lower corner clipping offset
                    for i in too_large[::-1]:
                        mask, local_blocks = domain_masks.pop(i)
                        local_meshes = meshes[mask]

                        corners = np.array(
                            [
                                block_to_coords(
                                    b, source.resolution, source.data, clp_of
                                )
                                for b in local_blocks
                            ]
                        )
                        lower_corner = corners.min(axis=0)[0]
                        upper_corner = corners.max(axis=0)[1]
                        size = upper_corner - lower_corner

                        local_masks = make_domains(
                            size=size,
                            max_dist=max_dist,
                            start_corner=lower_corner,
                            meshes=local_meshes,
                            client=project.client,
                        )
                        for local_mask in local_masks:
                            global_mask = np.zeros_like(meshes, dtype=bool)
                            global_mask[mask] = local_mask
                            domain_masks.append((global_mask, ()))

                        project.logger.debug(
                            f"decomposed {np.count_nonzero(mask)} orgs into {len(local_masks)}"
                        )

                    project.logger.debug(
                        f"Decomposed {len(too_large)} chunks into {len(domain_masks)} domains in {time() - t0} s"
                    )
                masks = [m for m, _ in domain_masks]

        else:
            # no domain decomposition -> all to all
            masks = [np.ones((num_rows,))]

        project.logger.debug(f"Masks created: {len(masks)}")
        results = []
        empty_cubes = 0
        tasks = []
        for mask in masks:
            indices = np.nonzero(mask)[0]
            local_meshes_d = [o.mesh for o in organelles[indices]]
            local_ids = organelles_ids[indices]

            if len(indices) < 2:
                empty_cubes += 1
                continue

            tasks.append((local_meshes_d, local_ids, max_dist))

        project.logger.debug(f"n empty cube: {empty_cubes}")
        project.logger.debug(f"n tasks get_min_dist: {len(tasks)}")

        with span("dist_matrix_min_dists"):
            t0 = time()
            results = map(delayed_domain_min_dists, tasks)
            results = compute(results)[0]
            results = [r for res in results for r in res]
            t1 = time()

        project.logger.debug(f"Min distances calculated in: {t1 - t0} s")
        for res in results:
            distance_df.loc[res[0]] = res[1]
            distance_df.loc[res[0][::-1]] = res[1]
        distance_df.fillna(-1, inplace=True)

        cache["distance_matrix"] = distance_df
        cache["max_distance_computed"] = max_dist

    else:
        project.logger.info("Retrieving distance matrix from cache")

    project.max_distance = max_dist
    return cache["distance_matrix"]


def generate_mcs(
    project,
    ids_filter_1: str,
    ids_filter_2: str,
    max_distance: float,
    min_distance: float = 0,
    overwrite=False,
) -> str:
    """Generates the MCS (Membrane Contact Site) pairs for a given project.
    The MCSs are calculated between two sets of organelles, defined by the
    two filter strings provided.


    Args:
        project (Project): The project object containing the distance
            matrix and organelles.
        max_distance (float): The maximum distance for the MCS pairs.
        min_distance (float, optional): The minimum distance for the MCS pairs.
            Defaults to 0.
        ids_filter_1 (str, optional): Filter for the first set of
            organelles using glob-style patterns. Defaults to "*".
        ids_filter_2 (str, optional): Filter for the second set of
            organelles using glob-style patterns. Defaults to "*".

        overwrite (bool, optional): Whether to overwrite existing MCS data.
            Defaults to False.

    Returns:
        str: The label of this MCS search
    """

    if max_distance > project.max_distance:
        project.max_distance = max_distance
    label_1 = ids_filter_1.replace("*", "")
    label_2 = ids_filter_2.replace("*", "")
    mcs_label = f"{min_distance}-{max_distance},{label_1}-{label_2}"
    if project._mcs_labels.get(mcs_label) and not overwrite:
        return mcs_label
    mcs_meta = {
        "ids_filter_1": ids_filter_1,
        "ids_filter_2": ids_filter_2,
        "min_distance": min_distance,
        "max_distance": max_distance,
    }

    ids_1 = project.get_organelle_ids(ids_filter_1)
    ids_2 = project.get_organelle_ids(ids_filter_2)
    distance_matrix = project.distance_matrix

    # filter dataframe and get row, column where the distance is between min_distance and max_distance
    rows, columns = np.where(
        (distance_matrix >= min_distance) & (distance_matrix <= max_distance)
    )

    # generate the pairs of organelles
    pairs = set()
    for i, j in zip(rows, columns):
        org_1_id = distance_matrix.index[i]
        org_2_id = distance_matrix.columns[j]

        if org_1_id == org_2_id:
            continue

        if org_1_id not in ids_1 or org_2_id not in ids_2:
            continue

        pairs.add(tuple(sorted([org_1_id, org_2_id])))

    project.logger.info(f"Found {len(pairs)} pairs of organelles to calculate MCS")

    meshes = {}
    for ids in pairs:
        for ind in ids:
            if ind not in meshes:
                meshes[ind] = project.get_organelles(ind)[0].mesh

    tasks = {}
    for id_1, id_2 in pairs:
        mcs_calc_delayed = calc_mcs_delayed(
            mcs_label,
            id_1,
            id_2,
            meshes[id_1],
            meshes[id_2],
            max_distance,
            min_distance,
        )
        tasks[id_1 + id_2] = mcs_calc_delayed

    logger.debug(f"Running {len(tasks)} mcs calculations..")
    results = compute(tasks)[0]
    logger.debug("Finished mcs calculations")

    org_ids = set()
    t0 = time()
    for id_1, id_2 in pairs:
        org1 = project.get_organelles(id_1)[0]
        org2 = project.get_organelles(id_2)[0]
        org_ids.add(id_1)
        org_ids.add(id_2)

        mcs_source, mcs_target = results[id_1 + id_2]

        # add mcs to organelle
        if org1.id == mcs_source["self_id"]:
            org1.add_mcs(mcs_source, mcs_meta)
            org2.add_mcs(mcs_target, mcs_meta)
        else:
            org1.add_mcs(mcs_target, mcs_meta)
            org2.add_mcs(mcs_source, mcs_meta)

    for org_id in org_ids:
        org = project.get_organelles(org_id)[0]
        org.calc_mcs_dict_entry(mcs_label)
    logger.debug(f"Transfered mcs to organelles in {time() - t0}")

    project._mcs_labels[mcs_label] = {
        "max_distance": max_distance,
        "min_distance": min_distance,
        "organelles": org_ids,
    }
    return mcs_label


def _mcs_patch_dict(mcs_label, self_id, partner_id, mesh, patches):
    """Temporary stopgap record shape consumed by ``Organelle.add_mcs``.

    Ticket 03 replaces the aggregate keys with per-patch records. Until then
    keep the old consumer keys (``vertices_index``, ``distances``, ``area``)
    plus the new ``patches`` list.
    """
    if patches:
        distances = np.concatenate(
            [np.asarray(patch["distances"], dtype=np.float64) for patch in patches]
        )
        area = float(sum(patch["area"] for patch in patches))
        faces_index = np.unique(
            np.concatenate(
                [np.asarray(patch["faces_index"], dtype=np.int64) for patch in patches]
            )
        )
        vertices_index = np.unique(
            np.asarray(mesh.faces, dtype=np.int64)[faces_index].reshape(-1)
        )
    else:
        distances = np.empty(0, dtype=np.float64)
        area = 0.0
        vertices_index = np.empty(0, dtype=np.int64)

    return {
        "self_id": self_id,
        "partner_id": partner_id,
        "mcs_label": mcs_label,
        "vertices_index": vertices_index,
        "distances": distances,
        "area": area,
        "patches": patches,
    }


@delayed
def calc_mcs_delayed(
    mcs_label,
    id_1,
    id_2,
    mesh_1,
    mesh_2,
    max_dist,
    min_dist,
):
    patches_1, patches_2 = pair_patches(
        id_1,
        id_2,
        mesh_1,
        mesh_2,
        min_dist,
        max_dist,
    )
    mcs_1 = _mcs_patch_dict(mcs_label, id_1, id_2, mesh_1, patches_1)
    mcs_2 = _mcs_patch_dict(mcs_label, id_2, id_1, mesh_2, patches_2)
    return mcs_1, mcs_2
