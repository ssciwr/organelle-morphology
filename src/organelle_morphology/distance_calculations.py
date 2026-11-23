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
    get_neighboring_chunks,
)

logger = logging.getLogger(__name__)

# chunk domains containing more organelles than this are decomposed further
# in `generate_distance_matrix` (chunk_dd branch)
MAX_ORGS_PER_DOMAIN = 500


def get_min_dist(args):
    """Bidirectional vertex-to-surface bounds for one organelle pair.

    ``d_AB`` is the smallest exact distance from a vertex of mesh A to the
    surface of mesh B, ``d_BA`` the same with the roles swapped. Their minimum
    is an upper bound on the true surface-to-surface minimum (error bounded by
    the sampling radius of the queried mesh), while

    ``max(d_AB - h_A, d_BA - h_B, 0)``

    is a rigorous lower bound (``h`` = sampling radius from ticket 01). The
    lower bound is clamped at zero because surface distances are non-negative;
    this also keeps zero a valid, distinguishable bound.

    Args:
        args: ``(id_1, id_2, mesh_1, mesh_2, max_dist)``.

    Returns:
        tuple: ``((id_1, id_2), upper, lower)``. ``upper`` is ``nan`` when the
        pair lies beyond ``max_dist`` (the matrix stores it as ``-1``); the
        lower bound is kept so the pair gate can still rule the pair in.
    """
    id_1, id_2, mesh_1, mesh_2, max_dist = args
    geometry_1 = get_mesh_geometry(mesh_1)
    geometry_2 = get_mesh_geometry(mesh_2)
    d_1, d_2 = min_dist_upper(geometry_1, geometry_2)
    upper = min(d_1, d_2)
    lower = max(
        d_1 - geometry_1.sampling_radius,
        d_2 - geometry_2.sampling_radius,
        0.0,
    )
    # Keep the historical matrix semantics: pairs beyond max_dist stay
    # uncomputed (-1) instead of storing an out-of-band distance.
    if max_dist is not None and upper > max_dist:
        upper = np.nan
    if not np.isfinite(upper):
        upper = np.nan
    if not np.isfinite(lower):
        lower = np.nan
    return (id_1, id_2), upper, lower


def _check_overlap(box, mins, maxs):
    """Boolean mask: which axis-aligned boxes overlap ``box``.

    ``mins``/``maxs`` are the lower and upper corners of all boxes, one row
    per box, so the whole mask is built without a Python loop.
    """
    if len(mins) == 0:
        return np.zeros((0,), dtype=bool)
    lower, upper = np.asarray(box[0]), np.asarray(box[1])
    return np.all((maxs >= lower) & (mins <= upper), axis=1)


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
    # invert the chunk -> organelle relation once instead of scanning every
    # organelle's chunk list for every domain and neighbour
    chunk_to_orgs = {}
    for org_index, org in enumerate(organelles):
        for chunk in set(org.chunks):
            chunk_to_orgs.setdefault(tuple(chunk), []).append(org_index)

    domain_masks = []
    for idx in domain_idxs:
        idx = tuple(idx)
        neighbor_chunks = get_neighboring_chunks(idx, dshape)
        center_orgs = chunk_to_orgs.get(idx)
        if not center_orgs:
            continue
        member_orgs = set(center_orgs)
        for chunk in neighbor_chunks:
            member_orgs.update(chunk_to_orgs.get(chunk, ()))

        mask = np.zeros(len(organelles), dtype=bool)
        mask[list(member_orgs)] = True
        domain_masks.append((mask.tolist(), neighbor_chunks))
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

    # corners once, so each pool task carries two small (n, 3) arrays instead
    # of pickling the full nested bounding box structure
    bounds = np.asarray(bounding_boxes, dtype=np.float64)
    if bounds.size:
        mins = np.ascontiguousarray(bounds[:, 0, :])
        maxs = np.ascontiguousarray(bounds[:, 1, :])
    else:
        mins = np.empty((0, 3), dtype=np.float64)
        maxs = np.empty((0, 3), dtype=np.float64)

    tasks = []
    for x, y, z in zip(xg, yg, zg):
        start = np.array((x, y, z))
        end = start + cube_size
        box = (start, end)
        tasks.append((box, mins, maxs))
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
    """Compute (and cache) the pairwise organelle distance matrix.

    The returned matrix holds, for every computed pair, the minimum of the two
    bidirectional vertex-to-surface minima (``min(d_AB, d_BA)``). That value is
    an **upper bound** on the true surface-to-surface minimum; the error is
    bounded by ``max(h_A, h_B)``, the sampling radii of the two meshes. Entries
    are ``-1`` when the pair was not evaluated (domain decomposition) or lies
    beyond ``max_distance``.

    Beside the upper-bound matrix, a second cached DataFrame
    (``cache["distance_matrix_lower"]``) holds the rigorous lower bound
    ``max(d_AB - h_A, d_BA - h_B, 0)`` for every evaluated pair. The pair gate
    in :func:`generate_mcs` uses it to decide whether a pair can possibly come
    within ``max_distance``. Both caches are keyed by ``max_distance_computed``;
    a cache that predates the lower matrix is recomputed.

    Args:
        project: project holding organelles and the mesh cache.
        domain_decomposition: restrict pair calculations to overlapping
            spatial domains (default) instead of all-to-all.
        chunk_dd: use the chunk-grid domain decomposition path.

    Returns:
        pandas.DataFrame: symmetric upper-bound distance matrix indexed by
        organelle id, in micrometres.
    """
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

    cache_complete = (
        "distance_matrix" in cache
        and "distance_matrix_lower" in cache
        and project.use_cache
        and max_dist <= max_cached
    )
    if not cache_complete:
        project.logger.info("Initializing distance matrix")

        organelles = np.array(project.organelles)
        organelles_ids = np.array([o.id for o in organelles])
        meshes = []
        for organelle in organelles:
            meshes.append(organelle.mesh)
        meshes = np.array(meshes)

        project.logger.info("Calculating distance matrix")
        num_rows = len(meshes)

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

        # Fill both matrices by index arrays. A pair can appear in several
        # overlapping domains; duplicate writes are identical values.
        id_to_index = {org_id: i for i, org_id in enumerate(organelles_ids)}
        upper_values = np.full((num_rows, num_rows), -1.0)
        lower_values = np.full((num_rows, num_rows), -1.0)
        if results:
            rows = np.fromiter(
                (id_to_index[res[0][0]] for res in results),
                dtype=np.intp,
                count=len(results),
            )
            columns = np.fromiter(
                (id_to_index[res[0][1]] for res in results),
                dtype=np.intp,
                count=len(results),
            )
            upper = np.fromiter(
                (res[1] for res in results), dtype=np.float64, count=len(results)
            )
            lower = np.fromiter(
                (res[2] for res in results), dtype=np.float64, count=len(results)
            )
            upper_values[rows, columns] = upper
            upper_values[columns, rows] = upper
            lower_values[rows, columns] = lower
            lower_values[columns, rows] = lower

        distance_df = pd.DataFrame(
            np.where(np.isnan(upper_values), -1.0, upper_values),
            index=organelles_ids,
            columns=organelles_ids,
        )
        lower_df = pd.DataFrame(
            np.where(np.isnan(lower_values), -1.0, lower_values),
            index=organelles_ids,
            columns=organelles_ids,
        )

        cache["distance_matrix"] = distance_df
        cache["distance_matrix_lower"] = lower_df
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
    subdivision_depth: int = 3,
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
        subdivision_depth (int, optional): Maximum midpoint refinement depth
            for faces straddling a band/facing boundary. Defaults to 3, the
            documented accuracy setting; raising it refines boundary areas
            further at 4x the leaves per level.

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
    if overwrite:
        # Replace, never append: drop every organelle's in-memory patches for
        # this label and unregister its aggregate records before recomputing.
        for organelle in project.organelles:
            organelle.clear_mcs(mcs_label)
        for record in list(project.registry.get_by_type("McsData")):
            if record.meta.mcs_label == mcs_label:
                project.registry.clear_record(record)
    mcs_meta = {
        "ids_filter_1": ids_filter_1,
        "ids_filter_2": ids_filter_2,
        "min_distance": min_distance,
        "max_distance": max_distance,
    }

    ids_1 = project.get_organelle_ids(ids_filter_1)
    ids_2 = project.get_organelle_ids(ids_filter_2)
    distance_matrix = project.distance_matrix
    distance_lower = project.cache["distance_matrix_lower"]

    # A pair can only be excluded when it provably lies beyond max_distance:
    # gate on the rigorous lower bound, never on the upper-bound matrix and
    # never on min_distance (the band filters surface points inside the
    # geometry core). -1/NaN mark uncomputed entries and are not distances.
    lower_values = distance_lower.to_numpy(dtype=np.float64, copy=False)
    candidate = (
        np.isfinite(lower_values)
        & (lower_values >= 0.0)
        & (lower_values <= max_distance)
    )
    rows, columns = np.nonzero(np.triu(candidate, k=1))

    matrix_ids = np.asarray(distance_matrix.index)
    in_filter_1 = np.isin(matrix_ids, ids_1)
    in_filter_2 = np.isin(matrix_ids, ids_2)
    pair_mask = (in_filter_1[rows] & in_filter_2[columns]) | (
        in_filter_2[rows] & in_filter_1[columns]
    )

    pairs = [
        tuple(sorted((matrix_ids[i], matrix_ids[j])))
        for i, j in zip(rows[pair_mask], columns[pair_mask])
    ]

    project.logger.info(f"Found {len(pairs)} pairs of organelles to calculate MCS")

    needed_ids = {org_id for pair in pairs for org_id in pair}
    meshes = {
        org_id: project.get_organelles(org_id)[0].mesh for org_id in needed_ids
    }

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
            subdivision_depth,
        )
        tasks[(id_1, id_2)] = mcs_calc_delayed

    logger.debug(f"Running {len(tasks)} mcs calculations..")
    results = compute(tasks)[0]
    logger.debug("Finished mcs calculations")

    org_ids = set()
    t0 = time()
    empty_pairs = 0
    for id_1, id_2 in pairs:
        mcs_source, mcs_target = results[(id_1, id_2)]
        org1 = project.get_organelles(id_1)[0]
        org2 = project.get_organelles(id_2)[0]
        if org1.id == mcs_source["self_id"]:
            mcs_1, mcs_2 = mcs_source, mcs_target
        else:
            mcs_1, mcs_2 = mcs_target, mcs_source

        # The two sides of an apposition are independent: one side may have no
        # facing surface point in the band while the other does. Store each
        # side on its own and only skip the pair when both are empty.
        added = False
        if mcs_1["patches"]:
            org1.add_mcs(mcs_1, mcs_meta)
            org_ids.add(id_1)
            added = True
        if mcs_2["patches"]:
            org2.add_mcs(mcs_2, mcs_meta)
            org_ids.add(id_2)
            added = True
        if not added:
            # The conservative gate also yields pairs without any in-band
            # surface point; they do not form an MCS on either side.
            empty_pairs += 1
    if empty_pairs:
        logger.debug(f"Skipped {empty_pairs} gated pairs without an MCS")

    project._mcs_labels[mcs_label] = {
        "max_distance": max_distance,
        "min_distance": min_distance,
        "organelles": org_ids,
    }

    for org_id in org_ids:
        org = project.get_organelles(org_id)[0]
        org.calc_mcs_dict_entry(mcs_label, mcs_meta)
    logger.debug(f"Transfered mcs to organelles in {time() - t0}")
    return mcs_label


def _mcs_patch_dict(mcs_label, self_id, partner_id, patches):
    """Per-side MCS result: identity plus the side's patch list."""
    return {
        "self_id": self_id,
        "partner_id": partner_id,
        "mcs_label": mcs_label,
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
    # Depth 3 is the documented accuracy setting; its cost on the largest
    # benchmark pair was accepted (ADR 0002), so it stays the default.
    subdivision_depth=3,
):
    patches_1, patches_2 = pair_patches(
        id_1,
        id_2,
        mesh_1,
        mesh_2,
        min_dist,
        max_dist,
        subdivision_depth,
    )
    mcs_1 = _mcs_patch_dict(mcs_label, id_1, id_2, patches_1)
    mcs_2 = _mcs_patch_dict(mcs_label, id_2, id_1, patches_2)
    return mcs_1, mcs_2
