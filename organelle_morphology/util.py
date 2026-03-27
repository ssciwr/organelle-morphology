from pathlib import Path
from typing import Optional, Iterable
import logging
import networkx

from dask.base import compute
from dask.delayed import Delayed, delayed
from trimesh import Trimesh
import trimesh
import matplotlib as mpl
import xdg
import pickle
from collections import defaultdict, deque
import time
import numpy as np


logger = logging.getLogger(__name__)
CACHE_DIR = xdg.xdg_cache_home() / "organelle_morphology"


class Disk_Store:
    def __init__(self, cache_name: str, cache_root: Path, mem_cache: dict):
        self.cache_root = cache_root
        self.cache_name = cache_name
        self.path: Path = cache_root / cache_name
        self.mem_cache = mem_cache
        if not self.path.exists():
            self.path.mkdir(parents=True)

    def __setitem__(self, key, value):
        with open(self.path / str(key), "wb") as f:
            pickle.dump(value, f)

    def __getitem__(self, key):
        try:
            with open(self.path / str(key), "rb") as f:
                value = pickle.load(f)
            self.mem_cache[key] = value
            return value
        except FileNotFoundError as e:
            raise KeyError(f"Key: {key} not found in {self.path}!\n{e}")

    def __contains__(self, key):
        return (self.path / str(key)).exists()

    def __len__(self):
        return len(list(self.path.iterdir()))

    def __delitem__(self, key):
        (self.path / str(key)).unlink()

    def clear(self):
        for f in self.path.iterdir():
            f.unlink()
        if self.path.exists():
            logger.debug(f"Removing cachedir: {self.path}")
            self.path.rmdir()
        for dir in self.path.parents:
            if dir == self.cache_root:
                break
            try:
                logger.debug(f"Trying to remove cacheparent: {dir}")
                dir.rmdir()
            except OSError:
                break

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default


class Cache:
    def __init__(
        self,
        cache_name: str,
        disk=True,
        cache_root: Optional[Path] = None,
    ):
        """Cache supporting memory and disk storages

        Args:
            cache_name: Unique identifier for this cache. Must consist of four
                levels like cache_project_name/type/compression_level/clipping
            disk: Whether disk storage for this cache is used
            cache_root: Where the cache is saved on disk if disk storage
                is enabled. The default, None, uses the xdg_cache_home.
        """
        self.cache_name = cache_name
        self.stores: list = [{}]
        self.disk = disk
        self.cache_root = cache_root if cache_root else CACHE_DIR
        if disk:
            self.stores.append(
                Disk_Store(self.cache_name, self.cache_root, self.stores[0])
            )

    def __setitem__(self, key, value):
        for store in self.stores:
            store[key] = value

    def __getitem__(self, key):
        error = None
        for store in self.stores:
            try:
                return store[key]
            except KeyError as e:
                error = e
                pass
        raise KeyError(f"Key {key} not in cache! {error}")

    def __contains__(self, key):
        for store in self.stores:
            if key in store:
                return True
        return False

    def __len__(self):
        return len(self.stores[0])

    def __delitem__(self, key):
        for store in self.stores:
            del store[key]

    def clear_all(self):
        """Deletes all caches, in memory and on disk"""
        for store in self.stores:
            store.clear()

    def clear_disk_cache(self):
        """Delete this cache from disk.
        Does nothing if it was not saved to disk.
        """
        if not self.disk:
            self.stores.append(
                Disk_Store(self.cache_name, self.cache_root, self.stores[0])
            )
        ds: Disk_Store = self.stores.pop(-1)
        ds.clear()
        self.disk = False

    def clear_memory_cache(self):
        """Clear the memory cache. Other caches (disk) remain untouched"""
        self.stores[0].clear()


@delayed
def measure_gaussian_curvature_delayed(tmesh: Trimesh, radius: float):
    """Return curvature of mesh normalized by the circle area or radius"""
    # radius can be 0 if vertices are used as sample points.
    curvature_vertices = trimesh.curvature.discrete_gaussian_curvature_measure(
        tmesh, tmesh.vertices, radius=radius
    ) / (np.pi * radius * radius)
    return curvature_vertices


@delayed
def reset_color_delayed(tmesh):
    tmesh.visual.vertex_colors[:] = [100, 100, 100, 255]
    return tmesh


@delayed
def color_delayed_trimesh_vertices(tmesh, vertices, color):
    tmesh.visual.vertex_colors[vertices] = color
    return tmesh


@delayed
def color_delayed_trimesh_rgba(
    tmesh: Trimesh,
    values,
    vmin: float,
    vmax: float,
    log=True,
) -> Trimesh:
    cm = mpl.colormaps["RdBu"].copy().reversed()
    cm.set_extremes(under=(1, 0, 1, 1), over=(0, 1, 0, 1))
    if log:
        norm = mpl.colors.SymLogNorm(
            linthresh=0.01, linscale=0.01, base=10, vmin=vmin, vmax=vmax
        )
    else:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    colors = cm(norm(values))
    tmesh.visual.face_colors = (0, 0, 0, 0)
    tmesh.visual.vertex_colors = colors
    return tmesh


@delayed
def color_delayed_trimesh(tmesh: Trimesh, color: int, transp: bool) -> Trimesh:
    if color:
        if color == 1:
            tmesh.visual.face_colors = trimesh.visual.random_color()
        elif color == 2:
            viridis = mpl.colormaps.get("viridis")
            tmesh.visual.face_colors = viridis.resampled(len(tmesh.faces)).colors
        elif color < 0:
            cm = mpl.colormaps.get("tab20")
            tmesh.visual.face_colors = cm.colors[-color % 20]

    if transp:
        tmesh.visual.face_colors[:, 3] = 100

    return tmesh


@delayed
def merge_delayed_trimeshes(tmeshes: list[Trimesh]):
    tmesh = Trimesh()
    for mesh in tmeshes:
        tmesh += mesh
    tmesh.merge_vertices(
        merge_tex=True,
        merge_norm=True,
        digits_vertex=2,
        digits_norm=2,
        digits_uv=2,
    )
    return tmesh


def sample_skeleton(skeleton, path_sample_dist: float = 0.1):
    # the sample points are points along the skeleton arms
    # and the reference points are the vertices of the skeleton from which
    # these samples have been generated. we need these to later calculate
    # the normal vector for the plane which will intersect our mesh
    sampled_path = []
    reference_point = []
    for edge in skeleton.edges:
        edge_len = np.linalg.norm(
            np.array(skeleton.vertices[edge[0]]) - np.array(skeleton.vertices[edge[1]])
        )

        if edge_len > path_sample_dist:
            p1 = np.array(skeleton.vertices[edge[0]])
            p2 = np.array(skeleton.vertices[edge[1]])

            # find number of points to add between the two vertices
            n_points = np.ceil(edge_len / path_sample_dist).astype(int)
            factors = np.linspace(0, 1, n_points)

            # Compute the interpolated points
            interpolated_points = (1 - factors[:, np.newaxis]) * p1 + factors[
                :, np.newaxis
            ] * p2
            sampled_path.extend(interpolated_points)
            reference_point.append(p1)

    if len(sampled_path) == 0:
        sampled_path.append(np.array(skeleton.vertices[0]))
        reference_point.append(np.array(skeleton.vertices[0]))

    sampled_path = np.asarray(sampled_path)
    reference_point = np.asarray(reference_point)
    _sampled_skeleton = sampled_path, reference_point
    return _sampled_skeleton


def get_skeleton_info(skeleton):
    skeleton_info = {}
    graph = skeleton.get_graph()
    if len(graph.nodes) == 0:
        return 0

    skeleton_info["num_nodes"] = len(graph.nodes)
    skeleton_info["num_branch_points"] = len(
        [node for node, degree in graph.degree() if degree > 2]
    )
    skeleton_info["end_points"] = len(
        [node for node, degree in graph.degree() if degree == 1]
    )
    skeleton_info["total_length"] = sum(
        d["weight"] for u, v, d in graph.edges(data=True)
    )
    skeleton_info["longest_path"] = networkx.dag_longest_path_length(graph)

    lengths = [d["weight"] for u, v, d in graph.edges(data=True)]
    skeleton_info["mean_length"] = np.mean(lengths)
    skeleton_info["std_length"] = np.std(lengths)

    skeleton_info["mean_radius"] = np.mean(skeleton.radius[0])
    skeleton_info["std_radius"] = np.std(skeleton.radius[0])
    return skeleton_info


def merge_meshes(
    meshes: Iterable[Delayed], color: Optional[int] = None, transp=False
) -> Delayed:
    """Merges delayed meshes into one delayed Mesh object

    Needs overlapping meshes, otherwise the intersections will not be
    connected.
    """

    meshes = list(meshes)
    if len(meshes) == 0:
        return delayed(Trimesh())
    if color is None:
        color = 0
    if color or transp:
        meshes = [color_delayed_trimesh(m, color, transp) for m in meshes]

    batch_size = 1000
    while len(meshes) > 1:
        if len(meshes) <= batch_size:
            return merge_delayed_trimeshes(meshes)
        else:
            new_meshes = []
            for i in range(0, len(meshes), batch_size):
                batch = meshes[i : i + batch_size]
                if len(batch) == 1:
                    new_meshes.append(batch[0])
                else:
                    merged = merge_delayed_trimeshes(batch)
                    new_meshes.append(merged)
            meshes = new_meshes
    return meshes[0]


def corners_to_edges(lower, upper):
    return np.array(upper) - np.array(lower)


@delayed
def bounding_box_delayed(mesh: Trimesh):
    """Calculate the corner with the smallest and the one
    with the biggest corrdinates.

    Args:
        mesh: Trimesh

    Returns:
        min: np.ndarray, First corner
        max: np.ndarray, Second corner
    """
    min = np.min(mesh.vertices, axis=0)
    max = np.max(mesh.vertices, axis=0)
    return min, max


def show(meshes):
    """Set up a scene with proper camera settings and show it"""
    scene = trimesh.Scene()
    scene.camera.z_far = 100000
    # scene.add_geometry(trimesh.creation.axis(origin_size=10))

    if not isinstance(meshes, (list, tuple)):
        meshes = [meshes]
    print("About to compute meshes in show")
    meshes = compute(*meshes, traverse=False)
    print("Done")

    if len(meshes[0].vertices) > 0:
        scene.camera_transform = scene.camera.look_at(meshes[0].vertices[:200])

    for mesh in meshes:
        scene.add_geometry(mesh)

    # scale to make the view behave properly
    scene = scene.scaled(1 / scene.scale)
    scene.show()
    return scene


def boxes_overlap(box1, box2):
    """
    Determine if two bounding boxes overlap.

    Parameters:
    box1, box2: tuple of two numpy arrays
        Each box is defined by two points: (min_point, max_point)
        where min_point has the lowest x, y, z coordinates
        and max_point has the highest x, y, z coordinates

    Returns:
    bool: True if boxes overlap, False otherwise
    """
    min1, max1 = box1
    min2, max2 = box2

    if (
        max1[0] < min2[0]
        or max2[0] < min1[0]
        or max1[1] < min2[1]
        or max2[1] < min1[1]
        or max1[2] < min2[2]
        or max2[2] < min1[2]
    ):
        return False

    return True


class FrequencyFilter(logging.Filter):
    """Accumulate repeating log messages"""

    def __init__(self, threshold=10, burst_threshold=3, window_size=4):
        self.threshold = threshold  # Log every nth occurrence in a short period
        self.burst_threshold = (
            burst_threshold  # Emit log immediately if below this threshold
        )
        self.window_size = window_size  # Time window for counting occurrences (seconds)
        self.cache = defaultdict(lambda: {"count": 0, "queue": deque()})

    def filter(self, record):
        current_time = time.time()
        entry = self.cache[record.getMessage()]

        # Remove old entries from the queue
        n_removed = -self.burst_threshold
        while entry["queue"] and current_time - entry["queue"][0] > self.window_size:
            n_removed += 1
            entry["count"] -= 1
            entry["queue"].popleft()

        entry["count"] += 1
        entry["queue"].append(current_time)

        if entry["count"] <= self.burst_threshold:
            return True
        elif (
            entry["count"] % self.threshold == 0
            and current_time - entry["queue"][0] < self.window_size
        ):
            print(
                f"(skipped {self.threshold - self.burst_threshold} duplicate log entries)"
            )

        return False


frequency_filter = FrequencyFilter(threshold=103, burst_threshold=3, window_size=2)


def setup_logging(loglevel: str = "INFO", log_file: Optional[Path] = None):
    """Configure the root logger for the entire application

    Args:
        loglevel: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file. If None, logs only to console.
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, loglevel.upper()))
    root_logger.handlers.clear()

    # Create formatters
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, loglevel.upper()))
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(frequency_filter)

    root_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Controll other loggers
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("MARKDOWN").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("embreex").setLevel(logging.ERROR)
    logging.getLogger("client").setLevel(logging.ERROR)
    logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
    logging.getLogger("application").setLevel(logging.ERROR)
    logging.getLogger("trimesh").setLevel(logging.ERROR)
