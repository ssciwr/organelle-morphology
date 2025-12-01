from pathlib import Path
from typing import Optional, Iterable
import logging

from dask.delayed import Delayed, delayed
import numpy as np
from trimesh import Trimesh
import trimesh
import matplotlib as mpl
import xdg
import pickle
from collections import defaultdict, deque
import time


CACHE_DIR = xdg.xdg_cache_home() / "organelle_morphology"


class Disk_Store:
    def __init__(self, cache_name: str, cache_root: Path):
        self.path: Path = cache_root / cache_name
        if not self.path.exists():
            self.path.mkdir(parents=True)

    def __setitem__(self, key, value):
        with open(self.path / str(key), "wb") as f:
            pickle.dump(value, f)

    def __getitem__(self, key):
        if (self.path / str(key)).exists():
            with open(self.path / str(key), "rb") as f:
                return pickle.load(f)
        else:
            raise KeyError(f"Key: {key} not found in {self.path}!")

    def __contains__(self, key):
        return (self.path / str(key)).exists()

    def __len__(self):
        return len(list(self.path.iterdir()))

    def __delitem__(self, key):
        (self.path / str(key)).unlink()

    def clear(self):
        for f in self.path.iterdir():
            f.unlink()

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default


class Cache:
    def __init__(
        self,
        project_path: Path,
        source: str,
        level: str,
        clipping: str,
        disk=True,
        cache_root: Optional[Path] = None,
    ):
        self.cache_name = f"cache_{project_path.name}/{source}/{level}/{clipping}"
        self.stores: list = [{}]
        self.disk = disk
        self.cache_root = cache_root if cache_root else CACHE_DIR
        if disk:
            self.stores.append(Disk_Store(self.cache_name, self.cache_root))

    def __setitem__(self, key, value):
        for store in self.stores:
            store[key] = value

    def __getitem__(self, key):
        for store in self.stores:
            if value := store.get(key, None):
                return value
        raise KeyError(f"Key {key} not in cache!")

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
            self.stores.append(Disk_Store(self.cache_name, self.cache_root))
        ds: Disk_Store = self.stores.pop(-1)
        ds.clear()
        ds.path.rmdir()
        self.disk = False

    def clear_memory_cache(self):
        """Clear the memory cache. Other caches (disk) remain untouched"""
        self.stores[0].clear()


@delayed
def mesure_gaussian_curvature_delayed(tmesh: Trimesh):
    # morph radius can be 0 if vertices are used as sample points.
    morph_radius = 0.0
    curvature_vertices = trimesh.curvature.discrete_gaussian_curvature_measure(
        tmesh, tmesh.vertices, radius=morph_radius
    )
    return curvature_vertices


@delayed
def color_delayed_trimesh_rgba(tmesh: Trimesh, values, log=True) -> Trimesh:
    cm = mpl.colormaps["RdBu"].copy().reversed()
    cm.set_extremes(under=(1, 0, 1, 1), over=(0, 1, 0, 1))
    if log:
        norm = mpl.colors.SymLogNorm(
            linthresh=0.01, linscale=0.01, base=10, vmin=-5, vmax=5
        )
    else:
        norm = mpl.colors.Normalize(vmin=-5, vmax=5)
    colors = cm(norm(values))
    tmesh.visual.vertex_colors = colors
    return tmesh


@delayed
def color_delayed_trimesh(tmesh: Trimesh, color: int) -> Trimesh:
    if color:
        if color == 1:
            tmesh.visual.vertex_colors = trimesh.visual.random_color()
        elif color == 2:
            viridis = mpl.colormaps.get("viridis")
            tmesh.visual.face_colors = viridis.resampled(len(tmesh.faces)).colors
        elif color < 0:
            cm = mpl.colormaps.get("tab20")
            tmesh.visual.face_colors = cm.colors[-color % 20]

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


def merge_meshes(meshes: Iterable[Delayed], color: Optional[int] = None) -> Delayed:
    """Merges delayed meshes into one concrete new Mesh object

    Needs overlapping meshes, otherwise the intersections will not be
    connected.
    """

    meshes = list(meshes)
    if color is None:
        color = 0
    if color:
        meshes = [color_delayed_trimesh(m, color) for m in meshes]

    while (length := len(meshes)) > 1:
        merged = []
        for i, _ in enumerate(meshes[::2]):
            j = length - (i + 1)
            # odd-length: indices meet in the middle
            if i == j:
                merged.append(meshes[i])
                break
            merged.append(merge_delayed_trimeshes([meshes[i], meshes[j]]))
        meshes = merged

    return meshes[0]


class FrequencyFilter(logging.Filter):
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


def get_logger(file: Path):
    logger = logging.getLogger(file.stem)
    logger.setLevel(logging.DEBUG)  # Set logger's level to INFO
    logger.propagate = False
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(file)

    # Set levels - INFO for console, DEBUG for file
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.DEBUG)

    # Create formatters and add it to handlers
    c_format = logging.Formatter("%(levelname)s - %(message)s")
    f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # frequency filtering
    c_handler.addFilter(frequency_filter)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    return logger
