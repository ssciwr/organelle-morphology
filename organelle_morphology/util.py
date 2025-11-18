import contextlib
from pathlib import Path
import cachetools
import hashlib
import logging

from multiprocess.pool import Pool
import shelved_cache
import xdg
from tqdm import tqdm
import pickle

import organelle_morphology

CACHE_DIR = xdg.xdg_cache_home() / "organelle_morphology"


class Disk_Store:
    def __init__(self, cache_name: str):
        self.path: Path = CACHE_DIR / cache_name
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
    ):
        self.cache_name = f"{project_path.name}/{source}/{level}/{clipping}"
        self.stores: list = [{}]
        self.disk = disk
        if disk:
            self.stores.append(Disk_Store(self.cache_name))

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

    def clear(self):
        """Deletes all caches, in memory and on disk"""
        for store in self.stores:
            store.clear()

    def delete_disk_cache(self):
        """Delete this cache from disk.
        Does nothing if it was not saved to disk.
        """
        if not self.disk:
            self.stores.append(Disk_Store(self.cache_name))
        ds: Disk_Store = self.stores.pop(-1)
        ds.clear()
        ds.path.rmdir()
        self.disk = False


def get_project_caches(project: "organelle_morphology.Project"):
    caches = []

    project.logger.info(f"{CACHE_DIR / project.path.name}")
    for source in filter(
        lambda f: f.is_dir(), (CACHE_DIR / project.path.name).iterdir()
    ):
        project.logger.info(f"├─ /{source.name}")

        for level in filter(lambda f: f.is_dir(), source.iterdir()):
            project.logger.info(f"│  ├─ /{level.name}")
            for clip_dir in filter(lambda f: f.is_dir, level.iterdir()):
                project.logger.info(f"│  │  ├─ /{clip_dir.name}")
                caches.append(
                    Cache(
                        project_path=project.path,
                        source=source.name,
                        level=level.name,
                        clipping=clip_dir.name,
                        disk=True,
                    )
                )
    project.logger.info(f"Found {len(caches)} caches for project {project.path.name}")
    return caches


@contextlib.contextmanager
def disk_cache(project_path: Path, name, maxsize=1000000):
    # Define the cache
    cache = shelved_cache.PersistentCache(
        cachetools.LRUCache,
        str(
            xdg.xdg_cache_home()
            / "organelle_morphology"
            / hashlib.sha256(str(project_path.absolute()).encode("utf-8")).hexdigest()
            / name
        ),
        maxsize=maxsize,
    )

    # Ensure that the cache contains a timestamp
    if "timestamp" not in cache:
        cache["timestamp"] = project_path.stat().st_mtime

    # Maybe decide to invalidate the cache based on the data timestamp
    if cache.get("timestamp") != project_path.stat().st_mtime:
        cache.clear()

    # Return the cache
    yield cache

    # Close the cache file handle
    cache.close()


# not sure yet wether to fully remove it or try to integrate it again.


@contextlib.contextmanager
def parallel_pool(total=None, cores=None):
    """A context manager that runs the code in parallel"""
    # Create a process pool

    pool = Pool(cores)

    # Run the code in parallel
    if total:
        pbar = tqdm(total=total)
        yield pool, pbar

    else:
        yield pool

    # Close the pool
    pool.close()
    pool.join()


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

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    return logger
