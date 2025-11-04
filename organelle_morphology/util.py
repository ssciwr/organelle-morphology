import contextlib
from pathlib import Path
import cachetools
import hashlib
import logging

from multiprocess.pool import Pool
import shelved_cache
import xdg
from tqdm import tqdm


@contextlib.contextmanager
def disk_cache(project_path: Path, name, maxsize=10000):
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
