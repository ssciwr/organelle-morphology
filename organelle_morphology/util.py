import contextlib
import cachetools
import hashlib

# import multiprocess
import shelved_cache
import xdg
from tqdm import tqdm


@contextlib.contextmanager
def disk_cache(project, name, maxsize=10000):
    # Define the cache
    cache = shelved_cache.PersistentCache(
        cachetools.LRUCache,
        str(
            xdg.xdg_cache_home()
            / "organelle_morphology"
            / hashlib.sha256(str(project.path.absolute()).encode("utf-8")).hexdigest()
            / name
        ),
        maxsize=maxsize,
    )

    # Ensure that the cache contains a timestamp
    if "timestamp" not in cache:
        cache["timestamp"] = project.path.stat().st_mtime

    # Maybe decide to invalidate the cache based on the data timestamp
    if cache.get("timestamp") != project.path.stat().st_mtime:
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

    pool = multiprocess.Pool(cores)

    # Run the code in parallel
    if total:
        pbar = tqdm(total=total)
        yield pool, pbar

    else:
        yield pool

    # Close the pool
    pool.close()
    pool.join()
