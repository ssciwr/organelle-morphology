import contextlib
import cachetools
import hashlib
import multiprocessing
import shelved_cache
import xdg


# Store the number of parallel cores used
_multiprocessing_cores = multiprocessing.cpu_count()


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


def set_parallel_cores():
    """Set the number of cores used for parallel processing"""

    global _multiprocessing_cores
    _multiprocessing_cores = multiprocessing.cpu_count()


@contextlib.contextmanager
def parallel_pool():
    """A context manager that runs the code in parallel"""

    # Create a process pool
    pool = multiprocessing.Pool(_multiprocessing_cores)

    # Run the code in parallel
    yield pool

    # Close the pool
    pool.close()
    pool.join()
