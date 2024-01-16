import contextlib
import cachetools
import shelved_cache
import xdg


@contextlib.contextmanager
def disk_cache(project, name, maxsize=10000):
    # Define the cache
    cache = shelved_cache.PersistentCache(
        cachetools.LRUCache,
        str(xdg.xdg_cache_home() / "organelle_morphology" / name),
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
