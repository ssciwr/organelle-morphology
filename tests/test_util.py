from organelle_morphology.util import disk_cache


def test_cache(cebra_project):
    with disk_cache(cebra_project, "test_cache") as cache:
        assert cache["timestamp"] == cebra_project.path.stat().st_mtime

        cache["foo"] = 42
        assert cache["foo"] == 42

    with disk_cache(cebra_project, "test_cache") as cache:
        assert cache["foo"] == 42
