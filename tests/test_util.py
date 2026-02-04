from organelle_morphology.util import measure_gaussian_curvature_delayed


def test_measure_curvature(project_with_sources):
    s = project_with_sources.sources["synth_data"]
    mesh = s.meshes[1]
    delayed = measure_gaussian_curvature_delayed(tmesh=mesh, radius=1)
    curv = delayed.compute()
    assert curv.shape == (866,)
