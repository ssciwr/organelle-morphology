import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium", layout_file="layouts/ui.grid.json")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    import organelle_morphology as om
    from pathlib import Path
    import numpy as np
    from organelle_morphology.util import bounding_box_delayed
    from dask.base import compute
    import pandas as pd
    from collections import defaultdict


@app.cell
def _():
    mo.md(r"""
    # Organelle Morphology
    """)
    return


@app.cell
def _():
    # Inputs
    path_ui = mo.ui.file_browser(
        selection_mode="directory", multiple=False, label="Project directory"
    )

    run_project = mo.ui.run_button(label="Load Project")

    mo.vstack(
        [
            mo.md("###Load/Create a Project"),
            path_ui,
            run_project,
        ]
    )
    return path_ui, run_project


@app.cell
def _(path_ui, run_project):
    mo.stop(not run_project.value, "Load a project first")
    mo.stop(not len(path_ui.value) > 0, "Select a project directory!")

    project = om.Project(
        project_path=path_ui.path(), clipping=None, compression_level="s3"
    )
    return (project,)


@app.cell
def _(project):
    # add sources
    def load_sources(button_value):
        for entry in source_path_ui.value:
            project.add_source(
                xml_path=entry.path, organelle=new_organelle_name_ui.value
            )

    def get_source_header():
        sources = "<br>".join([s.xml_path.name for s in project.sources.values()])
        return mo.md(f"<h3>Current sources:</h3>{sources}")

    source_path_ui = mo.ui.file_browser(
        filetypes=[".xml"],
        selection_mode="file",
        label="Source xml",
        multiple=False,
    )
    new_organelle_name_ui = mo.ui.text(placeholder="Organelle name [mito, er, ..]")

    run_add_source = mo.ui.button(label="Load sources", on_click=load_sources)

    mo.vstack(
        [
            mo.md("<h3>Add Sources</h3>"),
            source_path_ui,
            new_organelle_name_ui,
            run_add_source,
        ]
    )
    return get_source_header, run_add_source


@app.cell
def _():
    om.organelle.organelle_registry
    return


@app.cell
def _(get_source_header, run_add_source):
    run_add_source
    get_source_header()
    return


@app.cell
def _(project, run_add_source):
    mo.stop(len(project.sources) < 1, "Add a source first!")
    run_add_source

    cl_switch = mo.ui.switch(value=False, label="Clipping")
    cl_d = mo.md(
        "Lower corner:<br>{low_x} {low_y} {low_z} <br>Higher corner:<br>{high_x} {high_y} {high_z}"
    ).batch(
        low_x=mo.ui.number(start=0.0, stop=1.0, value=0.0),
        low_y=mo.ui.number(start=0.0, stop=1.0, value=0.0),
        low_z=mo.ui.number(start=0.0, stop=1.0, value=0.0),
        high_x=mo.ui.number(start=0.0, stop=1.0, value=1.0),
        high_y=mo.ui.number(start=0.0, stop=1.0, value=1.0),
        high_z=mo.ui.number(start=0.0, stop=1.0, value=1.0),
    )

    def _get_levels():
        s = list(project.sources.values())[0]
        return s.metadata["levels"]

    level_ui = mo.ui.radio(
        label="Compression level",
        options=_get_levels(),
        inline=True,
        value=_get_levels()[0],
    )

    def _update_clip_level(_):
        clipping = [
            [
                cl_d["low_x"].value,
                cl_d["low_y"].value,
                cl_d["low_z"].value,
            ],
            [
                cl_d["high_x"].value,
                cl_d["high_y"].value,
                cl_d["high_z"].value,
            ],
        ]
        clipping = clipping if cl_switch.value else None
        project.clipping = clipping
        project.compression_level = level_ui.value

    change_settings_button = mo.ui.button(
        label="Update project settings", on_click=_update_clip_level
    )

    mo.vstack(
        [
            mo.md("<h3>Change compression and clipping</h3>"),
            cl_switch,
            cl_d,
            level_ui,
            change_settings_button,
        ]
    )
    return (change_settings_button,)


@app.cell
def _(change_settings_button, project):
    change_settings_button
    mo.md(
        f"<h3>Current settings</h3>Compression level: <b>{project.compression_level}</b><br>Clipping: <b>{project.clipping}</b>"
    )
    return


@app.cell
def _():
    run_progress = mo.ui.button(
        label="Refresh Progress View", value=0, on_click=lambda v: v + 1
    )
    run_progress
    return (run_progress,)


@app.cell
def _(run_progress):
    mo.stop(run_progress.value == 0, "Please refresh once the project is created")
    url = "http://localhost:8787/individual-progress"
    mo.iframe(
        f'<iframe src="{url}" width="100%" height="600" frameborder="0"></iframe>',
        width="100%",
        height="900px",
    )
    return


@app.cell
def _():
    run_show_mesh = mo.ui.run_button(label="Show Mesh")

    mesh_id_filter = mo.ui.text(value="*", label="Organelle id filter")
    highlight_filter = mo.ui.text(value="", label="Highligh ids")

    box_dict = mo.ui.dictionary(
        {
            "draw box": mo.ui.checkbox(value=False),
            "lower_x": mo.ui.number(value=0.0, start=0.0, stop=1.0),
            "lower_y": mo.ui.number(value=0.0, start=0.0, stop=1.0),
            "lower_z": mo.ui.number(value=0.0, start=0.0, stop=1.0),
            "upper_x": mo.ui.number(value=1.0, start=0.0, stop=1.0),
            "upper_y": mo.ui.number(value=1.0, start=0.0, stop=1.0),
            "upper_z": mo.ui.number(value=1.0, start=0.0, stop=1.0),
        },
        label="Box settings",
    )

    curvature_check = mo.ui.checkbox(label="Curvature", value=False)
    log_check = mo.ui.checkbox(label="log scale", value=True)
    skeleton_check = mo.ui.checkbox(label="Skeleton", value=False)
    popout_viewer_check = mo.ui.checkbox(label="High-quality viewer", value=False)

    curv_radius_slider = mo.ui.slider(label="radius", value=4.0, start=0.0, stop=15, step=0.1)

    mo.vstack(
        [
            mo.md("## Show Mesh"),
            mesh_id_filter,
            highlight_filter,
            skeleton_check,
            mo.hstack([
                curvature_check,
                log_check,
                curv_radius_slider,
            ], justify="start"),
            box_dict,
            mo.hstack(
                [
                    run_show_mesh,
                    popout_viewer_check,
                ],
                justify="start",
            ),
        ]
    )
    return (
        box_dict,
        curv_radius_slider,
        curvature_check,
        highlight_filter,
        log_check,
        mesh_id_filter,
        popout_viewer_check,
        run_show_mesh,
        skeleton_check,
    )


@app.cell
def show_mesh(
    box_dict,
    curv_radius_slider,
    curvature_check,
    highlight_filter,
    log_check,
    mesh_id_filter,
    popout_viewer_check,
    project,
    run_show_mesh,
    skeleton_check,
):
    mo.stop(not run_show_mesh.value, "Mesh will be displayed here")

    box = None
    if box_dict["draw box"].value:
        box = (
            (
                box_dict["lower_x"].value,
                box_dict["lower_y"].value,
                box_dict["lower_z"].value,
            ),
            (
                box_dict["upper_x"].value,
                box_dict["upper_y"].value,
                box_dict["upper_z"].value,
            ),
        )
    highlight = highlight_filter.value if highlight_filter.value else None
    print(mesh_id_filter.value, highlight_filter.value)

    for source in project.sources.values():
        source.curvature_radius = curv_radius_slider.value

    scene = project.show(
        box=box,
        ids=mesh_id_filter.value,
        ids_highlight=highlight,
        curvature=curvature_check.value,
        skeleton=skeleton_check.value,
        curv_log=log_check.value,
    )
    viewer = "marimo"
    if popout_viewer_check.value:
        viewer = "gl"
    scene.show(viewer=viewer)
    return


@app.cell
def _():
    skel_dict = mo.ui.dictionary(
        {
            "theta": mo.ui.number(value=0.4),
            "waves": mo.ui.number(value=1, step=1, label="(Wavefront only)"),
            "epsilon": mo.ui.number(value=0.1, label="(Vertex cluster only)"),
            "path_sample_dist": mo.ui.number(value=0.1),
        }
    )
    run_skeleton_button = mo.ui.run_button()
    skel_form = mo.md(r"""
    ## Skeletonize

    Method: {method}  
    id filter: {ids}  
    Recompute: {recompute}  
    Settings: {settings}  

    """).batch(
        method=mo.ui.radio(
            options=["wavefront", "vertex cluster"], inline=True, value="wavefront"
        ),
        ids=mo.ui.text(value="*"),
        recompute=mo.ui.checkbox(value=False),
        settings=skel_dict,
    )

    mo.vstack([skel_form, run_skeleton_button])
    return run_skeleton_button, skel_form


@app.cell
def _(project, run_skeleton_button, skel_form):
    mo.stop(not run_skeleton_button.value, "Output of Skeletonization")
    form = skel_form.value

    with mo.redirect_stderr():
        settings = dict(form["settings"])
        settings["ids"] = form["ids"]
        settings["recompute"] = form["recompute"]
        if form["method"] == "wavefront":
            settings.pop("epsilon")
            out = project.skeletonize_wavefront(**settings)
        elif form["method"] == "vertex cluster":
            settings.pop("waves")
            out = project.skeletonize_vertex_clusters(**settings)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(project):
    def clear_caches(_):
        project.clear_caches(clear_disk_check.value)

    clear_disk_check = mo.ui.checkbox(value=False, label="Also clear disk cache")
    clear_disk_button = mo.ui.button(
        on_click=clear_caches, label="Clear Cache", kind="warn"
    )
    cache_info_button = mo.ui.run_button(label="Cache Information")

    mo.vstack(
        [
            mo.md("## Cache"),
            cache_info_button,
            mo.hstack(
                [
                    clear_disk_button,
                    clear_disk_check,
                ],
                justify="start",
            ),
        ]
    )
    return (cache_info_button,)


@app.cell
def _(cache_info_button, project):
    mo.stop(not cache_info_button.value, "Cache information")
    with mo.redirect_stderr():
        project.get_caches()
    return


@app.cell
def _(box_dict, mesh_id_filter, project):
    def ids_in_box(_):
        orgs = project.get_organelles(ids=mesh_id_filter.value)
        _scaling = np.array(list(project.sources.values())[0].metadata["size"])
        _box = (
            np.array(
                (
                    box_dict["lower_x"].value,
                    box_dict["lower_y"].value,
                    box_dict["lower_z"].value,
                )
            )
            * _scaling,
            np.array(
                (
                    box_dict["upper_x"].value,
                    box_dict["upper_y"].value,
                    box_dict["upper_z"].value,
                )
            )
            * _scaling,
        )
        result = []
        bbs = [(o, bounding_box_delayed(o.mesh)) for o in orgs]
        bbs = compute(*bbs)
        for o, bb in bbs:
            if np.all(bb[0] >= _box[0]) and np.all(bb[1] <= _box[1]):
                result.append(o.id)

        first = [n.split("_")[0] for n in result]
        orgs, counts = np.unique(first, return_counts=True)
        orgs = " , ".join([o + "_\*" for o in orgs])
        output = mo.md(f"Organelles: {orgs}<br>Counts: {counts}"), pd.DataFrame(result)

        mo.output.replace(output)
        return result

    box_dict  # control flow
    mesh_id_filter  # control flow
    mo.ui.button(on_click=ids_in_box, label="Get IDs of organelles in box", value=None)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
