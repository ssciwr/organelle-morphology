import marimo

__generated_with = "0.18.3"
app = marimo.App(width="medium", layout_file="layouts/ui.grid.json")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    import organelle_morphology as om
    from pathlib import Path


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
def _(get_source_header, run_add_source):
    run_add_source
    get_source_header()
    return


@app.cell
def _(project, run_add_source):
    mo.stop(len(project.sources)<1, "Add a source first!")
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

    level_ui = mo.ui.radio(label="Compression level", options=_get_levels(),inline=True,value=_get_levels()[0])

    def _update_clip_level(_):
        clipping = [[
            cl_d["low_x"].value,
            cl_d["low_y"].value,
            cl_d["low_z"].value,
            ],[
            cl_d["high_x"].value,
            cl_d["high_y"].value,
            cl_d["high_z"].value,
        ]]
        clipping = clipping if cl_switch.value else None
        project.clipping = clipping
        project.compression_level = level_ui.value

    change_settings_button = mo.ui.button(label="Update project settings", on_click=_update_clip_level)

    mo.vstack([
        mo.md("<h3>Change compression and clipping</h3>"),
        cl_switch,
        cl_d,
        level_ui,
        change_settings_button,
    ])
    return (change_settings_button,)


@app.cell
def _(change_settings_button, project):
    change_settings_button
    mo.md(f"<h3>Current settings</h3>Compression level: <b>{project.compression_level}</b><br>Clipping: <b>{project.clipping}</b>")
    return


@app.cell
def _():
    run_progress = mo.ui.run_button(label="Refresh Progress View")
    run_progress
    return (run_progress,)


@app.cell
def _(run_progress):
    run_progress
    url = "http://localhost:8787/individual-progress"
    mo.iframe(
        f'<iframe src="{url}" width="100%" height="600" frameborder="0"></iframe>',
        width="100%",
        height="900px",
    )
    return


@app.cell
def _(project):
    def clear_caches(_):
        project.clear_caches(clear_disk_check.value)

    # Does not work, as capturing is broken
    def cache_info(_):
        with mo.redirect_stderr():
            project.get_caches()

    clear_disk_check = mo.ui.checkbox(value=False, label="Also clear disk cache")
    clear_disk_button = mo.ui.button(on_click=clear_caches, label="Clear Cache",kind="warn")
    cache_info_button = mo.ui.run_button(label="Cache Information")

    run_show_mesh = mo.ui.run_button(label="Show Mesh")

    box_dict = mo.ui.dictionary({
        "draw box": mo.ui.checkbox(value=False),
        "lower_x": mo.ui.number(value=0.0, start=0.0, stop=1.0),
        "lower_y": mo.ui.number(value=0.0, start=0.0, stop=1.0),
        "lower_z": mo.ui.number(value=0.0, start=0.0, stop=1.0),
        "upper_x": mo.ui.number(value=1.0, start=0.0, stop=1.0),
        "upper_y": mo.ui.number(value=1.0, start=0.0, stop=1.0),
        "upper_z": mo.ui.number(value=1.0, start=0.0, stop=1.0),
    }, label="Box settings")

    mo.output.append(mo.vstack([
        mo.md("## Controlls"),
        run_show_mesh,
        box_dict,
        cache_info_button,
        mo.hstack([
            clear_disk_button, 
            clear_disk_check,
        ],justify="start"),
    ]))
    None
    return box_dict, cache_info_button, run_show_mesh


@app.cell
def show_mesh(box_dict, project, run_show_mesh):
    mo.stop(not run_show_mesh.value, "Mesh will be displayed here")
    box = None
    if box_dict["draw box"].value:
        box = (
            (box_dict["lower_x"].value, box_dict["lower_y"].value, box_dict["lower_z"].value),
            (box_dict["upper_x"].value, box_dict["upper_y"].value, box_dict["upper_z"].value),
        )
    
    project.show(box=box).show()
    return


@app.cell
def _(cache_info_button, project):
    mo.stop(not cache_info_button.value, "Cache information")
    with mo.redirect_stderr():
        project.get_caches()
    return


@app.cell
def _():
    return


@app.cell
def _():


    return


if __name__ == "__main__":
    app.run()
