import marimo

__generated_with = "0.19.6"
app = marimo.App(
    width="medium",
    app_title="Organelle Morphology",
    layout_file="layouts/ui.grid.json",
)

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    import organelle_morphology as om
    import numpy as np
    from organelle_morphology.util import bounding_box_delayed
    from dask.base import compute
    import pandas as pd
    import traceback
    from organelle_morphology.statistics import Statistics


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

    hint = mo.md(
        "<small>(To select a folder, click on the &nbsp;::lucide:folder::&nbsp; icon.)</small>"
    ).style(margin_top="-1.0rem") # adjust margin if the text ever disappears

    run_project = mo.ui.run_button(label="Load Project")

    mo.vstack(
        [
            mo.md("###Load/Create a Project"),
            path_ui,
            hint,
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
        entry = source_path_ui.value[0]
        print("entry:", entry, project)
        project.add_source(xml_path=entry.path, organelle=new_organelle_name_ui.value)
        return True

    def get_source_header():
        sources = "<br>".join([s.xml_path.name for s in project.sources.values()])
        return mo.md(f"<h3>Current sources:</h3>{sources}")

    source_path_ui = mo.ui.file_browser(
        initial_path=str(project.path),
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
def _(get_source_header, project, run_add_source):
    run_add_source
    sources = list(project.sources.values())
    get_source_header()
    return (sources,)


@app.cell
def _(project, run_add_source, sources):
    mo.stop(len(sources) < 1, "Add a source first!")
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
def _(project):
    import webbrowser

    def open_url_in_new_tab(url):
        try:
            webbrowser.open_new_tab(url)
            return True
        except Exception as e:
            print(f"Failed to open URL: {e}")
            return False

    show_dashboard_button = mo.ui.button(
        label="Show Dashboard",
        on_click=lambda _: open_url_in_new_tab(project.client.dashboard_link),
    )
    reset_client_button = mo.ui.button(
        label="Reset dask",
        on_click=lambda _: project.recreate_client(),
    )

    mo.hstack([show_dashboard_button, reset_client_button], justify="center")
    return


@app.cell
def _(project):
    # mo.stop(run_progress.value == 0, "Please refresh once the project is created")
    url = project.client.dashboard_link.split("/")
    url = "/".join(url[:-1]) + "/individual-progress"
    mo.iframe(
        f'<iframe src="{url}" width="100%" height="600" frameborder="0"></iframe>',
        width="100%",
        height="900px",
    )
    return


@app.cell
def _(sources):
    mo.stop(len(sources) < 1, "Add a source first!")
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

    skeleton_check = mo.ui.checkbox(label="Skeleton", value=False)
    curvature_check = mo.ui.checkbox(label="Curvature", value=False)
    log_check = mo.ui.checkbox(label="log scale", value=True)

    mcs_checkbox = mo.ui.checkbox(label="MCS", value=False)
    mcs_min_ui = mo.ui.number(label="Min dist", value=0.0, step=0.001)
    mcs_max_ui = mo.ui.number(label="Max dist", value=0.1, step=0.001)

    rad = sources[0].curvature_radius

    curv_radius_slider = mo.ui.slider(
        label="radius", value=rad, start=0.0, stop=10 * rad, step=rad / 10
    )
    color_indiv_check = mo.ui.checkbox(label="Color individual organelles", value=False)
    popout_viewer_check = mo.ui.checkbox(label="High-quality viewer", value=False)

    mo.vstack(
        [
            mo.md("## Show Mesh"),
            mesh_id_filter,
            highlight_filter,
            skeleton_check,
            mo.hstack(
                [
                    curvature_check,
                    log_check,
                    curv_radius_slider,
                ],
                justify="start",
            ),
            color_indiv_check,
            mo.hstack(
                [mcs_checkbox, mo.md(f"(Resolution: {sources[0].resolution})")],
                justify="start",
            ),
            mcs_min_ui,
            mcs_max_ui,
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
        color_indiv_check,
        curv_radius_slider,
        curvature_check,
        highlight_filter,
        log_check,
        mcs_checkbox,
        mcs_max_ui,
        mcs_min_ui,
        mesh_id_filter,
        popout_viewer_check,
        run_show_mesh,
        skeleton_check,
    )


@app.cell
def show_mesh(
    box_dict,
    color_indiv_check,
    curv_radius_slider,
    curvature_check,
    highlight_filter,
    log_check,
    mcs_checkbox,
    mcs_max_ui,
    mcs_min_ui,
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

    mcs_min = None
    mcs_max = None
    if mcs_checkbox.value:
        mcs_min = mcs_min_ui.value
        mcs_max = mcs_max_ui.value

    scene = project.show(
        box=box,
        ids=mesh_id_filter.value,
        ids_highlight=highlight,
        curvature=curvature_check.value,
        skeleton=skeleton_check.value,
        curv_log=log_check.value,
        color_instances=color_indiv_check.value,
        mcs_min=mcs_min,
        mcs_max=mcs_max,
    )
    viewer = "marimo"
    if popout_viewer_check.value:
        viewer = "gl"
    scene.show(viewer=viewer)
    return


@app.cell
def _(sources):
    mo.stop(len(sources) < 1, "Add a source first!")
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
            project.skeletonize_wavefront(**settings)
        elif form["method"] == "vertex cluster":
            settings.pop("waves")
            project.skeletonize_vertex_clusters(**settings)
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
        orgs = " , ".join([o + r"_\*" for o in orgs])
        output = mo.md(f"Organelles: {orgs}<br>Counts: {counts}"), pd.DataFrame(result, columns=["IDs"])

        mo.output.replace(output)
        return result

    box_dict  # control flow
    mesh_id_filter  # control flow
    mo.ui.button(on_click=ids_in_box, label="Get IDs of organelles in box", value=None)
    return


@app.cell
def _(sources):
    mo.stop(len(sources) < 1, "Add a source first!")
    of_ids_source = mo.ui.text(label="Labels 1", value="*")
    of_ids_target = mo.ui.text(label="Labels 2", value="*")
    of_filter_dist = mo.ui.number(label="Filter distance [um]", value=1)
    of_attribute = mo.ui.dropdown(
        label="Return type", options=["labels", "contacts", "objects"], value="labels"
    )
    of_run_button = mo.ui.run_button(label="Run filter")

    mo.vstack(
        [
            mo.md("Filter Organelles"),
            of_ids_source,
            of_ids_target,
            of_filter_dist,
            of_attribute,
            of_run_button,
        ]
    )
    return (
        of_attribute,
        of_filter_dist,
        of_ids_source,
        of_ids_target,
        of_run_button,
    )


@app.cell
def _(
    of_attribute,
    of_filter_dist,
    of_ids_source,
    of_ids_target,
    of_run_button,
    project,
):
    mo.stop(not of_run_button.value, "Organelle filter results")

    project.distance_filtering(
        of_ids_source.value,
        of_ids_target.value,
        of_filter_dist.value,
        of_attribute.value,
    )
    return


@app.cell
def _(of_run_button, project):
    of_run_button.value
    mo.ui.table(project.distance_matrix,
                selection=None,
                pagination=False,
                max_height=400,
                )
    return


@app.cell
def prop_selector_cell(project):

    stats =Statistics(project)
    available_properties = stats.get_properties()

    # Convert the list of available_properties keys into a dictionary of checkboxes
    properties_checkboxes = {}
    for key in available_properties:
        properties_checkboxes[key] = mo.ui.checkbox(value=True, label=f"{key}")


    prop_selector = mo.ui.dictionary(properties_checkboxes)
    calc_stats_btn = mo.ui.run_button(label="Calculate Statistics")

    display = mo.vstack([
        mo.md("## Select properties"),
        prop_selector,
        calc_stats_btn
    ])

    display
    return calc_stats_btn, prop_selector


@app.cell
def stats_display_cell(calc_stats_btn, mesh_id_filter, project, prop_selector):
    # Standard prompt before the button is clicked
    stats_output = mo.md("Click on \"Calculate Statistics\" to compute geometry properties.")

    if calc_stats_btn.value:
        try:
            display_stats = Statistics(project)

            # Get the list of internal keys from the checkbox dictionary
            selected_properties = [
                key for key, checked in prop_selector.value.items() if checked
            ]

            # Generate the raw data table
            df_data = display_stats.get_dataframe(
                ids=mesh_id_filter.value, 
                properties=selected_properties
            )

            # Generate the statistical summary table
            df_summary = display_stats.get_summary_dataframe(df_data)

            if not df_data.empty:
                # Identify column types for specialized formatting
                bool_cols = df_data.select_dtypes(include=[bool]).columns.tolist()
                float_cols = df_data.select_dtypes(include=['float', 'float64']).columns.tolist()

                # Define formatting for the Summary Table
                # Booleans show as percentage in 'Average' and '-' elsewhere
                summary_formats = {}
                for col in df_summary.columns:
                    if col in bool_cols:
                        summary_formats[col] = lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "-"
                    elif col in float_cols:
                        summary_formats[col] = "{:.3f}".format

                # Define formatting for the Raw Data Table
                data_formats = {col: "{:.3f}".format for col in float_cols}

                stats_output = mo.vstack([
                    mo.md("### Statistical Summary"),
                    mo.ui.table(
                        df_summary,
                        selection=None,
                        pagination=False,
                        format_mapping=summary_formats,
                        text_justify_columns={col: "right" for col in (bool_cols + float_cols)},
                    ),
                    mo.md(f"### Raw Organelle Data ({len(df_data)} items)"),
                    mo.ui.table(
                        df_data,
                        selection=None,
                        pagination=False,
                        max_height=400,
                        format_mapping=data_formats,
                        text_justify_columns={col: "right" for col in float_cols},
                    ),
                ])

        except NameError:
            stats_output = mo.md("## No Project was loaded.\n ## Please load a project first.")
        except Exception:
            # Capture errors and display them
            stats_output = mo.md(f"## Unable to calculate statistics\n\n```\n{traceback.format_exc()}\n```")
    stats_output 
    return


@app.cell
def _(sources):
    mo.stop(len(sources) < 1, "Add a source first!")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
