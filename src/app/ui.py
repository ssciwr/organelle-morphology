import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    app_title="Organelle Morphology",
    layout_file="layouts/ui.grid.json",
    auto_download=["html"],
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
    from organelle_morphology.analysis import Misc_Analysis, Mcs_Analysis
    from organelle_morphology.profile_calculations import ProfileCalculator
    import matplotlib.pyplot as plt
    from organelle_morphology.position import Position_Analysis
    from organelle_morphology.records import PropertyBlock
    from organelle_morphology.skeleton_analysis import Skeleton_Analysis
    from organelle_morphology.curvature_analysis import CurvatureAnalysis
    from collections import defaultdict


@app.cell
def _():
    mo.md(r"""
    # Organelle Morphology
    """)
    return


@app.cell
def _():
    project_path_ui = mo.ui.file_browser(
        selection_mode="directory", multiple=False, label="Project directory"
    )

    project_hint_loading = mo.md(
        "<small>(To select a folder, click on the &nbsp;::lucide:folder::&nbsp; icon.)</small>"
    ).style(margin_top="-1.0rem")  # adjust margin if the text ever disappears

    run_project_button = mo.ui.run_button(label="Load Project")

    mo.vstack(
        [
            mo.md("###Load/Create a Project"),
            project_path_ui,
            project_hint_loading,
            mo.hstack(
                [
                    run_project_button,
                ],
                justify="start",
            ),
        ]
    )
    return project_path_ui, run_project_button


@app.cell
def project_load_old_records(run_project_button):
    mo.stop(not run_project_button.value, "")

    run_project_button.value
    project_load_records_button = mo.ui.run_button(
        label="Load existing analysis records"
    )
    project_load_records_button
    return (project_load_records_button,)


@app.cell
def _(project, project_load_records_button):
    mo.stop(
        not project_load_records_button.value,
        f"Number of records loaded: {len(project.registry.get_all())}",
    )

    project.registry.load_all_from_yaml()
    mo.md(f"Number of records loaded: {len(project.registry.get_all())}")
    return


@app.cell
def _(project_path_ui, run_project_button):
    mo.stop(not run_project_button.value, "Load a project first")
    mo.stop(not len(project_path_ui.value) > 0, "Select a project directory!")

    project = om.Project(
        project_path=project_path_ui.path(), clipping=None, compression_level="s3"
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
        "Lower corner [%]:<br>{low_x} {low_y} {low_z} <br>Higher corner [%]:<br>{high_x} {high_y} {high_z}"
    ).batch(
        low_x=mo.ui.number(start=0.0, stop=100, value=0),
        low_y=mo.ui.number(start=0.0, stop=100, value=0),
        low_z=mo.ui.number(start=0.0, stop=100, value=0),
        high_x=mo.ui.number(start=0.0, stop=100, value=100),
        high_y=mo.ui.number(start=0.0, stop=100, value=100),
        high_z=mo.ui.number(start=0.0, stop=100, value=100),
    )

    def _get_levels():
        s = list(project.sources.values())[0]
        return s.metadata.levels

    level_ui = mo.ui.radio(
        label="Compression level",
        options=_get_levels(),
        inline=True,
        value=_get_levels()[0],
    )

    def _update_clip_level(_):
        clipping = [
            [
                cl_d["low_x"].value / 100,
                cl_d["low_y"].value / 100,
                cl_d["low_z"].value / 100,
            ],
            [
                cl_d["high_x"].value / 100,
                cl_d["high_y"].value / 100,
                cl_d["high_z"].value / 100,
            ],
        ]
        clipping = clipping if cl_switch.value else None
        project.clipping = clipping
        project.compression_level = level_ui.value
        project.simplify = simplify_ui.value

    def _get_simplify():
        return project.simplify

    simplify_ui = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.02,
        value=_get_simplify(),
        label="Mesh simplification [%]",
        show_value=True,
    )

    change_settings_button = mo.ui.button(
        label="Update project settings", on_click=_update_clip_level
    )

    mo.vstack(
        [
            mo.md("<h3>Change compression, simplification and clipping</h3>"),
            cl_switch,
            cl_d,
            level_ui,
            simplify_ui,
            change_settings_button,
        ]
    )
    return (change_settings_button,)


@app.cell
def _(change_settings_button, project):
    change_settings_button
    mo.md(
        f"<h3>Current settings</h3>Compression level: <b>{project.compression_level}</b>"
        f"<br>Clipping: <b>{project.clipping}</b>"
        f"<br>Mesh simplification: <b>{project.simplify}</b>"
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
def _(change_settings_button, project, sources):
    mo.stop(len(sources) < 1, "Add a source first!")

    # trigger update
    change_settings_button

    run_show_mesh = mo.ui.run_button(label="Show Mesh")
    mesh_id_filter = mo.ui.text(value="*", label="Organelle id filter")
    highlight_filter = mo.ui.text(value="", label="Highlight ids")

    box_dict = mo.ui.dictionary(
        {
            "box_checkbox": mo.ui.checkbox(value=False),
            "lower_x": mo.ui.number(value=0, start=0.0, stop=100, label="lower x"),
            "lower_y": mo.ui.number(value=0, start=0.0, stop=100, label="lower y"),
            "lower_z": mo.ui.number(value=0, start=0.0, stop=100, label="lower z"),
            "upper_x": mo.ui.number(value=100, start=0.0, stop=100, label="upper x"),
            "upper_y": mo.ui.number(value=100, start=0.0, stop=100, label="upper y"),
            "upper_z": mo.ui.number(value=100, start=0.0, stop=100, label="upper z"),
        },
        label="Box settings",
    )

    skeleton_check = mo.ui.checkbox(label="Skeleton", value=False)
    curvature_check = mo.ui.checkbox(label="Curvature", value=False)
    log_check = mo.ui.checkbox(label="log scale", value=True)

    mcs_checkbox = mo.ui.checkbox(label="MCS", value=False)
    mcs_min_ui = mo.ui.number(label="Min dist", value=0.0, step=0.001)
    mcs_max_ui = mo.ui.number(label="Max dist", value=0.1, step=0.001)
    mcs_filter_1_ui = mo.ui.text(value="*", label="Filter 1")
    mcs_filter_2_ui = mo.ui.text(value="*", label="Filter 2")

    color_volume_ui = mo.ui.number(label="Min volume", value=0.0, step=0.0001, start=0)

    rad = sources[0].curvature_radius

    curv_radius_slider = mo.ui.slider(
        label="radius", value=rad, start=0.0, stop=10 * rad, step=rad / 10
    )
    color_indiv_check = mo.ui.checkbox(label="Color individual organelles", value=False)
    popout_viewer_check = mo.ui.checkbox(label="High-quality viewer", value=False)

    mesh_rot_axis_ui = mo.ui.dropdown(
        options=["x", "y", "z"],
        label="Show rotation around axis: ",
        allow_select_none=True,
    )
    mesh_rot_angle_ui = mo.ui.number(
        label="Angle", start=-360, stop=360, value=0, step=1
    )

    mesh_export_toggle_ui = mo.ui.checkbox(label="Export Scene", value=False)
    mesh_export_name_ui = mo.ui.text(label="Scene Name", value="Organelles")

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
            mo.md(f"{mcs_checkbox} (Resolution: {project.resolution}) [$um$]"),
            mo.md(
                f"{mcs_min_ui}[$um$]<br>{mcs_max_ui}[$um$]<br>{mcs_filter_1_ui}<br>{mcs_filter_2_ui}"
            ),
            mo.md(
                f"Show helper box {box_dict['box_checkbox']}<br>"
                f" * {box_dict['lower_x']}[%]<br>"
                f" * {box_dict['lower_y']}[%]<br>"
                f" * {box_dict['lower_z']}[%]<br>"
                f" * {box_dict['upper_x']}[%]<br>"
                f" * {box_dict['upper_y']}[%]<br>"
                f" * {box_dict['upper_z']}[%]<br>"
            ),
            color_volume_ui,
            mo.hstack([mesh_rot_axis_ui, mesh_rot_angle_ui], justify="start"),
            mo.md("(yellow: reference 0°, orange: rotatated axis)"),
            mo.hstack(
                [
                    mesh_export_toggle_ui,
                    mesh_export_name_ui,
                ]
            ),
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
        color_volume_ui,
        curv_radius_slider,
        curvature_check,
        highlight_filter,
        log_check,
        mcs_checkbox,
        mcs_max_ui,
        mcs_min_ui,
        mesh_export_name_ui,
        mesh_export_toggle_ui,
        mesh_id_filter,
        mesh_rot_angle_ui,
        mesh_rot_axis_ui,
        popout_viewer_check,
        run_show_mesh,
        skeleton_check,
    )


@app.cell
def show_mesh(
    box_dict,
    color_indiv_check,
    color_volume_ui,
    curv_radius_slider,
    curvature_check,
    highlight_filter,
    log_check,
    mcs_checkbox,
    mcs_max_ui,
    mcs_min_ui,
    mesh_export_name_ui,
    mesh_export_toggle_ui,
    mesh_id_filter,
    mesh_rot_angle_ui,
    mesh_rot_axis_ui,
    popout_viewer_check,
    project,
    run_show_mesh,
    skeleton_check,
):
    mo.stop(not run_show_mesh.value, "Mesh will be displayed here")

    box = None
    if box_dict["box_checkbox"].value:
        box = (
            (
                box_dict["lower_x"].value / 100,
                box_dict["lower_y"].value / 100,
                box_dict["lower_z"].value / 100,
            ),
            (
                box_dict["upper_x"].value / 100,
                box_dict["upper_y"].value / 100,
                box_dict["upper_z"].value / 100,
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
        rot_axis=mesh_rot_axis_ui.value,
        rot_angle=mesh_rot_angle_ui.value,
        volume=color_volume_ui.value,
        export=mesh_export_name_ui.value if mesh_export_toggle_ui.value else None,
    )
    viewer = "marimo"
    if popout_viewer_check.value:
        viewer = "gl"
    scene.show(viewer=viewer)
    return


@app.cell
def set_volume_cutoff(project):
    volume_cutoff_ui = mo.ui.number(label="Minimum volume [$um^3$]", value=0.0, start=0)
    volume_to_bl_button = mo.ui.run_button(label="Add to blacklist")
    volume_clear_blacklist_button = mo.ui.button(
        label="Clear blacklist", on_click=lambda _: project.clear_blacklist()
    )
    mo.md(
        f"<h2>Set minimum Volume</h2>{volume_cutoff_ui}<br>"
        f"{volume_to_bl_button} {volume_clear_blacklist_button}"
    )
    return volume_clear_blacklist_button, volume_cutoff_ui, volume_to_bl_button


@app.cell
def calc_blacklist(project, volume_cutoff_ui, volume_to_bl_button):
    mo.stop(not volume_to_bl_button.value, "Add some organelles to the blacklist")
    project.blacklist_by_volume(volume_cutoff_ui.value)
    return


@app.cell
def show_blacklist(
    project,
    volume_clear_blacklist_button,
    volume_to_bl_button,
):
    volume_to_bl_button
    volume_clear_blacklist_button

    mo.md(
        "<h2>Blacklisted Organelles</h2>"
        f"Blocked: {len(project.permanent_blacklist)} Remaining: {len(project.organelles)}<br>"
    )
    return


@app.cell
def skeleton_run_button(project):
    skel_analysis = Skeleton_Analysis(project)
    run_skeleton_button = mo.ui.run_button(label="Run Skeletonization")
    run_skeleton_button
    return run_skeleton_button, skel_analysis


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
    skel_form = mo.md(r"""
    ## Skeletonize

    Method: {method} id filter: {ids}
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

    skel_form
    return (skel_form,)


@app.cell
def skeleton_progress(project, run_skeleton_button, skel_form):
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
def _(record_counts):
    record_counts
    return


@app.cell
def _(record_counts, skel_analysis):
    mo.stop(len(record_counts) < 1, "Skeleton Statistics")

    mo.vstack(
        [
            mo.md("## Skeletonization Statistics"),
            skel_analysis.get_dataframe(),
        ]
    )
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
        source = list(project.sources.values())[0]
        _scaling = np.array(source.metadata.size) * source.data_resolution
        _box = (
            np.array(
                (
                    box_dict["lower_x"].value,
                    box_dict["lower_y"].value,
                    box_dict["lower_z"].value,
                )
            )
            * _scaling
            * 0.01,
            np.array(
                (
                    box_dict["upper_x"].value,
                    box_dict["upper_y"].value,
                    box_dict["upper_z"].value,
                )
            )
            * _scaling
            * 0.01,
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
        output = (
            mo.md(f"Organelles: {orgs}<br>Counts: {counts}"),
            pd.DataFrame(result, columns=["IDs"]),
        )

        mo.output.replace(output)
        return result

    box_dict  # control flow
    mesh_id_filter  # control flow
    get_ids_box_ui = mo.ui.button(on_click=ids_in_box, label="Get IDs", value=None)
    mo.md(
        "<h3>Show IDs of Organelles in the box.</h3>"
        "Change the box in the `Show mesh` panel.<br>"
        f"{get_ids_box_ui}"
    )
    return


@app.cell
def _(sources):
    mo.stop(len(sources) < 1, "Add a source first!")
    of_ids_source = mo.ui.text(label="Labels 1", value="*")
    of_ids_target = mo.ui.text(label="Labels 2", value="*")
    of_filter_dist = mo.ui.number(label="Filter distance [$um$]", value=0.1)
    of_attribute = mo.ui.dropdown(
        label="Return type", options=["labels", "contacts", "objects"], value="labels"
    )
    of_run_button = mo.ui.run_button(label="Run filter")

    mo.vstack(
        [
            mo.md("Filter Organelles for distance-based calculations"),
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
def calculation_cell(
    of_attribute,
    of_filter_dist,
    of_ids_source,
    of_ids_target,
    of_run_button,
    project,
):
    calc_status = mo.md("Select filter settings and click 'Run filter'")
    mo.stop(not of_run_button.value, "Organelle filter results")
    calc_status = mo.md("Calculating...")
    try:
        filtered_distance_results = project.distance_filtering(
            of_ids_source.value,
            of_ids_target.value,
            of_filter_dist.value,
            of_attribute.value,
        )
        calc_status = mo.md("Distance calculation is done.")
    except Exception:
        calc_status = mo.md(
            f"Unable to run filter. Clear Cache and try again.\n\n```\n{traceback.format_exc()}\n```"
        )
    calc_status
    return (filtered_distance_results,)


@app.cell
def table_display_cell(filtered_distance_results, of_run_button):
    table_display = mo.md('Click on "Run filter" to see results here.')
    if of_run_button.value and not filtered_distance_results:
        table_display = mo.md("No results. Change filter to see results here.")
    elif of_run_button.value:
        table_display = pd.DataFrame.from_dict(
            filtered_distance_results, orient="index"
        )
    table_display
    return


@app.cell
def mcs_calc_button():
    run_mcs_btn = mo.ui.run_button(label="Calculate MCS")
    run_mcs_btn
    return (run_mcs_btn,)


@app.cell
def mcs_calc_ui_cell(change_settings_button, project, sources):
    mo.stop(len(sources) < 1, "MCS Analysis, add source first!")
    # trigger updates
    change_settings_button

    mcs_max_dist_ui = mo.ui.number(value=0.10, label="Max distance threshold")
    mcs_min_dist_ui = mo.ui.number(value=0.0, label="Min distance threshold")
    mcs_filter1_ui = mo.ui.text(label="Labels 1", value="*")
    mcs_filter2_ui = mo.ui.text(label="Labels 2", value="*")
    mcs_overwrite_ui = mo.ui.checkbox(
        value=False, label="Overwrite existing mcs results"
    )
    mo.vstack(
        [
            mo.md("## Membrane Contact Sites (MCS)"),
            mo.md(f"(Project resolution: {project.resolution} [$um$]"),
            mcs_max_dist_ui,
            mcs_min_dist_ui,
            mcs_filter1_ui,
            mcs_filter2_ui,
            mcs_overwrite_ui,
        ]
    )
    return (
        mcs_filter1_ui,
        mcs_filter2_ui,
        mcs_max_dist_ui,
        mcs_min_dist_ui,
        mcs_overwrite_ui,
    )


@app.cell
def _():
    return


@app.cell
def create_mcs_analysis(project):
    mcs_analysis = Mcs_Analysis(project=project)
    return (mcs_analysis,)


@app.cell
def mcs_execute_cell(
    mcs_filter1_ui,
    mcs_filter2_ui,
    mcs_max_dist_ui,
    mcs_min_dist_ui,
    mcs_overwrite_ui,
    project,
    run_mcs_btn,
):
    mo.stop(not run_mcs_btn.value, mo.md(""))
    project.search_mcs(
        min_distance=mcs_min_dist_ui.value,
        max_distance=mcs_max_dist_ui.value,
        ids_filter_1=mcs_filter1_ui.value,
        ids_filter_2=mcs_filter2_ui.value,
        overwrite_mcs_label=mcs_overwrite_ui.value,
    )  # Calculate the contact sites
    mcs_execute_status = mo.md(
        f"MCS successfully calculated for distance {mcs_min_dist_ui.value}-{mcs_max_dist_ui.value}."
    )
    mcs_execute_status
    return


@app.cell
def mcs_analysis_set_filter(mcs_analysis, project, record_counts):
    mo.stop(
        not project.registry.get_by_type("McsData"),
        mo.md("No MCS calculations run yet"),
    )
    record_counts
    mcs_labels = {r.meta.mcs_label for r in mcs_analysis.own_records}

    mcs_clear_records_button_ui = mo.ui.button(
        label="Remove all MCS records",
        kind="warn",
        on_click=lambda _: mcs_analysis.clean_own_records(),
    )

    mo.vstack(
        [
            mo.md("### MCS Analysis"),
            mo.md(f"{len(mcs_analysis.own_records)} mcs records loaded"),
            mo.md("<br>".join(mcs_labels)),
            mcs_clear_records_button_ui,
        ]
    )
    return


@app.cell
def mcs_analysis_overview(mcs_analysis, project, record_counts):
    mo.stop(
        not project.registry.get_by_type("McsData"),
        mo.md("No MCS calculations run yet"),
    )
    record_counts
    mo.ui.table(
        mcs_analysis.get_mcs_overview().reset_index(),
        page_size=14,
        selection=None,
        show_column_summaries=False,
    )
    return


@app.cell
def mcs_analysis_properties(mcs_analysis, project, record_counts):
    mo.stop(
        not project.registry.get_by_type("McsData"),
        mo.md("No MCS calculations run yet"),
    )
    record_counts
    mo.ui.table(mcs_analysis.get_mcs_properties(), selection=None, page_size=15)
    return


@app.cell
def geo_calc_ui_cell(sources):
    mo.stop(len(sources) < 1, "")
    run_geo_btn = mo.ui.run_button(label="Calculate Geometry")
    geo_calc_ui_layout = mo.vstack(
        [
            mo.md("## Geometry Properties"),
            mo.md("Compute voxel-based data (voxel_solidity, voxel_extent)."),
            run_geo_btn,
        ]
    )
    geo_calc_ui_layout
    return (run_geo_btn,)


@app.cell
def geo_execute_cell(project, run_geo_btn):
    mo.stop(not run_geo_btn.value, mo.md(""))
    geo_df = (
        project.geometric_properties
    )  # Accessing the property triggers the computation and caching
    geo_execute_status = mo.md(
        f"Geometry properties computed for {len(geo_df)} organelles."
    )
    geo_execute_status
    return


@app.cell
def create_run_profile_btn():
    run_profile_btn = mo.ui.run_button(label="Calculate Profiles")
    run_profile_btn
    return (run_profile_btn,)


@app.cell
def profile_calc_ui_cell(project, sources):
    mo.stop(len(sources) < 1, "")
    profile_calculator = ProfileCalculator(project)

    profile_ids_ui = mo.ui.text(value="*", label="Organelle ID filter (e.g., er_*)")

    profile_method_ui = mo.ui.radio(
        options=["Fixed Axis", "Random Planes", "Skeleton Perpendicular"],
        value="Fixed Axis",
        inline=True,
        label="Slicing Method",
    )

    fixed_axis_dict = mo.ui.dictionary(
        {
            "axis": mo.ui.text(value="z", label="Axis (x, y, z)"),
            "num_slices": mo.ui.number(
                value=20, start=1, step=1, label="Number of slices"
            ),
        }
    )

    random_planes_dict = mo.ui.dictionary(
        {
            "num_planes": mo.ui.number(
                value=20, start=1, step=1, label="Number of planes"
            ),
            "seed": mo.ui.number(value=42, start=0, step=1, label="Random seed"),
        }
    )

    skeleton_dict = mo.ui.dictionary(
        {
            "sample_distance": mo.ui.number(
                value=0.1, start=0.01, step=0.01, label="Sample distance"
            )
        }
    )

    profile_calc_ui_layout = mo.vstack(
        [
            mo.md(
                "## 2D Profile Calculations\nCalculate perimeter, width, and ratio of 2D cross-sections to determine tubule vs. sheet morphology."
            ),
            profile_ids_ui,
            profile_method_ui,
            mo.md("**Fixed Axis Settings:**"),
            fixed_axis_dict,
            mo.md("**Random Planes Settings:**"),
            random_planes_dict,
            mo.md("**Skeleton Perpendicular Settings:**"),
            skeleton_dict,
        ]
    )

    profile_calc_ui_layout  # display the layout
    return (
        fixed_axis_dict,
        profile_calculator,
        profile_ids_ui,
        profile_method_ui,
        random_planes_dict,
        skeleton_dict,
    )


@app.cell
def profile_execute_cell(
    fixed_axis_dict,
    profile_calculator,
    profile_ids_ui,
    profile_method_ui,
    random_planes_dict,
    run_profile_btn,
    skeleton_dict,
):
    mo.stop(
        not run_profile_btn.value, mo.md("Select method and click 'Calculate Profiles'")
    )

    profile_execute_cell_status = mo.md(
        "Calculating profiles... (Check Dask dashboard for progress)"
    )

    try:
        with mo.redirect_stderr():
            if profile_method_ui.value == "Fixed Axis":
                profile_calculator.calculate_profile_lengths(
                    ids=profile_ids_ui.value,
                    axis=fixed_axis_dict["axis"].value,
                    num_slices=fixed_axis_dict["num_slices"].value,
                )
            elif profile_method_ui.value == "Random Planes":
                profile_calculator.calculate_random_profiles(
                    ids=profile_ids_ui.value,
                    num_planes=random_planes_dict["num_planes"].value,
                    seed=random_planes_dict["seed"].value,
                )
            elif profile_method_ui.value == "Skeleton Perpendicular":
                profile_calculator.calculate_skeleton_profiles(
                    ids=profile_ids_ui.value,
                    sample_distance=skeleton_dict["sample_distance"].value,
                )
            else:
                raise ValueError(
                    f"Unknown profile calculation method: {profile_method_ui.value}"
                )
            df = profile_calculator.get_dataframe()

        profile_execute_cell_status = mo.vstack(
            [
                mo.md(
                    "**Success!** Profile properties "
                    f"computed using {profile_method_ui.value}."
                ),
                mo.ui.table(df, selection=None, pagination=True, page_size=15),
            ]
        )

    except Exception:
        profile_execute_cell_status = mo.md(
            f"## Unable to calculate profiles\n\n```\n{traceback.format_exc()}\n```"
        )

    profile_execute_cell_status  # display the status/table
    return


@app.cell
def prop_selector_cell(project, sources):
    mo.stop(len(sources) < 1, "Property statistics need loaded sources!")
    stats = Misc_Analysis(project, PropertyBlock)
    available_properties = (
        stats.get_mesh_properties()
        + stats.get_skeleton_properties()
        + stats.get_geometry_properties()
    )

    # Convert the list of available_properties keys into a dictionary of checkboxes
    properties_checkboxes = {}
    for key in available_properties:
        properties_checkboxes[key] = mo.ui.checkbox(value=False, label=f"{key}")

    prop_selector = mo.ui.dictionary(properties_checkboxes)
    calc_stats_btn = mo.ui.run_button(label="Calculate Statistics")

    display = mo.vstack([mo.md("## Select properties"), prop_selector, calc_stats_btn])

    display
    return calc_stats_btn, prop_selector


@app.cell
def prop_display_cell(calc_stats_btn, mesh_id_filter, project, prop_selector):
    stats_output = mo.md('Click on "Calculate Statistics" to show properties.')

    if calc_stats_btn.value:
        try:
            display_stats = Misc_Analysis(project, PropertyBlock)

            # Get the list of internal keys from the checkbox dictionary
            selected_properties = [
                key for key, checked in prop_selector.value.items() if checked
            ]

            # Generate the raw data table
            df_data = display_stats.get_dataframe(
                ids=mesh_id_filter.value, properties=selected_properties
            )

            # Generate the statistical summary table
            df_summary = display_stats.get_summary_dataframe(df_data)

            if not df_data.empty:
                # Identify column types for specialized formatting
                bool_cols = df_data.select_dtypes(include=[bool]).columns.tolist()
                float_cols = df_data.select_dtypes(
                    include=["float", "float64"]
                ).columns.tolist()

                # Define formatting for the Summary Table
                # Booleans show as percentage in 'Average' and '-' elsewhere
                summary_formats = {}
                for col in df_summary.columns:
                    if col in bool_cols:
                        summary_formats[col] = lambda x: (
                            f"{x * 100:.1f}%" if pd.notnull(x) else "-"
                        )
                    elif col in float_cols:
                        summary_formats[col] = "{:.3f}".format

                # Define formatting for the Raw Data Table
                data_formats = {col: "{:.3f}".format for col in float_cols}

                df_display = df_data.copy()
                for col in float_cols:
                    df_display[col] = df_display[col].replace([np.inf, -np.inf], np.nan)

                for col in ["mesh_centroid", "mesh_inertia"]:
                    if col in df_display.columns:
                        df_display[col] = df_display[col].astype(str)

                stats_output = mo.vstack(
                    [
                        mo.md("### Statistical Summary"),
                        mo.ui.table(
                            df_summary,
                            selection=None,
                            pagination=False,
                            format_mapping=summary_formats,
                            text_justify_columns={
                                col: "right" for col in (bool_cols + float_cols)
                            },
                        ),
                        mo.md(f"### Raw Organelle Data ({len(df_data)} items)"),
                        mo.ui.table(
                            df_display,
                            selection=None,
                            pagination=True,
                            max_height=400,
                            format_mapping=data_formats,
                            text_justify_columns={col: "right" for col in float_cols},
                        ),
                    ]
                )

        except NameError:
            stats_output = mo.md(
                "## No Project was loaded.\n ## Please load a project first."
            )
        except Exception:
            # Capture errors and display them
            stats_output = mo.md(
                f"## Unable to calculate statistics\n\n```\n{traceback.format_exc()}\n```"
            )
    stats_output
    return (df_data,)


@app.cell
def plot_selector_cell(calc_stats_btn, df_data):
    mo.stop(
        not calc_stats_btn.value,
        mo.md("Calculate statistics first to enable visualization."),
    )

    # Which properties can be plotted?
    plotable_columns = [
        col for col in df_data.columns if pd.api.types.is_numeric_dtype(df_data[col])
    ]
    mo.stop(not plotable_columns, mo.md("No properties selected for plotting."))

    # Create the dropdown for the property to plot
    plot_property_ui = mo.ui.dropdown(
        options=plotable_columns, value=plotable_columns[0], label="property:"
    )

    # Create dropdown for secondary plot options (e.g., histogram or scatter with another property)
    plot_selector_histogram = "freq. (->histogram)"
    plot_selector_secondary_options = [plot_selector_histogram] + plotable_columns
    plot_secondary_property_ui = mo.ui.dropdown(
        options=plot_selector_secondary_options,
        value=plot_selector_histogram,
        label="Y-axis:",
    )

    plot_selector_layout = mo.vstack(
        [
            mo.md("## Visualization"),
            mo.md("Choose a property for the plot:"),
            plot_property_ui,
            plot_secondary_property_ui,
        ]
    )
    plot_selector_layout  # show the layout
    return (
        plot_property_ui,
        plot_secondary_property_ui,
        plot_selector_histogram,
    )


@app.cell
def plot_display_cell(
    df_data,
    plot_property_ui,
    plot_secondary_property_ui,
    plot_selector_histogram,
):
    mo.stop(df_data is None or df_data.empty, mo.md("No data calculated yet."))

    prop_x = plot_property_ui.value
    prop_y = plot_secondary_property_ui.value

    fig, ax = plt.subplots(figsize=(8, 4))

    if prop_y == plot_selector_histogram:
        valid_data = df_data[prop_x].replace([np.inf, -np.inf], np.nan).dropna()
        mo.stop(
            not pd.api.types.is_numeric_dtype(valid_data), mo.md("Data is not numeric.")
        )

        counts, bins, patches = ax.hist(
            valid_data, bins=10, edgecolor="white", linewidth=1.2
        )
        ax.set_xticks(bins)
        ax.set_title(f"Distribution of {prop_x}", fontsize=14, pad=15)
        ax.set_xlabel(prop_x, fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    else:
        valid_data = (
            df_data[[prop_x, prop_y]].replace([np.inf, -np.inf], np.nan).dropna()
        )
        mo.stop(
            not pd.api.types.is_numeric_dtype(valid_data[prop_x])
            or not pd.api.types.is_numeric_dtype(valid_data[prop_y]),
            mo.md("Data is not numeric."),
        )

        ax.scatter(valid_data[prop_x], valid_data[prop_y], alpha=0.7)
        ax.set_title(f"{prop_y} vs {prop_x}", fontsize=14, pad=15)
        ax.set_xlabel(prop_x, fontsize=12)
        ax.set_ylabel(prop_y, fontsize=12)
        ax.grid(axis="both", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig
    return


@app.cell
def create_pa_run_button():
    pa_run_button = mo.ui.run_button(label="Run position analysis")
    pa_run_button
    return (pa_run_button,)


@app.cell
def _(change_settings_button, project, sources):
    mo.stop(not project, "Position Analysis, add a source first!")

    # trigger updates
    change_settings_button

    pa_source_ui = mo.ui.dropdown(
        options=[s.org_name for s in sources],
        label="Organelle types",
        value=[s.org_name for s in sources][0],
    )
    res_hint = f"(Project resolution: {project.resolution} [$um$])"
    pa_resolution_ui = mo.md(
        "Binning resolution [$um$]<br>" + res_hint + ": <br>{x} {y} {z}"
    ).batch(
        x=mo.ui.number(start=0.0, stop=1.0, value=0.1),
        y=mo.ui.number(start=0.0, stop=1.0, value=0.1),
        z=mo.ui.number(start=0.0, stop=1.0, value=0.1),
    )
    pa_marginal_axis_ui = mo.ui.dropdown(
        options=["x", "y", "z"],
        value="z",
        label="2D axis to average over",
        allow_select_none=False,
    )
    pa_axis1d_ui = mo.ui.dropdown(
        options=["x", "y", "z"],
        value="x",
        label="1D axis to measure along",
        allow_select_none=False,
    )
    pa_rot_angle_ui = mo.ui.number(
        start=-360, stop=360, value=0.0, label="Rotation angle of 3D volume [deg]"
    )
    pa_rot_axis_ui = mo.ui.dropdown(
        options=["x", "y", "z"],
        value="x",
        label="Rotation Axis",
        allow_select_none=False,
    )

    pa_dim_tab_ui = mo.ui.tabs(
        label="Dimensionality",
        tabs={
            "2D": pa_marginal_axis_ui,
            "1D": pa_axis1d_ui,
        },
        value="2D",
    )
    mo.vstack(
        [
            mo.md("## Position Analysis"),
            pa_source_ui,
            pa_resolution_ui,
            pa_dim_tab_ui,
            mo.hstack(
                [
                    pa_rot_angle_ui,
                    pa_rot_axis_ui,
                ],
                justify="start",
            ),
        ]
    )
    return (
        pa_axis1d_ui,
        pa_dim_tab_ui,
        pa_marginal_axis_ui,
        pa_resolution_ui,
        pa_rot_angle_ui,
        pa_rot_axis_ui,
        pa_source_ui,
    )


@app.cell
def _(
    pa,
    pa_axis1d_ui,
    pa_dim_tab_ui,
    pa_marginal_axis_ui,
    pa_resolution_ui,
    pa_rot_angle_ui,
    pa_rot_axis_ui,
    pa_run_button,
    pa_source_ui,
    sources,
):
    mo.stop(not pa_run_button.value, "Run a position analysis")
    if pa_run_button.value:
        pa_source = [s for s in sources if s.org_name == pa_source_ui.value][0]
        pa_bin_res = (
            pa_resolution_ui["x"].value,
            pa_resolution_ui["y"].value,
            pa_resolution_ui["z"].value,
        )

        if pa_dim_tab_ui.value == "2D":
            pa_marginal_axis = {"x": 0, "y": 1, "z": 2}[pa_marginal_axis_ui.value]
            pa_rot_axis = {"x": 0, "y": 1, "z": 2}[pa_rot_axis_ui.value]
            print("axis", pa_rot_axis)
            pa.density2D(
                source=pa_source,
                bin_resolution=pa_bin_res,
                marginal_axis=pa_marginal_axis,
                rot_angle=pa_rot_angle_ui.value,
                rot_axis=pa_rot_axis,
            )
        if pa_dim_tab_ui.value == "1D":
            pa_axis1d = {"x": 0, "y": 1, "z": 2}[pa_axis1d_ui.value]
            pa_rot_axis = {"x": 0, "y": 1, "z": 2}[pa_rot_axis_ui.value]
            pa.density1D(
                source=pa_source,
                bin_resolution=pa_bin_res,
                axis=pa_axis1d,
                rot_angle=pa_rot_angle_ui.value,
                rot_axis=pa_rot_axis,
            )
        else:
            "Unknown dimensionality"
    return


@app.cell
def create_position_analysis(project):
    pa = Position_Analysis(project)
    return (pa,)


@app.cell
def position_analysi_table(pa, record_counts):
    # update from loading and calculating
    record_counts

    pa_metas = []
    for i, r in enumerate(pa.own_records):
        pa_meta = r.meta.__dict__
        pa_meta["index"] = i
        pa_metas.append(pa_meta)

    pa_plot_densities_button = mo.ui.run_button(label="Plot densities")
    pa_records_table = mo.ui.table(pa_metas)

    mo.vstack(
        [
            mo.md("#### Select which entries to plot. 3D plots are expensive to show"),
            pa_records_table,
            pa_plot_densities_button,
        ]
    )
    return pa_plot_densities_button, pa_records_table


@app.cell
def _(pa, pa_plot_densities_button, pa_records_table):
    mo.stop(not pa_plot_densities_button.value, "Density plots")
    pa_idxs = [m["index"] for m in pa_records_table.value]
    pa_records_filterd = [r for i, r in enumerate(pa.own_records) if i in pa_idxs]
    if any([r.meta.dimensionality == 3 for r in pa_records_filterd]):
        mo.output.replace(
            mo.mpl.interactive(pa.plot_multiple_densities(pa_records_filterd)[0])
        )
    else:
        mo.output.replace(pa.plot_multiple_densities(pa_records_filterd)[0])
    return


@app.cell
def _(record_counts, records_save_button, records_update_button):
    record_count_to_table = []
    if len(record_counts):
        record_count_to_table = record_counts

    mo.vstack(
        [
            mo.md("## Analysis Records"),
            mo.ui.table(record_count_to_table, selection=None),
            mo.hstack(
                [
                    records_update_button,
                    records_save_button,
                ],
                justify="start",
            ),
        ]
    )
    return


@app.cell
def _(
    ca_run_button,
    pa_run_button,
    project,
    project_load_records_button,
    records_update_button,
    run_mcs_btn,
    run_profile_btn,
    run_skeleton_button,
):
    # Count the records
    # Get's triggered by any calculation or loading of records.
    # Should be used to trigger the update of any record visualization, like tables

    project_load_records_button
    run_mcs_btn
    pa_run_button
    run_profile_btn
    run_skeleton_button
    records_update_button
    ca_run_button

    [r.meta for r in project.records]
    record_counts = defaultdict(int)
    for rec in project.records:
        record_counts[rec.name] += 1

    record_counts
    return (record_counts,)


@app.cell
def _():
    records_update_button = mo.ui.button(
        label="Update records",
    )
    records_save_button = mo.ui.run_button(label="Save all records")
    return records_save_button, records_update_button


@app.cell
def _(project, records_save_button):
    mo.stop(not records_save_button.value, "Save all records")
    project.registry.save_all_to_yaml()
    mo.md("Saved successfully!")
    return


@app.cell
def curveanalysis_obj(project):
    # curvature analysis
    ca = CurvatureAnalysis(project)
    return (ca,)


@app.cell
def create_ca_run_button():
    ca_run_button = mo.ui.run_button(label="Run curvature analysis")
    ca_run_button
    return (ca_run_button,)


@app.cell
def _(sources):
    # UI for curvature analyis
    mo.stop(len(sources) < 1, "Add sources first!")
    _rad = sources[0].resolution[0] * 2

    ca_radius_ui = mo.ui.number(
        label="Radius $[um]$", value=_rad, start=0.0, stop=10 * _rad, step=_rad / 10
    )
    ca_filter_ui = mo.ui.text(label="Organelle filter", value="*")
    mo.vstack(
        [
            mo.md("## Curvature Analysis"),
            mo.md("Based on the sum of angles within a circle on the mesh surface"),
            ca_radius_ui,
            ca_filter_ui,
        ]
    )
    return ca_filter_ui, ca_radius_ui


@app.cell
def _(ca, ca_filter_ui, ca_radius_ui, ca_run_button, project):
    # Run curvature analysis
    mo.stop(not ca_run_button.value, "Run curvature analysis")

    project.set_curvature_radius(ca_radius_ui.value)
    ca.calculate_curvature_stats(ids=ca_filter_ui.value)
    return


@app.cell
def _(ca, record_counts):
    # Curvature analysis outputs
    record_counts
    mo.ui.table(
        ca.get_dataframe(),
        page_size=15,
        format_mapping={
            "min_curvature": "{:.5g}",
            "max_curvature": "{:.5g}",
            "mean_curvature": "{:.5g}",
            "std_curvature": "{:.5g}",
            "median_curvature": "{:.5g}",
            "mean_absolute_curvature": "{:.5g}",
        },
        selection=None,
    )
    return


@app.cell
def ca_record_control(ca, record_counts):
    record_counts

    ca_clean_own_button = mo.ui.button(
        label="Remove all curvature records",
        on_click=lambda _: ca.clean_own_records(),
        kind="warn",
    )

    mo.vstack(
        [
            mo.md("## Curvature Records"),
            mo.md(f"{len(ca.own_records)} curvature records loaded."),
            ca_clean_own_button,
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
