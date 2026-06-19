import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium", app_title="Organelle Morphology - Analysis")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    import organelle_morphology as om
    import numpy as np
    from organelle_morphology.statistics import Stats


@app.cell
def _():
    mo.md(r"""
    # Organelle Morphology - Analysis
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
    ).style(margin_top="-1.0rem")  # adjust margin if the text ever disappears

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

    stat_yamls = list((path_ui.path() / "analysis").rglob("*.yaml"))
    print(f"Found {len(stat_yamls)} stats")

    stats = [Stats.from_yaml(s) for s in stat_yamls]
    return (stat_yamls,)


@app.cell
def _(stat_yamls):
    stat_yamls

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
