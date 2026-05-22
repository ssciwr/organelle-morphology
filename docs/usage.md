---
title: Usage
---

# Using Organelle Morphology

## Marimo GUI

Organelle Morphology comes with a graphical interface which runs in the browser
and is produced by marimo.

Make sure your conda environment is active and run:

```
marimo run notebooks/ui.py
```

Organelle Morphology should open in a new browser tab.

## Walkthrough

Lets go one by one from left to right, from top to bottom through the available panels in OM.

### Create a project

A `project` is a directory where OM can save data to. It *can* contain the microscopy data belonging to this project.
OM will write analysis outputs and its cache to the `project` directory.

In the UI, to select a directory make sure to click on the :lucide-folder: folder icon!
Create the `project` by clicking `Load Project`.

!!! info "Running computations"
    Note the moving hour glas in the top left indicating running computations.

### Loading Sources

OM works with three dimensional segmentations of microscopy images.
Currently, OM can only handle outputs from `CebraEM` in the `n5` format.
`CebraEM` stores the actual data in the `.n5` file and creates an additional `.xml` file in the same directory as the `n5`.

To load an image, in the `Add Source` panel, navigate to the directory containing your data.
Select one `xml` file and type the name you want to give the corresponding type of organelle below, then click `Load Source`.

Repeat until you have all sources loaded you need.

You can see the currently loaded sources below the loading panel.

!!! warning
    Loading many sources makes most computations slow, so only load what you need!

Having sources loaded activates most other functionality in the app.

### Compression and Clipping

To make computations faster and previewing more practical, OM allows to crop the data and not load everything. This is called clipping and works by defining the lowest corner and the highest corner of a cube. Everything outside is not loaded.
Make sure to activate the switch if you want to clip your data.

If your data comes with multiple compression levels, you can select here on which one you'd like to work.
`s3` is the corresponds to the lowest resolution, `s0` is full resolution.

OM computes meshes from the given segmentations. These meshes are by default very dense, they have a constant density of one vertex per voxel.
You can choose to simplify the mesh by a percent value, removing this amount of vertices from each mesh.

Mesh simplification can significantly speed up most computations in OM, but it also introduces a bit of error, especially to area computations.

### Progress

Here the progress of jobs running in the parallelization backend are shown.
You can open the dashboard to see more information (only clickable while the app is idle).

### Load Analysis Records

When the `Project` contains some analysis records from a previous run, you can load them here.

### Show Mesh

Here you can preview the mesh in 3D. By default the meshes are colored by organelle type.

* Organelle id filter: filter what is displayed. `*` can be any text. Separate multiple filters with a comma.
* Highlight ids: Choose which ids to highlight. Makes everything else gray.
If empty, the color is unchanged
* Skeleton: Display the skeleton, if computed. Make sure to activate `High-quality viewer` to make the mesh transparent.
* Curvature: Color the mesh by its local curvature.
* Color individual organelles: Each organelle instance is colored differently.
* MCS: Membrane contact site. See below for an explanation of all options.
* Box settings: Draw a box to help decide on clipping settings for higher resolution levels. Also used in `Get IDs of organelles in box`
* Show rotation: Display two lines indicating a rotation by a given angle. Yellow is the reference 0° line, orange is the rotated line.
* High-quality viewer: Opens the view in a separate window with better quality. (Transparency only works here)

Additional to the meshes, the chosen clipping box is shown, as well as an box enclosing the unclipped data.

The (0,0,0) corner is marked with three arrows along the axes. The red one points in x direction, the green one in y direction and the blue one in z direction.

### Cache

Display the contents of the project cache or clean it.

### Skeletonize

Calculate a skeleton for the selected organelles.

For information about the algorithms and options see the [skeletor documentation](https://navis-org.github.io/skeletor/skeletor/skeletonize.html)

### Membrane Contact Sites (MCS)

Calculate sites where two meshes are close.
Choose an upper and lower distance threshold to define what counts as a contact.
Then select between which groups of organelles to search for contacts.

You could set `Labels 1` to `er_*` and `Labels 2` to `*` to find all contacts with at least one `er` organelle involved.
`er_0001` and `mito_*` would find all contacts between the `er_0001` organelle and any `mito` organelle.

### Position Analysis

Create 2d or 1d histograms of organelle densities. In contrast to most analysis, this one works *not* with the mesh data, but with the raw volume. The `compression` setting is still respected.

First you have to choose which organelle type to consider.
Then select the bin size, choosing a higher bin size can speed up the calculations.

Select whether you want a 2d or 1d histogram.
For 2d you can select which axis to collapse, for 1d you can select the axis which you want to retain.

Before running the calculations, you can rotate the 3d volume if necessary to align it better to other data.

### Analysis Records

Running analysis generates records of these analysis. They contain the measurements and metadata of the settings how they were generated, as well as metadata about the sources and project.

Here you see an overview of all currently loaded records, and you can save them to the project folder.

### 2D Profile Calculations

Analyse the shape of organelles by looking at 2d cross-sections.

* `Fixed Axis` cuts perpendicular to a chosen axis
* `Random Planes` cuts using randomly generated planes
* `Skeleton` cuts perpendicular to a skeleton. The skeleton needs to be generated already!
