# Installation

## Prerequisites

To install, you need to have `conda` and `git` installed. We recommend using `miniforge`, get it [here from conda-forge](https://conda-forge.org/download/).
Install `git` if you don't have it already from [here](https://git-scm.com/install/)

## Installation

On Windows, use your conda installation to open a shell,
On linux/mac open a terminal.

The following lines download the organelle morphology and install all necessary dependencies:

```
git clone https://github.com/ssciwr/organelle-morphology.git
cd organelle-morphology
conda env create -f environment-dev.yml
conda activate morph
```


## Run Organelle Morphology

Make sure your `conda` environment is active, and navigate to the repository:

```
conda activate morph
```

Then start Organelle Morphology with:

```
om_app
```


or navigate to the repository and run marimo directly:

```
cd organelle-morphology
marimo run src/app/ui.py
```
