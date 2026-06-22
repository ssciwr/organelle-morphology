# Installation

## Prerequisites

To install on windows, we suggest you use `conda` and `git`.
We recommend using `miniforge`, get it [here from conda-forge](https://conda-forge.org/download/).
Install `git` if you don't have it already from [here](https://git-scm.com/install/).

On Linux/Mac, you can also use `conda`, but we recommend using `uv`.
Install `uv` by running: `curl -LsSf https://astral.sh/uv/install.sh | sh`  
For more information, see [uv docs](https://docs.astral.sh/uv/getting-started/installation/)

## Installation

=== "Windows"

    On Windows, use your conda installation to open a shell.
    The following lines download organelle morphology and install all necessary dependencies:

    ```
    git clone https://github.com/ssciwr/organelle-morphology.git
    cd organelle-morphology
    conda env create -f environment-dev.yml
    conda activate morph
    ```

=== "Linux/Mac"

    After having installed `uv`, run:
    ```bash
    uv tool install --from git+https://github.com/ssciwr/organelle-morphology organelle-morphology
    ```

## Run Organelle Morphology

=== "Conda"

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

=== "uv"

    After installing organelle-morphology as a `uv tool`, you can run the app from anywhere:

    ```bash
    om_app
    ```
