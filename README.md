# Welcome to organelle-morphology

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ssciwr/organelle-morphology/ci.yml?branch=main)](https://github.com/ssciwr/organelle-morphology/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/organelle-morphology/badge/)](https://organelle-morphology.readthedocs.io/)
[![codecov](https://codecov.io/gh/ssciwr/organelle-morphology/branch/main/graph/badge.svg)](https://codecov.io/gh/ssciwr/organelle-morphology)

**This project is currently under construction**

## Installation
To get started, clone the repository and set up the Conda environment:

```bash
git clone https://github.com/ssciwr/organelle-morphology.git
cd organelle-morphology
conda env create -f environment-dev.yml
conda activate morph
```

### Interactive UI
To start the graphical user interface, run:

```bash
marimo run notebooks/ui.py
```
Click on the displayed link or navigate to http://localhost:2718 in your web browser.

### Tests
The test suite can be run using pytest:
```bash
python -m pytest
```

## Running on Helix

### Installation (Run on Helix Login Node)
```bash
ssh <your_helix_username>@helix.bwservices.uni-heidelberg.de
```

```bash
git clone https://github.com/ssciwr/organelle-morphology.git
cd organelle-morphology
module load devel/miniforge/24.9.2
mamba env create -f environment-dev.yml
mamba clean --all -y
```

### Move Data to Helix (Run on Local Machine)
```bash
scp -r /path/to/local/data <your_helix_username>@helix.bwservices.uni-heidelberg.de:~/target_directory/
```

### Option A: Run the User Interface interactively:
#### Request Resources (Run on Helix Login Node)
```bash
salloc --partition=cpu-single --ntasks=1 --cpus-per-task=16 --time=02:00:00
```
This will drop you into the compute node. Check the name of this compute node.

#### Start Web Interface (Run on Helix Compute Node)
```bash
cd organelle-morphology
module load devel/miniforge/24.9.2
conda activate morph
marimo run notebooks/ui.py --host 0.0.0.0 --port 2718
```
#### Connect Browser (Run on Local WSL)
```bash
ssh -N -L 2718:<name_of_the_compute_node>:2718 <your_helix_username>@helix.bwservices.uni-heidelberg.de
```
This will look like it hangs.
Open http://localhost:2718 in your local web browser.


### Option B: Run a Python script via Slurm
For reproducable results it is best to use scripts.
create a python script for your work, for inspiration see:
organelle-morphology/scripts/helix-demo.py

To run that python script you need a slurm script, for inspiration see:
organelle-morphology/scripts/om-helix.slurm

Submit your job to the cluster scheduler from the login node:
```bash
sbatch om-helix.slurm
```
