---
title: organelle-morphology on the HPC
---

# Why use the HPC?
Large amounts of data cannot always be processed on a normal computer. In such cases, a High-Performance Computing (HPC) cluster is needed. This guide explains how to use the [Helix cluster](https://wiki.bwhpc.de/e/Helix).

# Installation on Helix
First, log into Helix:
```bash
ssh your_helix_user_name@helix.bwservices.uni-heidelberg.de
```

Then, load the Conda module:
```bash
module purge
module load devel/miniforge/24.9.2
```
(If `devel/miniforge/24.9.2` is not available, search for another version of Conda.)

Next, clone the `organelle-morphology` repository:
```bash
git clone https://github.com/ssciwr/organelle-morphology.git
cd organelle-morphology
```

Create the Conda environment from the provided file and activate it:
```bash
conda env create -f environment-dev.yml
conda activate morph
```

Finally, install the package in editable mode:
```bash
pip install -e .
```

# Moving Data to the HPC
To move your data to the HPC, you can use command-line tools like `scp` or `rsync`.

`scp` is useful for copying single files. For example, to copy a local file to your home directory on Helix, run:
```bash
scp /path/to/your/local/data.h5 your_helix_user_name@helix.bwservices.uni-heidelberg.de:~/
```

`rsync` is better for synchronizing entire directories. It only transfers the changes, which is efficient for large datasets.
```bash
rsync -avz /path/to/your/local/data_directory/ your_helix_user_name@helix.bwservices.uni-heidelberg.de:~/data_directory/
```

# Creating a Script
For reproducible results, it is best to write a script to run your analysis. You can find examples in the `organelle-morphology/scripts` directory.

# Interactive Sessions
Please do not run heavy computations on the login node. For development and testing, request an interactive compute node:
```bash
salloc --partition=cpu-single --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=00:30:00 --job-name=om-interactive
```

After your allocation is granted, your shell prompt will change to a compute node. Load the environment again:
```bash
module purge
module load devel/miniforge/24.9.2
conda activate morph
```

You can run tests with:
```bash
python -m pytest
```

# Batch Processing with Slurm
For long-running analyses, submit a batch job using a Slurm script.

### Create a Slurm Script
Create a file named `run_om.slurm` with the following content:
```slurm
#!/bin/bash
#SBATCH --job-name=om
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=cpu-single
#SBATCH --output=om_%j.out
#SBATCH --error=om_%j.err

# Clear environment and load modules
module purge
module load devel/miniforge/24.9.2

# Activate the conda environment
conda activate morph

# Run your analysis script
python $HOME/organelle-morphology/scripts/benchmark.py
```

### Submit the Job
Submit the script to the Slurm scheduler:
```bash
sbatch run_om.slurm
```

# Running Marimo Notebooks on the HPC
Running Marimo notebooks on the HPC is possible but requires setting up an SSH tunnel.

### Request Resources
On the Helix login node, request an interactive session:
```bash
salloc --partition=cpu-single --ntasks=1 --cpus-per-task=16 --time=00:30:00
```

### Start Marimo on the Compute Node
Once on the compute node, note its hostname. Then, start the Marimo server, binding to `0.0.0.0` to make it accessible on the internal network.
```bash
cd organelle-morphology
module load devel/miniforge/24.9.2
conda activate morph
marimo run notebooks/ui.py --host 0.0.0.0 --port 2718
```

### Create an SSH Tunnel
To access the Marimo server, create an SSH tunnel from your local machine. Open a **new terminal** and run the following command, replacing `COMPUTE_NODE_HOSTNAME` with the actual hostname.
```bash
ssh -N -L 2718:COMPUTE_NODE_HOSTNAME:2718  your_helix_user_name@helix.bwservices.uni-heidelberg.de
```
After entering your credentials, the command will not produce further output. This is expected.

### Access Marimo in Your Browser
Open your local web browser and navigate to:
[http://localhost:2718](http://localhost:2718)
