# Running on HPC

## Why use the HPC?

The main bottleneck for Organelle Morphology is memory.
Loading a large part of an image at a high resolution might therefore not be possible on a usual PC.

HPC clusters like Helix offer more RAM, so Organelle Morphology can be used to work on larger data.

You can run analysis on the cluster, save the `records` and import them on a local machine again, without needing to load the actual sources again.


## Installation on Helix

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

Next, clone the `organelle-morphology` repository, and move into the new directory:

```bash
git clone https://github.com/ssciwr/organelle-morphology.git
cd organelle-morphology
```

Create the Conda environment from the provided file and activate it:

```bash
conda env create -f environment-dev.yml
conda activate morph
```

## Moving Data to the HPC

### Workspaces

On Helix, large data should be stored in temporary workspaces.
To create a workspace for organelle morphology valid for 30 days, run:

```bash
ws_allocate -r 7 -m <email> morph 30
```

Create a link to the workspace with:

```bash
ws_register workspaces
```

Now you should have a directory `workspaces/gpfs/` containing your workspace `<username>-morph`.
Software can be installed in your home directory (`~`), large data should be in the workspace.

See [helix documentation](https://wiki.bwhpc.de/e/Helix/Filesystems) for more informations.

### File transfer

To move your data to the HPC, you can use command-line tools like `scp` or `rsync`.

`scp` is useful for copying single files. For example, to copy a local file to your home directory on Helix, run:

```bash
scp /path/to/your/local/data.h5 your_helix_user_name@helix.bwservices.uni-heidelberg.de:~/
```

`rsync` is better for synchronizing entire directories. It only transfers the changes, which is efficient for large datasets.

```bash
rsync -avz /path/to/your/local/data_directory/ your_helix_user_name@helix.bwservices.uni-heidelberg.de:~/data_directory/

```

## Submitting to a cluster

#### 1. Creating a Script

You control Organelle Morphology through a python script.
First create and navigate to a project folder. This will contain the analysis results and run scripts.

```bash
mkdir ~/workspaces/gpfs/<workspace name>/<project name>
cd ~/workspaces/gpfs/<workspace name>/<project name>
```


Use the `om_init_hpc` command to copy a template scripts to your current working directory.

```bash
# For using a single node
om_init_hpc single

# For using two nodes
om_init_hpc multi

# For benchmark processing
om_init_hpc benchmark
```

This will copy two files to your current directory:
- `hpc_example.py` - The main Python script template
- `helix_run_[choice].sh` - The Slurm batch script template

Modify `hpc_example.py` to create the analysis you want to run.
You can use `nano` or `vim` to edit the files on the hpc,
or you can prepare them locally and copy them over using the file
transfer tools described above.


* Change compression and clipping
* Change the sources you want to load
* Change the analysis steps you want to run
* Remove the other steps you don't need.

In the `.sh` slurm submit script, change

* the path to the project
* the path to the python script (`hpc_exampel.py`)
* the path to the data
* if necessary, the runtime and/or processes (`ntasks`)


#### 2. Submitting the script

After creating and modifying the `hpc_example.py` and the `helix_run_single.sh`, you need to submit the job to the cluster and wait for it to run:

```bash
sbatch helix_run_single.sh
```

You can read the log of organelle morphology with `less om2.log`.
Press `shift-f` to auto-refresh while the computations are running.


## Interactive Sessions

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


## Running Marimo Notebooks on the HPC

Running Marimo notebooks on the HPC is possible but requires setting up an SSH tunnel.

#### 1. Request Resources

On the Helix login node, request an interactive session:

```bash
salloc --partition=cpu-single --ntasks=1 --cpus-per-task=16 --time=00:30:00
```

#### 2. Start Marimo on the Compute Node

Once on the compute node, note its hostname. Then, start the Marimo server, binding to `0.0.0.0` to make it accessible on the internal network.

```bash
cd organelle-morphology
module load devel/miniforge/24.9.2
conda activate morph
marimo run src/app/ui.py --host 0.0.0.0 --port 2718
```

#### 3. Create an SSH Tunnel

To access the Marimo server, create an SSH tunnel from your local machine. Open a **new terminal** and run the following command, replacing `COMPUTE_NODE_HOSTNAME` with the actual hostname.

```bash
ssh -N -L 2718:COMPUTE_NODE_HOSTNAME:2718  your_helix_user_name@helix.bwservices.uni-heidelberg.de
```

After entering your credentials, the command will not produce further output. This is expected.

#### 4. Access Marimo in Your Browser

Open your local web browser and navigate to:
[http://localhost:2718](http://localhost:2718)
