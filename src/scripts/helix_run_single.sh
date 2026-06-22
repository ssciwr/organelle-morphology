#!/bin/bash
#SBATCH --job-name=vem-single
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=4
#SBATCH --mem=230gb
#SBATCH --time=04:00:00

# This is an example how a slurm script might look like on the helix cluster.
# Copy the hpc_example.py to your project directory
# Adjust the sbatch section above
# Adjust the paths to your data and project below

module purge
module load compiler/gnu
module load mpi/openmpi

## for conda:
module load devel/miniforge/24.9.2
conda activate morph

cd /home/hd/hd_hd/hd_vw182/workspaces/gpfs/hd_vw182-vem || exit
mpirun -np "$SLURM_NTASKS" \
  python PATH_TO_YOUR_PYTHON_RUN_FILE.py \
  --mpi --threads "$SLURM_CPUS_PER_TASK" \
  --data PATH_TO_DIR_CONTAINING_XMLs \
  --projectpath PATH_TO_PROJECT_DIR
