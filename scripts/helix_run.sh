#!/bin/bash
#SBATCH --job-name=helix-demo
#SBATCH --partition=devel
#SBATCH --nodes=2
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=4
#SBATCH --mem=230gb
#SBATCH --time=00:30:00

# This is an example how a slurm script might look like on the helix cluster.
# You will likely have to adjust the paths to the script and to the data.

# module load devel/miniforge/24.9.2
conda activate morph
cd /home/hd/hd_hdc/hd_vw182/workspaces/gpfs/hd_vw182-vem || exit
mpirun -np 16 vem --threads 4 --data data/interphase_4T -p vem_test_run
