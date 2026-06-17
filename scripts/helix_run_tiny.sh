#!/bin/bash
#SBATCH --job-name=helix-demo
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH --mem=80gb
#SBATCH --time=00:10:00

# This is an example how a slurm script might look like on the helix cluster.
# You will likely have to adjust the paths to the script and to the data.

module load devel/miniforge/24.9.2
conda activate morph
cd /home/hd/hd_hdc/hd_vw182/workspaces/gpfs/hd_vw182-vem || exit
mpirun -np 8 vem --mpi --threads 2 --data data/interphase_4T -p vem_test_run
