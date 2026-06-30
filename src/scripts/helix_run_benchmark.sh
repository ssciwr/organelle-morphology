#!/bin/bash
#SBATCH --job-name=vem-single
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH --mem=60gb
#SBATCH --time=00:05:00

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
  python cli_benchmark.py \
  --mpi --threads "$SLURM_CPUS_PER_TASK" \
  -- workers "$SLURM_NTASKS" \
  --data data/interphase_4T \
  --projectpath vem_benchmark
