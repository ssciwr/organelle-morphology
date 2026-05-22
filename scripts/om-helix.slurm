#!/bin/bash
#SBATCH --job-name=helix-demo
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=230gb
#SBATCH --time=02:00:00

# This is an example how a slurm script might look like on the helix cluster.
# You will likely have to adjust the paths to the script and to the data.

module load devel/miniforge/24.9.2
conda activate morph
python ~/organelle-morphology/scripts/helix-demo.py --workers 16 --threads 4 --data ~/om2-projects/interphase_4T/interphase_4T/images/bdv-n5
