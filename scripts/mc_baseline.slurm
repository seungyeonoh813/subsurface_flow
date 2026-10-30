#!/bin/bash
#SBATCH --job-name=qfs_mc
#SBATCH --account=<your_OSC_project>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # one MATLAB engine per worker
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out

module load matlab/r2025b
module load miniconda3/24.1.2-py310
source activate rl_env              # or: conda activate rl_env
export MRST_PATH=/path/to/mrst      # folder with MRST's startup.m

cd "$SLURM_SUBMIT_DIR"
python qfs_mc_baseline.py --map-range 951 1000 --workers "$SLURM_CPUS_PER_TASK"
