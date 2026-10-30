#!/bin/bash
#SBATCH --job-name=qfs_ppo
#SBATCH --account=<your_OSC_project>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10          # 8 training envs + 1 eval env + main process
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out

module load matlab/r2025b
module load miniconda3/24.1.2-py310
source activate rl_env              # or: conda activate rl_env
export MRST_PATH=/path/to/mrst      # folder with MRST's startup.m

cd "$SLURM_SUBMIT_DIR"
python qfs_ppo.py
