#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=v100:2
#SBATCH --mem-per-gpu=20G
#SBATCH --account=def-hup-ab
#SBATCH --output=logs/job_log.out

echo "Loading rust"
module load rust/1.70.0

echo "Setting up python venv"
module load python/3.8.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
pip install --no-index -U 'tensorboardX'
pip install --no-index -U 'tensorboard'

srun ./train.py

