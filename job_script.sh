#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=def-hup-ab
echo "Training ML model..."
./train.py 
