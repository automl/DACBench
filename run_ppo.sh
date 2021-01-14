#!/bin/bash
#
#SBATCH --partition=cpu_cluster
#SBATCH --job-name=dacbenchlines
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=8000M

python train_ppo.py --outdir ppo_baselines --benchmarks $1 
