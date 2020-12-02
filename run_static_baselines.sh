#!/bin/bash
#
#SBATCH --partition=cpu_cluster
#SBATCH --job-name=dacbenchlines
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=8000M

source ~/anaconda3/tmp/bin/activate dacbench

python dacbench/run_baselines.py --outdir baselines --static --num_episodes 1000 --benchmarks $1
