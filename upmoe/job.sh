#!/bin/bash
#SBATCH --job-name=draft
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=job_%j.out

source /data2/shared/apps/conda/etc/profile.d/conda.sh

conda activate draft_env1
export PYTHONUNBUFFERED=1

CUDA_LAUNCH_BLOCKING=1 python test.py

