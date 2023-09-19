#!/bin/bash
#SBATCH --partition a100_dev
#SBATCH --nodes 1
#SBATCH --nodelist=a100-4002
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 8G
#SBATCH --job-name mc
#SBATCH --output=test.log

source ../.bashrc

source mdj.sh

which python
xvfb-run python env_test.py
