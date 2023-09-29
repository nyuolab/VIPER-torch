#!/bin/bash
#SBATCH --partition a100_dev
#SBATCH --nodes 1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node 1
#SBATCH --mem 32G
#SBATCH --job-name traj

source ../.bashrc

source mdj.sh

python traj_cmp.py --configs crafter --task crafter_reward --logdir ./logdir/crafter_reward1
# python traj_cmp.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk
# python traj_cmp.py --configs dmc_vision --task dmc_hopper_hop --logdir ./logdir/dmc_hopper_hop

# End of script