#!/bin/bash
#SBATCH --partition gpu8_medium
#SBATCH --nodes 1
#SBATCH --nodelist=gpu-0003
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node 2
#SBATCH --mem 256G
#SBATCH --job-name mdj

source ../.bashrc 

source mdj.sh

# python dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk
# python traj_cmp.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk
# python dreamer.py --configs crafter --task crafter_reward --logdir ./logdir/crafter_reward
# accelerate launch --multi_gpu --mixed_precision=fp16 dreamer.py --configs crafter --task crafter_reward --logdir ./logdir/crafter_reward
torchrun dreamer.py --configs minecraft --task minecraft_diamond --logdir ./logdir/minecraft_diamond

# End of script