#!/bin/bash
# load modules
source /gpfs/share/apps/anaconda3/gpu/2022.10/etc/profile.d/conda.sh
# module load miniconda3/gpu/4.9.2
module load git
module load cuda/11.8
module load gcc/11.2.0 
module load nccl
module load jdk/8u181

# activate ~local~ conda environment
conda activate /gpfs/data/oermannlab/users/qp2040/.conda/envs/mdj
export PYTHONPATH=/gpfs/data/oermannlab/users/qp2040/.conda/envs/mdj/lib/python3.10/site-packages

# create /tmp/$SLURM_JOB_ID directory .gradle symlink
# mkdir -p /tmp/$SLURM_JOB_ID/.gradle
# rm -f ~/.gradle
# ln -s /tmp/$SLURM_JOB_ID/.gradle ~/.gradle

export GRADLE_USER_HOME=/tmp/$SLURM_JOB_ID/.gradle