# VIPER_RL-torch
Pytorch implementation of [Video Prediction Models as Rewards for Reinforcement Learning](https://arxiv.org/pdf/2305.14343.pdf). VIPER leverages the next-frame log likelihoods of a pre-trained video prediction model as rewards for downstream reinforcement learning tasks. The method is flexible to the particular choice of video prediction model and reinforcement learning algorithm. The general method outline is shown below:

## Install:

Create a conda environment with Python 3.10:

```
conda create -n viper python=3.10
conda activate viper
```

Install dependencies:
```
pip install -r requirements.txt
```

## Downloading Data

Download the DeepMind Control Suite expert dataset with the following command:

```
python -m viper_rl_data.download dataset dmc
```

and the Atari dataset with:

```
python -m viper_rl_data.download dataset atari
```

This will produce datasets in `<VIPER_INSTALL_PATH>/viper_rl_data/datasets/` which are used for training the video prediction model. The location of the datasets can be retrieved via the `viper_rl_data.VIPER_DATASET_PATH` variable.

## Video Model Training

Use the following command to first train a VQ-GAN:
```
python scripts/train_vqgan.py -o viper_rl_data/checkpoints/dmc_vqgan -c viper_rl/configs/vqgan/dmc.yaml
```

To train the VideoGPT, update `ae_ckpt` in `viper_rl/configs/dmc.yaml` to point to the VQGAN checkpoint, and then run:
```
python scripts/train_videogpt.py -o viper_rl_data/checkpoints/dmc_videogpt_l16_s1 -c viper_rl/configs/videogpt/dmc.yaml
```

## Policy training

Checkpoints for various models can be found in `viper_rl/videogpt/reward_models/__init__.py`. To use one of these video models during policy optimization, simply specify it with the `--reward_model` argument.  e.g.

```
python scripts/train_dreamer.py --configs=dmc_vision videogpt_prior_rb --task=dmc_cartpole_balance --reward_model=dmc_clen16_fskip4 --logdir=~/logdir
```

Custom checkpoint directories can be specified with the `$VIPER_CHECKPOINT_DIR` environment variable. The default checkpoint path is set to `viper_rl_data/checkpoints/`.

## Acknowledgments
This code is heavily inspired by the following works:
- Alejandro's viper_rl jax implementation: https://github.com/Alescontrela/viper_rl
- Dreamer-v3 torch implementation: https://github.com/NM512/dreamerv3-torch
- danijar's Dreamer-v3 jax implementation: https://github.com/danijar/dreamerv3
- danijar's Dreamer-v2 tensorflow implementation: https://github.com/danijar/dreamerv2
- jsikyoon's Dreamer-v2 pytorch implementation: https://github.com/jsikyoon/dreamer-torch
- RajGhugare19's Dreamer-v2 pytorch implementation: https://github.com/RajGhugare19/dreamerv2
- denisyarats's DrQ-v2 original implementation: https://github.com/facebookresearch/drqv2
