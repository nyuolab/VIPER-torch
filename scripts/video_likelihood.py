import argparse
import functools
import os
import pathlib
import sys
import glob

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml

directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
#sys.path.append(str(pathlib.Path(__file__).parent))

import torch
from torch import nn
from torch import distributions as torchd

import viper_rl.dreamerv3.tools as tools
from viper_rl.videogpt.reward_models import LOAD_REWARD_MODEL_DICT



to_np = lambda x: x.detach().cpu().numpy()



class dict2obj:
    def __init__(self, data):
        for key, value in data.items():
            setattr(self, key, value)

def main(config):
    device = "cpu"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device

    transformer_config = dict2obj(yaml.safe_load(open("viper_rl/configs/videogpt/dmc.yaml", 'r')))

    reward_model = LOAD_REWARD_MODEL_DICT[config.reward_model](
                    task=config.task,
                    ae_config=config.ae,
                    config=transformer_config,
                    compute_joint=config.reward_model_compute_joint,
                    minibatch_size=config.reward_model_batch_size,
                    encoding_minibatch_size=config.reward_model_batch_size,
                    reward_model_device=config.device) # "cpu"
    
    transformer_config.device = device

    reward_model.gpt.device = device
    reward_model.gpt.model.device = device
    reward_model.gpt.ae.device = device
    
    reward_model.gpt.model.to(device)
    reward_model.gpt.optimizer = torch.optim.AdamW(reward_model.gpt.model.parameters(), lr=transformer_config.lr)
    reward_model.gpt.ae.ae.to(device)
    reward_model.gpt.model.position_bias_to_device()
    reward_model.gpt.init_ema_params()

    env = "cheetah_run"
    # env = "cartpole_balance"
    path = "viper_rl_data/datasets/dmc/{}/test/*.npz".format(env)
    fns = glob.glob(path)

    traj_len = 501
    for video_path in fns:
        video = np.load(video_path)['arr_0']
        max_idx = video.shape[0] - traj_len + 1
        max_idx = min(max_idx, traj_len)
        np.random.seed(config.seed+video.shape[0])
        idx = np.random.randint(0, max_idx)
        video = video[idx:idx+traj_len]
        print(video.shape)
        is_first = np.zeros(traj_len)
        is_first[0] = 1
        data = dict(image=video, is_first=is_first)
        data = reward_model(data)
        data["density"] = np.array(data["density"])
        print("The viper return is {}".format(sum(data["density"])))
        print("The viper reward mean is {}".format(data["density"].mean()))
        print("The viper reward std is {}".format(data["density"].std()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs='+')
    args, remaining = parser.parse_known_args()
    
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent.parent / "viper_rl/configs/dreamer/configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    print(args.configs)
    name_list = ["defaults", "videogpt_prior_rb", *args.configs] if args.configs else ["defaults", "videogpt_prior_rb"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))

    main(parser.parse_args(remaining))
