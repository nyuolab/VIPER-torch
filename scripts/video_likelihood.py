import argparse
import functools
import os
import pathlib
import sys
import glob

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml

from gym.spaces import MultiDiscrete

directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
#sys.path.append(str(pathlib.Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch import distributions as torchd

import viper_rl.dreamerv3.tools as tools
from viper_rl.dreamerv3.parallel import Parallel, Damy

from viper_rl.videogpt.reward_models import LOAD_REWARD_MODEL_DICT

from scripts.train_dreamerv3 import make_env


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

    # random traj
    make = lambda mode: make_env(config, mode)
    envs = [make("eval") for _ in range(config.envs)]

    if config.envs > 1:
        envs = [Parallel(env, "process") for env in eval_envs]
    else:
        envs = [Damy(env) for env in envs]
        
    acts = envs[0].action_space
    if 'minecraft' not in config.task:
        config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    else:
        config.num_actions = list(acts.nvec)
        config.action_dims = len(acts.nvec)
        # print(config.num_actions)
        # print(type(config.num_actions))
        config.action_idxs = [0]
        idx = 0
        for action_dim in config.num_actions:
            idx += action_dim
            config.action_idxs.append(idx)
        assert len(config.action_idxs) == len(config.num_actions)+1

        config.action_weights = 1/acts.nvec.astype(float)
    

    if isinstance(acts, MultiDiscrete):
        print("MineCraft MultiDiscrete")
        
        random_actor = tools.MultiOneHotDist(
            [torch.zeros(num_actions).repeat(config.envs, 1) for num_actions in acts.nvec]
        )
        # action sample is fine
        # print(random_actor.sample())

    elif hasattr(acts, "discrete"):
        # random_actor = tools.OneHotDist(
        #     torch.zeros(config.video_len*config.num_actions).repeat(config.envs, 1)
        # )
        random_actor = tools.OneHotDist(
            torch.zeros(config.num_actions).repeat(config.envs, 1)
        )
    else:
        random_actor = torchd.independent.Independent(
            torchd.uniform.Uniform(
                torch.Tensor(acts.low).repeat(config.envs, 1),
                torch.Tensor(acts.high).repeat(config.envs, 1),
            ),
            1,
        )

    class RandomAgent(object):
        def __init__(self, config, reward_model=None): # video_encoder=None):
            super(RandomAgent, self).__init__()
            self.reward_model = reward_model
            # self.video_encoder = video_encoder
            self._config = config
        
        def __call__(self, o, d, state, training):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None
    
    seeds = 10
    traj_len = 501

    random_density = np.zeros((seeds, traj_len))
    is_first = np.zeros(traj_len)
    is_first[0] = 1

    random_agent = RandomAgent(config)

    for i in range(seeds):
        rand = tools.eval_rollout(random_agent, envs)
        
        rand["is_first"] = is_first
        rand = reward_model(rand)
        random_density[i] = np.array(rand["density"])

        print("The random viper return is {}".format(sum(random_density[i])))
        print("The random viper reward mean is {}".format(random_density[i].mean()))
        print("The random viper reward std is {}".format(random_density[i].std()))

    # expert traj
    # env = "cheetah_run"
    env = config.task.split("_", 1)[-1]
    path = "viper_rl_data/datasets/dmc/{}/test/*.npz".format(env)
    fns = glob.glob(path)

    expert_density = np.zeros((seeds, traj_len))
    # for video_path in fns:
    for i in range(seeds):
        video_path = fns[i]
        video = np.load(video_path)['arr_0']
        # max_idx = video.shape[0] - traj_len + 1
        # max_idx = min(max_idx, traj_len)
        # np.random.seed(config.seed+video.shape[0])
        # idx = np.random.randint(0, max_idx)
        video = video[:traj_len]
        # print(video.shape)

        expert = dict(image=video, is_first=is_first)
        expert = reward_model(expert)
        expert_density[i] = np.array(expert["density"])

    
        print("The expert viper return is {}".format(sum(expert_density[i])))
        print("The expert viper reward mean is {}".format(expert_density[i].mean()))
        print("The expert viper reward std is {}".format(expert_density[i].std()))

    x = np.arange(traj_len)

    rand_avs = np.mean(random_density, axis=0)
    rand_maxs = np.max(random_density, axis=0)
    rand_mins = np.min(random_density, axis=0)

    expert_avs = np.mean(expert_density, axis=0)
    expert_maxs = np.max(expert_density, axis=0)
    expert_mins = np.min(expert_density, axis=0)

    plt.figure(figsize=(10, 6))  # Set the figure size for better visibility
    
    plt.fill_between(x, rand_mins, rand_maxs, color='red', alpha=0.35)
    plt.plot(x, rand_avs, '-o', color='red', markersize=1, label="random")
    
    plt.fill_between(x, expert_mins, expert_maxs, color='green', alpha=0.35)
    plt.plot(x, expert_avs, '-o', color='green', markersize=1, label='expert')  # Plot y2 vs. x with red color
    
    plt.title('VIPER {} trajectory'.format(config.task))  # Title of the plot
    plt.xlabel('Trajectory Step')  # X-axis label
    plt.ylabel('r VIPER')  # Y-axis label
    plt.legend()  # Show legend to differentiate the two lines

    plt.savefig("plots/viper_{}_traj.png".format(config.task))




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
