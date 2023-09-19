import argparse
import functools
import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd

from dreamer import Dreamer, make_dataset, make_env

to_np = lambda x: x.detach().cpu().numpy()

def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))

def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path("experiments").expanduser()
    config.evaldir = config.evaldir or logdir / "eval_eps"
    # config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat
    config.size = [256, 256]

    print("DDP Activation is {}!".format(config.ddp))

    if config.ddp:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        print(f"Start running basic DDP on rank {rank}.")

        # create model and move it to GPU with id rank
        config.device = rank % torch.cuda.device_count()


    print("Logdir", logdir) # logdir/env
    logdir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True) # logdir/env/eval_eps
    # step = count_steps(config.traindir) # How many training steps by far
    # step in logger is environmental step
    logger = tools.Logger(logdir, 0)

    print("Create envs.")
    # if config.offline_traindir:
    #     directory = config.offline_traindir.format(**vars(config))
    # else:
    #     directory = config.traindir

    # train_eps = tools.load_episodes(directory, limit=config.dataset_size) # train transition buffer/dataset
    
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir

    eval_eps = tools.load_episodes(directory, limit=1) # single trajectory?

    # train on different envs
    make = lambda mode: make_env(config, mode)
    
    config.envs = 1
    # train_envs = [make("train") for _ in range(config.envs)]
    eval_envs = [make("eval") for _ in range(config.envs)]
    # if config.envs > 1:
    #     # train_envs = [Parallel(env, "process") for env in train_envs]
    #     eval_envs = [Parallel(env, "process") for env in eval_envs]
    # else:
        # train_envs = [Damy(env) for env in train_envs]
    
    eval_envs = [Damy(env) for env in eval_envs]
    acts = eval_envs[0].action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    # state = None

    # if not config.offline_traindir:
        # prefill = max(0, config.prefill - count_steps(config.traindir)) # prefill steps if not sufficient
        # print(f"Prefill dataset ({prefill} steps).")
        
        # check discrete or continuous action space
        # if hasattr(acts, "discrete"):
        #     random_actor = tools.OneHotDist(
        #         torch.zeros(config.num_actions).repeat(config.envs, 1) # config.envs rows of same actions
        #     )
        # else:
        #     random_actor = torchd.independent.Independent(
        #         torchd.uniform.Uniform(
        #             torch.Tensor(acts.low).repeat(config.envs, 1),
        #             torch.Tensor(acts.high).repeat(config.envs, 1),
        #         ),
        #         1,
        #     )

        # def random_agent(o, d, s):
        #     action = random_actor.sample()
        #     logprob = random_actor.log_prob(action)
        #     return {"action": action, "logprob": logprob}, None

        # state = tools.simulate(
        #     random_agent,
        #     train_envs,
        #     train_eps,
        #     config.traindir,
        #     logger,
        #     limit=config.dataset_size,
        #     steps=prefill,
        # )

    print("Simulate agent.")
    # train_dataset = make_dataset(train_eps, config) 
    train_dataset = None
    eval_dataset = make_dataset(eval_eps, config)
    
    agent = Dreamer(
        eval_envs[0].observation_space,
        eval_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)

    agent.requires_grad_(requires_grad=False)

    # load saved world model weights
    model_dir = logdir / "{}/latest_model.pt".format(config.task)
    if (model_dir).exists():
        print("Load weights")
        agent.load_state_dict(torch.load(model_dir))
        agent._should_pretrain._once = False

    # evaluate policy via real rollout
    # make sure eval will be executed once after config.steps
    
    print("Start evaluation.")
    eval_policy = agent
    tools.eval_rollout(
        eval_policy,
        eval_envs,
        eval_eps,
        config.evaldir,
        logger, 
        episodes=config.eval_episode_num,
        real=False
    )


    # if config.video_pred_log:
    #     video_pred = agent._wm.video_pred(next(eval_dataset))
    #     logger.video("eval_openl", to_np(video_pred))
    

    for env in eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    print(torch.cuda.device_count())
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )
    
    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))