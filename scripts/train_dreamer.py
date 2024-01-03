import argparse
import functools
import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import viper_rl.dreamerv3.exploration as expl
import viper_rl.dreamerv3.models
import viper_rl.dreamerv3.tools
import viper_rl.dreamerv3.envs.wrappers as wrappers
from viper_rl.dreamerv3.dreamer import Dreamer
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd

import torch
import torch.distributed as dist

dist.init_process_group(backend='nccl')

from gym.spaces import MultiDiscrete
# from vmae_encoder import VMAEEncoder

to_np = lambda x: x.detach().cpu().numpy()



def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, mode):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(task, config.action_repeat, config.size)
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            seed=config.seed,
        )
        env = wrappers.OneHotAction(env)

    elif suite == "dmlab":
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task, 
            mode if "train" in mode else "test", 
            config.action_repeat,
            seed=config.seed,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed,)
        env = wrappers.OneHotAction(env)
    elif suite == "minecraft":
        # print("Creating minecraft env")
        import envs.minecraft as minecraft

        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        
        # This is misleading
        # print(env.observation_space.spaces.items())
        # print({k: tuple(v.shape) for k, v in env.observation_space.spaces.items()})

        # print(env.action_space)
        env = wrappers.MultiDiscreteAction(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action") # action dict to action
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)
    # env = wrappers.RewardObs(env)
    return env


def main(config):
    is_master_process = dist.get_rank() == 0

    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    config.logdir += str(config.seed)
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    # if is_master_process:
    #     wandb.init(project='dreamer', config=config,
    #                id=config.run_id, resume='allow', mode='online')
    #     wandb.run.name = config.run_id
    #     wandb.run.save()

    # print(config.expl_amount)
    # print(config.expl_decay_rate)

    print("DDP Activation is {}!".format(config.ddp))

    if config.ddp:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        print(f"Start running basic DDP on rank {rank}.")

        # create model and move it to GPU with id rank
        config.device = rank % torch.cuda.device_count()

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)
    
    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    make = lambda mode: make_env(config, mode)
    train_envs = [make("train") for _ in range(config.envs)]
    eval_envs = [make("eval") for _ in range(config.envs)]
    if config.envs > 1:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]
    acts = train_envs[0].action_space
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

    # print(train_envs[0].observation_space.spaces.items())
    # print({k: tuple(v.shape) for k, v in train_envs[0].observation_space.spaces.items()})

    # print(acts)
    # video_encoder = VMAEEncoder(num_frames=config.video_len)
    video_encoder = None
    
    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
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
            def __init__(self, config): # video_encoder=None):
                super(RandomAgent, self).__init__()
                # self.video_encoder = video_encoder
                self._config = config
            
            def __call__(self, o, d, state):
                action = random_actor.sample(flatten=False)
                logprob = random_actor.log_prob(action)
                return {"action": action, "logprob": logprob}, None

        # def random_agent(o, d, s):
        #     action = random_actor.sample()
        #     logprob = random_actor.log_prob(action)
        #     return {"action": action, "logprob": logprob}, None
        # random_agent = RandomAgent(config, video_encoder)
        random_agent = RandomAgent(config)

        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
        )

        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)


    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
        # video_encoder=video_encoder
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest_model.pt").exists():
        print("Load weights")
        agent.load_state_dict(torch.load(logdir / "latest_model.pt", map_location=config.device))
        agent._should_pretrain._once = False

    # make sure eval will be executed once after config.steps
    while agent._step < config.steps + config.eval_every:
        logger.write()
        if config.eval_episode_num > 0:
            print("Start evaluation.")
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                config.evaldir,
                logger,
                is_eval=True,
                episodes=config.eval_episode_num,
            )
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))
        
        # Checkpoint for minedojo
        print("Start training.")
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
        )
        # if config.ddp:
        #     torch.save(agent.module.state_dict(), logdir / "latest_model.pt")
        # else:
        # if is_master_process:
        #     wandb.log({'train/lr': agent.scheduler.get_last_lr()[-1]}, step=iteration)
        #     wandb.log({**{f'train/{metric}': val
        #                 for metric, val in metrics.items()}
        #             }, step=iteration)

        torch.save(agent.state_dict(), logdir / "latest_model.pt")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    print("The number of available gpus is {}".format(torch.cuda.device_count()))
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "viper_rl/configs/dreamer/configs.yaml").read_text()
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
