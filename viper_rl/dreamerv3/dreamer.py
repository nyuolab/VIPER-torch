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

from gym.spaces import MultiDiscrete
# from vmae_encoder import VMAEEncoder


to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset, reward_model=None): # , video_encoder=None):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        # self.video_encoder = video_encoder
        self._dataset = dataset
        # self.n_skip = config.transformer["frame_skip"]
        # self.seq_len = config.transformer["seq_len"] * self.n_skip
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(
            config, self._wm, config.behavior_stop_grad
        )
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        if self._config.task_behavior == "prior":
            reward = lambda f, s, a: self._wm.heads["density"](f).mean()
        else:    
            reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            prior=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

        self.reward_model = reward_model
        # print(self._config.expl_amount)

    # real trajectory
    def __call__(self, obs, reset, state=None, training=True, train=True):
        # print(self._config.expl_amount)
        step = self._step
        if self._should_reset(step):
            state = None
        if state is not None and reset.any():
            mask = 1 - reset
            for key in state[0].keys():
                for i in range(state[0][key].shape[0]):
                    state[0][key][i] *= mask[i]
            for i in range(len(state[1])):
                state[1][i] *= mask[i]
        if training and train:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for i in range(steps):
                # checkpoint
                # print("Training iter {}".format(i))
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                # progress
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True) # print all metrics
                
        if self._step > (self._config.prefill * self._config.action_repeat) and not (self._step % 1000):
            self._config.expl_amount = max(self._config.expl_amount*self._config.expl_decay_rate, self._config.expl_min)
            # print("The current exploration epsilon is {}".format(self._config.expl_amount))
            self._logger.scalar("epsilon", float(self._config.expl_amount))
        
        
        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state # (latent, action)

    # real trajectory as well, output action and logprob in dict
    def _policy(self, obs, state, training):
        # agent_state
        if state is None:
            # if self._config.video_len > 1:
            #     batch_size = len(obs["video"])
            # else:
            batch_size = len(obs["image"])
            # print(batch_size)
            latent = self._wm.dynamics.initial(batch_size)
            if isinstance(self._config.num_actions, list):
                action = torch.zeros((batch_size, sum(self._config.num_actions))).to(
                    self._config.device
                )
            else:
                # if self._config.video_len > 1:
                #     action = torch.zeros((batch_size, self._config.video_len, self._config.num_actions)).to(
                #         self._config.device
                #     )
                # else:
                action = torch.zeros((batch_size, self._config.num_actions)).to(
                    self._config.device
                )

        else:
            # action from agent_state
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        
        latent, _ = self._wm.dynamics.obs_step(
            latent, action, embed, obs["is_first"], self._config.collect_dyn_sample
        ) # action.flatten(start_dim=-2)
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        # checkpoint
        if isinstance(action, list):
            for i in range(len(action)):
                action[i] = action[i].detach()
                # print(action[i].shape)
        else:
            action = action.detach()

        # if self._config.actor_dist == "onehot_gumble":
        #     action = torch.one_hot(
        #         torch.argmax(action, dim=-1), self._config.num_actions
        #     )
        action = self._exploration(action, training)
        # if isinstance(action, list):
        #     action = torch.cat(action, dim=-1)
        # from tensor to dictionary
        policy_output = {"action": action, "logprob": logprob}
        
        if isinstance(action, list): 
            state = (latent, torch.cat(action, dim=-1))
        else:
            state = (latent, action)
        return policy_output, state

    def _exploration(self, action, training):
        amount = self._config.expl_amount if training else self._config.eval_noise
        # print(amount)
        if amount == 0:
            return action
        if self._config.actor_dist == "multionehot":
            probs_list = []
            for i in range(self._config.action_dims):
                probs_list.append(amount / self._config.num_actions[i] + (1 - amount) * action[i])
            return tools.MultiOneHotDist(probs_list=probs_list).sample()
        elif "onehot" in self._config.actor_dist:
            probs = amount / self._config.num_actions + (1 - amount) * action
            return tools.OneHotDist(probs=probs).sample()
        else:
            return torch.clip(torchd.normal.Normal(action, amount).sample(), -1, 1)

    def _train(self, data):
        metrics = {}
        # Get posterior from world model
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        # why the virtual reward function doesn't use action as input?
        if self._config.task_behavior == "prior":
            reward = lambda f, s, a: self._wm.heads["density"](f).mean()
        else:    
            reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        
        metrics.update(self._task_behavior._train(start, reward)[-1]) # imagbehavior
        if self._config.expl_behavior not in ["greedy", "prior"]:
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


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
                action = random_actor.sample()
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
