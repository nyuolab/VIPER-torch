import argparse
import functools
import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools

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
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        if self._config.task_behavior == "prior":
            reward = lambda f, s, a: self._wm.heads["density"](f).mean()
        else:    
            reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        
        if self._config.expl_behavior != "greedy":
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
        
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._config.expl_behavior != "greedy" and self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
            # e-greedy
            if self._config.expl_amount > 0:
                action = self._exploration(action, training)

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
            reward = lambda f, s, a: self._wm.heads["density"](f).mode()
            # reward = lambda f, s, a: self._wm.heads["density"](
            #     self._wm.dynamics.get_feat(s)
            # ).mode()
        else:
            reward = lambda f, s, a: self._wm.heads["reward"](f).mode()
            # reward = lambda f, s, a: self._wm.heads["reward"](
            #     self._wm.dynamics.get_feat(s)
            # ).mode()
            # reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        
        metrics.update(self._task_behavior._train(start, reward)[-1]) # imagbehavior
        if self._config.expl_behavior not in ["greedy", "prior"]:
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)
