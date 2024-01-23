import copy
import torch
from torch import nn
import numpy as np
from PIL import ImageColor, Image, ImageDraw, ImageFont

import viper_rl.dreamerv3.networks as networks
import viper_rl.dreamerv3.tools as tools

# from ding.reward_model.rnd_reward_model import RndRewardModel

to_np = lambda x: x.detach().cpu().numpy()

# exponentially weighted avg
class RewardEMA(object):
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.values = torch.zeros((2,)).to(device)
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        self.values = self.alpha * x_quantile + (1 - self.alpha) * self.values
        scale = torch.clip(self.values[1] - self.values[0], min=1.0)
        offset = self.values[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        # if config.video_len > 1:
        #     shapes.pop("image")
        #     shapes["video"] = (config.video_embed_dim,)
        # q(z|h,x)
        self.encoder = networks.MultiEncoder(shapes, config.device, **config.encoder)
        self.embed_size = self.encoder.outdim
        self.dynamics = networks.RSSM( # p(z|h)
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_input_layers,
            config.dyn_output_layers,
            config.dyn_rec_depth,
            config.dyn_shared,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_temp_post,
            config.dyn_min_std,
            config.dyn_cell,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
            # config.video_len
        )
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            # 32 * 32 + 512 = 1536
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        # decoder p(x|h,z)
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, config.device, **config.decoder
        )
        if config.task_behavior == "prior" or config.expl_behavior == "prior":
            self.heads["density"] = networks.MLP(
                feat_size, 
                (255,),
                **config.density_head, 
                name="density",
            )

        # reward predictor p(r|h,z)
        if config.reward_head == "symlog_disc":
            self.heads["reward"] = networks.MLP(
                feat_size,  # pytorch version
                (255,),
                config.reward_layers,
                config.units,
                config.act,
                config.norm,
                dist=config.reward_head,
                outscale=0.0,
                device=config.device,
            )
        else:
            self.heads["reward"] = networks.MLP(
                feat_size,  # pytorch version
                [],
                config.reward_layers,
                config.units,
                config.act,
                config.norm,
                dist=config.reward_head,
                outscale=0.0,
                device=config.device,
            )

        # continue predictor p(c|h,z)
        self.heads["cont"] = networks.MLP(
            feat_size,  # pytorch version
            [],
            config.cont_layers,
            config.units,
            config.act,
            config.norm,
            dist="binary",
            device=config.device,
        )

        self.amp = (config.task_behavior == "MotionPrior") or (
            config.expl_behavior == "MotionPrior"
        )
        if self.amp:
            self.discriminator = networks.Discriminator(
                shapes, **config.discriminator, name="discriminator"
            )
            self.heads["discriminator_reward"] = networks.MLP(
                feat_size, 
                [], 
                **config.discriminator_head, 
                name="discriminator_head"
            )

        for name in config.grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        self._scales = dict(reward=config.reward_scale, cont=config.cont_scale, density=config.density_scale)

        #RND network
        if self._config.curiosity == 'rnd':
            self.rnd_cfg = dict(
                                obs_shape=config.dyn_hidden+config.dyn_deter,
                                # (str) Reward model register name, refer to registry ``REWARD_MODEL_REGISTRY``.
                                type='rnd',
                                # (str) The intrinsic reward type, including add, new, or assign.
                                intrinsic_reward_type='add',
                                # (float) The step size of gradient descent.
                                learning_rate=1e-3,
                                # (float) Batch size.
                                batch_size=64,
                                # (list(int)) Sequence of ``hidden_size`` of reward network.
                                # If obs.shape == 1,  use MLP layers.
                                # If obs.shape == 3,  use conv layer and final dense layer.
                                hidden_size_list=[64, 64, 128],
                                # (int) How many updates(iterations) to train after collector's one collection.
                                # Bigger "update_per_collect" means bigger off-policy.
                                # collect data -> update policy-> collect data -> ...
                                update_per_collect=1,
                                # (bool) Observation normalization: transform obs to mean 0, std 1.
                                obs_norm=True,
                                # (int) Min clip value for observation normalization.
                                obs_norm_clamp_min=-1,
                                # (int) Max clip value for observation normalization.
                                obs_norm_clamp_max=1,
                                # Means the relative weight of RND intrinsic_reward.
                                # (float) The weight of intrinsic reward
                                # r = intrinsic_reward_weight * r_i + r_e.
                                intrinsic_reward_weight=0.01,
                                # (bool) Whether to normlize extrinsic reward.
                                # Normalize the reward to [0, extrinsic_reward_norm_max].
                                extrinsic_reward_norm=True,
                                # (int) The upper bound of the reward normalization.
                                extrinsic_reward_norm_max=1,
                            )
            self.rnd_model = RndRewardModel(self.rnd_cfg, self._config.device)
            self.rnd_model.reward_model.apply(tools.weight_init)

    def _train(self, data, reference_data=None):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)
        # print(data.keys())

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = tools.schedule(self._config.kl_free, self._step)
                dyn_scale = tools.schedule(self._config.dyn_scale, self._step)
                rep_scale = tools.schedule(self._config.rep_scale, self._step)
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                preds = {}
                for name, head in self.heads.items():
                    if name == "discriminator_reward":
                        continue
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                
                losses = {}
                for name, pred in preds.items():
                    like = pred.log_prob(data[name])
                    # print("World model data loss for {}".format(name))
                    losses[name] = -torch.mean(like) * self._scales.get(name, 1.0)

                model_loss = sum(losses.values()) + kl_loss

            metrics = self._model_opt(model_loss, self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics


    def preprocess(self, obs, img_norm=True): # video=True):
        obs = obs.copy()
        # if video:
        #     obs["video"] = torch.Tensor(obs["video"])
        # else:
        if img_norm:
            obs["image"] = torch.Tensor(obs["image"]) / 255.0 - 0.5
        else:
            obs["image"] = torch.Tensor(obs["image"])
        # (batch_size, batch_length) -> (batch_size, batch_length, 1)
        # obs["reward"] = torch.Tensor(obs["reward"]).unsqueeze(-1)
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = torch.Tensor(obs["discount"]).unsqueeze(-1)
        # if "is_terminal" in obs:
        #     # this label is necessary to train cont_head
        #     obs["cont"] = torch.Tensor(1.0 - obs["is_terminal"]).unsqueeze(-1)
        # else:
        #     raise ValueError('"is_terminal" was not found in observation.')
       
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = torch.Tensor(1.0 - obs["is_terminal"]).unsqueeze(-1)
        obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
        return obs
    

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6] + 0.5
        model = model + 0.5
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)


class ImagBehavior(nn.Module):
    def __init__(self, config, world_model, stop_grad_actor=True, reward=None):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        self._stop_grad_actor = stop_grad_actor
        self._reward = reward
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.actor = networks.ActionHead(
            feat_size,
            config.num_actions,
            config.actor_layers,
            config.units,
            config.act,
            config.norm,
            config.actor_dist,
            config.actor_init_std,
            config.actor_min_std,
            config.actor_max_std,
            config.actor_temp,
            outscale=1.0,
            unimix_ratio=config.action_unimix_ratio,
        )
        if config.value_head == "symlog_disc":
            self.value = networks.MLP(
                feat_size,
                (255,),
                config.value_layers,
                config.units,
                config.act,
                config.norm,
                config.value_head,
                outscale=0.0,
                device=config.device,
            )
        else:
            self.value = networks.MLP(
                feat_size,
                [],
                config.value_layers,
                config.units,
                config.act,
                config.norm,
                config.value_head,
                outscale=0.0,
                device=config.device,
            )
        if config.slow_value_target:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor_lr,
            config.ac_opt_eps,
            config.actor_grad_clip,
            **kw,
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.value_lr,
            config.ac_opt_eps,
            config.value_grad_clip,
            **kw,
        )
        if self._config.reward_EMA:
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
        self,
        start,
        objective=None,
        action=None,
        reward=None,
        imagine=None,
        tape=None,
        repeats=None,
    ):
        objective = objective or self._reward
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                # virtual rollout for H steps
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon, repeats
                )
                # print(imag_action.shape)

                # if self._config.video_len > 1:
                #     # imag_action = imag_action.reshape(-1, self._config.video_len, self._config.num_actions)
                #     imag_action = imag_action.reshape(imag_action.shape[:-1] + (self._config.video_len, self._config.num_actions))
                #     # print(imag_action.shape)

                if isinstance(self._config.num_actions, list):
                    imag_action = [imag_action[..., self._config.action_idxs[i]:self._config.action_idxs[i+1]]
                                   for i in range(self._config.action_dims)]
                    # print(torch.stack([torch.argmax(imag_action[i], axis=-1) for i in range(self._config.action_dims)], axis=-1).shape)

                reward = objective(imag_feat, imag_state, imag_action)
                
                # print(imag_feat.reshape(-1, imag_feat.size(-1)).shape)

                if self._config.curiosity == 'rnd':
                    self._world_model.rnd_model.collect_data(imag_feat.reshape(-1, imag_feat.size(-1)).detach())
                    
                    rnd_loss = self._world_model.rnd_model.train()

                    rnd_reward = self._world_model.rnd_model.estimate(imag_feat)
                    metrics["rnd_loss"] = to_np(rnd_loss)
                    metrics.update(tools.tensorstats(rnd_reward, "rnd_reward"))
                    reward += rnd_reward

                
                policy = self.actor(imag_feat)
                actor_ent = policy.entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                # this target is not scaled
                # slow is flag to indicate whether slow_target is used for lambda-return
                # gpu mem boost
                value = self.value(imag_feat).mode()

                if not isinstance(self._config.num_actions, list):
                    actor_ent = [actor_ent]
                    imag_action = [imag_action]


                target, weights, base = self._compute_target(
                    # imag_feat, 
                    value,
                    imag_state, 
                    imag_action, # flattened/cat action sequence
                    reward, 
                    actor_ent, # unflattened
                    state_ent
                )
    
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    value,
                    policy, # last usage of policy torch
                    imag_state,
                    imag_action,
                    target,
                    actor_ent, # last usage of actor_ent torch
                    state_ent,
                    weights,
                    base,
                )
                # print(actor_loss)

                metrics.update(mets)
                value_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                # value duplicate
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.slow_value_target:
                    value_loss = value_loss - value.log_prob(
                        slow_target.mode().detach()
                    )
                if self._config.value_decay:
                    value_loss += self._config.value_decay * value.mode()
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.actor_dist in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action[0], dim=-1).float(), "imag_action"
                )
            )
        elif self._config.actor_dist in ["multionehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.cat([torch.argmax(act, dim=-1).float() for act in imag_action], dim=-1), "imag_action"
                    # torch.cat(imag_action, dim=-1), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action[0], "imag_action"))

        # Checkpoint!
        metrics["actor_entropy"] = to_np(torch.mean(torch.stack(actor_ent)))
        with tools.RequiresGrad(self):
            # actor and value loss backprop
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
            
            if self._config.curiosity == 'rnd':
                self._world_model.rnd_model.opt.zero_grad()
                rnd_loss.backward()
                self._world_model.rnd_model.opt.step()

        return imag_feat, imag_state, imag_action, weights, metrics


    def _imagine(self, start, actor, horizon, repeats=None):
        dynamics = self._world_model.dynamics
        if repeats:
            raise NotImplemented("repeats is not implemented in this version")
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach() if self._stop_grad_actor else feat
            # actionhead takes in z+h and outputs action
            action = actor(inp).sample()
            if isinstance(action, list):
                action = torch.cat(action, dim=-1)
            # print(action.shape)f
            succ = dynamics.img_step(state, action, sample=self._config.imag_sample)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
        if repeats:
            raise NotImplemented("repeats is not implemented in this version")

        return feats, states, actions

    def _compute_target(
        # self, imag_feat, imag_state, imag_action, reward, actor_ent, state_ent
        # gpu mem boost
        self, value, imag_state, imag_action_list, reward, actor_ent_list, state_ent
    ):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        
        # if self._config.future_entropy and self._config.actor_entropy > 0:    
        #     for actor_ent in actor_ent_list:   
        #         reward += self._config.actor_entropy * actor_ent
        #     # Do I need to average?
        #     reward /= len(actor_ent_list)
            
        # if self._config.future_entropy and self._config.actor_state_entropy > 0:
        #     reward += self._config.actor_state_entropy * state_ent

        # value = self.value(imag_feat).mode()
        
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )

        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()

        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        # gpu mem boost
        value, # value = self.value(imag_feat).mode()
        policy,
        imag_state,
        imag_action_list,
        target,
        actor_ent_list, # actor_ent = self.actor(imag_feat).entropy()
        state_ent,
        weights,
        base,
    ):
        metrics = {}
        inp = imag_feat.detach() if self._stop_grad_actor else imag_feat
        # print(inp.shape)
        policy = self.actor(inp)
        # actor_ent = policy.entropy()
        # gpu mem boost
        if self._stop_grad_actor:
            # policy = policy.detach()
            for i in range(len(actor_ent_list)):
                actor_ent_list[i] = actor_ent_list[i].detach()
            
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            values = self.reward_ema.values
            metrics["EMA_005"] = to_np(values[0])
            metrics["EMA_095"] = to_np(values[1])

        actor_loss_list = []
        i = 0
        for imag_action, actor_ent in zip(imag_action_list, actor_ent_list):
            if self._config.imag_gradient == "dynamics":
                actor_target = adv
            elif self._config.imag_gradient == "reinforce":
                # print(imag_action)
                # print(type(imag_action))
                # print(imag_action.shape) [15, 1600]
                if len(imag_action_list) > 1:
                    actor_target = (
                        # policy_logprob
                        policy.log_prob(imag_action, i)[:-1][:, :, None]
                        # * (target - self.value(imag_feat[:-1]).mode()).detach()
                        # gpu mem boost
                        * (target - value[:-1]).detach()
                    )
                else:
                    # print(imag_action.shape)
                    # print(policy.log_prob(imag_action).shape)
                    # if self._config.video_len > 1:
                    #     actor_target = (
                    #         # policy_logprob
                    #         policy.log_prob(imag_action)[:-1]
                    #         # * (target - self.value(imag_feat[:-1]).mode()).detach()
                    #         # gpu mem boost
                    #         * (target - value[:-1]).detach()
                    #     )
                    # else:
                    actor_target = (
                        # policy_logprob
                        policy.log_prob(imag_action)[:-1][:, :, None]
                        # * (target - self.value(imag_feat[:-1]).mode()).detach()
                        # gpu mem boost
                        * (target - value[:-1]).detach()
                    )
            elif self._config.imag_gradient == "both":
                if len(imag_action_list) > 1:    
                    actor_target = (
                        # policy_logprob
                        policy.log_prob(imag_action, i)[:-1][:, :, None]
                        # * (target - self.value(imag_feat[:-1]).mode()).detach()
                        # gpu mem boost
                        * (target - value[:-1]).detach()
                    )
                else:
                    actor_target = (
                        # policy_logprob
                        policy.log_prob(imag_action)[:-1][:, :, None]
                        # * (target - self.value(imag_feat[:-1]).mode()).detach()
                        # gpu mem boost
                        * (target - value[:-1]).detach()
                    )
                mix = self._config.imag_gradient_mix
                actor_target = mix * target + (1 - mix) * actor_target
                metrics["imag_gradient_mix"] = mix
            else:
                raise NotImplementedError(self._config.imag_gradient)

            if not self._config.future_entropy and (self._config.actor_entropy > 0):
                # if self._config.video_len > 1:
                #     actor_entropy = self._config.actor_entropy * actor_ent[:-1]
                # else:    
                actor_entropy = self._config.actor_entropy * actor_ent[:-1][:, :, None]
                # print(actor_target.shape)
                # print(actor_entropy.shape)
                actor_target += actor_entropy
            if not self._config.future_entropy and (self._config.actor_state_entropy > 0):
                state_entropy = self._config.actor_state_entropy * state_ent[:-1]
                actor_target += state_entropy
                metrics["actor_state_entropy"] = to_np(torch.mean(state_entropy))

            actor_loss = -torch.mean(weights[:-1] * actor_target)
            actor_loss_list.append(actor_loss)

            i += 1

        # print("The losses are {}".format(np.array([loss.detach().cpu().numpy() for loss in actor_loss_list])))
        # torch.sum(torch.stack(actor_loss_list))
        return sum(actor_loss_list), metrics

    def _update_slow_target(self):
        if self._config.slow_value_target:
            if self._updates % self._config.slow_target_update == 0:
                mix = self._config.slow_target_fraction
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
