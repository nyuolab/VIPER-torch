import copy
import torch
from torch import nn

import viper_rl.dreamerv3.networks as networks
import viper_rl.dreamerv3.tools as tools

# from ding.reward_model.rnd_reward_model import RndRewardModel

to_np = lambda x: x.detach().cpu().numpy()

# exponentially weighted avg
class RewardEMA(object):
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
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
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
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
                device=config.device,
            )

        # reward predictor p(r|h,z)
        
        self.heads["reward"] = networks.MLP(
            feat_size,  # pytorch version
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            **config.reward_head,
            device=config.device,
        )

        # continue predictor p(c|h,z)
        self.heads["cont"] = networks.MLP(
            feat_size,
            (),
            **config.cont_head,
            device=config.device,
        )

        # self.amp = (config.task_behavior == "MotionPrior") or (
        #     config.expl_behavior == "MotionPrior"
        # )
        # if self.amp:
        #     self.discriminator = networks.Discriminator(
        #         shapes, **config.discriminator, name="discriminator"
        #     )
        #     self.heads["discriminator_reward"] = networks.MLP(
        #         feat_size, 
        #         [], 
        #         **config.discriminator_head, 
        #         name="discriminator_head"
        #     )

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
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
            density=config.density_head["loss_scale"],
        )
        

        #RND network for latent state exploration
        # if self._config.curiosity == 'rnd':
        #     self.rnd_cfg = dict(
        #                         obs_shape=config.dyn_hidden+config.dyn_deter,
        #                         # (str) Reward model register name, refer to registry ``REWARD_MODEL_REGISTRY``.
        #                         type='rnd',
        #                         # (str) The intrinsic reward type, including add, new, or assign.
        #                         intrinsic_reward_type='add',
        #                         # (float) The step size of gradient descent.
        #                         learning_rate=1e-3,
        #                         # (float) Batch size.
        #                         batch_size=64,
        #                         # (list(int)) Sequence of ``hidden_size`` of reward network.
        #                         # If obs.shape == 1,  use MLP layers.
        #                         # If obs.shape == 3,  use conv layer and final dense layer.
        #                         hidden_size_list=[64, 64, 128],
        #                         # (int) How many updates(iterations) to train after collector's one collection.
        #                         # Bigger "update_per_collect" means bigger off-policy.
        #                         # collect data -> update policy-> collect data -> ...
        #                         update_per_collect=1,
        #                         # (bool) Observation normalization: transform obs to mean 0, std 1.
        #                         obs_norm=True,
        #                         # (int) Min clip value for observation normalization.
        #                         obs_norm_clamp_min=-1,
        #                         # (int) Max clip value for observation normalization.
        #                         obs_norm_clamp_max=1,
        #                         # Means the relative weight of RND intrinsic_reward.
        #                         # (float) The weight of intrinsic reward
        #                         # r = intrinsic_reward_weight * r_i + r_e.
        #                         intrinsic_reward_weight=0.01,
        #                         # (bool) Whether to normlize extrinsic reward.
        #                         # Normalize the reward to [0, extrinsic_reward_norm_max].
        #                         extrinsic_reward_norm=True,
        #                         # (int) The upper bound of the reward normalization.
        #                         extrinsic_reward_norm_max=1,
        #                     )
        #     self.rnd_model = RndRewardModel(self.rnd_cfg, self._config.device)
        #     self.rnd_model.reward_model.apply(tools.weight_init)

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
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
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
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

            #         like = pred.log_prob(data[name])
            #         # print("World model data loss for {}".format(name))
            #         losses[name] = -torch.mean(like) * self._scales.get(name, 1.0)
            #     model_loss = sum(losses.values()) + kl_loss
            # metrics = self._model_opt(model_loss, self.parameters())

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
            obs["image"] = torch.Tensor(obs["image"]) / 255.0
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
        truth = data["image"][:6]
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)


class ImagBehavior(nn.Module):
    def __init__(self, config, world_model):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
    
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.actor = networks.MLP(
            feat_size,
            (config.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            config.actor["dist"],
            "learned",
            config.actor["min_std"],
            config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor",
        )
        self.value = networks.MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )

        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
            
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )
        if self._config.reward_EMA:
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            self.register_buffer("ema_vals", torch.zeros((2,)).to(self._config.device))
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
        self,
        start,
        objective=None,
    ):
        objective = objective or self._reward
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                # virtual rollout for H steps
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon
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

                # if self._config.curiosity == 'rnd':
                #     self._world_model.rnd_model.collect_data(imag_feat.reshape(-1, imag_feat.size(-1)).detach())
                    
                #     rnd_loss = self._world_model.rnd_model.train()

                #     rnd_reward = self._world_model.rnd_model.estimate(imag_feat)
                #     metrics["rnd_loss"] = to_np(rnd_loss)
                #     metrics.update(tools.tensorstats(rnd_reward, "rnd_reward"))
                #     reward += rnd_reward

                
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
                    reward, 
                )
    
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    value,
                    policy, # last usage of policy torch
                    imag_action,
                    target,
                    weights,
                    base,
                )
                # print(actor_loss)
                actor_loss -= self._config.actor["entropy"] * actor_ent[0][:-1, ..., None]
                actor_loss = torch.mean(actor_loss)
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
                if self._config.critic["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode().detach())
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


    def _imagine(self, start, actor, horizon):
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            # actionhead takes in z+h and outputs action
            action = actor(inp).sample()
            if isinstance(action, list):
                action = torch.cat(action, dim=-1)
            # print(action.shape)f
            succ = dynamics.img_step(state, action)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        return feats, states, actions

    def _compute_target(
        # self, imag_feat, imag_state, imag_action, reward, actor_ent, state_ent
        # gpu mem boost
        self, value, imag_state, reward
    ):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)

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
        imag_action_list,
        target,
        weights,
        base,
    ):
        metrics = {}
        inp = imag_feat.detach()
        # print(inp.shape)
        policy = self.actor(inp)
        # actor_ent = policy.entropy()
        # gpu mem boost
            
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])

        actor_loss_list = []
        i = 0
        for imag_action in imag_action_list:
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

            actor_loss = -weights[:-1] * actor_target
            actor_loss_list.append(actor_loss)

            i += 1

        # print("The losses are {}".format(np.array([loss.detach().cpu().numpy() for loss in actor_loss_list])))
        # torch.sum(torch.stack(actor_loss_list))
        return sum(actor_loss_list), metrics

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
