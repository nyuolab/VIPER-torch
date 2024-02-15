import torch
from torch import nn
from torch import distributions as torchd

import viper_rl.dreamer.models as models
import viper_rl.dreamer.nets as nets
import viper_rl.dreamer.tools as tools



class Disag(nn.Module):
    def __init__(self, config, world_model):
        super(Plan2Explore, self).__init__()
        self.config = config
        self._use_amp = True if config.precision == 16 else False
        
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
            stoch = config.dyn_stoch * config.dyn_discrete
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
            stoch = config.dyn_stoch
        size = {
            "embed": world_model.embed_size,
            "stoch": stoch,
            "deter": config.dyn_deter,
            "feat": config.dyn_stoch + config.dyn_deter,
        }[self._config.disag_target]

        kw = dict(
            inp_dim=feat_size + (config.num_actions
            if config.disag_action_cond
            else 0),  # pytorch version
            shape=size,
            layers=config.disag_layers,
            units=config.disag_units,
            act=config.act,
        )
        self.nets = nn.ModuleList(
            [networks.MLP(**kw) for _ in range(config.disag_models)]
        )
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self.opt = tools.Optimizer(
            "explorer",
            self._networks.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            **kw
        )
    
    def forward(self, inputs, action=None):
        if self.config.disag_action_cond:
            inputs = torch.concat([inputs, action], -1)
        preds = torch.cat(
            [head(inputs, torch.float32).mode()[None] for head in self.nets], 0
        )
        disag = torch.mean(torch.std(preds, 0), -1)[..., None]
        if self.config.disag_log:
            disag = torch.log(disag)
        reward = self.config.expl_intr_scale * disag
        return reward

    def loss(self, inputs, targets):
        with torch.cuda.amp.autocast(self._use_amp):
            if self.config.disag_offset:
                targets = targets[:, self.config.disag_offset :]
                inputs = inputs[:, : -self.config.disag_offset]
            targets = targets.detach()
            inputs = inputs.detach()
            preds = [head(inputs) for head in self.nets]
            likes = torch.cat(
                [torch.mean(pred.log_prob(targets))[None] for pred in preds], 0
            )
            loss = -torch.mean(likes)
        return loss


    def train(self, start, context, data):
        with tools.RequiresGrad(self.nets):
            stoch = start["stoch"]
            if self._config.dyn_discrete:
                stoch = torch.reshape(
                    stoch, (stoch.shape[:-2] + ((stoch.shape[-2] * stoch.shape[-1]),))
                )
            target = {
                "embed": context["embed"],
                "stoch": stoch,
                "deter": start["deter"],
                "feat": context["feat"],
            }[self._config.disag_target]

            inputs = context["feat"]
            if self.config.disag_action_cond:
                inputs = torch.concat(
                    [inputs, torch.Tensor(data["action"]).to(self._config.device)], -1
                )

            loss = self.loss(inputs, target)

        return self.opt(loss, self._networks.parameters())
