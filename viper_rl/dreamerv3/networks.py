import math
import numpy as np
import re

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import CLIPProcessor, CLIPModel

import viper_rl.dreamerv3.tools as tools

# World Model: Recurrent State-Space Model (RSSM)
class RSSM(nn.Module):
    def __init__(
        self,
        stoch=30,
        deter=200,
        hidden=200,
        rec_depth=1,
        discrete=False,
        act="SiLU", # silu(x) = x * sigmoid(x)
        norm=True,
        mean_act="none",
        std_act="softplus",
        min_std=0.1,
        unimix_ratio=0.01,
        initial="learned",
        num_actions=None,
        embed=None,
        device=None,
        # video_len=1,
    ):
        super(RSSM, self).__init__()
        self._stoch = stoch # z: stochastic state dim 
        self._deter = deter # h: deterministic state dim
        self._hidden = hidden # h: hidden state dim
        self._min_std = min_std
        self._rec_depth = rec_depth
        self._discrete = discrete # discrete action space?
        act = getattr(torch.nn, act) # torch.nn.SiLU
        self._mean_act = mean_act
        self._std_act = std_act
        self._unimix_ratio = unimix_ratio # Unimix categoricals: Mixtures of 1% uniform and 99% neural network output
        self._initial = initial
        self._num_actions = num_actions
        self._embed = embed
        self._device = device
        # self.video_len = video_len

        if isinstance(num_actions, list):
            num_actions = int(sum(num_actions).item())
        print(num_actions)

        inp_layers = []

        # Why not just self._stoch + num_actions? self._discrete is boolean
        if self._discrete:
            inp_dim = self._stoch * self._discrete + num_actions # *self.video_len
        else:
            inp_dim = self._stoch + num_actions # *self.video_len

        inp_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            inp_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        inp_layers.append(act())

        self._img_in_layers = nn.Sequential(*inp_layers)
        self._img_in_layers.apply(tools.weight_init)
        # Deterministic GRU state
        self._cell = GRUCell(self._hidden, self._deter, norm=norm)
        self._cell.apply(tools.weight_init)

        # input hidden deterministic state dim
        # activations: linear + layernorm + SiLU
        img_out_layers = []
        inp_dim = self._deter
        img_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            img_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        img_out_layers.append(act())
        
        self._img_out_layers = nn.Sequential(*img_out_layers)
        self._img_out_layers.apply(tools.weight_init)

        # Same as above, but why?
        obs_out_layers = []
        inp_dim = self._deter + self._embed
        obs_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            obs_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        obs_out_layers.append(act())
        self._obs_out_layers = nn.Sequential(*obs_out_layers)
        self._obs_out_layers.apply(tools.weight_init)

        # Why do ims_stat_layer(image?) and obs_stat_layer(vector obs?) have the same logistics?
        if self._discrete:
            self._ims_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            # self._ims_stat_layer.apply(tools.weight_init)
            self._imgs_stat_layer = nn.Linear(
                self._hidden, self._stoch * self._discrete
            )
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))

            self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))
            # self._obs_stat_layer.apply(tools.weight_init)
        else:
            self._imgs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            # self._ims_stat_layer.apply(tools.weight_init)
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            # self._obs_stat_layer.apply(tools.weight_init)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))


        if self._initial == "learned":
            self.W = torch.nn.Parameter(
                torch.zeros((1, self._deter), device=torch.device(self._device)),
                requires_grad=True,
            )


    def initial(self, batch_size):
        deter = torch.zeros(batch_size, self._deter).to(self._device)
        if self._discrete:
            state = dict(
                logit=torch.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                stoch=torch.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                deter=deter,
            )
        else:
            state = dict(
                mean=torch.zeros([batch_size, self._stoch]).to(self._device),
                std=torch.zeros([batch_size, self._stoch]).to(self._device),
                stoch=torch.zeros([batch_size, self._stoch]).to(self._device),
                deter=deter,
            )
        if self._initial == "zeros":
            return state
        elif self._initial == "learned":
            state["deter"] = torch.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
            # (batch, time, ch) -> (time, batch, ch)
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        # prev_state[0] means selecting posterior of return(posterior, prior) from obs_step
        post, prior = tools.static_scan(
            lambda prev_state, prev_act, embed, is_first: self.obs_step(
                prev_state[0], prev_act, embed, is_first
            ),
            (action, embed, is_first),
            (state, state),
        )

        # (batch, time, stoch, discrete_num) -> (batch, time, stoch, discrete_num)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine(self, action, state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape)))) # swap 1st & 2nd dim of x, others unchanged
        if state is None:
            state = self.initial(action.shape[0])
        assert isinstance(state, dict), state
        action = action
        action = swap(action)
        prior = tools.static_scan(self.img_step, [action], state)
        prior = prior[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior
    
    # input stochastic z and deterministic h in dictionary
    # concats z & deterministic h along the last dim
    def get_feat(self, state):
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state["deter"]], -1) # concats z & h along the last dim

    def get_dist(self, state, dtype=None):
        if self._discrete:
            logit = state["logit"]
            # print(logit.shape)
            dist = torchd.independent.Independent( 
                tools.OneHotDist(logit, unimix_ratio=self._unimix_ratio, num_seg=1), 1
            )
        else:
            mean, std = state["mean"], state["std"]
            dist = tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, std), 1)
            )
        return dist

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        # initialize all prev_state
        if prev_state == None or torch.sum(is_first) == len(is_first):
            prev_state = self.initial(len(is_first))
            prev_action = torch.zeros((len(is_first), self._num_actions)).to(
                self._device
            )
        # overwrite the prev_state only where is_first=True
        elif torch.sum(is_first) > 0:
            is_first = is_first[:, None]
            prev_action *= 1.0 - is_first
            init_state = self.initial(len(is_first))
            for key, val in prev_state.items():
                is_first_r = torch.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )

                # print(val.shape) # torch.Size([16, 32, 32])
                # print(init_state[key].shape) # torch.Size([16, 32, 32])
                # print(is_first.shape) # torch.Size([16, 1])
                prev_state[key] = (
                    val * (1.0 - is_first_r) + init_state[key] * is_first_r
                )

        prior = self.img_step(prev_state, prev_action)
        x = torch.cat([prior["deter"], embed], -1)
        # (batch_size, prior_deter + embed) -> (batch_size, hidden)
        x = self._obs_out_layers(x)
        # (batch_size, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("obs", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior

    # this is used for making future image, outputs next step z and h
    def img_step(self, prev_state, prev_action, sample=True):
        # (batch, stoch, discrete_num)
        # if isinstance(prev_action , list):
        #     prev_action = tools.for_loop_parallel(prev_action, lambda pre_act: pre_act*(1.0 / torch.clip(torch.abs(prev_act), min=1.0)).detach())
        #     # prev_action *= (1.0 / torch.clip(torch.abs(prev_action), min=1.0)).detach()
        # else:
        
        # prev_action *= (1.0 / torch.clip(torch.abs(prev_action), min=1.0)).detach()
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            # (batch, stoch, discrete_num) -> (batch, stoch * discrete_num)
            prev_stoch = prev_stoch.reshape(shape)
        # (batch, stoch * discrete_num) -> (batch, stoch * discrete_num + action)
        x = torch.cat([prev_stoch, prev_action], -1)
        # (batch, stoch * discrete_num + action, embed) -> (batch, hidden)
        x = self._img_in_layers(x)
        for _ in range(self._rec_depth):  # rec depth is not correctly implemented
            deter = prev_state["deter"]
            # (batch, hidden), (batch, deter) -> (batch, deter), (batch, deter)
            x, deter = self._cell(x, [deter])
            deter = deter[0]  # Keras wraps the state in a list.
        # (batch, deter) -> (batch, hidden)
        x = self._img_out_layers(x)
        # (batch, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("ims", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    def get_stoch(self, deter):
        x = self._img_out_layers(deter)
        stats = self._suff_stats_layer("ims", x)
        dist = self.get_dist(stats)
        return dist.mode()

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            mean, std = torch.split(x, [self._stoch] * 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: torch.softplus(std),
                "abs": lambda: torch.abs(std + 1),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = value = kld(
            dist(post) if self._discrete else dist(post)._dist,
            dist(sg(prior)) if self._discrete else dist(sg(prior))._dist,
        )
        dyn_loss = kld(
            dist(sg(post)) if self._discrete else dist(sg(post))._dist,
            dist(prior) if self._discrete else dist(prior)._dist,
        )
        # this is implemented using maximum at the original repo as the gradients are not backpropagated for the out of limits.
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)

        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss

# class Discriminator(nn.Module):
#     def __init__(
#         self,
#         shapes,
#         cnn_keys=r".*",
#         act="SiLU",
#         norm="LayerNorm",
#         mlp_layers=4,
#         mlp_units=512,
#         cnn="resnet",
#         cnn_depth=48,
#         kernel_size=4,
#         cnn_blocks=2,
#         resize="stride",
#         symlog_inputs=False,
#         minres=4,
#         # **kw,
#     ):
#         excluded = r"(is_first|is_last)"
#         shapes = {
#             k: v
#             for k, v in shapes.items()
#             if (not re.match(excluded, k) and not k.startswith("log_"))
#         }
        
#         input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
#         input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
        
#         self.cnn_shapes = {
#             k: v for k, v in shapes.items() if (len(v) == 3 and re.match(cnn_keys, k))
#         }
#         self.shapes = self.cnn_shapes
#         print("Discriminator CNN shapes:", self.cnn_shapes)
        
#         input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
#         input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)

#         if cnn == "resnet":
#             self._cnn = ConvEncoder(input_shape, depth=cnn_depth, act=act, norm=norm, kernel_size=4, minres=4)
#         else:
#             raise NotImplementedError(cnn)
#         # logit_kw = {**kw, "symlog_inputs": symlog_inputs, "name": "logit_mlp"}
#         input_dim = input_shape[:2] + (self._cnn.outdim,)

#         self._logit_mlp = MLP(
#             input_dim, 
#             None, 
#             layers=mlp_layers, 
#             units=mlp_units, 
#             act=act,
#             norm=norm,
#             symlog_inputs=symlog_inputs,
#             name="discMLP",
#         )
#         # self._logit_mlp.outdim
#         self.disc_logit = nn.Linear(in_features=mlp_units, out_features=1, bias=False)

#     def forward(self, data):
#         some_key, some_shape = list(self.shapes.items())[0]
#         batch_dims = data[some_key].shape[: -(1 + len(some_shape))]
#         data = {
#             k: v.reshape((-1,) + v.shape[len(batch_dims) :]) for k, v in data.items()
#         }
#         inputs = torch.cat([data[k] for k in self.cnn_shapes], -1)
#         inputs = torch.transpose(inputs, (0, 2, 3, 1, 4))
#         inputs = torch.reshape(inputs, inputs.shape[:3] + (np.prod(inputs.shape[3:]),))
#         output = self._cnn(inputs)
#         print(output.shape)
#         output = output.reshape((output.shape[0], -1))
#         output = output.reshape(batch_dims + output.shape[1:])
#         logits = self._logit_mlp(output)
#         projection = self.disc_logit(logits)
#         # projection = projection.reshape(projection.shape[:-1])
#         return projection

class MultiEncoder(nn.Module):
    def __init__(
        self,
        shapes,
        device,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        symlog_inputs,
    ):
        super(MultiEncoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal", "reward")
        # excluded = ("is_first", "is_last", "is_terminal")
        shapes = {
            k: v
            for k, v in shapes.items()
            if k not in excluded and not k.startswith("log_")
        }
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)

        self.outdim = 0
        if self.cnn_shapes:
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            self._cnn = ConvEncoder(
                input_shape, cnn_depth, act, norm, kernel_size, minres
            )
            # if torch.cuda.device_count() > 1:
            #     self._cnn = DDP(self._cnn.to(device), device_ids=[device])
            #     self.outdim += self._cnn.module.outdim
            # else:
            self.outdim += self._cnn.outdim
        
        if self.mlp_shapes:
            input_size = sum([sum(v) for v in self.mlp_shapes.values()])
            # print("The input size is {}".format(input_size))
            self._mlp = MLP(
                input_size,
                None,
                mlp_layers,
                mlp_units,
                act,
                norm,
                symlog_inputs=symlog_inputs,
                name="Encoder",
            )
            self.outdim += mlp_units

    def forward(self, obs):
        outputs = []
        if self.cnn_shapes:
            inputs = torch.cat([obs[k] for k in self.cnn_shapes], -1)
            # print("MultiEncoder CNN Input shape: {}".format(inputs.shape))
            outputs.append(self._cnn(inputs))
        if self.mlp_shapes:
            # print([obs[k].shape for k in self.mlp_shapes])
            # [torch.Size([16, 100, 1]), torch.Size([16, 100, 4, 6]),
            # torch.Size([16, 100, 1]), torch.Size([16, 100, 1]), 
            # torch.Size([16, 100, 4, 36]), torch.Size([16, 100, 4, 36])]
            
            # print(self.mlp_shapes)
            # {'breath': (1,), 'equipped': (4,), 
            # 'health': (1,), 'hunger': (1,), 
            # 'inventory': (4,), 'inventory_max': (4,)}
            # for k in self.mlp_shapes:
            #     print("{0} has shape {1}".format(k, obs[k].shape))
            #     print(tuple(obs[k].shape[:2])+(-1,)) # (16, 100, -1)
            inputs = torch.cat([obs[k].reshape(tuple(obs[k].shape[:-1])+(-1,)) for k in self.mlp_shapes], -1)
            # print("MultiEncoder MLP Input shape: {}".format(inputs.shape))
            outputs.append(self._mlp(inputs))
        outputs = torch.cat(outputs, -1)
        return outputs

class CLIPEncoder(nn.Module):
    def __init__(
        self,
        # image_shape,
        device,
        layers=2,
        units=512,
        act='SiLU',
        norm=True,
        dist='normal',
        symlog_inputs=True,
    ):
        super(CLIPEncoder, self).__init__()
        self.device = device
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        for param in self.clip.parameters():
            param.requires_grad = False
        # Verify
        for name, param in self.clip.named_parameters():
            assert not param.requires_grad, f"Parameter {name} is not frozen."

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self._mlp = MLP(
                    self.clip.config.projection_dim,
                    None,
                    layers=layers,
                    units=units,
                    act=act,
                    norm=norm,
                    dist='normal',
                    symlog_inputs=symlog_inputs,
                    name="CLIP_out",
                ).to(device)
        self.outdim = units

    def forward(self, obs):
        # Get the image embeddings
        x = obs['image']
        old_dim = len(x.shape)
        if old_dim > 4:
            pre_shape = x.shape[:-3]
            x = x.view(-1, *x.shape[-3:])
        with torch.no_grad():
            x = self.processor(images=x, return_tensors="pt", do_rescale=False).to(self.device)
            x = self.clip.get_image_features(**x)
        x = self._mlp(x)
        if old_dim > 4:
            x = x.view(*pre_shape, -1)
        return x # [batch_size, time, self.clip.config.projection_dim]

class MultiDecoder(nn.Module):
    def __init__(
        self,
        feat_size,
        shapes,
        device,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        cnn_sigmoid,
        image_dist,
        vector_dist,
        outscale,
    ):
        super(MultiDecoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal")
        # excluded = ("is_first", "is_last", "is_terminal")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Decoder CNN shapes:", self.cnn_shapes)
        print("Decoder MLP shapes:", self.mlp_shapes)

        if self.cnn_shapes:
            some_shape = list(self.cnn_shapes.values())[0]
            shape = (sum(x[-1] for x in self.cnn_shapes.values()),) + some_shape[:-1]
            self._cnn = ConvDecoder(
                feat_size,
                shape,
                cnn_depth,
                act,
                norm,
                kernel_size,
                minres,
                outscale=outscale,
                cnn_sigmoid=cnn_sigmoid,
            )
            # if torch.cuda.device_count() > 1:
            #     self._cnn = DDP(self._cnn.to(device), device_ids=[device])

        if self.mlp_shapes:
            self._mlp = MLP(
                feat_size,
                self.mlp_shapes,
                mlp_layers,
                mlp_units,
                act,
                norm,
                vector_dist,
                outscale=outscale,
                name="Decoder",
            )
        self._image_dist = image_dist

    def forward(self, features):
        dists = {}
        if self.cnn_shapes:
            feat = features
            outputs = self._cnn(feat)
            split_sizes = [v[-1] for v in self.cnn_shapes.values()]
            outputs = torch.split(outputs, split_sizes, -1)
            dists.update(
                {
                    key: self._make_image_dist(output)
                    for key, output in zip(self.cnn_shapes.keys(), outputs)
                }
            )
        if self.mlp_shapes:
            dists.update(self._mlp(features))
        return dists

    def _make_image_dist(self, mean):
        if self._image_dist == "normal":
            return tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, 1), 3)
            )
        if self._image_dist == "mse":
            return tools.MSEDist(mean)
        raise NotImplementedError(self._image_dist)


class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_shape,
        depth=32,
        act="SiLU",
        norm=True,
        kernel_size=4,
        minres=4,
    ):
        super(ConvEncoder, self).__init__()
        act = getattr(torch.nn, act)
        h, w, input_ch = input_shape
        stages = int(np.log2(h) - np.log2(minres))
        in_dim = input_ch
        out_dim = depth
        layers = []
        for i in range(stages):
            layers.append(
                Conv2dSamePad(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    bias=False,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            layers.append(act())
            in_dim = out_dim
            out_dim *= 2
            h, w = h // 2, w // 2

        self.outdim = out_dim // 2 * h * w
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

    def forward(self, obs):
        # obs -= 0.5
        # (batch, time, h, w, ch) -> (batch * time, h, w, ch)
        # print(obs.shape)
        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))
        # print("ConvEncoder input shape: {}".format(x.shape))
        # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
            # print("ConvEncoder input shape after permute: {}".format(x.shape))
        x = self.layers(x)
        # (batch * time, ...) -> (batch * time, -1)
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        # (batch * time, -1) -> (batch, time, -1)
        return x.reshape(list(obs.shape[:-3]) + [x.shape[-1]])


class ConvDecoder(nn.Module):
    def __init__(
        self,
        feat_size,
        shape=(3, 64, 64),
        depth=32,
        act=nn.ELU,
        norm=True,
        kernel_size=4,
        minres=4,
        outscale=1.0,
        cnn_sigmoid=False,
    ):
        super(ConvDecoder, self).__init__()
        act = getattr(torch.nn, act)
        self._shape = shape
        self._cnn_sigmoid = cnn_sigmoid
        layer_num = int(np.log2(shape[1]) - np.log2(minres))
        self._minres = minres
        out_ch = minres**2 * depth * 2 ** (layer_num - 1)
        self._embed_size = out_ch
        # 16 * 96 * 2**4 = 24576
        self._linear_layer = nn.Linear(feat_size, out_ch)
        self._linear_layer.apply(tools.uniform_weight_init(outscale))
        in_dim = out_ch // (minres**2)
        out_dim = in_dim // 2

        layers = []
        h, w = minres, minres
        for i in range(layer_num):
            bias = False
            if i == layer_num - 1:
                out_dim = self._shape[0]
                act = False
                bias = True
                norm = False

            if i:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth
            pad_h, outpad_h = self.calc_same_pad(k=kernel_size, s=2, d=1)
            pad_w, outpad_w = self.calc_same_pad(k=kernel_size, s=2, d=1)
            layers.append(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    2,
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=bias,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            if act:
                layers.append(act())
            in_dim = out_dim
            out_dim //= 2
            h, w = h * 2, w * 2

        [m.apply(tools.weight_init) for m in layers[:-1]]
        layers[-1].apply(tools.uniform_weight_init(outscale))
        self.layers = nn.Sequential(*layers)

    def calc_same_pad(self, k, s, d):
        val = d * (k - 1) - s + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad

    def forward(self, features, dtype=None):
        # print(features.shape) [16, 100, 5120]
        x = self._linear_layer(features)
        # (batch, time, -1) -> (batch * time, h, w, ch)
        x = x.reshape(
            [-1, self._minres, self._minres, self._embed_size // self._minres**2]
        )
        # print(x.shape) [1600, 4, 4, 1536]
        # print(self._shape) (3, 160, 160)

        # (batch, time, -1) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2) # [1600, 1536, 4, 4]
        x = self.layers(x)
        # print(features.shape[:-1] + self._shape) # [16, 100, 3, 160, 160]
        # (batch, time, -1) -> (batch * time, ch, h, w)
        mean = x.reshape(features.shape[:-1] + self._shape)
        # print(x.shape)
        # (batch * time, ch, h, w) -> (batch * time, h, w, ch)
        # mean = mean.permute(0, 1, 3, 4, 2)
        k = len(mean.shape) - 3
        permuted_dims = list(range(k)) + [k+1, k+2, k]
        mean = mean.permute(*permuted_dims)
        # mean = mean.permute(0, 2, 3, 1)
        if self._cnn_sigmoid:
            mean = F.sigmoid(mean) - 0.5
        #     mean = F.sigmoid(mean)
        # else:
        #     mean += 0.5
        return mean

# density_head: 
#     {layers: 5, units: 1024, act: "SiLU", 
#     norm: True, dist: 'symlog_disc', loss_scale: 1.0, 
#     outscale: 0.0, name: "Density"}
class MLP(nn.Module):
    def __init__(
        self,
        inp_dim,
        shape,
        layers=5,
        units=1024,
        act="SiLU",
        norm=True,
        dist="normal",
        loss_scale=1.0,
        std=1.0,
        min_std=0.1,
        max_std=1.0,
        absmax=None,
        temp=0.1,
        unimix_ratio=0.01,
        outscale=1.0,
        symlog_inputs=False,
        name="NoName",
        device="cuda",
    ):
        super(MLP, self).__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape

        if self._shape is None:
            self.outdim = units
        else:
            if len(self._shape) == 0:
                self._shape = (1,)

        act = getattr(torch.nn, act)
        self._dist = dist
        self._std = std if isinstance(std, str) else torch.tensor((std,), device=device)
        self._min_std = min_std
        self._max_std = max_std
        self._absmax = absmax
        self._temp = temp
        self._unimix_ratio = unimix_ratio
        self._symlog_inputs = symlog_inputs
        self._device = device

        self.layers = nn.Sequential()
        for i in range(layers):
            self.layers.add_module(f"{name}_linear{i}", nn.Linear(inp_dim, units, bias=False))
            if norm:
                self.layers.add_module(
                    f"{name}_norm{i}", nn.LayerNorm(units, eps=1e-03)
                )
            self.layers.add_module(f"{name}_act{i}", act())
            if i == 0:
                inp_dim = units
        self.layers.apply(tools.weight_init)
        
        if isinstance(self._shape, dict):
            self.mean_layer = nn.ModuleDict()
            for name, shape in self._shape.items():
                self.mean_layer[name] = nn.Linear(inp_dim, np.prod(shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                self.std_layer = nn.ModuleDict()
                for name, shape in self._shape.items():
                    self.std_layer[name] = nn.Linear(inp_dim, np.prod(shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))
        elif self._shape is not None:
            self.mean_layer = nn.Linear(inp_dim, np.prod(self._shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.Linear(units, np.prod(self._shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))
            # if self._dist == "multionehot":
            #     self._dist_layer = []
            #     for dim in self._size:
            #         post_layers = []
            #         for index in range(out_layers):
            #             post_layers.append(nn.Linear(inp_dim, self._units, bias=False))
            #             post_layers.append(norm(self._units, eps=1e-03))
            #             post_layers.append(act())
            #         head = nn.Sequential(*post_layers)
            #         head.apply(tools.weight_init)

            #         dl = nn.Linear(self._units, dim)
            #         dl.apply(tools.uniform_weight_init(outscale))
                    
            #         head.add_module('dist', dl)
                    
            #         self._dist_layer.append(head)

            #     self._dist_layer = nn.ModuleList(self._dist_layer)

    def forward(self, features, dtype=None):
        x = features
        if self._symlog_inputs:
            x = tools.symlog(x)
        out = self.layers(x)
        if self._shape is None:
            return out
        if isinstance(self._shape, dict):
            dists = {}
            for name, shape in self._shape.items():
                mean = self.mean_layer[name](out)
                if self._std == "learned":
                    std = self.std_layer[name](out)
                else:
                    std = self._std
                dists.update({name: self.dist(self._dist, mean, std, shape)})
            return dists
        else:
            mean = self.mean_layer(out)
            if self._std == "learned":
                std = self.std_layer(out)
            else:
                std = self._std
            return self.dist(self._dist, mean, std, self._shape)
        

    def dist(self, dist, mean, std, shape):
        # print(self._dist)
        if self._dist == "tanh_normal":
            mean = torch.tanh(mean)
            std = F.softplus(std) + self._min_std
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(
                dist, tools.TanhBijector()
            )
            dist = torchd.independent.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == "normal":
            std = (self._max_std - self._min_std) * torch.sigmoid(
                std + 2.0
            ) + self._min_std
            dist = torchd.normal.Normal(torch.tanh(mean), std)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "normal_std_fixed":
            dist = torchd.normal.Normal(mean, self._std)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "trunc_normal":
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "onehot":
            dist = tools.OneHotDist(mean, unimix_ratio=self._unimix_ratio)
        elif self._dist == "onehot_gumble":
            dist = tools.ContDist(
                torchd.gumbel.Gumbel(mean, 1 / self._temp), absmax=self._absmax
            )
        elif self._dist == "multionehot":
            xlist = []
            for dl in self._dist_layer:
                xtemp = dl(x)
                xlist.append(xtemp)
            dist = tools.MultiOneHotDist(xlist, unimix_ratio=self._unimix_ratio)
        elif self._dist == "huber":
            return tools.ContDist(
                torchd.independent.Independent(
                    tools.UnnormalizedHuber(mean, std, 1.0), len(shape)
                )
            )
        elif self._dist == "binary":
            return tools.Bernoulli(
                torchd.independent.Independent(
                    torchd.bernoulli.Bernoulli(logits=mean), len(shape)
                )
            )
        elif self._dist == "symlog_disc":
            return tools.DiscDist(logits=mean, device=self._device)
        elif self._dist == "symlog_mse":
            return tools.SymlogDist(mean)
        else:
            raise NotImplementedError(dist)
        return dist


# class ActionHead(nn.Module):
#     def __init__(
#         self,
#         inp_dim,
#         size,
#         layers,
#         units,
#         act=nn.ELU,
#         norm=nn.LayerNorm,
#         dist="trunc_normal",
#         init_std=0.0,
#         min_std=0.1,
#         max_std=1.0,
#         temp=0.1,
#         outscale=1.0,
#         unimix_ratio=0.01,
#         out_layers=1,
#         action_space=None,
#         # video_len=5
#     ):
#         super(ActionHead, self).__init__()
#         self._size = size
#         self._layers = layers
#         self._units = units
#         self._dist = dist
#         act = getattr(torch.nn, act)
#         norm = getattr(torch.nn, norm)
#         self._min_std = min_std
#         self._max_std = max_std
#         self._init_std = init_std
#         self._unimix_ratio = unimix_ratio
#         self._temp = temp() if callable(temp) else temp
#         # self.video_len = video_len

#         pre_layers = []
#         for index in range(self._layers):
#             pre_layers.append(nn.Linear(inp_dim, self._units, bias=False))
#             pre_layers.append(norm(self._units, eps=1e-03))
#             pre_layers.append(act())
#             if index == 0:
#                 inp_dim = self._units
#         self._pre_layers = nn.Sequential(*pre_layers)
#         self._pre_layers.apply(tools.weight_init)

#         if self._dist in ["tanh_normal", "tanh_normal_5", "normal", "trunc_normal"]:
#             self._dist_layer = nn.Linear(self._units, 2 * self._size) # * video_len)
#             self._dist_layer.apply(tools.uniform_weight_init(outscale))

#         elif self._dist in ["normal_1", "onehot", "onehot_gumbel"]:
#             self._dist_layer = nn.Linear(self._units, self._size) # * video_len)
#             self._dist_layer.apply(tools.uniform_weight_init(outscale))
#         elif self._dist == "multionehot":
#             self._dist_layer = []
#             for dim in self._size:
#                 post_layers = []
#                 for index in range(out_layers):
#                     post_layers.append(nn.Linear(inp_dim, self._units, bias=False))
#                     post_layers.append(norm(self._units, eps=1e-03))
#                     post_layers.append(act())
#                 head = nn.Sequential(*post_layers)
#                 head.apply(tools.weight_init)

#                 dl = nn.Linear(self._units, dim)
#                 dl.apply(tools.uniform_weight_init(outscale))
                
#                 head.add_module('dist', dl)
                
#                 self._dist_layer.append(head)

#             self._dist_layer = nn.ModuleList(self._dist_layer)

#     def forward(self, features, dtype=None):
#         x = features
#         x = self._pre_layers(x)
#         if self._dist == "tanh_normal":
#             x = self._dist_layer(x)
#             mean, std = torch.split(x, 2, -1)
#             mean = torch.tanh(mean)
#             std = F.softplus(std + self._init_std) + self._min_std
#             dist = torchd.normal.Normal(mean, std)
#             dist = torchd.transformed_distribution.TransformedDistribution(
#                 dist, tools.TanhBijector()
#             )
#             dist = torchd.independent.Independent(dist, 1)
#             dist = tools.SampleDist(dist)
#         elif self._dist == "tanh_normal_5":
#             x = self._dist_layer(x)
#             mean, std = torch.split(x, 2, -1)
#             mean = 5 * torch.tanh(mean / 5)
#             std = F.softplus(std + 5) + 5
#             dist = torchd.normal.Normal(mean, std)
#             dist = torchd.transformed_distribution.TransformedDistribution(
#                 dist, tools.TanhBijector()
#             )
#             dist = torchd.independent.Independent(dist, 1)
#             dist = tools.SampleDist(dist)
#         elif self._dist == "normal":
#             x = self._dist_layer(x)
#             mean, std = torch.split(x, [self._size] * 2, -1)
#             std = (self._max_std - self._min_std) * torch.sigmoid(
#                 std + 2.0
#             ) + self._min_std
#             dist = torchd.normal.Normal(torch.tanh(mean), std)
#             dist = tools.ContDist(
#                 torchd.independent.Independent(dist, 1), absmax=self._absmax
#             )
#         elif self._dist == "normal_1":
#             mean = self._dist_layer(x)
#             dist = torchd.normal.Normal(mean, 1)
#             dist = tools.ContDist(
#                 torchd.independent.Independent(dist, 1), absmax=self._absmax
#             )
#         elif self._dist == "trunc_normal":
#             x = self._dist_layer(x)
#             mean, std = torch.split(x, [self._size] * 2, -1)
#             mean = torch.tanh(mean)
#             std = 2 * torch.sigmoid(std / 2) + self._min_std
#             dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
#             dist = tools.ContDist(
#                 torchd.independent.Independent(dist, 1), absmax=self._absmax
#             )
#         elif self._dist == "onehot":
#             x = self._dist_layer(x)
#             dist = tools.OneHotDist(x, unimix_ratio=self._unimix_ratio)
#         elif self._dist == "onehot_gumble":
#             x = self._dist_layer(x)
#             temp = self._temp
#             dist = tools.ContDist(
#                 torchd.gumbel.Gumbel(mean, 1 / self._temp), absmax=self._absmax
#             )
#         elif self._dist == "multionehot":
#             xlist = []
#             for dl in self._dist_layer:
#                 xtemp = dl(x)
#                 xlist.append(xtemp)
#             dist = tools.MultiOneHotDist(xlist, unimix_ratio=self._unimix_ratio)
#         elif dist == "huber":
#             dist = tools.ContDist(
#                 torchd.independent.Independent(
#                     tools.UnnormalizedHuber(mean, std, 1.0), len(shape)
#                     tools.UnnormalizedHuber(mean, std, 1.0),
#                     len(shape),
#                     absmax=self._absmax,
#                 )
#             )
#         elif dist == "binary":
#             dist = tools.Bernoulli(
#                 torchd.independent.Independent(
#                     torchd.bernoulli.Bernoulli(logits=mean), len(shape)
#                 )
#             )
#         elif dist == "symlog_disc":
#             dist = tools.DiscDist(logits=mean, device=self._device)
#         elif dist == "symlog_mse":
#             dist = tools.SymlogDist(mean)
#         else:
#             raise NotImplementedError(dist)
#         return dist


class GRUCell(nn.Module):
    def __init__(self, inp_size, size, norm=True, act=torch.tanh, update_bias=-1):
        super(GRUCell, self).__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._update_bias = update_bias
        self.layers = nn.Sequential()
        self.layers.add_module(
            "GRU_linear", nn.Linear(inp_size + size, 3 * size, bias=False)
        )
        if norm:
            self.layers.add_module("GRU_norm", nn.LayerNorm(3 * size, eps=1e-03))

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self.layers(torch.cat([inputs, state], -1))
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


class Conv2dSamePad(torch.nn.Conv2d):
    def calc_same_pad(self, i, k, s, d):
        return max((math.ceil(i/s) - 1) * s + (k-1) * d + 1 - i, 0)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        ret = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return ret


class ImgChLayerNorm(nn.Module):
    def __init__(self, ch, eps=1e-03):
        super(ImgChLayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x
