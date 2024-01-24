from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import Transformer

import pathlib
import sys


directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))

from viper_rl.videogpt import weight_init



class VideoGPT(nn.Module):
    def __init__(self, config, ae):
        super(VideoGPT, self).__init__()
        self.config = config
        # print(self.config)
        self.ae = ae
        # self.device = config.device if device is None else device
        self.shape = (config.seq_len, *ae.latent_shape(config.image_size)) # (16, 8, 8)
        self.model = Transformer(
            config.image_size,
            config.ae["embed_dim"],
            **self.config.transformer,
            shape=self.shape,
            out_dim=self.ae.n_embed,
            n_classes=self.config.n_classes,
        )
        self.ema_decay = config.ema

        self.model.apply(weight_init)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)

        # def lambda_fn(step):
        #     return min(step / float(config.warmup_steps), 1)
            
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.total_steps,
            gamma=1.0
        )

    def init_ema_params(self):
        self.ema_params = {name: param.clone() for name, param in self.model.named_parameters()}

    def update_ema(self, ema_decay=None):
        if not ema_decay:
            ema_decay = self.ema_decay
        model = self.model.module if self.config.ddp else self.model
        for name, param in model.named_parameters():
            # print("name is {}".format(name))
            # print("ema param device is {}".format(self.ema_params[name].device))
            # print("param device is {}".format(param.device))
            if param.requires_grad:
                self.ema_params[name] = self.ema_decay * self.ema_params[name] + (1.0 - self.ema_decay) * param


    def forward(self, embeddings, label=None, decode_step=None, training=False):
        if self.config.class_cond:
            assert label is not None, "label is required for class conditioned model"

        # Create mask (torch.tril can be used for triangular mask)
        L = np.prod(self.shape) # 1024
        mask = torch.tril(torch.ones((L, L), dtype=torch.bool)).to(self.device)

        if self.config.class_cond:
            label = F.one_hot(label.long(), num_classes=self.config.n_classes).float()

        return self.model(embeddings, mask=mask, label=label, decode_step=decode_step, training=training)

    @property
    def metrics(self):
        return ['loss']

    def log_prob(self, embeddings, encodings, label=None, training=False, reduce_sum=True):
        logits = self.forward(embeddings, label=label, training=training)
        # print(logits.shape)        # print(logits.shape) # [64, 16, 8, 8, 256]
        # print(labels.shape) # [64, 16, 8, 8, 256]
        # nll = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1, labels.size(-1)), reduction='none')
        # flatten_dim = np.prod(logits.shape[:-1])
        labels = F.one_hot(encodings.long(), num_classes=self.ae.n_embed).float()  # Assuming 'encodings' are in a suitable format

        labels = torch.argmax(labels, dim=-1)
        # nll = F.cross_entropy(logits, labels)
        # print(logits.view(flatten_dim, -1).shape)
        # print(labels.view(flatten_dim, -1).shape)
        nll = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none').view(logits.shape[:-1])
        # print(nll.shape)
        if self.config.class_cond:
            nll = nll.view(*nll.shape[:2], -1)
            nll = (nll.max(-1)[0] * np.prod(encodings.shape[2:]) + nll.sum(-1)) / (2 * np.prod(encodings.shape[2:]))
        else:
            if reduce_sum:
                nll = nll.view(*nll.shape[:2], -1).sum(dim=-1)
        # print(nll.shape)
        return -nll # .mean()  # Taking mean if required, based on how loss is calculated

    def loss(self, batch, training=True):
        embeddings = batch["embeddings"]
        # print(embeddings)
        # print("embeddings shape is {}".format(embeddings.shape))
        encodings = batch["encodings"]
        # print(encodings)
        # print("encodings shape is {}".format(encodings.shape))
        label = batch["label"] if "label" in batch else None
        # print("encodings shape is {}".format(label.shape))
        loss = -self.log_prob(
            embeddings, encodings, label, training=training
        ).mean() / np.prod(self.shape[1:])
        return dict(loss=loss)