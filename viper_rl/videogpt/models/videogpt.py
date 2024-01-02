from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import Transformer

class VideoGPT(nn.Module):
    def __init__(self, config, ae):
        super(VideoGPT, self).__init__()
        self.config = config
        self.ae = ae
        self.shape = (config.seq_len, *ae.latent_shape(config.image_size))
        self.model = Transformer(
            **self.config.transformer,
            shape=self.shape,
            out_dim=self.ae.n_embed
        )
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: min((epoch + 1) / config.warmup_steps, 1)
        )


    def forward(self, embeddings, label=None, decode_step=None, training=False):
        if self.config.class_cond:
            assert label is not None, "label is required for class conditioned model"

        # Create mask (torch.tril can be used for triangular mask)
        L = np.prod(self.shape)
        mask = torch.tril(torch.ones((L, L), dtype=torch.bool))

        if self.config.class_cond:
            label = F.one_hot(label, num_classes=self.config.n_classes)

        return self.model(embeddings, mask=mask, label=label, decode_step=decode_step, deterministic=not training)

    @property
    def metrics(self):
        return ['loss']

    def log_prob(self, embeddings, encodings, label=None, training=False, reduce_sum=True):
        # wtf is embeddings? encodings = batch
        logits = self.forward(embeddings, label=label, training=training)
        labels = F.one_hot(encodings.long(), num_classes=self.ae.n_embed).float()  # Assuming 'encodings' are in a suitable format
        nll = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1, labels.size(-1)), reduction='none')

        if self.config.class_cond:
            nll = nll.view(*nll.shape[:2], -1)
            nll = (nll.max(-1)[0] * np.prod(encodings.shape[2:]) + nll.sum(-1)) / (2 * np.prod(encodings.shape[2:]))
        else:
            if reduce_sum:
                nll = nll.view(*nll.shape[:2], -1).sum(-1)

        return -nll # .mean()  # Taking mean if required, based on how loss is calculated

    def loss(self, embeddings, encodings, label=None, training=True):
        loss = -self.log_prob(
            embeddings, encodings, label, training=training
        ).mean() / np.prod(self.shape[1:])
        return dict(loss=loss)