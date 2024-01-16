from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import Transformer



class VideoGPT(nn.Module):
    def __init__(self, config, ae, device=None):
        super(VideoGPT, self).__init__()
        self.config = config
        # print(self.config)
        self.ae = ae
        self.device = config.device if device is None else device
        self.shape = (config.seq_len, *ae.latent_shape(config.image_size)) # (16, 8, 8)
        self.model = Transformer(
            config.image_size,
            config.ae["embed_dim"],
            **self.config.transformer,
            shape=self.shape,
            out_dim=self.ae.n_embed,
            device=self.device,
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        # progress
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: min((epoch + 1) / (config.warmup_steps+1), 1)
        )


    def forward(self, embeddings, label=None, decode_step=None, training=False):
        if self.config.class_cond:
            assert label is not None, "label is required for class conditioned model"

        # Create mask (torch.tril can be used for triangular mask)
        L = np.prod(self.shape) # 1024
        mask = torch.tril(torch.ones((L, L), dtype=torch.bool)).to(self.device)

        if self.config.class_cond:
            label = F.one_hot(label.long(), num_classes=self.config.n_classes)

        return self.model(embeddings, mask=mask, label=label, decode_step=decode_step, training=training)

    @property
    def metrics(self):
        return ['loss']

    def log_prob(self, embeddings, encodings, label=None, training=False, reduce_sum=True):
        # wtf is embeddings? encodings = batch
        logits = self.forward(embeddings, label=label, training=training)
        # print(logits.shape)
        labels = F.one_hot(encodings.long(), num_classes=self.ae.n_embed).float()  # Assuming 'encodings' are in a suitable format
        # print(logits.shape) # [64, 16, 8, 8, 256]
        # print(labels.shape) # [64, 16, 8, 8, 256]
        # nll = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1, labels.size(-1)), reduction='none')
        # flatten_dim = np.prod(logits.shape[:-1])
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