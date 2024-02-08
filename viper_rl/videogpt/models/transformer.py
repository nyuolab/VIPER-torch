import numpy as np
import torch.nn as nn

from .attention import AttentionBlock, LayerNorm, BroadcastPositionBiases, RightShift


class Transformer(nn.Module):
    def __init__(self, image_size, ae_embed_dim, embed_dim, mlp_dim, n_head, n_layer, dropout, attention_dropout, attn_type, shape, out_dim, n_classes=17):
        super(Transformer, self).__init__()
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.n_layer = n_layer
        self.shape = shape # (seq_len=16, 8, 8)
        self.n_classes = n_classes

        self.dense_in = nn.Linear(in_features=ae_embed_dim, out_features=embed_dim)
        self.right_shift = RightShift(embed_dim)  # Assuming RightShift is already defined
        self.position_bias = BroadcastPositionBiases(shape, self.embed_dim)  # Assuming BroadcastPositionBiases is already defined
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            AttentionBlock(shape, embed_dim, mlp_dim, n_head, n_layer, dropout, attention_dropout, attn_type, n_classes)  # Assuming TransformerLayer is already defined
            for _ in range(n_layer)
        ])

        self.norm = LayerNorm(embed_dim, n_classes)  # Assuming LayerNorm is already defined
        self.dense_out = nn.Linear(embed_dim, out_dim)

    def position_bias_to_device(self, device=None):
        # self.position_bias = BroadcastPositionBiases(self.shape, self.embed_dim, self.device)
        device = self.device if device is None else device
        self.position_bias.embs.to(device)

    def forward(self, x, label=None, decode_step=None, decode_idx=None, training=False):
        old_shape = x.shape[1:-1]
        # print(x.shape) # [64, 16, 8, 8, 64]
        x = x.view(x.shape[0], -1, x.shape[-1])
        # print(x.shape) # [64, 1024, 64]

        x = self.dense_in(x)
        if decode_step is None or x.shape[1] > 1:
            x = self.right_shift(x)
        else:
            x_shift = self.right_shift(x)
            x = x if decode_step > 0 else x_shift

        position_bias = self.position_bias(x, decode_step=decode_step, decode_idx=decode_idx)

        # if decode_step is not None and x.shape[1] == 1:
        #     position_bias = position_bias[decode_step]
        # else:
        #     position_bias = position_bias[:x.shape[1]]
        
        # print(x.shape) # [64, 1024, 256])
        # print(position_bias.shape) # [1024, 256]
        # print(x.get_device())
        # print(position_bias.get_device())
        # print(x.shape)
        # print(position_bias.shape)
        x += position_bias

        x = self.dropout(x)

        # progress
        # print(mask.shape)
        for layer in self.layers:
            x = layer(x, label=label, decode_step=decode_step, decode_idx=decode_idx, training=training)
        
        x = self.norm(x, cond=label)
        x = self.dense_out(x)
        x = x.view(x.shape[0], *old_shape, x.shape[-1])
        return x

