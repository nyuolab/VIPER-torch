from typing import Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class GELU2(nn.Module):
    def __init__(self):
        super(GELU2, self).__init__()
    
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x

class Transformer(nn.Module):
    def __init__(self, image_size, ae_embed_dim, embed_dim, mlp_dim, num_heads, num_layers, dropout, attention_dropout, shape, out_dim, device):
        super(Transformer, self).__init__()
        self.device = device
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.shape = shape

        self.dense_in = nn.Linear(in_features=ae_embed_dim, out_features=embed_dim)
        self.right_shift = RightShift(embed_dim)  # Assuming RightShift is already defined
        self.position_bias = BroadcastPositionBiases(shape)  # Assuming BroadcastPositionBiases is already defined
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, mlp_dim, num_heads, dropout, attention_dropout)  # Assuming TransformerLayer is already defined
            for _ in range(num_layers)
        ])

        self.norm = LayerNorm(embed_dim)  # Assuming LayerNorm is already defined
        self.dense_out = nn.Linear(embed_dim, out_dim)

    def forward(self, x, mask=None, deterministic=False, label=None, decode_step=None):
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

        position_bias = self.position_bias(x).to(self.device)

        if decode_step is not None and x.shape[1] == 1:
            position_bias = position_bias[decode_step]
        else:
            position_bias = position_bias[:x.shape[1]]
        
        # print(x.shape) # [64, 1024, 256])
        # print(position_bias.shape) # [1024, 256]
        # print(x.get_device())
        # print(position_bias.get_device())

        x += position_bias

        x = self.dropout(x)

        # progress
        for layer in self.layers:
            x = layer(x, mask=mask, label=label, decode_step=decode_step, deterministic=deterministic)
        
        x = self.norm(x)
        x = self.dense_out(x)
        x = x.view(x.shape[0], *old_shape, x.shape[-1])
        return x


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, mlp_dim, num_heads, dropout, attention_dropout):
        super(TransformerLayer, self).__init__()
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout

        # Assuming LayerNorm and MultiHeadAttention are already defined
        self.norm1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(num_heads=num_heads, head_dim=embed_dim // num_heads, dropout_rate=attention_dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            # Assuming gelu2 is already defined
            GELU2(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None, label=None, decode_step=None, deterministic=False):
        h = self.norm1(x)
        h = self.attn(h, mask=mask, decode_step=decode_step)
        h = self.dropout1(h)
        x = x + h

        h = self.norm2(x)
        h = self.mlp(h)
        h = self.dropout2(h)
        x = x + h

        return x

class QKVTransform(nn.Module):
    def __init__(self, input_dim, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.linear = nn.Linear(input_dim, num_heads * head_dim * 3)

    def forward(self, x):
        # Apply the linear transformation
        output = self.linear(x)

        # Reshape the output to get separate query, key, value tensors
        batch_size, seq_length = x.shape[:2]
        qkv = output.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)

        # Split the last dimension into query, key, and value
        query, key, value = qkv.chunk(3, dim=-1)
        return query, key, value

# class DenseGeneral(nn.Module):
#     def __init__(self, num_features):
#         super(DenseGeneral, self).__init__()
#         self.num_features = num_features
#         self.linear = None

#     def forward(self, x):
#         # Dynamically create the linear layer based on the input shape
#         *initial_dims, dim1, dim2 = x.shape
#         if self.linear is None:
#             combined_dim = dim1 * dim2
#             self.linear = nn.Linear(combined_dim, self.num_features)

#         # Reshape the input to combine the last two dimensions
#         x_reshaped = x.view(-1, combined_dim)

#         # Apply the linear transformation
#         out = self.linear(x_reshaped)

#         # Reshape the output to split the last dimension into two dimensions
#         # We infer the second-to-last dimension, assuming num_features is divisible by dim2
#         output_shape = initial_dims + [self.num_features // dim2, dim2]
#         return out.view(output_shape)

# class DotProductAttention(nn.Module):
#     def __init__(self, dropout_rate=0.0):
#         super(DotProductAttention, self).__init__()
#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, query, key, value, mask=None, deterministic=True):
#         d_k = query.size(-1)

#         # Compute attention scores
#         scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k).float())

#         # Apply mask (if provided)
#         # if mask is not None:
#         #     scores = scores.masked_fill(mask == 0, float('-inf'))

#         # Apply softmax to get attention weights
#         attention_weights = F.softmax(scores, dim=-1)

#         # Apply dropout if not in deterministic mode
#         if not deterministic:
#             attention_weights = self.dropout(attention_weights)

#         # Apply the attention to the value tensor
#         output = torch.matmul(attention_weights, value)
#         return output

class DotProductAttention(nn.Module):
    def __init__(self, head_dim, num_heads, dropout_rate=0.0):
        super(DotProductAttention, self).__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.embed_dim = self.head_dim * self.num_heads

        # Output projection layer
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, query, key, value, qkv_proj, attn_mask=None, key_padding_mask=None, deterministic=False):
        # Ensure query, key, value are batched 3D tensors [batch_size, seq_length, embed_dim]
        batch_size, seq_length, _, _ = query.size() # torch.Size([64, 1024, 8, 32])


        # Reshape query, key, value for multi-head attention
        # Split the embed_dim into (num_heads, head_dim)
        query = query.reshape(batch_size, seq_length, -1)
        key = key.reshape(batch_size, seq_length, -1)
        value = value.reshape(batch_size, seq_length, -1)

        attn_output, _ = F.multi_head_attention_forward(
            query=query, key=key, value=value,
            embed_dim_to_check=self.embed_dim, num_heads=self.num_heads,
            in_proj_weight=qkv_proj.weight, in_proj_bias=qkv_proj.bias,
            bias_k=None, bias_v=None,
            add_zero_attn=False, 
            dropout_p=self.dropout_rate,
            out_proj_weight=self.out_proj.weight, 
            out_proj_bias=self.out_proj.bias,
            training=not deterministic,
            key_padding_mask=key_padding_mask, 
            need_weights=True,
            attn_mask=None, # attn_mask.unsqueeze(0)
        )

        # print(attn_output.shape) # torch.Size([64, 1024, 256])

        # Apply the output projection
        # attn_output = attn_output.contiguous().view(batch_size, seq_length, -1)
        attn_output = self.out_proj(attn_output)
        
        return attn_output

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_dim, dropout_rate=0.0):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.embed_dim = num_heads * head_dim
        self.dropout_rate = dropout_rate

        # self.qkv_proj = QKVTransform(num_heads * head_dim, num_heads, head_dim)
        # self.out_proj = DenseGeneral(num_heads * head_dim)
        self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        # self.out_proj = nn.Linear(head_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        # self.dropout = nn.Dropout(dropout_rate)

        self.register_buffer("cached_key", None)
        self.register_buffer("cached_value", None)

        self.dot_product_attention = DotProductAttention(head_dim, num_heads, dropout_rate=dropout_rate)
    
    # def dot_product_attention(self, query, key, value, mask=None, dropout_rate=0.0, deterministic=True):
    #     # Calculate the dot product attention (scaled by key dimension)
    #     d_k = query.size(-1)
    #     scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k).float())
    #     # (scores.shape) # torch.Size([64, 1024, 8, 8])
    #     # print(mask.shape) # torch.Size([1024, 1024])
    #     # Apply the mask (if any)
    #     # if mask is not None:
    #     #     scores = scores.masked_fill(mask == 0, float('-inf'))

    #     # Softmax to get attention weights
    #     attention_weights = F.softmax(scores, dim=-1)

    #     # Apply dropout if not in deterministic mode
    #     if not deterministic and dropout_rate > 0.0:
    #         attention_weights = F.dropout(attention_weights, p=dropout_rate)

    #     # Apply the attention to the value tensor
    #     output = torch.matmul(attention_weights, value)
    #     return output
        
    def forward(self, inputs, mask=None, decode_step=None, deterministic=False):
        qkv = self.qkv_proj(inputs)
        # total_dim = qkv.size(-1)
        qkv = qkv.view(*qkv.shape[:-1], self.num_heads, 3 * self.head_dim)
        query, key, value = qkv.chunk(3, dim=-1)
        
        batch_size, seq_length, _, _ = query.size()
        # print(query.shape)
        # print(key.shape)
        # print(value.shape)
        # torch.Size([64, 1024, 8, 32])

        if decode_step is not None:
            if self.cached_key is None or self.cached_value is None:
                self.cached_key = key
                self.cached_value = value
            else:
                if inputs.shape[1] == 1:
                    self.cached_key[:, :, decode_step:decode_step+1, :] = key
                    self.cached_value[:, :, decode_step:decode_step+1, :] = value
                else:
                    self.cached_key = key
                    self.cached_value = value

            key = self.cached_key
            value = self.cached_value

            if mask is not None and inputs.shape[1] == 1:
                mask = mask[decode_step:decode_step+1, :]

        # Attention computation
        # attn_output, _ = F.multi_head_attention_forward(
        #     query=query,
        #     key=key,
        #     value=value,
        #     embed_dim_to_check=total_dim,
        #     num_heads=self.num_heads,
        #     dropout_p=self.dropout_rate,
        #     training=not deterministic,
        #     need_weights=False,
        #     attn_mask=mask
        # )

        attn_output, _ = self.dot_product_attention(
            query=query,
            key=key,
            value=value,
            qkv_proj=self.qkv_proj,
            attn_mask=mask,
            deterministic=deterministic,
        )
        
        # print(attn_output.shape) # torch.Size([64, 1024, 256])


        # Final projection
        out = self.out_proj(attn_output) # .reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        return out

         
class ConditionalDense(nn.Module):
    def __init__(self, features, use_bias=False):
        super(ConditionalDense, self).__init__()
        self.linear = nn.Linear(features, features, bias=use_bias)

    def forward(self, x):
        return self.linear(x).unsqueeze(-2)


class LayerNorm(nn.Module):
    def __init__(self, features, epsilon=1e-6, use_bias=True, use_scale=True):
        super(LayerNorm, self).__init__()
        self.epsilon = epsilon
        self.use_bias = use_bias
        self.use_scale = use_scale
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(features))
        if use_scale:
            self.scale = nn.Parameter(torch.ones(features))
        # Conditional layers
        self.conditional_scale = ConditionalDense(features, use_bias=False)
        self.conditional_bias = ConditionalDense(features, use_bias=False)

    def forward(self, x, cond=None):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.var(dim=-1, keepdim=True, unbiased=False).sqrt()

        y = (x - mean) / (std + self.epsilon)

        if self.use_scale:
            if cond is None:
                y *= self.scale
            else:
                scale = self.conditional_scale(cond)
                y *= 1 + scale

        if self.use_bias:
            if cond is None:
                y += self.bias
            else:
                bias = self.conditional_bias(cond)
                y += bias

        return y


class BroadcastPositionBiases(nn.Module):
    def __init__(self, shape: Tuple[int]):
        super(BroadcastPositionBiases, self).__init__()
        self.shape = shape
        n_dim = len(shape)
        self.n_dim = n_dim
        self.embs = nn.ParameterList()

        # The embedding dimension will be defined later in the forward pass
        self.embed_dim = None

        for i in range(n_dim):
            # Placeholders for actual sizes which will be set in the forward method
            self.embs.append(nn.Parameter(torch.randn(1) * 0.02))

    def forward(self, x):
        if self.embed_dim is None:
            self.embed_dim = x.shape[-1]
            chunk_sizes = [self.embed_dim // self.n_dim + (i < (self.embed_dim % self.n_dim))
                           for i in range(self.n_dim)]
            assert sum(chunk_sizes) == self.embed_dim, f'sum({chunk_sizes}) = {sum(chunk_sizes)} != {self.embed_dim}'

            for i, emb in enumerate(self.embs):
                self.embs[i] = nn.Parameter(torch.randn(self.shape[i], chunk_sizes[i]) * 0.02)

        out = []
        for i, e in enumerate(self.embs):
            e = e.view((1,) + (1,) * i + (self.shape[i],) + (1,) * (self.n_dim - i - 1) + (-1,))
            e = e.expand((1, *self.shape, e.shape[-1]))
            out.append(e)

        out = torch.cat(out, dim=-1)
        out = out.view((-1, self.embed_dim))
        return out    


class RightShift(nn.Module):
    def __init__(self, channel_size):
        super(RightShift, self).__init__()
        self.sos = nn.Parameter(torch.randn(channel_size) * 0.02)

    def forward(self, x):
        # Creating a tensor of shape [batch_size, 1, channel_size] for sos
        sos = self.sos.unsqueeze(0).unsqueeze(1).expand(x.size(0), 1, -1)
        # Concatenating sos at the beginning of each sequence in the batch
        x_shifted = torch.cat([sos, x[:, :-1]], dim=1)
        return x_shifted
