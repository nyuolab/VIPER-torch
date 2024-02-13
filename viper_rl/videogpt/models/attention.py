from typing import Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pathlib
import sys

directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
from viper_rl.videogpt.train_utils import view_range, shift_dim, tensor_slice


class AttentionBlock(nn.Module):
    def __init__(self, shape, embed_dim, mlp_dim, n_head, n_layer, dropout, attn_dropout, attn_type, n_classes):
        super(AttentionBlock, self).__init__()
        self.shape = shape
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.n_head = n_head
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.n_layer = n_layer

        # Assuming LayerNorm and MultiHeadAttention are already defined
        self.norm1 = LayerNorm(embed_dim, n_classes)
        self.attn = MultiHeadAttention(
            shape, embed_dim, embed_dim, n_head,
            n_layer, causal=True, attn_type=attn_type,
            dropout_rate=attn_dropout
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(embed_dim, n_classes)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            # Assuming gelu2 is already defined
            GELU2(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, label=None, decode_step=None, decode_idx=None, training=True):
        h = self.norm1(x, cond=label)
        h = self.attn(h, h, h, decode_step=decode_step, decode_idx=decode_idx, training=training)
        h = self.dropout1(h)
        x = x + h

        h = self.norm2(x, cond=label)
        h = self.mlp(h)
        h = self.dropout2(h)
        x = x + h

        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, shape, dim_q, dim_kv, n_head, n_layer,
                 causal, attn_type, dropout_rate=0.0):
        super(MultiHeadAttention, self).__init__()
        self.causal = causal
        self.shape = shape
        self.init_seq_len = np.prod(shape[1:])
        self.attn_seq_len = np.prod(shape)
        self.dropout_rate = dropout_rate

        self.d_k = dim_q // n_head
        self.d_v = dim_kv // n_head
        self.n_head = n_head

        self.w_qs = nn.Linear(dim_q, n_head * self.d_k, bias=False) # q
        self.w_qs.weight.data.normal_(std=1.0 / np.sqrt(dim_q))

        self.w_ks = nn.Linear(dim_kv, n_head * self.d_k, bias=False) # k
        self.w_ks.weight.data.normal_(std=1.0 / np.sqrt(dim_kv))

        self.w_vs = nn.Linear(dim_kv, n_head * self.d_v, bias=False) # v
        self.w_vs.weight.data.normal_(std=1.0 / np.sqrt(dim_kv))

        self.fc = nn.Linear(n_head * self.d_v, dim_q, bias=True) # c
        self.fc.weight.data.normal_(std=1.0 / np.sqrt(dim_q * n_layer))

        if attn_type == 'full':
            self.attn = FullAttention(self.attn_seq_len, causal, dropout_rate)
        elif attn_type == 'axial':
            assert not causal, 'causal axial attention is not supported'
            self.attn = AxialAttention(len(shape), dropout_rate)
        elif attn_type == 'sparse':
            self.attn = SparseAttention(shape, n_head, causal, dropout_rate)
        
        # self.cache = None
        
    def forward(self, q, k, v, decode_step=None, decode_idx=None, training=True):
        """ Compute multi-head attention
        Args
            q, k, v: a [b, d1, ..., dn, c] tensor or
                     a [b, 1, ..., 1, c] tensor if decode_step is not None

        Returns
            The output after performing attention
        """

        is_slice = q.shape[1] == 1
        # compute k, q, v
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        q = view_range(self.w_qs(q), -1, None, (n_head, d_k))
        k = view_range(self.w_ks(k), -1, None, (n_head, d_k))
        v = view_range(self.w_vs(v), -1, None, (n_head, d_v))
        
        # b x n_head x seq_len x d
        # (b, *d_shape, n_head, d) -> (b, n_head, *d_shape, d)
        q = shift_dim(q, -2, 1)
        k = shift_dim(k, -2, 1)
        v = shift_dim(v, -2, 1)

        # print(q.shape)

        # print("key dim is {}".format(k.shape))
        # print("value dim is {}".format(v.shape))

        # is_slice = q.shape[2] == 1
        # fast decoding
        # if self.cache is not None:
        #     print(type(self.cache))
        # else:
        #     print(type(self.cache['k']))

        if decode_step is not None:
            # if self.cache is None:
            if decode_step == 0:
                # assert self.cache is None
                # if self.causal:
                #     k_shape = (q.shape[0], n_head, attn_seq_len, self.d_k)
                #     v_shape = (q.shape[0], n_head, attn_seq_len, self.d_v)
                #     self.cache = dict(k=torch.zeros(k_shape, dtype=k.dtype, device=q.device),
                #                     v=torch.zeros(v_shape, dtype=v.dtype, device=q.device))
                #     self.cache['k'][:, :, :self.init_seq_len]
                # else:
                    # cache only once in the non-causal case
                self.cache = dict(k=k.clone(), v=v.clone())

            else:
                if self.causal:
                    # progress
                    # idx = (slice(None, None), slice(None, None), *[decode_step])
                    # self.cache['k'][idx] = k
                    # self.cache['v'][idx] = v
                    self.cache['k'][..., decode_step:decode_step+1, :] = k
                    self.cache['v'][..., decode_step:decode_step+1, :] = v
            

            k, v = self.cache['k'], self.cache['v']
            # print("query shape is {}".format(q.shape))
            # print("key shape is {}".format(k.shape))
            # print("value shape is {}".format(v.shape))
        
        a = self.attn(q, k, v, decode_step, decode_idx, training=training)

        # (b, n_head, *d_shape, d) -> (b, *d_shape, n_head, d)
        # (b, *d_shape, n_head, d) -> (b, *d_shape, n_head * d)
        a = shift_dim(a, 1, -2).flatten(start_dim=-2)
        # print(attn_output.shape) # torch.Size([64, 1024, 256])

        # Final projection
        a = self.fc(a) # (b x seq_len x embed_dim)
        # print(a.shape) # [75, 1024, 256]
        # .reshape(batch_size, seq_length, self.n_head, self.head_dim)
        return a

############## Attention #######################
class FullAttention(nn.Module):
    def __init__(self, attn_seq_len, causal, attn_dropout):
        super().__init__()
        self.causal = causal
        self.attn_dropout = attn_dropout

        # seq_len = np.prod(shape)
        if self.causal:
            self.register_buffer('mask', torch.tril(torch.ones(attn_seq_len, attn_seq_len)))

    def forward(self, q, k, v, decode_step, decode_idx, training=True):
        mask = self.mask if self.causal else None
        if decode_step is not None and mask is not None:
            mask = mask[[decode_step]]

        # old_shape = q.shape[2:-1]
        # q = q.flatten(start_dim=2, end_dim=-2)
        # k = k.flatten(start_dim=2, end_dim=-2)
        # v = v.flatten(start_dim=2, end_dim=-2)

        out = scaled_dot_product_attention(q, k, v, mask=mask,
                                           attn_dropout=self.attn_dropout,
                                           training=training)

        return out # view_range(out, 2, 3, old_shape)

class AxialAttention(nn.Module):
    def __init__(self, n_dim, axial_dim):
        super().__init__()
        if axial_dim < 0:
            axial_dim = 2 + n_dim + 1 + axial_dim
        else:
            axial_dim += 2 # account for batch, head, dim
        self.axial_dim = axial_dim

    def forward(self, q, k, v, decode_step, decode_idx):
        q = shift_dim(q, self.axial_dim, -2).flatten(end_dim=-3)
        k = shift_dim(k, self.axial_dim, -2).flatten(end_dim=-3)
        v = shift_dim(v, self.axial_dim, -2)
        old_shape = list(v.shape)
        v = v.flatten(end_dim=-3)

        out = scaled_dot_product_attention(q, k, v, training=self.training)
        out = out.view(*old_shape)
        out = shift_dim(out, -2, self.axial_dim)
        return out


class SparseAttention(nn.Module):
    ops = dict()
    attn_mask = dict()
    block_layout = dict()

    def __init__(self, shape, n_head, causal, num_local_blocks=4, block=32,
                 attn_dropout=0.): # does not use attn_dropout
        super().__init__()
        self.causal = causal
        self.shape = shape

        self.sparsity_config = StridedSparsityConfig(shape=shape, n_head=n_head,
                                                     causal=causal, block=block,
                                                     num_local_blocks=num_local_blocks)

        if self.shape not in SparseAttention.block_layout:
            SparseAttention.block_layout[self.shape] = self.sparsity_config.make_layout()
        if causal and self.shape not in SparseAttention.attn_mask:
            SparseAttention.attn_mask[self.shape] = self.sparsity_config.make_sparse_attn_mask()

    def get_ops(self):
        try:
            from deepspeed.ops.sparse_attention import MatMul, Softmax
        except:
            raise Exception('Error importing deepspeed. Please install using `DS_BUILD_SPARSE_ATTN=1 pip install deepspeed`')
        if self.shape not in SparseAttention.ops:
            sparsity_layout = self.sparsity_config.make_layout()
            sparse_dot_sdd_nt = MatMul(sparsity_layout,
                                       self.sparsity_config.block,
                                       'sdd',
                                       trans_a=False,
                                       trans_b=True)

            sparse_dot_dsd_nn = MatMul(sparsity_layout,
                                       self.sparsity_config.block,
                                       'dsd',
                                       trans_a=False,
                                       trans_b=False)

            sparse_softmax = Softmax(sparsity_layout, self.sparsity_config.block)

            SparseAttention.ops[self.shape] = (sparse_dot_sdd_nt,
                                               sparse_dot_dsd_nn,
                                               sparse_softmax)
        return SparseAttention.ops[self.shape]

    def forward(self, q, k, v, decode_step=None, decode_idx=None, training=True):
        if self.training and self.shape not in SparseAttention.ops:
            self.get_ops()

        SparseAttention.block_layout[self.shape] = SparseAttention.block_layout[self.shape].to(q)
        if self.causal:
            SparseAttention.attn_mask[self.shape] = SparseAttention.attn_mask[self.shape].to(q).type_as(q)
        attn_mask = SparseAttention.attn_mask[self.shape] if self.causal else None

        # old_shape = q.shape[2:-1]
        # q = q.flatten(start_dim=2, end_dim=-2)
        # k = k.flatten(start_dim=2, end_dim=-2)
        # v = v.flatten(start_dim=2, end_dim=-2)

        if decode_step is not None:
            mask = self.sparsity_config.get_non_block_layout_row(SparseAttention.block_layout[self.shape], decode_step)
            out = scaled_dot_product_attention(q, k, v, mask=mask, training=self.training)
        else:
            if q.shape != k.shape or k.shape != v.shape:
                raise Exception('SparseAttention only support self-attention')
            sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax = self.get_ops()
            scaling = float(q.shape[-1]) ** -0.5

            attn_output_weights = sparse_dot_sdd_nt(q, k)
            if attn_mask is not None:
                attn_output_weights = attn_output_weights.masked_fill(attn_mask == 0,
                                                                      float('-inf'))
            attn_output_weights = sparse_softmax(
                attn_output_weights,
                scale=scaling
            )

            out = sparse_dot_dsd_nn(attn_output_weights, v)

        return out # view_range(out, 2, 3, old_shape)


class StridedSparsityConfig(object):
    """
    Strided Sparse configuration specified in https://arxiv.org/abs/1904.10509 that
    generalizes to arbitrary dimensions
    """
    def __init__(self, shape, n_head, causal, block, num_local_blocks):
        self.n_head = n_head
        self.shape = shape
        self.causal = causal
        self.block = block
        self.num_local_blocks = num_local_blocks

        assert self.num_local_blocks >= 1, 'Must have at least 1 local block'
        assert self.seq_len % self.block == 0, 'seq len must be divisible by block size'

        self._block_shape = self._compute_block_shape()
        self._block_shape_cum = self._block_shape_cum_sizes()

    @property
    def seq_len(self):
        return np.prod(self.shape)

    @property
    def num_blocks(self):
        return self.seq_len // self.block

    def set_local_layout(self, layout):
        num_blocks = self.num_blocks
        for row in range(0, num_blocks):
            end = min(row + self.num_local_blocks, num_blocks)
            for col in range(
                    max(0, row - self.num_local_blocks),
                    (row + 1 if self.causal else end)):
                layout[:, row, col] = 1
        return layout

    def set_global_layout(self, layout):
        num_blocks = self.num_blocks
        n_dim = len(self._block_shape)
        for row in range(num_blocks):
            assert self._to_flattened_idx(self._to_unflattened_idx(row)) == row
            cur_idx = self._to_unflattened_idx(row)
            # no strided attention over last dim
            for d in range(n_dim - 1):
                end = self._block_shape[d]
                for i in range(0, (cur_idx[d] + 1 if self.causal else end)):
                    new_idx = list(cur_idx)
                    new_idx[d] = i
                    new_idx = tuple(new_idx)

                    col = self._to_flattened_idx(new_idx)
                    layout[:, row, col] = 1

        return layout

    def make_layout(self):
        layout = torch.zeros((self.n_head, self.num_blocks, self.num_blocks), dtype=torch.int64)
        layout = self.set_local_layout(layout)
        layout = self.set_global_layout(layout)
        return layout

    def make_sparse_attn_mask(self):
        block_layout = self.make_layout()
        assert block_layout.shape[1] == block_layout.shape[2] == self.num_blocks

        num_dense_blocks = block_layout.sum().item()
        attn_mask = torch.ones(num_dense_blocks, self.block, self.block)
        counter = 0
        for h in range(self.n_head):
            for i in range(self.num_blocks):
                for j in range(self.num_blocks):
                    elem = block_layout[h, i, j].item()
                    if elem == 1:
                        assert i >= j
                        if i == j: # need to mask within block on diagonals
                            attn_mask[counter] = torch.tril(attn_mask[counter])
                        counter += 1
        assert counter == num_dense_blocks

        return attn_mask.unsqueeze(0)

    def get_non_block_layout_row(self, block_layout, row):
        block_row = row // self.block
        block_row = block_layout[:, [block_row]] # n_head x 1 x n_blocks
        block_row = block_row.repeat_interleave(self.block, dim=-1)
        block_row[:, :, row + 1:] = 0.
        return block_row

    ############# Helper functions ##########################

    def _compute_block_shape(self):
        n_dim = len(self.shape)
        cum_prod = 1
        for i in range(n_dim - 1, -1, -1):
            cum_prod *= self.shape[i]
            if cum_prod > self.block:
                break
        assert cum_prod % self.block == 0
        new_shape = (*self.shape[:i], cum_prod // self.block)

        assert np.prod(new_shape) == np.prod(self.shape) // self.block

        return new_shape

    def _block_shape_cum_sizes(self):
        bs = np.flip(np.array(self._block_shape))
        return tuple(np.flip(np.cumprod(bs)[:-1])) + (1,)

    def _to_flattened_idx(self, idx):
        assert len(idx) == len(self._block_shape), f"{len(idx)} != {len(self._block_shape)}"
        flat_idx = 0
        for i in range(len(self._block_shape)):
            flat_idx += idx[i] * self._block_shape_cum[i]
        return flat_idx

    def _to_unflattened_idx(self, flat_idx):
        assert flat_idx < np.prod(self._block_shape)
        idx = []
        for i in range(len(self._block_shape)):
            idx.append(flat_idx // self._block_shape_cum[i])
            flat_idx %= self._block_shape_cum[i]
        return tuple(idx)


################ Spatiotemporal broadcasted positional embeddings ###############
class BroadcastPositionBiases(nn.Module):
    def __init__(self, shape: Tuple[int], embed_dim):
        super(BroadcastPositionBiases, self).__init__()
        self.shape = shape
        # self.device = device
        n_dim = len(shape)
        self.n_dim = n_dim
        self.embs = nn.ParameterList()

        # The embedding dimension will be defined later in the forward pass
        self.embed_dim = embed_dim

        # for i in range(n_dim):
        #     # Placeholders for actual sizes which will be set in the forward method
        #     self.embs.append(nn.Parameter(torch.randn(1) * 0.02)).to(self.device)
        
        # self.embed_dim = x.shape[-1]
        chunk_sizes = [self.embed_dim // self.n_dim + (i < (self.embed_dim % self.n_dim))
                        for i in range(self.n_dim)]
        assert sum(chunk_sizes) == self.embed_dim, f'sum({chunk_sizes}) = {sum(chunk_sizes)} != {self.embed_dim}'

        for i in range(self.n_dim):
            self.embs.append(nn.Parameter(torch.randn(self.shape[i], chunk_sizes[i]) * 0.02))

    def forward(self, x, decode_step=None, decode_idx=None):
        out = []
        for i, e in enumerate(self.embs):
            # e = e.view((1,) + (1,) * i + (self.shape[i],) + (1,) * (self.n_dim - i - 1) + (-1,))
            e = e.view(1, *((1,) * i), self.shape[i], *((1,) * (self.n_dim - i - 1)), -1)
            e = e.expand((1, *self.shape, e.shape[-1]))
            out.append(e)

        out = torch.cat(out, dim=-1)
        out = out.view((-1, self.embed_dim))
        # print(out.shape) # [1024, 256]
        if decode_step is not None and decode_step:
            out = out[decode_step]
        # else:
        #     out = out[:x.shape[1]]
        # if decode_step is not None:
        #     out = tensor_slice(out, [0, *decode_idx, 0], [x.shape[0], *(1,) * -1, x.shape[-1]])
        # out = out.view((-1, self.embed_dim))
        # if decode_step is not None and x.shape[1] == 1:
        #     out = out[decode_step]
        # else:
        #     out = out[:x.shape[1]]

        return out


################# Helper Functions ###################################
def scaled_dot_product_attention(q, k, v, mask=None, attn_dropout=0., training=True):
    # Performs scaled dot-product attention over the second to last dimension dn

    # (b, n_head, d1, ..., dn, d)
    attn = torch.matmul(q, k.transpose(-1, -2)) # [batch_size, num_heads, 1, 1024]
    #attn = torch.einsum('bhqd,bhdk->bhqk', q, k)

    attn = attn / np.sqrt(q.shape[-1])
    if mask is not None:
        attn = attn.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(attn, dim=-1)
    attn = attn.type_as(attn) # b x n_head x d1 x ... x dn x d
    attn = F.dropout(attn, p=attn_dropout, training=training)
    # print(attn.shape)
    # print(v.shape)
    a = torch.einsum('bhqk,bhkd->bhqd', attn, v) # b x n_head x d1 x ... x dn x d
    # torch.einsum('bhqk,bhkd->bhqd', attn, v) # b x n_head x d1 x ... x dn x d

    return a

class GELU2(nn.Module):
    def __init__(self):
        super(GELU2, self).__init__()
    
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x

class RightShift(nn.Module):
    def __init__(self, channel_size):
        super(RightShift, self).__init__()
        self.embed_dim = channel_size
        self.sos = nn.Parameter(torch.FloatTensor(channel_size).normal_(std=0.02), requires_grad=True)

    def forward(self, x, decode_step=None):
        if decode_step is not None and decode_step > 0:
            return x
        # # Creating a tensor of shape [batch_size, 1, channel_size] for sos
        # sos = self.sos.unsqueeze(0).unsqueeze(1).expand(x.size(0), 1, -1)
        # # Concatenating sos at the beginning of each sequence in the batch
        # x_shifted = torch.cat([sos, x[:, :-1]], dim=1)
        
        # x_shape = list(x.shape)
        # x = x.flatten(start_dim=1, end_dim=-2) # (batch, seq_len, embed_dim)
        sos = torch.ones(x.shape[0], 1, self.embed_dim, dtype=torch.float32).to(self.sos) * self.sos
        sos = sos.type_as(x)
        x = torch.cat([sos, x[:, :-1, :]], axis=1)
        # x = x.view(*x_shape)

        return x

class LayerNorm(nn.Module):
    def __init__(self, embed_dim, n_classes, epsilon=1e-6):
        super(LayerNorm, self).__init__()
        self.embed_dim = embed_dim
        self.n_classes = n_classes
        self.epsilon = epsilon
        self.conditional = n_classes is not None
        
        if self.conditional:
            self.w = nn.Linear(n_classes, embed_dim, bias=False)
            nn.init.constant_(self.w.weight.data, 1. / np.sqrt(n_classes))
            self.wb = nn.Linear(n_classes, embed_dim, bias=False)
        else:
            self.g = nn.Parameter(torch.ones(embed_dim, dtype=torch.float32), requires_grad=True)
            self.b = nn.Parameter(torch.zeros(embed_dim, dtype=torch.float32), requires_grad=True)

    def forward(self, x, cond):
        if self.conditional:  # (b, cond_dim)
            g = 1 + self.w(cond).view(x.shape[0], *(1,)*(len(x.shape)-2), x.shape[-1]) # (b, ..., embed_dim)
            b = self.wb(cond).view(x.shape[0], *(1,)*(len(x.shape)-2), x.shape[-1])
        else:
            g = self.g  # (embed_dim,)
            b = self.b

        x_float = x.float()

        mu = x.mean(dim=-1, keepdims=True)
        s = (x - mu).square().mean(dim=-1, keepdims=True)
        x_float = (x - mu) * (1e-5 + s.rsqrt())  # (b, ..., embed_dim)
        x = x * g + b

        x = x.type_as(x)
        return x


