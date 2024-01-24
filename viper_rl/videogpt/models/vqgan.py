from typing import Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from typing import Any, Optional, Tuple
# import numpy as np
# import flax.linen as nn
# import jax
# import jax.numpy as jnp


class VQGAN(nn.Module):
    def __init__(self, image_size, ch, ch_mult, num_res_blocks, attn_resolutions, z_channels, double_z, dropout, n_embed, embed_dim, patch_size, ddp=False, channels=None):
        super(VQGAN, self).__init__()
        self.channels = channels if channels is not None else 3
        self.encoder = Encoder(image_size=image_size, ch=ch, 
                               ch_mult=ch_mult,
                               num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions,
                               z_channels=z_channels,
                               double_z=double_z,
                               dropout=dropout,
                               downsample=patch_size)

        self.decoder = Decoder(image_size=image_size, ch=ch, ch_mult=ch_mult,
                               out_ch=self.channels, embed_dim=embed_dim, 
                               num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions,
                               dropout=dropout,
                               upsample=patch_size)

        self.quantize = VectorQuantizer(n_e=n_embed, e_dim=embed_dim)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.z_channels = z_channels
        self.ddp = ddp
        ndims = len(self.patch_size)
        
        if ndims == 1:
            conv_layer = nn.Conv1d
        elif ndims == 2:
            conv_layer = nn.Conv2d
        elif ndims == 3:
            conv_layer = nn.Conv3d
        else:
            raise ValueError("ndims must be 1, 2, or 3")

        self.quant_conv = conv_layer(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=1)
        self.post_quant_conv = conv_layer(in_channels=self.z_channels, out_channels=self.z_channels, kernel_size=1)


    @property
    def metrics(self):
        return ['vq_loss', 'ae_loss', 'perplexity']

    def latent_shape(self, image_size):
        return tuple([image_size // p for p in self.patch_size]) # (8, 8)

    def codebook_lookup(self, encodings, permute=True):
        return self.quantize(None, encodings, permute=permute)    

    def reconstruct(self, image):
        vq_out = self.encode(image, deterministic=True)
        recon = self.decode(vq_out['encodings'], deterministic=True)
        return recon

    def encode(self, image, deterministic=True):
        # print(image.shape) # (128, 3, 64, 64)
        h = self.encoder(image) #, deterministic=deterministic)
        # print(h.shape)
        h = self.quant_conv(h)
        # print(h.shape)
        # progress
        vq_out = self.quantize(h)
        return vq_out
    
    def decode(self, encodings, is_embed=False, deterministic=True):
        # print(encodings.shape)
        encodings = encodings if is_embed else self.codebook_lookup(encodings)
        # print(encodings.shape) # [16, 8, 8, 64]
        recon = self.decoder(self.post_quant_conv(encodings)) #, deterministic)
        return recon
 
    def forward(self, image, deterministic=True):
        vq_out = self.encode(image, deterministic=deterministic)
        # print(vq_out['embeddings'].shape) [128, 64, 8, 8]
        recon = self.decode(vq_out['embeddings'], is_embed=True,
                            deterministic=deterministic)
        return {
            'recon': recon,
            'vq_loss': vq_out['vq_loss'],
            'perplexity': vq_out['perplexity']
        }

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta=0.25):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        
        self.embeddings = nn.Parameter(torch.rand(n_e, e_dim) * 2.0 / n_e - 1.0 / n_e)
        # (256, 64)

    def forward(self, z, encoding_indices=None, permute=True):
        if encoding_indices is not None:
            # print(self.embeddings.shape)
            # print(encoding_indices.shape)
            # print(self.embeddings[encoding_indices].shape)
            embeddings = self.embeddings[encoding_indices]
            if permute:
                # (..., H, W, C) -> (..., C, H, W)
                axis_order = tuple(range(embeddings.ndim - 3)) + (embeddings.ndim - 1, embeddings.ndim - 3, embeddings.ndim - 2)
                return embeddings.permute(axis_order).contiguous()
            else:
                return embeddings
        # print(z.shape)
        # torch.Size([128, 64, 8, 8])

        z_flattened = z.permute(0, 2, 3, 1).contiguous()
        z_e_size = z_flattened.shape
        z_flattened = z_flattened.view(-1, z_flattened.shape[-1])
        # print(z_flattened.shape) # [8192, 64]
        # progress
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embeddings.t() ** 2, dim=0, keepdim=True) - \
            2 * torch.einsum('bd,nd->bn', z_flattened, self.embeddings)
        
        # print(d.shape) # [8192, 256]
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embeddings[min_encoding_indices]
        z_q = z_q.view(z_e_size).permute(0, 3, 1, 2).contiguous()
        
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
               torch.mean((z_q - z.detach()) ** 2)
        z_q = z + (z_q - z).detach()

        encodings_one_hot = F.one_hot(min_encoding_indices, num_classes=self.n_e).float()
        avg_probs = torch.mean(encodings_one_hot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        min_encoding_indices = min_encoding_indices.view(*z_e_size[:-1])
        # print(f'Latents of shape {min_encoding_indices.shape}') # [batch_size, 8, 8]

        return {
            'embeddings': z_q,
            'encodings': min_encoding_indices,
            'vq_loss': loss,
            'perplexity': perplexity
        }
        

class Encoder(nn.Module):
    def __init__(self, image_size, ch, ch_mult, num_res_blocks, downsample, attn_resolutions, z_channels, double_z=True, dropout=0., resample_with_conv=True):
        super(Encoder, self).__init__()
        self.image_size = image_size
        self.ch = ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks # dmc 1 resnet block
        self.downsample = downsample
        self.attn_resolutions = attn_resolutions
        self.z_channels = z_channels
        self.double_z = double_z
        self.dropout = dropout
        self.resample_with_conv = resample_with_conv

        # Compute strides for downsampling
        num_resolutions = len(ch_mult)
        all_strides = []
        ds = downsample
        while not all(d == 1 for d in ds):
            strides = tuple(2 if d > 1 else 1 for d in ds)
            ds = tuple(max(1, d // 2) for d in ds)
            all_strides.append(strides)

        # Create layers dynamically
        self.layers = nn.ModuleList()
        cur_channels = self.ch
        self.layers.append(nn.Conv2d(3, cur_channels, kernel_size=3, padding=1))

        cur_res = self.image_size
        for i_level, mult in enumerate(ch_mult):
            block_out = self.ch * mult
            # print("Level {} Multiply {}".format(i_level, mult))
            for _ in range(num_res_blocks):
                self.layers.append(ResnetBlock(cur_channels, block_out, dropout=dropout))
                cur_channels = block_out
                if cur_res in attn_resolutions:
                    self.layers.append(AttnBlock(cur_channels))

            if i_level != num_resolutions - 1:
                self.layers.append(Downsample(all_strides[i_level], resample_with_conv, cur_channels))
                cur_res //= 2

        self.final_layers = nn.Sequential(
            ResnetBlock(cur_channels, cur_channels, dropout=dropout),
            AttnBlock(cur_channels),
            ResnetBlock(cur_channels, cur_channels, dropout=dropout),
            nn.GroupNorm(num_groups=32, num_channels=cur_channels),
            nn.SiLU(),  # SiLU is the Swish activation function
            nn.Conv2d(cur_channels, 2 * z_channels if double_z else z_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # print(self.layers)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            # print("Encoder {}th layer passed".format(i)) # 2 resnet layers passed
        x = self.final_layers(x)
        # print(x.shape)
        return x


class Decoder(nn.Module):
    def __init__(self, image_size, ch, ch_mult, out_ch, embed_dim, num_res_blocks, upsample, attn_resolutions, dropout=0., resamp_with_conv=True):
        super(Decoder, self).__init__()
        self.image_size = image_size
        self.ch = ch
        self.ch_mult = ch_mult
        self.out_ch = out_ch
        self.num_res_blocks = num_res_blocks
        self.upsample = upsample
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        self.resamp_with_conv = resamp_with_conv

        # Compute strides for upsampling
        num_resolutions = len(ch_mult)
        all_strides = []
        while not all(d == 1 for d in upsample):
            strides = tuple(2 if d > 1 else 1 for d in upsample)
            upsample = tuple(max(1, d // 2) for d in upsample)
            all_strides.append(strides)
        assert len(all_strides) + 1 == num_resolutions

        # Create layers dynamically
        self.layers = nn.ModuleList()
        block_in = ch * ch_mult[num_resolutions - 1]
        self.layers.append(nn.Conv2d(embed_dim, block_in, kernel_size=3, padding=1))
        self.layers.append(ResnetBlock(block_in, block_in, dropout=dropout))
        self.layers.append(AttnBlock(block_in))
        self.layers.append(ResnetBlock(block_in, block_in, dropout=dropout))
        
        cur_res = image_size
        cur_channels = block_in
        for i_level in reversed(range(num_resolutions)):
            block_out = self.ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
                self.layers.append(ResnetBlock(cur_channels, block_out, dropout=dropout))
                cur_channels = block_out
                if i_level in attn_resolutions:
                    self.layers.append(AttnBlock(cur_channels))
            if i_level != 0:
                self.layers.append(Upsample(all_strides[i_level - 1], resamp_with_conv, cur_channels))
                cur_res *= 2

        # (cur_channels)
        # print(out_ch)
        self.final_layers = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=cur_channels),
            nn.SiLU(),  # SiLU is the Swish activation function
            nn.Conv2d(cur_channels, out_ch, kernel_size=3, padding=1)
        )
        # ModuleList(
        #     (0): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     (1): ResnetBlock(
        #         (norm1): GroupNorm(32, 256, eps=1e-05, affine=True)
        #         (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (norm2): GroupNorm(32, 256, eps=1e-05, affine=True)
        #         (dropout_layer): Dropout(p=0.0, inplace=False)
        #         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     )
        #     (2): AttnBlock(
        #         (norm): GroupNorm(32, 256, eps=1e-05, affine=True)
        #         (q_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        #         (k_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        #         (v_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        #         (out_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        #     )
        #     (3-5): 3 x ResnetBlock(
        #         (norm1): GroupNorm(32, 256, eps=1e-05, affine=True)
        #         (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (norm2): GroupNorm(32, 256, eps=1e-05, affine=True)
        #         (dropout_layer): Dropout(p=0.0, inplace=False)
        #         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     )
        #     (6): Upsample(
        #         (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     )
        #     (7-8): 2 x ResnetBlock(
        #         (norm1): GroupNorm(32, 256, eps=1e-05, affine=True)
        #         (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (norm2): GroupNorm(32, 256, eps=1e-05, affine=True)
        #         (dropout_layer): Dropout(p=0.0, inplace=False)
        #         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     )
        #     (9): Upsample(
        #         (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     )
        #     (10-11): 2 x ResnetBlock(
        #         (norm1): GroupNorm(32, 256, eps=1e-05, affine=True)
        #         (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (norm2): GroupNorm(32, 256, eps=1e-05, affine=True)
        #         (dropout_layer): Dropout(p=0.0, inplace=False)
        #         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     )
        #     (12): Upsample(
        #         (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     )
        #     (13-14): 2 x ResnetBlock(
        #         (norm1): GroupNorm(32, 256, eps=1e-05, affine=True)
        #         (conv1): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (norm2): GroupNorm(32, 128, eps=1e-05, affine=True)
        #         (dropout_layer): Dropout(p=0.0, inplace=False)
        #         (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (shortcut_conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, z):
        h = z
        # progress
        # print(self.layers)
        # print("Number of decoder layers {}".format(len(self.layers)))
        for i in range(len(self.layers)):
            h = self.layers[i](h)
            #print("{}th layer passed".format(i))
            # print(h.shape)
        h = self.final_layers(h)
        # print(h.shape) # (batch, 3, 512, 1024)
        return h


class Upsample(nn.Module):
    def __init__(self, strides, with_conv, channels):
        super(Upsample, self).__init__()
        self.strides = strides
        self.with_conv = with_conv
        ndims = len(strides)
        print("upsample stride dimension is {}".format(ndims))
        if ndims == 1:
            self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        elif ndims == 2:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        elif ndims == 3:
            self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        else:
            raise ValueError("Unsupported number of dimensions")

    # def forward(self, x):
    #     # Upsample
    #     ndims = len(x.shape[2:])
    #     assert len(self.strides) == ndims

    #     # Calculate output shape for each dimension
    #     output_shape = [x.shape[0:2]] + [d * s for d, s in zip(x.shape[2:], self.strides)]

    #     # Upsample
    #     x = F.interpolate(x, size=output_shape[2:], mode='nearest')
    #     # x = F.interpolate(x, scale_factor=self.strides, mode='nearest')


    #     # Optional convolution
    #     if self.with_conv:
    #         x = self.conv(x)

    #     return x
    def forward(self, x):
        ndims = len(x.shape[2:])
        assert len(self.strides) == ndims

        # Upsample
        scale_factor = self.strides
        x = F.interpolate(x, scale_factor=scale_factor, mode='nearest')

        # Optional convolution
        if self.with_conv:
            x = self.conv(x)

        return x

                
class Downsample(nn.Module):
    def __init__(self, strides, with_conv, channels):
        super(Downsample, self).__init__()
        self.strides = strides
        self.with_conv = with_conv

        if with_conv:
            # The number of dimensions is inferred from the length of the strides
            ndims = len(strides)
            print("downsample stride dimension is {}".format(ndims))
            if ndims == 1:
                self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=strides, padding=1)
            elif ndims == 2:
                self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=strides, padding=1)
            elif ndims == 3:
                self.conv = nn.Conv3d(channels, channels, kernel_size=3, stride=strides, padding=1)
            else:
                raise ValueError("Unsupported number of dimensions")

    def forward(self, x):
        if self.with_conv:
            x = self.conv(x)
        else:
            # PyTorch doesn't directly support tuple strides in avg_pool, so we handle each dimension case separately
            if len(self.strides) == 1:
                x = F.avg_pool1d(x, kernel_size=self.strides[0], stride=self.strides, padding=0)
            elif len(self.strides) == 2:
                x = F.avg_pool2d(x, kernel_size=self.strides, stride=self.strides, padding=0)
            elif len(self.strides) == 3:
                x = F.avg_pool3d(x, kernel_size=self.strides, stride=self.strides, padding=0)
            else:
                raise ValueError("Unsupported number of dimensions")
        return x


class ResnetBlock(nn.Module):
    def __init__(self, channels, out_channels=None, use_conv_shortcut=False, dropout=0., deterministic=False):
        super(ResnetBlock, self).__init__()
        self.use_conv_shortcut = use_conv_shortcut
        self.dropout = dropout
        self.deterministic = deterministic
        # self.deterministic = deterministic
        self.out_channels = out_channels if out_channels is not None else channels
        # print(channels)
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.conv1 = nn.Conv2d(channels, self.out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=self.out_channels)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)

        if use_conv_shortcut or channels != self.out_channels:
            self.shortcut_conv = nn.Conv2d(channels, self.out_channels, kernel_size=1 if not use_conv_shortcut else 3, padding=0 if not use_conv_shortcut else 1)
        else:
            self.shortcut_conv = None

    def forward(self, x):
        identity = x

        out = self.norm1(x)
        out = F.silu(out)  # Swish activation, in PyTorch it's called SiLU
        out = self.conv1(out)

        out = self.norm2(out)
        out = F.silu(out)
        out = self.dropout_layer(out) if not self.deterministic else out
        out = self.conv2(out)

        if self.shortcut_conv is not None:
            identity = self.shortcut_conv(identity)

        out += identity
        return out

class AttnBlock(nn.Module):
    def __init__(self, channels):
        super(AttnBlock, self).__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.q_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        h = self.norm(x)
        q = self.q_conv(h)
        k = self.k_conv(h)
        v = self.v_conv(h)

        B, C, *z_shape = q.shape
        z_tot = np.prod(z_shape)
        q = q.view(B, C, z_tot).permute(0, 2, 1)
        k = k.view(B, C, z_tot).permute(0, 2, 1)
        w_ = torch.bmm(q, k.transpose(1, 2)) * (C ** (-0.5))
        w_ = F.softmax(w_, dim=-1)

        v = v.view(B, C, z_tot).permute(0, 2, 1)
        h = torch.bmm(w_, v).permute(0, 2, 1).view(B, C, *z_shape)

        # q = q.view(B, C, -1).permute(0, 2, 1)  # B, z_tot, C
        # k = k.view(B, C, -1)  # B, C, z_tot
        # w = torch.matmul(q, k) * (C ** -0.5)
        # w = F.softmax(w, dim=-1)

        # v = v.view(B, C, -1)  # B, C, z_tot
        # h = torch.matmul(w, v.permute(0, 2, 1)).view(B, C, *z_shape)  # B, C, z_shape

        h = self.out_conv(h)

        return x + h