from typing import Tuple, Union, Callable, List
import torch
import torch.nn as nn
import torch.nn.functional as F
# import functools as ft
from math import log2, sqrt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _apply_filter_2d(
    x: torch.Tensor,
    filter_kernel: torch.Tensor,
    padding: Tuple[int, int] = (0, 0)
) -> torch.Tensor:
    """
    Apply a 2D filter to a 4D input tensor.

    Args:
    - x (torch.Tensor): A 4D input tensor of shape (N, H, W, C).
    - filter_kernel (torch.Tensor): A 2D filter kernel of shape (H_k, W_k).
    - padding (Tuple[int, int]): Padding for height and width.

    Returns:
    - torch.Tensor: The filtered output tensor.
    """
    # Ensure the kernel has the right shape for PyTorch's conv2d (out_channels, in_channels, H, W)
    filter_kernel = filter_kernel[None, None, :, :].repeat(x.shape[1], 1, 1, 1)
    
    # PyTorch expects the tensor in the shape (N, C, H, W)
    # x = x.permute(0, 3, 1, 2)

    # Apply 2D convolution
    # print(x.is_cuda)
    y = F.conv2d(x, filter_kernel, padding=padding, groups=x.shape[1])

    # Revert tensor shape to (N, H, W, C)
    # y = y.permute(0, 2, 3, 1)
    return y


class ConvDownsample2D(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_shape, resample_kernel, downsample_factor=1, gain=1.0, dtype=torch.float32):
        super(ConvDownsample2D, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_shape = kernel_shape
        self.downsample_factor = downsample_factor
        self.gain = gain
        self.dtype = dtype

        # Initialize the convolution layer
        self.conv = nn.Conv2d(
            in_channels=self.input_channels,  # to be set during the first forward pass
            out_channels=output_channels,
            kernel_size=kernel_shape,
            stride=downsample_factor,
            padding=0,
            dtype=dtype
        )

        # Prepare the resample kernel
        if len(resample_kernel.shape) == 1:
            resample_kernel = resample_kernel[:, None] * resample_kernel[None, :]
        elif len(resample_kernel.shape) > 2:
            raise ValueError(f"Resample kernel has invalid shape {resample_kernel.shape}")

        self.resample_kernel = torch.tensor(resample_kernel * gain / resample_kernel.sum(), dtype=torch.float32)

    def forward(self, x):
        # Set in_channels for the first forward pass
        if self.conv.in_channels is None:
            self.conv.in_channels = x.shape[1]

        kh, kw = self.resample_kernel.shape
        ch, cw = self.kernel_shape, self.kernel_shape
        assert kh == kw
        assert ch == cw

        # Calculate padding
        pad_0 = (kw - self.downsample_factor + cw) // 2
        pad_1 = (kw - self.downsample_factor + cw - 1) // 2

        # print(x.shape)
        # Apply custom filter
        y = _apply_filter_2d(x, self.resample_kernel, padding=(pad_0, pad_1))
        # print(y.shape)

        # Apply convolution
        return self.conv(y)

def minibatch_stddev_layer(x, group_size=None, num_new_features=1):
    N, C, H, W = x.shape  # Assuming NCHW format (common in PyTorch)
    # N, H, W, C = x.shape

    if group_size is None or group_size > N:
        group_size = N

    C_ = C // num_new_features

    # Reshape and calculate std dev
    y = x.view(-1, group_size, C_, num_new_features, H, W)
    # print(y.shape) # [1, 2, 512, 1, 4, 4]
    y_centered = y - torch.mean(y, dim=1, keepdim=True)
    y_std = torch.sqrt(torch.mean(y_centered ** 2, dim=1) + 1e-8)

    y_std = torch.mean(y_std, dim=(1, 3, 4)).view(-1, 1, 1, num_new_features)
    y_std = y_std.repeat(N, 1, H, W)

    return torch.cat((x, y_std), dim=1)


class DiscriminatorBlock(nn.Module):
    def __init__(self, channels, in_features, out_features, activation_function=nn.LeakyReLU(), resample_kernel=torch.Tensor([1, 3, 3, 1])):
        super(DiscriminatorBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_function = activation_function
        self.resample_kernel = resample_kernel.to(device)

        # Assuming ConvDownsample2D is already defined as per earlier discussions
        self.conv_in = nn.Conv2d(channels, in_features, kernel_size=3, padding=1)
        self.downsample1 = ConvDownsample2D(in_features,
            out_features, kernel_shape=3, resample_kernel=self.resample_kernel, downsample_factor=2
        )
        self.downsample2 = ConvDownsample2D(channels,
            out_features, kernel_shape=1, resample_kernel=self.resample_kernel, downsample_factor=2
        )

    def forward(self, x):
        y = self.conv_in(x)
        y = self.activation_function(y)
        y = self.downsample1(y)
        y = self.activation_function(y)

        residual = self.downsample2(x)
        return (y + residual) / sqrt(2)
    

def _get_num_features(base_features: int, image_size: Tuple[int, int], max_hidden_feature_size: int) -> List[int]:
    """
    Gets number of features for the blocks. Each block includes a downsampling
    step by a factor of two and at the end, we want the resolution to be
    down to 4x4 (for square images)

    >>> features = _get_num_features(64, (512, 512), 1024)
    >>> 512 // 2**(len(features) - 1)
    4
    >>> features[3]
    512
    """
    for size in image_size:
        assert 2 ** int(log2(size)) == size, f"Image size must be a power of 2, got {image_size}"
    # determine the number of layers based on smaller side length
    shortest_side = min(*image_size)
    num_blocks = int(log2(shortest_side)) - 1
    num_features = (base_features * (2**i) for i in range(num_blocks))
    # we want to bring it down to 4x4 at the end of the last block
    return [min(n, max_hidden_feature_size) for n in num_features]


class StyleGANDisc(nn.Module):
    def __init__(self, image_size, base_features, max_hidden_feature_size, mbstd_group_size, mbstd_num_features, gradient_penalty_weight, dtype=torch.float32):
        super(StyleGANDisc, self).__init__()
        self.input_channels = 3
        self.base_features = base_features
        self.max_hidden_feature_size = max_hidden_feature_size
        self.mbstd_group_size = mbstd_group_size
        self.mbstd_num_features = mbstd_num_features
        self.gradient_penalty_weight = gradient_penalty_weight
        self.dtype = dtype

        self.image_size = image_size

        size_t = (self.image_size, self.image_size)  # Assuming NCHW format
        num_features = _get_num_features(2 * self.base_features, size_t, self.max_hidden_feature_size)

        self.conv1 = nn.Conv2d(self.input_channels, self.base_features, kernel_size=1, padding='same')
        self.leaky_relu1 = F.leaky_relu

        self.discblocks = nn.ModuleList([])

        input_channels = self.base_features
        for n_in, n_out in zip(num_features[1:], num_features[:-1]):
            self.discblocks.append(DiscriminatorBlock(input_channels, n_in, n_out, activation_function=F.leaky_relu))
            input_channels = n_out

        self.minibatch_stddev_layer = minibatch_stddev_layer

        self.conv2 = nn.Conv2d(num_features[-2]+self.mbstd_num_features, num_features[-2], kernel_size=3, padding="valid")

        self.leaky_relu2 = F.leaky_relu
        
        self.linear1 = nn.Linear(num_features[-2]*(2**2), num_features[-1])

        self.leaky_relu3 = F.leaky_relu

        self.linear2 = nn.Linear(num_features[-1], 1)

        # Initialize layers here
    @property
    def metrics(self):
        return ['g_loss', 'd_loss', 'd_grad_penalty']
    
    def compute_disc_logits(self, image):
        return self.forward(image)

    def loss_G(self, fake):
        logits_fake = self.compute_disc_logits(fake)
        return {'g_loss': torch.mean(F.softplus(-logits_fake))}
    
    def loss_D(self, real, fake):
        logits_real = self.compute_disc_logits(real)
        logits_fake = self.compute_disc_logits(fake)
        d_loss = torch.mean(F.softplus(logits_fake) + F.softplus(-logits_real))

        # Gradient penalty
        real.requires_grad_()
        logits_real = self.compute_disc_logits(real)
        grads = torch.autograd.grad(outputs=logits_real, inputs=real,
                                    grad_outputs=torch.ones_like(logits_real),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
        r1_penalty = grads.square().sum([1, 2, 3]).mean()
        r1_penalty *= self.gradient_penalty_weight

        loss_D = d_loss + r1_penalty
        return {
            'loss_D': loss_D,
            'd_loss': d_loss,
            'd_grad_penalty': r1_penalty,
        }

    def forward(self, x, deterministic=True):
        # Assuming x in NCHW format

        x = self.conv1(x)
        x = self.leaky_relu1(x)
        for i in range(len(self.discblocks)):
            x = self.discblocks[i](x)
            # print("{}th disc block passed".format(i))

        # Final block running on 4x4 feature maps
        assert min(x.shape[1:3]) == 4
        # print(x.shape)
        x = self.minibatch_stddev_layer(x, group_size=self.mbstd_group_size, num_new_features=self.mbstd_num_features)

        x = self.conv2(x)
        x = self.leaky_relu2(x)
        # print(x.shape)
        x = x.view(x.size(0), -1) # [batch, 512, 2, 2]
        x = self.linear1(x)
        x = self.leaky_relu3(x)
        x = self.linear2(x)

        return x

