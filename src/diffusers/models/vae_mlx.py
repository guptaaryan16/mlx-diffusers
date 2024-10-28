# Copyright 2024 Apple and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MLX implementation of VQGAN from taming-transformers https://github.com/CompVis/taming-transformers using JAX and pytorch API of diffusers
# API inspired from https://github.com/ml-explore/mlx-examples/blob/main/stable_diffusion/stable_diffusion/

import math
from functools import partial
from typing import Tuple, Optional
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..configuration_utils import ConfigMixin, mlx_register_to_config
from ..utils import BaseOutput
from .modeling_mlx_utils import MLXModelMixin


@dataclass
class MLXDecoderOutput(BaseOutput):
    """
    Output of decoding method.

    Args:
        sample (`mx.array` of shape `(batch_size, num_channels, height, width)`):
            The decoded output sample from the last layer of the model.
        dtype (`mx.Dtype`, *optional*, defaults to `mx.float32`):
            The `dtype` of the parameters.
    """

    sample: mx.array


@dataclass
class MLXAutoencoderKLOutput(BaseOutput):
    """
    Output of AutoencoderKL encoding method.

    Args:
        latent_dist (`MLXDiagonalGaussianDistribution`):
            Encoded outputs of `Encoder` represented as the mean and logvar of "MLXDiagonalGaussianDistribution`.
            `MLXDiagonalGaussianDistribution` allows for sampling latents from the distribution.
    """

    latent_dist: "MLXDiagonalGaussianDistribution"

# Taken from https://github.com/ml-explore/mlx-examples/blob/main/stable_diffusion/stable_diffusion/unet.py#L12
def upsample_nearest(x, scale: int = 2):
    B, H, W, C = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (B, H, scale, W, scale, C))
    x = x.reshape(B, H * scale, W * scale, C)
    return x


class MLXUpsample2D(nn.Module):
    """
    MLX implementation of 2D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        name (`str`, default `conv`):
            name of the upsampling 2D layer.
    """
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        dtype: mx.Dtype = mx.float32
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.dtype = dtype

        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype= self.dtype
        )

    def __call__(self, hidden_states):
        batch, height, width, channels = hidden_states.shape
        hidden_states = upsample_nearest(hidden_states, 2)
        hidden_states = self.conv(hidden_states)
        return hidden_states


class MLXDownsample2D(nn.Module):
    """
    MLX implementation of 2D Downsample layer

    Args:
        in_channels (`int`):
            Input channels
        out_channels (`int`) = None:
            Output channels
        dtype (:obj:`mx.Dtype`, *optional*, defaults to mx.float32):
            Parameters `dtype`
    """

    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        dtype: mx.Dtype = mx.float32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.dtype = dtype

        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        pad = ((0, 0), (0, 1), (0, 1), (0, 0))  # pad height and width dim
        hidden_states = mx.pad(hidden_states, pad_width=pad)
        hidden_states = self.conv(hidden_states)
        return hidden_states


class MLXResnetBlock2D(nn.Module):
    """
    MLX implementation of 2D Resnet Block.

    Args:
        Args:
        in_channels (`int`):
            Input channels
        out_channels (`int`):
            Output channels
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        groups (:obj:`int`, *optional*, defaults to `32`):
            The number of groups to use for group norm.
        use_nin_shortcut (:obj:`bool`, *optional*, defaults to `None`):
            Whether to use `nin_shortcut`. This activates a new layer inside ResNet block
        dtype (:obj:`mx.Dtype`, *optional*, defaults to mx.float32):
            Parameters `dtype`
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int = None,
            dropout: float = 0.0,
            groups: int = 32,
            use_nin_shortcut: bool = None,
            dtype: mx.Dtype = mx.float32,
        ):
        super().__init__()

        out_channels = out_channels or in_channels

        self.norm1 = nn.GroupNorm(groups, in_channels, ep=1e-6,pytorch_compatible=True)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype
        )

        self.norm2 = nn.GroupNorm(groups, out_channels, eps=1e-6, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype
        )

        use_nin_shortcut = self.in_channels != out_channels if self.use_nin_shortcut is None else self.use_nin_shortcut

        self.conv_shortcut = None
        if use_nin_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                dtype=self.dtype,
            )

    def __call__(self, hidden_states: mx.array, deterministic=True):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = nn.silu(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = nn.silu(hidden_states)
        hidden_states = self.dropout_layer(hidden_states, deterministic)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        return hidden_states + residual


class MLXDownEncoderBlock2D(nn.Module):
    r"""
    MLX Resnet blocks-based Encoder block for diffusion-based VAE.

    Parameters:
        in_channels (:obj:`int`):
            Input channels
        out_channels (:obj:`int`):
            Output channels
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Number of Resnet layer block
        resnet_groups (:obj:`int`, *optional*, defaults to `32`):
            The number of groups to use for the Resnet block group norm
        add_downsample (:obj:`bool`, *optional*, defaults to `True`):
            Whether to add downsample layer
        dtype (:obj:`mx.Dtype`, *optional*, defaults to mx.float32):
            Parameters `dtype`
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_groups: int = 32,
        add_downsample: bool = True,
        dtype: mx.Dtype = mx.float32,
    ):
        resnets = []
        for i in range(self.num_layers):
            in_channels = self.in_channels if i == 0 else self.out_channels

            res_block = MLXResnetBlock2D(
                in_channels=in_channels,
                out_channels=self.out_channels,
                dropout=self.dropout,
                groups=self.resnet_groups,
                dtype=self.dtype,
            )
            resnets.append(res_block)
        self.resnets = resnets

        if self.add_downsample:
            self.downsamplers_0 = MLXDownsample2D(self.out_channels, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic=True):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, deterministic=deterministic)

        if self.add_downsample:
            hidden_states = self.downsamplers_0(hidden_states)

        return hidden_states


class MLXUpDecoderBlock2D(nn.Module):
    r"""
    MLX Resnet blocks-based Decoder block for diffusion-based VAE.

    Parameters:
        in_channels (:obj:`int`):
            Input channels
        out_channels (:obj:`int`):
            Output channels
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Number of Resnet layer block
        resnet_groups (:obj:`int`, *optional*, defaults to `32`):
            The number of groups to use for the Resnet block group norm
        add_upsample (:obj:`bool`, *optional*, defaults to `True`):
            Whether to add upsample layer
        dtype (:obj:`mx.Dtype`, *optional*, defaults to mx.float32):
            Parameters `dtype`
    """

    

    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_groups: int = 32,
        add_upsample: bool = True,
        dtype: mx.Dtype = mx.float32,
    ):
        resnets = []
        for i in range(self.num_layers):
            in_channels = self.in_channels if i == 0 else self.out_channels
            res_block = MLXResnetBlock2D(
                in_channels=in_channels,
                out_channels=self.out_channels,
                dropout=self.dropout,
                groups=self.resnet_groups,
                dtype=self.dtype,
            )
            resnets.append(res_block)

        self.resnets = resnets

        if self.add_upsample:
            self.upsamplers_0 = MLXUpsample2D(self.out_channels, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic=True):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, deterministic=deterministic)

        if self.add_upsample:
            hidden_states = self.upsamplers_0(hidden_states)

        return hidden_states


class MLXUNetMidBlock2D(nn.Module):
    r"""
    MLX Unet Mid-Block module.

    Parameters:
        in_channels (:obj:`int`):
            Input channels
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Number of Resnet layer block
        resnet_groups (:obj:`int`, *optional*, defaults to `32`):
            The number of groups to use for the Resnet and Attention block group norm
        num_attention_heads (:obj:`int`, *optional*, defaults to `1`):
            Number of attention heads for each attention block
        dtype (:obj:`mx.Dtype`, *optional*, defaults to mx.float32):
            Parameters `dtype`
    """

    

    def __init__(
        self,
        in_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_groups: int = 32,
        num_attention_heads: int = 1,
        dtype: mx.Dtype = mx.float32
    ):
        resnet_groups = (
            self.resnet_groups
            if self.resnet_groups is not None
            else min(self.in_channels // 4, 32)
        )

        # there is always at least one resnet
        resnets = [
            MLXResnetBlock2D(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                dropout=self.dropout,
                groups=resnet_groups,
                dtype=self.dtype,
            )
        ]

        attentions = []

        for _ in range(self.num_layers):
            attn_block = MLXAttentionBlock(
                channels=self.in_channels,
                num_head_channels=self.num_attention_heads,
                num_groups=resnet_groups,
                dtype=self.dtype,
            )
            attentions.append(attn_block)

            res_block = MLXResnetBlock2D(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                dropout=self.dropout,
                groups=resnet_groups,
                dtype=self.dtype,
            )
            resnets.append(res_block)

        self.resnets = resnets
        self.attentions = attentions

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = self.resnets[0](hidden_states, deterministic=deterministic)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states)
            hidden_states = resnet(hidden_states, deterministic=deterministic)

        return hidden_states


class MLXEncoder(nn.Module):
    r"""
    MLX Implementation of VAE Encoder.

    Parameters:
        in_channels (:obj:`int`, *optional*, defaults to 3):
            Input channels
        out_channels (:obj:`int`, *optional*, defaults to 3):
            Output channels
        down_block_types (:obj:`Tuple[str]`, *optional*, defaults to `(DownEncoderBlock2D)`):
            DownEncoder block type
        block_out_channels (:obj:`Tuple[str]`, *optional*, defaults to `(64,)`):
            Tuple containing the number of output channels for each block
        layers_per_block (:obj:`int`, *optional*, defaults to `2`):
            Number of Resnet layer for each block
        norm_num_groups (:obj:`int`, *optional*, defaults to `32`):
            norm num group
        act_fn (:obj:`str`, *optional*, defaults to `silu`):
            Activation function
        double_z (:obj:`bool`, *optional*, defaults to `False`):
            Whether to double the last output channels
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """

    

    def __init__(
        self, 
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = False,
        dtype: mx.Dtype = mx.float32,
    ):
        block_out_channels = self.block_out_channels
        # in
        self.conv_in = nn.Conv2d(
            block_out_channels[0],
            block_out_channels[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        # downsampling
        down_blocks = []
        output_channel = block_out_channels[0]
        for i, _ in enumerate(self.down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = MLXDownEncoderBlock2D(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=self.layers_per_block,
                resnet_groups=self.norm_num_groups,
                add_downsample=not is_final_block,
                dtype=self.dtype,
            )
            down_blocks.append(down_block)
        self.down_blocks = down_blocks

        # middle
        self.mid_block = MLXUNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_groups=self.norm_num_groups,
            num_attention_heads=None,
            dtype=self.dtype,
        )

        # end
        conv_out_channels = (
            2 * self.out_channels if self.double_z else self.out_channels
        )
        self.conv_norm_out = nn.GroupNorm(num_groups=self.norm_num_groups, epsilon=1e-6)
        self.conv_out = nn.Conv2d(
            conv_out_channels,
            conv_out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

    def __call__(self, sample, deterministic: bool = True):
        # in
        sample = self.conv_in(sample)

        # downsampling
        for block in self.down_blocks:
            sample = block(sample, deterministic=deterministic)

        # middle
        sample = self.mid_block(sample, deterministic=deterministic)

        # end
        sample = self.conv_norm_out(sample)
        sample = nn.silu(sample)
        sample = self.conv_out(sample)

        return sample


class MLXDecoder(nn.Module):
    r"""
    MLX Implementation of VAE Decoder.

    Parameters:
        in_channels (:obj:`int`, *optional*, defaults to 3):
            Input channels
        out_channels (:obj:`int`, *optional*, defaults to 3):
            Output channels
        up_block_types (:obj:`Tuple[str]`, *optional*, defaults to `(UpDecoderBlock2D)`):
            UpDecoder block type
        block_out_channels (:obj:`Tuple[str]`, *optional*, defaults to `(64,)`):
            Tuple containing the number of output channels for each block
        layers_per_block (:obj:`int`, *optional*, defaults to `2`):
            Number of Resnet layer for each block
        norm_num_groups (:obj:`int`, *optional*, defaults to `32`):
            norm num group
        act_fn (:obj:`str`, *optional*, defaults to `silu`):
            Activation function
        double_z (:obj:`bool`, *optional*, defaults to `False`):
            Whether to double the last output channels
        dtype (:obj:`mx.dtype`, *optional*, defaults to mx.float32):
            parameters `dtype`
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: int = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        dtype: mx.Dtype = mx.float32
    ):
        block_out_channels = self.block_out_channels

        # z to block_in
        self.conv_in = nn.Conv2d(
            block_out_channels[-1],
            block_out_channels[-1],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        # middle
        self.mid_block = MLXUNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_groups=self.norm_num_groups,
            num_attention_heads=None,
            dtype=self.dtype,
        )

        # upsampling
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        up_blocks = []
        for i, _ in enumerate(self.up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = MLXUpDecoderBlock2D(
                in_channels=prev_output_channel,
                out_channels=output_channel,
                num_layers=self.layers_per_block + 1,
                resnet_groups=self.norm_num_groups,
                add_upsample=not is_final_block,
                dtype=self.dtype,
            )
            up_blocks.append(up_block)
            prev_output_channel = output_channel

        self.up_blocks = up_blocks

        # end
        self.conv_norm_out = nn.GroupNorm(num_groups=self.norm_num_groups, epsilon=1e-6)
        self.conv_out = nn.Conv2d(
            self.out_channels,
            self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

    def __call__(self, sample, deterministic: bool = True):
        # z to block_in
        sample = self.conv_in(sample)

        # middle
        sample = self.mid_block(sample, deterministic=deterministic)

        # upsampling
        for block in self.up_blocks:
            sample = block(sample, deterministic=deterministic)

        sample = self.conv_norm_out(sample)
        sample = nn.silu(sample)
        sample = self.conv_out(sample)

        return sample


class MLXDiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        # Last axis to account for channels-last
        self.mean, self.logvar = mx.split(parameters, 2, axis=-1)
        self.logvar = mx.clip(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = mx.exp(0.5 * self.logvar)
        self.var = mx.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = mx.zeros_like(self.mean)

    def sample(self, key):
        return self.mean + self.std * mx.random.normal(self.mean.shape,key=key)

    def kl(self, other=None):
        if self.deterministic:
            return mx.array([0.0])

        if other is None:
            return 0.5 * mx.sum(
                self.mean**2 + self.var - 1.0 - self.logvar, axis=[1, 2, 3]
            )

        return 0.5 * mx.sum(
            mx.square(self.mean - other.mean) / other.var
            + self.var / other.var
            - 1.0
            - self.logvar
            + other.logvar,
            axis=[1, 2, 3],
        )

    def nll(self, sample, axis=[1, 2, 3]):
        if self.deterministic:
            return mx.array([0.0])

        logtwopi = mx.log(2.0 * mx.pi)
        return 0.5 * mx.sum(
            logtwopi + self.logvar + mx.square(sample - self.mean) / self.var,
            axis=axis,
        )

    def mode(self):
        return self.mean


@mlx_register_to_config
class MLXAutoencoderKL(nn.Module, MLXModelMixin, ConfigMixin):
    r"""
    MLX implementation of a VAE model with KL loss for decoding latent representations.

    This model inherits from [`MLXModelMixin`]. Check the superclass documentation for it's generic methods
    implemented for all models (such as downloading or saving).

    Parameters:
        in_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input image.
        out_channels (`int`, *optional*, defaults to 3):
            Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `(DownEncoderBlock2D)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `(UpDecoderBlock2D)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[str]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        layers_per_block (`int`, *optional*, defaults to `2`):
            Number of ResNet layer for each block.
        act_fn (`str`, *optional*, defaults to `silu`):
            The activation function to use.
        latent_channels (`int`, *optional*, defaults to `4`):
            Number of channels in the latent space.
        norm_num_groups (`int`, *optional*, defaults to `32`):
            The number of groups for normalization.
        sample_size (`int`, *optional*, defaults to 32):
            Sample input size.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
        dtype (`mx.Dtype`, *optional*, defaults to `mx.float32`):
            The `dtype` of the parameters.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        dtype: mx.Dtype = mx.float32
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down_block_types = down_block_types
        self.up_block_types = up_block_types
        self.block_out_channels = block_out_channels
        self.layers_per_block = layers_per_block
        self.act_fn = act_fn
        self.latent_channels = latent_channels
        self.norm_num_groups = norm_num_groups
        self.sample_size = sample_size
        self.scaling_factor = scaling_factor
        self.dtype = dtype

        self.encoder = MLXEncoder(
            in_channels=config.in_channels,
            out_channels=config.latent_channels,
            down_block_types=config.down_block_types,
            block_out_channels=config.block_out_channels,
            layers_per_block=config.layers_per_block,
            act_fn=config.act_fn,
            norm_num_groups=config.norm_num_groups,
            double_z=True,
            dtype=dtype,
        )
        self.decoder = MLXDecoder(
            in_channels=config.latent_channels,
            out_channels=config.out_channels,
            up_block_types=config.up_block_types,
            block_out_channels=config.block_out_channels,
            layers_per_block=config.layers_per_block,
            norm_num_groups=config.norm_num_groups,
            act_fn=config.act_fn,
            dtype=dtype,
        )
        self.quant_conv = nn.Conv(
            2 * config.latent_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            dtype=dtype,
        )
        self.post_quant_conv = nn.Conv(
            config.latent_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            dtype=dtype,
        )

    def init_weights(self, rng: Optional[mx.array, int]):
        # init input tensors
        sample_shape = (1, self.in_channels, self.sample_size, self.sample_size)
        sample = mx.zeros(sample_shape, dtype=mx.float32)

        params_rng, dropout_rng, gaussian_rng = mx.random.split(rng, 3)
        rngs = {"params": params_rng, "dropout": dropout_rng, "gaussian": gaussian_rng}

        return self.init(rngs, sample)["params"]

    def encode(self, sample, deterministic: bool = True, return_dict: bool = True):
        sample = mx.transpose(sample, (0, 2, 3, 1))

        hidden_states = self.encoder(sample, deterministic=deterministic)
        moments = self.quant_conv(hidden_states)
        posterior = MLXDiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return MLXAutoencoderKLOutput(latent_dist=posterior)

    def decode(self, latents, deterministic: bool = True, return_dict: bool = True):
        if latents.shape[-1] != self.config.latent_channels:
            latents = jnp.transpose(latents, (0, 2, 3, 1))

        hidden_states = self.post_quant_conv(latents)
        hidden_states = self.decoder(hidden_states, deterministic=deterministic)

        hidden_states = mx.transpose(hidden_states, (0, 3, 1, 2))

        if not return_dict:
            return (hidden_states,)

        return MLXDecoderOutput(sample=hidden_states)

    def __call__(
        self,
        sample,
        sample_posterior=False,
        deterministic: bool = True,
        return_dict: bool = True,
    ):
        posterior = self.encode(
            sample, deterministic=deterministic, return_dict=return_dict
        )
        if sample_posterior:
            rng = self.make_rng("gaussian")
            hidden_states = posterior.latent_dist.sample(rng)
        else:
            hidden_states = posterior.latent_dist.mode()

        sample = self.decode(hidden_states, return_dict=return_dict).sample

        if not return_dict:
            return (sample,)

        return MLXDecoderOutput(sample=sample)
