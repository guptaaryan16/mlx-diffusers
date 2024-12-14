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

from typing import Tuple
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import math

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .modeling_mlx_utils import MLXModelMixin
from .resnet_mlx import MLXResnetBlock2D, MLXUpsample2D, MLXDownsample2D

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


class MLXAttention(nn.Module):
    """A single head unmasked attention for use with the VAE."""

    def __init__(self, dims: int, norm_groups: int = 32):
        super().__init__()

        self.group_norm = nn.GroupNorm(norm_groups, dims, pytorch_compatible=True)
        self.query = nn.Linear(dims, dims)
        self.key = nn.Linear(dims, dims)
        self.value = nn.Linear(dims, dims)
        self.proj_attn = nn.Linear(dims, dims)

    def __call__(self, x):
        B, H, W, C = x.shape

        y = self.group_norm(x)

        queries = self.query(y).reshape(B, H * W, C)
        keys = self.key(y).reshape(B, H * W, C)
        values = self.value(y).reshape(B, H * W, C)

        scale = 1 / math.sqrt(queries.shape[-1])
        scores = (queries * scale) @ keys.transpose(0, 2, 1)
        attn = mx.softmax(scores, axis=-1)
        y = (attn @ values).reshape(B, H, W, C)

        y = self.proj_attn(y)
        x = x + y

        return x


class MLXDownEncoderBlock2D(nn.Module):
    r"""
    MLX Resnet blocks-based Encoder block for diffusion-based VAE.

    Parameters:
        in_channels (:obj:`int`):
            Input channels
        out_channels (:obj:`int`):
            Output channels
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Number of Resnet layer block
        resnet_groups (:obj:`int`, *optional*, defaults to `32`):
            The number of groups to use for the Resnet block group norm
        add_downsample (:obj:`bool`, *optional*, defaults to `True`):
            Whether to add downsample layer
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        resnet_groups: int = 32,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True
    ):
        super().__init__()
        self.add_downsample = add_downsample
        resnets = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels

            res_block = MLXResnetBlock2D(
                in_channels=in_channels,
                out_channels=out_channels,
                groups=resnet_groups,
                output_scale_factor=output_scale_factor
            )
            resnets.append(res_block)
        self.resnets = resnets

        if self.add_downsample:
            self.downsamplers = [MLXDownsample2D(out_channels, stride=2, padding=0)]

    def __call__(self, hidden_states):
        
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
            
        
        if self.add_downsample:
            hidden_states = mx.pad(hidden_states, [(0, 0), (0, 1), (0, 1), (0, 0)])
            hidden_states = self.downsamplers[0](hidden_states)

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
    """
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        resnet_groups: int = 32,
        add_upsample: bool = True,
    ):
        super().__init__()
        self.add_upsample = add_upsample
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            res_block = MLXResnetBlock2D(
                in_channels=in_channels,
                out_channels=out_channels,
                groups=resnet_groups,
            )
            resnets.append(res_block)

        self.resnets = resnets

        if self.add_upsample:
            self.upsamplers = [MLXUpsample2D(out_channels)]

    def __call__(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        if self.add_upsample:
            hidden_states = self.upsamplers[0](hidden_states)
        return hidden_states


class MLXUNetMidBlock2D(nn.Module):
    r"""
    MLX Unet Mid-Block module.

    Parameters:
        in_channels (:obj:`int`):
            Input channels
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Number of Resnet layer block
        resnet_groups (:obj:`int`, *optional*, defaults to `32`):
            The number of groups to use for the Resnet and Attention block group norm
        num_attention_heads (:obj:`int`, *optional*, defaults to `1`):
            Number of attention heads for each attention block
    """
    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        resnet_groups: int = 32,
    ):
        super().__init__()
        resnet_groups = (
            resnet_groups
            if resnet_groups is not None
            else min(in_channels // 4, 32)
        )

        # there is always at least one resnet
        resnets = [
            MLXResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                groups=resnet_groups,
            )
        ]

        attentions = []

        for _ in range(num_layers):
            attn_block = MLXAttention(
                dims=in_channels,
                norm_groups=resnet_groups,
            )
            attentions.append(attn_block)

            res_block = MLXResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                groups=resnet_groups
            )
            resnets.append(res_block)

        self.resnets = resnets
        self.attentions = attentions

    def __call__(self, hidden_states):
        hidden_states = self.resnets[0](hidden_states)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states)
            hidden_states = resnet(hidden_states)
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
        double_z (:obj:`bool`, *optional*, defaults to `False`):
            Whether to double the last output channels
    """
    def __init__(
        self, 
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        double_z: bool = False
    ):
        super().__init__()
        
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1
        )

        # downsampling
        down_blocks = []
        output_channel = block_out_channels[0]
        for i, _ in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = MLXDownEncoderBlock2D(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=layers_per_block,
                resnet_groups=norm_num_groups,
                add_downsample=not is_final_block
            )
            down_blocks.append(down_block)
        self.down_blocks = down_blocks

        # middle
        self.mid_block = MLXUNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_groups=norm_num_groups
        )

        # end
        conv_out_channels = (
            2 * out_channels if double_z else out_channels
        )
        self.conv_norm_out = nn.GroupNorm(num_groups=norm_num_groups, dims=block_out_channels[-1], pytorch_compatible=True)
        self.conv_out = nn.Conv2d(
            block_out_channels[-1],
            conv_out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def __call__(self, sample):
        sample = self.conv_in(sample)
        
        # downsampling
        for block in self.down_blocks:
            sample = block(sample)
        
        # middle
        sample = self.mid_block(sample)
        
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
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: int = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
    ):
        super().__init__()
        # z to block_in
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1
        )

        # middle
        self.mid_block = MLXUNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_groups=norm_num_groups,
        )

        # upsampling
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        up_blocks = []
        for i, _ in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = MLXUpDecoderBlock2D(
                in_channels=prev_output_channel,
                out_channels=output_channel,
                num_layers=layers_per_block + 1,
                resnet_groups=norm_num_groups,
                add_upsample=not is_final_block
            )
            up_blocks.append(up_block)
            prev_output_channel = output_channel

        self.up_blocks = up_blocks

        # end
        self.conv_norm_out = nn.GroupNorm(num_groups=norm_num_groups, dims=block_out_channels[0], pytorch_compatible=True)
        self.conv_out = nn.Conv2d(
            block_out_channels[0], 
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def __call__(self, sample):
        sample = self.conv_in(sample)

        # middle
        sample = self.mid_block(sample)

        # upsampling
        for block in self.up_blocks:
            sample = block(sample)

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


class MLXAutoencoderKL(MLXModelMixin, ConfigMixin):
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
    """
    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        dtype: mx.Dtype = mx.float32
    ):
        super().__init__()
        self.latent_channels = latent_channels

        self.encoder = MLXEncoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )
        self.decoder = MLXDecoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
        )
        self.quant_conv = nn.Conv2d(
            2 * latent_channels,
            2 * latent_channels,
            kernel_size=1,
            stride=1,
        )
        self.post_quant_conv = nn.Conv2d(
            latent_channels,
            latent_channels,
            kernel_size=1,
            stride=1,
        )
        
    def encode(self, sample, deterministic: bool = True, return_dict: bool = True):
        sample = mx.transpose(sample, (0, 2, 3, 1))

        hidden_states = self.encoder(sample)

        moments = self.quant_conv(hidden_states)
        posterior = MLXDiagonalGaussianDistribution(moments, deterministic=deterministic)
        if not return_dict:
            return (posterior,)

        return MLXAutoencoderKLOutput(latent_dist=posterior)

    def decode(self, latents, deterministic: bool = True, return_dict: bool = True):
        if latents.shape[-1] != self.latent_channels:
            latents = mx.transpose(latents, (0, 2, 3, 1))

        hidden_states = self.post_quant_conv(latents)
        hidden_states = self.decoder(hidden_states)

        hidden_states = mx.transpose(hidden_states, (0, 3, 1, 2))

        if not return_dict:
            return (hidden_states,)

        return MLXDecoderOutput(sample=hidden_states)

    def __call__(
        self,
        sample,
        key = 0,
        sample_posterior=False,
        deterministic: bool = True,
        return_dict: bool = True,
    ):
        posterior = self.encode(
            sample, deterministic=deterministic, return_dict=return_dict
        )
        if sample_posterior:
            rng = mx.random.seed(key)
            hidden_states = posterior.latent_dist.sample(rng)
        else:
            hidden_states = posterior.latent_dist.mode()
        sample = self.decode(hidden_states, return_dict=return_dict).sample
        if not return_dict:
            return (sample,)

        return MLXDecoderOutput(sample=sample)
