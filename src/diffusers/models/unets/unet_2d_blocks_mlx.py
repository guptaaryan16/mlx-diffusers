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

import mlx.core as mx
import mlx.nn as nn

from ..attention_mlx import MLXTransformer2DModel
from ..resnet_mlx import MLXDownsample2D, MLXResnetBlock2D, MLXUpsample2D

class MLXDownBlock2D(nn.Module):
    r"""
    MLX 2D downsizing block

    Parameters:
        in_channels (:obj:`int`):
            Input channels
        out_channels (:obj:`int`):
            Output channels
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Number of attention blocks layers
        add_downsample (:obj:`bool`, *optional*, defaults to `True`):
            Whether to add downsampling layer before each final output
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        add_downsample: bool = True,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels

        resnets = []

        for i in range(num_layers):
            in_channels = self.in_channels if i == 0 else self.out_channels

            res_block = MLXResnetBlock2D(
                in_channels=in_channels,
                out_channels=self.out_channels,
            )
            resnets.append(res_block)
        self.resnets = resnets

        if self.add_downsample:
            self.downsamplers_0 = MLXDownsample2D(self.out_channels)

    def __call__(self, hidden_states, temb, deterministic=True):
        output_states = ()

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states += (hidden_states,)

        if self.add_downsample:
            hidden_states = self.downsamplers_0(hidden_states)
            output_states += (hidden_states,)

        return hidden_states, output_states

class MLXUpBlock2D(nn.Module):
    r"""
    MLX 2D upsampling block

    Parameters:
        in_channels (:obj:`int`):
            Input channels
        out_channels (:obj:`int`):
            Output channels
        prev_output_channel (:obj:`int`):
            Output channels from the previous block
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Number of attention blocks layers
        add_downsample (:obj:`bool`, *optional*, defaults to `True`):
            Whether to add downsampling layer before each final output
    """

    in_channels: int
    out_channels: int
    prev_output_channel: int
    dropout: float = 0.0
    num_layers: int = 1
    add_upsample: bool = True

    def setup(self):
        resnets = []

        for i in range(self.num_layers):
            res_skip_channels = (
                self.in_channels if (i == self.num_layers - 1) else self.out_channels
            )
            resnet_in_channels = (
                self.prev_output_channel if i == 0 else self.out_channels
            )

            res_block = FlaxResnetBlock2D(
                in_channels=resnet_in_channels + res_skip_channels,
                out_channels=self.out_channels,
                dropout_prob=self.dropout
            )
            resnets.append(res_block)

        self.resnets = resnets

        if self.add_upsample:
            self.upsamplers_0 = MLXUpsample2D(self.out_channels)

    def __call__(
        self, hidden_states, res_hidden_states_tuple, temb, deterministic=True
    ):
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = jnp.concatenate((hidden_states, res_hidden_states), axis=-1)

            hidden_states = resnet(hidden_states, temb, deterministic=deterministic)

        if self.add_upsample:
            hidden_states = self.upsamplers_0(hidden_states)

        return hidden_states


class MLXUNetMidBlock2DCrossAttn(nn.Module):
    r"""
    Cross Attention 2D Mid-level block - original architecture from Unet transformers: https://arxiv.org/abs/2103.06104

    Parameters:
        in_channels (:obj:`int`):
            Input channels
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Number of attention blocks layers
        num_attention_heads (:obj:`int`, *optional*, defaults to 1):
            Number of attention heads of each spatial transformer block
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            enable memory efficient attention https://arxiv.org/abs/2112.05682
        split_head_dim (`bool`, *optional*, defaults to `False`):
            Whether to split the head dimension into a new axis for the self-attention computation. In most cases,
            enabling this flag should speed up the computation for Stable Diffusion 2.x and Stable Diffusion XL.
    """
    def __init__(
        self,
        in_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        num_attention_heads: int = 1,
        use_linear_projection: bool = False,
        use_memory_efficient_attention: bool = False,
        split_head_dim: bool = False,
        transformer_layers_per_block: int = 1,
    ):

        # there is always at least one resnet
        resnets = [
            MLXResnetBlock2D(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                dropout_prob=self.dropout,
                dtype=self.dtype,
            )
        ]

        attentions = []

        for _ in range(self.num_layers):
            attn_block = MLXTransformer2DModel(
                in_channels=self.in_channels,
                n_heads=self.num_attention_heads,
                d_head=self.in_channels // self.num_attention_heads,
                depth=self.transformer_layers_per_block,
                use_linear_projection=self.use_linear_projection,
                use_memory_efficient_attention=self.use_memory_efficient_attention,
                split_head_dim=self.split_head_dim,
                dtype=self.dtype,
            )
            attentions.append(attn_block)

            res_block = MLXResnetBlock2D(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                dropout_prob=self.dropout,
                dtype=self.dtype,
            )
            resnets.append(res_block)

        self.resnets = resnets
        self.attentions = attentions

    def __call__(self, hidden_states, temb, encoder_hidden_states):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(
                hidden_states, encoder_hidden_states, 
            )
            hidden_states = resnet(hidden_states, temb)

        return hidden_states
