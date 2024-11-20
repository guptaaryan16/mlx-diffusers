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


class MLXCrossAttnDownBlock2D(nn.Module):
    r"""
    MLX port of Cross Attention 2D Downsizing block - original architecture from Unet transformers:
    https://arxiv.org/abs/2103.06104
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        num_attention_heads: int = 1,
        add_downsample: bool = True,
        transformer_layers_per_block: int = 1,
        only_cross_attention: bool= False,
        cross_attention_dim: int=None, 
        temb_channels: int = None,
        resnet_groups: int= 32
    ):
        self.add_downsample = add_downsample 
        resnets = []
        attentions = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels

            res_block = MLXResnetBlock2D(
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                groups=resnet_groups
            )
            resnets.append(res_block)

            attn_block = MLXTransformer2DModel(
                in_channels=out_channels,
                model_dims=out_channels,
                num_heads=num_attention_heads,
                num_layers=transformer_layers_per_block,
                encoder_dims=cross_attention_dim
            )
            attentions.append(attn_block)

        self.resnets = resnets
        self.attentions = attentions

        if add_downsample:
            self.downsamplers = [MLXDownsample2D(out_channels)]

    def __call__(self, hidden_states, temb, encoder_hidden_states):
        output_states = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states, encoder_hidden_states
            )
            output_states += (hidden_states,)

        if self.add_downsample:
            hidden_states = self.downsamplers[0](hidden_states)
            output_states += (hidden_states)

        return hidden_states, output_states


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
        num_layers: int = 1,
        add_downsample: bool = True,
        temb_channels: int = None,
        resnet_groups: int = 32
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_downsample = add_downsample
        resnets = []

        for i in range(num_layers):
            in_channels = self.in_channels if i == 0 else self.out_channels

            res_block = MLXResnetBlock2D(
                in_channels=in_channels,
                out_channels=self.out_channels,
                temb_channels=temb_channels,
                groups=resnet_groups
            )
            resnets.append(res_block)
        self.resnets = resnets

        if self.add_downsample:
            self.downsamplers = [MLXDownsample2D(self.out_channels)]

    def __call__(self, hidden_states, temb):
        output_states = ()

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states += (hidden_states,)

        if self.add_downsample:
            hidden_states = self.downsamplers[0](hidden_states)
            output_states += (hidden_states,)

        return hidden_states, output_states


class MLXCrossAttnUpBlock2D(nn.Module):
    r"""
    MLX port of Cross Attention 2D Upsampling block - original architecture from Unet transformers:
    https://arxiv.org/abs/2103.06104

    Parameters:
        in_channels (:obj:`int`):
            Input channels
        out_channels (:obj:`int`):
            Output channels
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Number of attention blocks layers
        num_attention_heads (:obj:`int`, *optional*, defaults to 1):
            Number of attention heads of each spatial transformer block
        add_upsample (:obj:`bool`, *optional*, defaults to `True`):
            Whether to add upsampling layer before each final output
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            prev_output_channel: int,
            num_layers: int = 1,
            num_attention_heads: int = 1,
            add_upsample: bool = True,
            transformer_layers_per_block: int = 1,
            temb_channels: int = None,
            resnet_groups: int = 32,
            cross_attention_dim: int = None
        ):
        self.add_upsample = add_upsample
        resnets = []
        attentions = []

        for i in range(num_layers):
            res_skip_channels = (
                in_channels if (i == num_layers - 1) else out_channels
            )
            resnet_in_channels = (
                prev_output_channel if i == 0 else out_channels
            )

            res_block = MLXResnetBlock2D(
                in_channels=resnet_in_channels + res_skip_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                groups=resnet_groups
            )
            resnets.append(res_block)

            attn_block = MLXTransformer2DModel(
                in_channels=out_channels,
                model_dims=out_channels,
                num_heads=num_attention_heads,
                encoder_dims=cross_attention_dim,
                num_layers=transformer_layers_per_block,
                norm_num_groups=resnet_groups
            )
            attentions.append(attn_block)

        self.resnets = resnets
        self.attentions = attentions

        if add_upsample:
            self.upsamplers = [MLXUpsample2D(out_channels)]

    def __call__(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb,
        encoder_hidden_states,
    ):
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = mx.concatenate((hidden_states, res_hidden_states), axis=-1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states, encoder_hidden_states
            )

        if self.add_upsample:
            hidden_states = self.upsamplers[0](hidden_states)

        return hidden_states


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
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            prev_output_channel: int,
            num_layers: int = 1,
            add_upsample: bool = True,
            temb_channels: int = None,
            resnet_groups: int = 32,
    ):
        self.add_upsample = add_upsample
        resnets = []

        for i in range(num_layers):
            res_skip_channels = (
                in_channels if (i == num_layers - 1) else out_channels
            )
            resnet_in_channels = (
                prev_output_channel if i == 0 else out_channels
            )

            res_block = MLXResnetBlock2D(
                in_channels=resnet_in_channels + res_skip_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                groups=resnet_groups
            )
            resnets.append(res_block)

        self.resnets = resnets

        if self.add_upsample:
            self.upsamplers = [MLXUpsample2D(out_channels)]

    def __call__(
        self, hidden_states, res_hidden_states_tuple, temb
    ):
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = mx.concatenate((hidden_states, res_hidden_states), axis=-1)

            hidden_states = resnet(hidden_states, temb)

        if self.add_upsample:
            hidden_states = self.upsamplers[0](hidden_states)

        return hidden_states


class MLXUNetMidBlock2DCrossAttn(nn.Module):
    r"""
    Cross Attention 2D Mid-level block - original architecture from Unet transformers: https://arxiv.org/abs/2103.06104

    Parameters:
        in_channels (:obj:`int`):
            Input channels
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Number of attention blocks layers
        num_attention_heads (:obj:`int`, *optional*, defaults to 1):
            Number of attention heads of each spatial transformer block
    """
    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        num_attention_heads: int = 1,
        transformer_layers_per_block: int = 1,
        resnet_groups: int = 32,
        temb_channels: int = None,
        cross_attention_dim: int = None,
    ):

        # there is always at least one resnet
        resnets = [
            MLXResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                groups=resnet_groups
            )
        ]
        attentions = []

        for _ in range(num_layers):
            attn_block = MLXTransformer2DModel(
                in_channels=in_channels,
                model_dims=in_channels,
                num_heads=num_attention_heads,
                encoder_dims=cross_attention_dim,
                num_layers=transformer_layers_per_block,
            )
            attentions.append(attn_block)

            res_block = MLXResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                groups=resnet_groups
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
