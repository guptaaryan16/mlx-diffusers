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
# limitations under the License.huggingface/diffusers.git
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import BaseOutput
from ..embeddings_mlx import MLXTimestepEmbedding, MLXTimesteps
from ..modeling_mlx_utils import MLXModelMixin
from .unet_2d_blocks_mlx import (
    MLXCrossAttnDownBlock2D,
    MLXCrossAttnUpBlock2D,
    MLXDownBlock2D,
    MLXUNetMidBlock2DCrossAttn,
    MLXUpBlock2D,
)


@dataclass
class MLXUNet2DConditionOutput(BaseOutput):
    """
    The output of [`MLXUNet2DConditionModel`].

    Args:
        sample (`mx.array` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: mx.array


class MLXUNet2DConditionModel(MLXModelMixin, ConfigMixin):
    r"""
    MLX implementation of conditional 2D UNet model that takes a noisy sample, conditional state, and a timestep and returns a sample
    shaped output.

    This model inherits from [`MLXModelMixin`]. Check the superclass documentation for it's generic methods
    implemented for all models (such as downloading or saving).


    Parameters:
        sample_size (`int`, *optional*):
            The size of the input sample.
        in_channels (`int`, *optional*, defaults to 4):
            The number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4):
            The number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("MLXCrossAttnDownBlock2D", "MLXCrossAttnDownBlock2D", "MLXCrossAttnDownBlock2D", "MLXDownBlock2D")`):
            The tuple of downsample blocks to use.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("MLXUpBlock2D", "MLXCrossAttnUpBlock2D", "MLXCrossAttnUpBlock2D", "MLXCrossAttnUpBlock2D")`):
            The tuple of upsample blocks to use.
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock2DCrossAttn"`):
            Block type for middle of UNet, it can be one of `UNetMidBlock2DCrossAttn`. If `None`, the mid block layer
            is skipped.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        attention_head_dim (`int` or `Tuple[int]`, *optional*, defaults to 8):
            The dimension of the attention heads.
        num_attention_heads (`int` or `Tuple[int]`, *optional*):
            The number of attention heads.
        cross_attention_dim (`int`, *optional*, defaults to 768):
            The dimension of the cross attention features.
        dropout (`float`, *optional*, defaults to 0):
            Dropout probability for down, up and bottleneck blocks.
        flip_sin_to_cos (`bool`, *optional*, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        freq_shift (`int`, *optional*, defaults to 0): The frequency shift to apply to the time embedding.
    """
    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        use_linear_projection: bool = False,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: Optional[int] = None,
        class_embeddings_concat: bool = False,
        dtype: mx.Dtype = mx.float32
    ):
        super().__init__()

        self.sample_size = sample_size
        self.addition_embed_type = addition_embed_type
        self.block_out_channels = block_out_channels
        self.time_embed_dim = block_out_channels[0] * 4
        self.layers_per_block = layers_per_block
        time_embed_dim = block_out_channels[0] * 4
        num_attention_heads = num_attention_heads or attention_head_dim
        
        # input
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=conv_in_kernel,
            padding=conv_in_padding
        )

        # time
        self.time_proj = MLXTimesteps(
            block_out_channels[0],
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
        )
        timestep_input_dim = block_out_channels[0]
        self.time_embedding = MLXTimestepEmbedding(timestep_input_dim, time_embed_dim)
        
        # transformer layers per block
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(
                down_block_types
            )

        # addition embed types
        if addition_embed_type is None:
            self.add_embedding = None
        elif addition_embed_type == "text_time":
            if addition_time_embed_dim is None:
                raise ValueError(
                    f"addition_embed_type {addition_embed_type} requires `addition_time_embed_dim` to not be None"
                )
            self.add_time_proj = MLXTimesteps(
                addition_time_embed_dim, flip_sin_to_cos, freq_shift
            )
            self.add_embedding = MLXTimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        else:
            raise ValueError(
                f"addition_embed_type: {addition_embed_type} must be None or `text_time`."
            )

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)
        
        if class_embeddings_concat:
            # The time embeddings are concatenated with the class embeddings. The dimension of the
            # time embeddings passed to the down, middle, and up blocks is twice the dimension of the
            # regular time embeddings
            blocks_time_embed_dim = time_embed_dim * 2
        else:
            blocks_time_embed_dim = time_embed_dim

        # down
        down_blocks = []
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            if down_block_type == "CrossAttnDownBlock2D":
                down_block = MLXCrossAttnDownBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    num_layers=layers_per_block[i],
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    num_attention_heads=num_attention_heads[i],
                    add_downsample=not is_final_block,
                    use_linear_projection=use_linear_projection,
                    cross_attention_dim=cross_attention_dim[i],
                    resnet_groups=norm_num_groups,
                    temb_channels=blocks_time_embed_dim,
                )
            else:
                down_block = MLXDownBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    num_layers=layers_per_block[i],
                    add_downsample=not is_final_block,
                    resnet_groups=norm_num_groups,
                    temb_channels=blocks_time_embed_dim,
                )
            down_blocks.append(down_block)
        self.down_blocks = down_blocks

        # mid
        if mid_block_type == "UNetMidBlock2DCrossAttn":
            self.mid_block = MLXUNetMidBlock2DCrossAttn(
                in_channels=block_out_channels[-1],
                resnet_groups=norm_num_groups,
                temb_channels=blocks_time_embed_dim,
                transformer_layers_per_block=transformer_layers_per_block[-1],
                num_attention_heads=num_attention_heads[-1],
                cross_attention_dim=cross_attention_dim[-1],
                use_linear_projection=use_linear_projection
            )
        elif mid_block_type is None:
            self.mid_block = None
        else:
            raise ValueError(f"Unexpected mid_block_type {self.config.mid_block_type}")
        
        self.num_upsamplers = 0

        # up
        up_blocks = []
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = (
            list(reversed(transformer_layers_per_block))
            if reverse_transformer_layers_per_block is None
            else reverse_transformer_layers_per_block
        )
        output_channel = reversed_block_out_channels[0]
    
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]

            is_final_block = i == len(block_out_channels) - 1# add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            if up_block_type == "CrossAttnUpBlock2D":
                up_block = MLXCrossAttnUpBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    num_layers=reversed_layers_per_block[i] + 1,
                    transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                    num_attention_heads=reversed_num_attention_heads[i],
                    add_upsample=add_upsample,
                    use_linear_projection=use_linear_projection,
                    temb_channels=blocks_time_embed_dim,
                    resnet_groups=norm_num_groups,
                    cross_attention_dim=reversed_cross_attention_dim[i]
                )
            else:
                up_block = MLXUpBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    num_layers=reversed_layers_per_block[i] + 1,
                    add_upsample=add_upsample,
                    temb_channels=blocks_time_embed_dim,
                    resnet_groups=norm_num_groups,
                )

            up_blocks.append(up_block)
            prev_output_channel = output_channel
        self.up_blocks = up_blocks

        # out
        if norm_num_groups is not None:
            self.conv_norm_out = nn.GroupNorm(
                dims=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
            )

        else:
            self.conv_norm_out = None
            self.conv_act = None

        conv_out_padding = (conv_out_kernel - 1) // 2
        self.conv_out = nn.Conv2d(
            block_out_channels[0], out_channels, kernel_size=conv_out_kernel, padding=conv_out_padding
        )

    def __call__(
        self,
        sample: mx.array,
        timesteps: Union[mx.array, float, int],
        encoder_hidden_states: mx.array,
        added_cond_kwargs: Optional[Dict] = None,
        down_block_additional_residuals: Optional[Tuple[mx.array, ...]] = None,
        mid_block_additional_residual: Optional[mx.array] = None,
        return_dict: bool = True,
        train: bool = False,
    ) -> Union[MLXUNet2DConditionOutput, Tuple[mx.array]]:
        r"""
        Args:
            sample (`mx.array`): (batch, channel, height, width) noisy inputs tensor
            timestep (`mx.array` or `float` or `int`): timesteps
            encoder_hidden_states (`mx.array`): (batch_size, sequence_length, hidden_size) encoder hidden states
            added_cond_kwargs: (`dict`, *optional*):
                A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
                are passed along to the UNet blocks.
            down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
                A tuple of tensors that if specified are added to the residuals of down unet blocks.
            mid_block_additional_residual: (`torch.Tensor`, *optional*):
                A tensor that if specified is added to the residual of the middle unet block.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unets.unet_2d_condition_mlx.MLXUNet2DConditionOutput`] instead of
                a plain tuple.
            train (`bool`, *optional*, defaults to `False`):
                Use deterministic functions and disable dropout when not training.

        Returns:
            [`~models.unets.unet_2d_condition_mlx.MLXUNet2DConditionOutput`] or `tuple`:
            [`~models.unets.unet_2d_condition_mlx.MLXUNet2DConditionOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is the sample tensor.
        """
        # 1. time
        if not isinstance(timesteps, mx.array):
            timesteps = mx.array([timesteps], dtype=mx.int32)
        elif isinstance(timesteps, mx.array) and len(timesteps.shape) == 0:
            timesteps = timesteps.astype(dtype=mx.float32)
            timesteps = mx.expand_dims(timesteps, 0)

        t_emb = self.time_proj(timesteps)
        t_emb = self.time_embedding(t_emb)

        # additional embeddings
        aug_emb = None
        if self.addition_embed_type == "text_time":
            if added_cond_kwargs is None:
                raise ValueError(
                    f"Need to provide argument `added_cond_kwargs` for {self.__class__} when using `addition_embed_type={self.addition_embed_type}`"
                )
            text_embeds = added_cond_kwargs.get("text_embeds")
            if text_embeds is None:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                )
            time_ids = added_cond_kwargs.get("time_ids")
            if time_ids is None:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                )
            # compute time embeds
            time_embeds = self.add_time_proj(
                mx.ravel(time_ids)
            )  # (1, 6) => (6,) => (6, 256)
            time_embeds = mx.reshape(time_embeds, (text_embeds.shape[0], -1))
            add_embeds = mx.concatenate([text_embeds, time_embeds], axis=-1)
            aug_emb = self.add_embedding(add_embeds)

        t_emb = t_emb + aug_emb if aug_emb is not None else t_emb

        # 2. pre-process
        sample = mx.transpose(sample, (0, 2, 3, 1))
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for down_block in self.down_blocks:
            if isinstance(down_block, MLXCrossAttnDownBlock2D):
                sample, res_samples = down_block(sample, t_emb, encoder_hidden_states)
            else:
                sample, res_samples = down_block(sample, t_emb)
            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample += down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = (new_down_block_res_samples,)
        
        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample, t_emb, encoder_hidden_states
            )

        if mid_block_additional_residual is not None:
            sample += mid_block_additional_residual

        # 5. up
        for up_block in self.up_blocks:
            res_samples = down_block_res_samples[-(self.layers_per_block + 1) :]
            down_block_res_samples = down_block_res_samples[: -(self.layers_per_block + 1)]
            if isinstance(up_block, MLXCrossAttnUpBlock2D):
                sample = up_block(
                    sample,
                    temb=t_emb,
                    encoder_hidden_states=encoder_hidden_states,
                    res_hidden_states_tuple=res_samples,
                )
            else:
                sample = up_block(sample, temb=t_emb, res_hidden_states_tuple=res_samples)
           
        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = nn.silu(sample)
        sample = self.conv_out(sample)
        sample = mx.transpose(sample, (0, 3, 1, 2))

        if not return_dict:
            return (sample,)

        return MLXUNet2DConditionOutput(sample=sample)
