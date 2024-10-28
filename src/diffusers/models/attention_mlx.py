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
import functools
import math

import mlx.core as mx
import mlx.nn as nn

## Inspired from example here: https://github.com/ml-explore/mlx-examples/blob/main/stable_diffusion/stable_diffusion/unet.py
## I am using complete implementations as in Flax and pyTorch implementation of diffusers to make it easy tp maintain the library in future with the similar API

class MLXAttention(nn.Module):
    r"""
    A MLX multi-head attention module as described in: https://arxiv.org/abs/1706.03762
    Used the mlx base MultiHeadAttention implementation with some changes.

    Parameters:
        dims (int): The model dimensions. This is also the default
            value for the queries, keys, values, and the output.
        num_heads (int): The number of attention heads to use.
        query_input_dims (int, optional): The input dimensions of the queries.
            Default: ``dims``.
        key_input_dims (int, optional): The input dimensions of the keys.
            Default: ``dims``.
        value_input_dims (int, optional): The input dimensions of the values.
            Default: ``key_input_dims``.
        value_dims (int, optional): The dimensions of the values after the
            projection. Default: ``dims``.
        value_output_dims (int, optional): The dimensions the new values will
            be projected to. Default: ``dims``.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        bias (bool, optional): Whether or not to use a bias in the projections.
            Default: ``False``.
    """
    def __init__(
        self,
        dims: int,
        num_heads: int,
        query_input_dims: Optional[int] = None,
        key_input_dims: Optional[int] = None,
        value_input_dims: Optional[int] = None,
        value_dims: Optional[int] = None,
        value_output_dims: Optional[int] = None,
        bias: bool = False,
    ):
        query_input_dims = query_input_dims or dims
        key_input_dims = key_input_dims or dims
        value_input_dims = value_input_dims or key_input_dims
        value_dims = value_dims or dims
        value_output_dims = value_output_dims or dims

        self.num_heads = num_heads

        inner_dim = self.dim_head * self.heads
        self.scale = self.dim_head**-0.5

        self.to_q = Linear(query_input_dims, dims, bias=bias)
        self.to_k = Linear(key_input_dims, dims, bias=bias)
        self.to_v = Linear(value_input_dims, value_dims, bias=bias)
        self.to_out_0 = Linear(value_dims, value_output_dims, bias=bias)

    def __call__(self, hidden_states, context=None, mask=None):
        context = hidden_states if context is None else context

        queries = self.to_q(hidden_states)
        keys = self.to_k(context)
        values = self.to_v(context)

        num_heads = self.num_heads
        B, L, D = queries.shape
        _, S, _ = keys.shape
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, num_heads, -1).transpose(0, 2, 3, 1)
        values = values.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)

        # Dimensions are [batch x num heads x sequence x hidden dim]
        scale = math.sqrt(1 / queries.shape[-1])
        scores = (queries * scale) @ keys
        if mask is not None:
            scores = scores + mask.astype(scores.dtype)
        scores = mx.softmax(scores, axis=-1)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.to_out_0(values_hat)


class MLXBasicTransformerBlock(nn.Module):
    r"""
    A MLX transformer block layer.

    Parameters:
        model_dims (:obj:`int`):
            Inner hidden states dimension
        n_heads (:obj:`int`):
            Number of heads
        hidden_dims (:obj:`int`):
            Hidden states dimension
    """
    def __init__(
        self,
        model_dims: int,
        n_heads: int,
        hidden_dims: Optional[int] = None,
        memory_dims: Optional[int] = None,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(model_dims)
        self.attn1 = MLXAttention(model_dims, n_heads)
        self.attn1.to_out_0.bias = mx.zeros(model_dims)

        memory_dims = memory_dims or model_dims
        self.norm2 = nn.LayerNorm(model_dims)
        self.attn2 = MLXAttention(
            model_dims, n_heads, key_input_dims=memory_dims
        )
        self.attn2.to_out_0.bias = mx.zeros(model_dims)

        hidden_dims = hidden_dims or 4 * model_dims
        self.norm3 = nn.LayerNorm(model_dims)
        self.ff = MLXFeedForward(model_dims, hidden_dims)

    def __call__(self, residual, hidden_states=None, attn_mask=None):
        # self attention
        residual = hidden_states
        hidden_states = self.attn1(self.norm1(hidden_states), attn_mask)
        hidden_states = hidden_states + residual

        # cross attention
        residual = hidden_states
        hidden_states = self.attn2(
            self.norm2(hidden_states), context, attn_mask
        )
        hidden_states = hidden_states + residual

        # feed forward
        residual = hidden_states
        hidden_states = self.ff(self.norm3(hidden_states))
        hidden_states = hidden_states + residual

        return hidden_states


class MLXTransformer2DModel(nn.Module):
    r"""
    A transformer model for inputs with 2 spatial dimensions.

    Parameters:
        in_channels (:obj:`int`):
            Input number of channels
        n_heads (:obj:`int`):
            Number of heads
        d_head (:obj:`int`):
            Hidden states dimension inside each head
        depth (:obj:`int`, *optional*, defaults to 1):
            Number of transformers block
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
    """
    def __init__(
        self,
        in_channels: int,
        model_dims: int,
        encoder_dims: int,
        num_heads: int,
        num_layers: int = 1,
        norm_num_groups: int = 32,
    ):
        super().__init__()

        self.norm = nn.GroupNorm(norm_num_groups, in_channels, pytorch_compatible=True)
        self.proj_in = nn.Linear(in_channels, model_dims)
        self.transformer_blocks = [
            MLXBasicTransformerBlock(model_dims, num_heads, memory_dims=encoder_dims)
            for i in range(num_layers)
        ]
        self.proj_out = nn.Linear(model_dims, in_channels)

    def __call__(self, hidden_states, context, attn_mask):
        # Save the input to add to the output
        residual = hidden_states
        dtype = hidden_statesx.dtype

        # Perform the input norm and projection
        B, H, W, C = hidden_states.shape
        hidden_states = self.norm(hidden_states).reshape(B, -1, C)
        hidden_states = self.proj_in(hidden_states)

        # Apply the transformer
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, context, attn_mask)

        # Apply the output projection and reshape
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(B, H, W, C)

        return hidden_states + residual


class MLXFeedForward(nn.Module):
    r"""
    MLX module that encapsulates two Linear layers separated by a non-linearity. It is the counterpart of PyTorch's
    [`FeedForward`] class, with the following simplifications:
    - The activation function is currently hardcoded to a gated linear unit from:
    https://arxiv.org/abs/2002.05202
    - `dim_out` is equal to `dim`.
    - The number of hidden dimensions is hardcoded to `dim * 4` in [`MLXGELU`].

    Parameters:
        model_dims (:obj:`int`):
            Model input states dimension
        hidden_dims (:obj:`int`):
            Inner hidden states dimension
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
    """
    def __init__(
        self,
        model_dims: int, 
        hidden_dims: int = None,
        dropout: float = 0.0,
    ):
        # The second linear layer needs to be called
        # net_2 for now to match the index of the Sequential layer
        hidden_dims = hidden_dims or 4 * model_dims
        self.net_0 = MLXGEGLU(model_dims, hidden_dims, dropout)
        self.net_2 = nn.Linear(hidden_dims, model_dims)

    def __call__(self, hidden_states):
        hidden_states = self.net_0(hidden_states)
        hidden_states = self.net_2(hidden_states)
        return hidden_states


class MLXGEGLU(nn.Module):
    r"""
    MLX implementation of a Linear layer followed by the variant of the gated linear unit activation function from
    https://arxiv.org/abs/2002.05202.

    Parameters:
        model_dims (:obj:`int`):
            Input hidden states dimension
        hidden_dims (:obj:`int`):
            hidden states dimension
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
    """
    def __init__(
        self, model_dims: int, hidden_dims: int=None, dropout: float = 0.0
    ):
        self.proj = nn.Linear(model_dims, hidden_dims*2)
        self.dropout_layer = nn.Dropout(rate=self.dropout)

    def __call__(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_linear, hidden_gelu = mx.split(hidden_states, 2, axis=2)
        return self.dropout_layer(
            hidden_linear * nn.gelu(hidden_gelu)
        )
