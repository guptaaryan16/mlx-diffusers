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

import math
import mlx.core as mx
import mlx.nn as nn

def get_sinusoidal_embeddings(
    timesteps: mx.array,
    embedding_dim: int,
    freq_shift: float = 1,
    min_timescale: float = 1,
    max_timescale: float = 1.0e4,
    flip_sin_to_cos: bool = False,
    scale: float = 1.0,
) -> mx.array:
    """Returns the positional encoding (same as Tensor2Tensor).

    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
        These may be fractional.
        embedding_dim: The number of output channels.
        min_timescale: The smallest time unit (should probably be 0.0).
        max_timescale: The largest time unit.
    Returns:
        a Tensor of timing signals [N, num_channels]
    """
    assert timesteps.ndim == 1, "Timesteps should be a 1d-array"
    assert embedding_dim % 2 == 0, f"Embedding dimension {embedding_dim} should be even"
    num_timescales = float(embedding_dim // 2)
    log_timescale_increment = math.log(max_timescale / min_timescale) / (
        num_timescales - freq_shift
    )
    inv_timescales = min_timescale * mx.exp(
        mx.arange(num_timescales, dtype=mx.float32) * -log_timescale_increment
    )
    emb = mx.expand_dims(timesteps, 1) * mx.expand_dims(inv_timescales, 0)

    # scale embeddings
    scaled_time = scale * emb

    if flip_sin_to_cos:
        signal = mx.concatenate([mx.cos(scaled_time), mx.sin(scaled_time)], axis=1)
    else:
        signal = mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=1)
    signal = mx.reshape(signal, [timesteps.shape[0], embedding_dim])
    return signal


class MLXTimestepEmbedding(nn.Module):
    r"""
    Time step Embedding Module. Learns embeddings for input time steps.

    Args:
        time_embed_dim (`int`, *optional*, defaults to `32`):
                Time step embedding dimension
    """

    def __init__(
        self, 
        in_channels: int,
        time_embed_dim: int = 32,
    ):
        self.time_embed_dim = time_embed_dim

        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def __call__(self, temb):
        temb = self.linear_1(temb)
        temb = nn.silu(temb)
        temb = self.linear_2(temb)
        return temb


class MLXTimesteps(nn.Module):
    r"""
    Wrapper Module for sinusoidal Time step Embeddings as described in https://arxiv.org/abs/2006.11239

    Args:
        dim (`int`, *optional*, defaults to `32`):
                Time step embedding dimension
    """

    def __init__(
        self,
        dim: int = 32,
        flip_sin_to_cos: bool = False,
        freq_shift: float = 1
    ):
        self.dim = dim
        self.flip_sin_to_cos = flip_sin_to_cos
        self.freq_shift = freq_shift

    def __call__(self, timesteps):
        return get_sinusoidal_embeddings(
            timesteps,
            embedding_dim=self.dim,
            flip_sin_to_cos=self.flip_sin_to_cos,
            freq_shift=self.freq_shift,
        )
