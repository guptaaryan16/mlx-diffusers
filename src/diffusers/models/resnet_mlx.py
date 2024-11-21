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

from typing import Optional

# Taken from https://github.com/ml-explore/mlx-examples/blob/main/stable_diffusion/stable_diffusion/unet.py
def upsample_nearest(x, scale: int = 2):
    B, H, W, C = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (B, H, scale, W, scale, C))
    x = mx.reshape(x, (B, H * scale, W * scale, C))
    return x


class MLXUpsample2D(nn.Module):
    def __init__(
        self,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        self.conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def __call__(self, hidden_states):
        hidden_states = upsample_nearest(hidden_states, 2)
        hidden_states = self.conv(hidden_states)
        return hidden_states


class MLXDownsample2D(nn.Module):
    def __init__(
        self,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1 
    ):
        self.conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def __call__(self, hidden_states):
        # pad = ((0, 0), (0, 1), (0, 1), (0, 0))  # pad height and width dim
        # hidden_states = mx.pad(hidden_states, pad_width=pad)
        hidden_states = self.conv(hidden_states)
        return hidden_states


class MLXResnetBlock2D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: Optional[int] = None,
        groups: int = 32,
    ):
        self.temb_channels = temb_channels
        out_channels = (
            in_channels if out_channels is None else out_channels
        )

        self.norm1 = nn.GroupNorm(
            num_groups=groups, dims=in_channels, pytorch_compatible=True
        )
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels)

        self.norm2 = nn.GroupNorm(
            num_groups=groups, dims=out_channels, pytorch_compatible=True
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def __call__(self, hidden_states, temb=None):
        temb = hidden_states if temb is None else temb
        
        if self.temb_channels is not None:
            temb = self.time_emb_proj(nn.silu(temb))
        
        residual = hidden_states

        hidden_states = self.norm1(hidden_states)
   
        hidden_states = nn.silu(hidden_states)
        hidden_states = self.conv1(hidden_states)
        
        if self.temb_channels is not None:
            hidden_states = hidden_states + temb[:, None, None, :]

        hidden_states = self.norm2(hidden_states)
        hidden_states = nn.silu(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            hidden_states = hidden_states + self.conv_shortcut(residual)

        return hidden_states
