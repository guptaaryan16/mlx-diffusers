# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""PyTorch - MLX general utilities."""

from mlx.utils import tree_flatten
from ..utils import logging

logger = logging.get_logger(__name__)


# Used code from the flax utils file due to the same design of Flax and MLX
#####################
# PyTorch => MLX #
#####################


def rename_key_and_reshape_tensor(pt_tuple_key, pt_tensor):
    """Rename PT weight names to corresponding MLX weight names and reshape tensor if necessary"""

    # conv layer
    if pt_tuple_key[-1] == "weight" and pt_tensor.ndim == 4:
        pt_tensor = pt_tensor.transpose(0, 2, 3, 1)
        return pt_tuple_key, pt_tensor

    return pt_tuple_key, pt_tensor


def convert_pytorch_state_dict_to_mlx(pt_state_dict):
    for pt_key, pt_tensor in pt_state_dict.items():
        pt_tuple_key = tuple(pt_key.split("."))
        mlx_key, mlx_tensor = rename_key_and_reshape_tensor(pt_tuple_key, pt_tensor)
        mlx_key = ".".join(mlx_key)
        pt_state_dict[mlx_key] = mlx_tensor

    return pt_state_dict