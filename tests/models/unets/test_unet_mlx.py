import unittest

from diffusers import MLXAutoencoderKL, AutoencoderKL, UNet2DConditionModel, MLXUNet2DConditionModel
from diffusers.utils import is_mlx_available
from diffusers.utils.testing_utils import require_mlx
import numpy as np
import torch

if is_mlx_available():
    import mlx.core as mx

### Used to test MLXUnetConditional Model from MLX and PyTorch for correct outputs

image = mx.random.normal(shape=(4, 4, 64, 64), dtype=mx.float32, key=mx.random.key(0))
image_torch = torch.tensor(np.array(image))
encoder_hidden_mlx = mx.random.normal(shape=(4, 77, 768),dtype=mx.float32, key=mx.random.key(0))
encoder_hidden_torch = torch.tensor(np.array(encoder_hidden_mlx))
model = MLXUNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", from_pt=True, dtype=mx.float32)
model_torch = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet",  torch_dtype=torch.float32)

mlx_output = model(image, mx.array(10, dtype=mx.int32), encoder_hidden_mlx).sample
torch_output = model_torch(image_torch, torch.tensor(10, dtype=torch.int32), encoder_hidden_torch).sample

# Compare outputs
np_torch_output = torch_output.detach().numpy()
np_mlx_output = np.array(mlx_output)

print("\nOutput Comparison:")
print("Torch Output Shape:", np_torch_output.shape)
print("MLX Output Shape:", np_mlx_output.shape)

# Calculate absolute differences
abs_diff = np.abs(np_torch_output - np_mlx_output)
print("\nMaximum Absolute Difference:", np.sum(abs_diff))
print("Mean Absolute Difference:", np.mean(abs_diff))

print("mlx output = ", np_mlx_output)
print("torch output = ", np_torch_output)