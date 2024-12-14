import unittest

from diffusers import MLXAutoencoderKL, AutoencoderKL
from diffusers.utils import is_mlx_available
from diffusers.utils.testing_utils import require_mlx
import numpy as np
import torch

if is_mlx_available():
    import mlx.core as mx

## Need to be used for testing MLX code from pytorch one 

image = mx.random.normal(shape=(4, 3, 128, 128), dtype=mx.float32, key=mx.random.key(0))
image_torch = torch.tensor(np.array(image))

model = MLXAutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae",  from_pt=True, dtype=mx.float32)
model_torch = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae",  torch_dtype=torch.float32)

mlx_output = model(image).sample
torch_output = model_torch(image_torch).sample

# Compare outputs
np_torch_output = torch_output.detach().numpy()
np_mlx_output = np.array(mlx_output)

print("\nOutput Comparison:")
print("Torch Output Shape:", np_torch_output.shape)
print("MLX Output Shape:", np_mlx_output.shape)

# Calculate absolute differences
abs_diff = np.abs(np_torch_output - np_mlx_output)
print("\nSum Maximum Absolute Difference:", np.sum(abs_diff))
print("Mean Absolute Difference:", np.mean(abs_diff))

print("mlx output = ", np_mlx_output)
print("torch output = ", np_torch_output)