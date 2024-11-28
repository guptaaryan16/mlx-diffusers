from diffusers import MLXUNet2DConditionModel
import mlx.core as mx
from diffusers.utils.testing_utils import load_hf_numpy, require_mlx, slow
import numpy as np

def get_file_format(seed, shape):
    return f"gaussian_noise_s={seed}_shape={'_'.join([str(s) for s in shape])}.npy"

def get_unet_model(fp16=False, model_id="stabilityai/stable-diffusion-2"):
    dtype = mx.bfloat16 if fp16 else mx.float32

    model = MLXUNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", dtype=dtype, from_pt=True
    )
    return model

def get_latents(seed, shape=(4, 4, 64, 64), fp16=False):
    dtype = mx.bfloat16 if fp16 else mx.float32
    image = mx.array(load_hf_numpy(get_file_format(seed, shape)), dtype=dtype)
    return image    

def get_encoder_hidden_states(seed=0, shape=(4, 77, 1024), fp16=False):
    dtype = mx.bfloat16 if fp16 else mx.float32
    hidden_states = mx.array(
        load_hf_numpy(get_file_format(seed, shape)), dtype=dtype
    )
    return hidden_states

model = get_unet_model(
    model_id="stabilityai/stable-diffusion-2", fp16=True
)

'''
 [17, 0.55, [0.1164, -0.0216, 0.0170, 0.1589, -0.3120, 0.1005, -0.0581, -0.1458]],
            [8, 0.89, [-0.1758, -0.0169, 0.1004, -0.1411, 0.1312, 0.1103, -0.1996, 0.2139]],
            [3, 1000,
'''
seed = 17
timestamp = 0.55
latents = get_latents(seed, fp16=True)
encoder_hidden_states = get_encoder_hidden_states(seed, fp16=True)

sample = model(
    latents,
    mx.array([timestamp], dtype=mx.int32),
    encoder_hidden_states=encoder_hidden_states,
).sample 

output_slice = mx.array(sample[-1, -2:, -2:, :2].flatten(), dtype=mx.float32)
print(np.array(output_slice))
