from diffusers import FlaxUNet2DConditionModel
from diffusers.utils.testing_utils import load_hf_numpy, require_mlx, slow

import jax
import jax.numpy as jnp

def get_file_format(seed, shape):
    return f"gaussian_noise_s={seed}_shape={'_'.join([str(s) for s in shape])}.npy"

def get_unet_model(fp16=False, model_id="CompVis/stable-diffusion-v1-4"):
    dtype = jnp.bfloat16 if fp16 else jnp.float32
    revision = "bf16" if fp16 else None

    model, params = FlaxUNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", dtype=dtype, revision=revision
    )
    return model, params

def get_latents(seed, shape=(4, 4, 64, 64), fp16=False):
    dtype = jnp.bfloat16 if fp16 else jnp.float32
    image = jnp.array(load_hf_numpy(get_file_format(seed, shape)), dtype=dtype)
    return image 

def get_encoder_hidden_states(seed=0, shape=(4, 77, 1024), fp16=False):
    dtype = jnp.bfloat16 if fp16 else jnp.float32
    hidden_states = jnp.array(load_hf_numpy(get_file_format(seed, shape)), dtype=dtype)
    return hidden_states

model, params = get_unet_model(
    model_id="stabilityai/stable-diffusion-2", fp16=True
)

latents = get_latents(3, fp16=True)
encoder_hidden_states = get_encoder_hidden_states(3, fp16=True)

sample = model.apply(
    {"params": params},
    latents,
    jnp.array([4], dtype=jnp.int32),
    encoder_hidden_states=encoder_hidden_states,
).sample

print(sample)
print(sample.shape)