from diffusers import MLXAutoencoderKL, MLXUNet2DConditionModel, MLXStableDiffusionPipeline, MLXDDIMScheduler 
from transformers import CLIPImageProcessor, CLIPTokenizer, FlaxCLIPTextModel
import mlx.core as mx
import mlx.nn as nn
import torch
from PIL import Image
import numpy as np
model_id = "CompVis/stable-diffusion-v1-4"
dtype = mx.float16

## Testing Stable Diffusion 1 with MLX Components

## MLX based Components
scheduler = MLXDDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
vae = MLXAutoencoderKL.from_pretrained(model_id, subfolder="vae", dtype=dtype, from_pt=True)
unet = MLXUNet2DConditionModel.from_pretrained(model_id, subfolder="unet", dtype=dtype, from_pt=True)

torch_dtype = torch.float16
## PyTorch based parts(to be converted to MLX)
text_encoder = FlaxCLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer",  torch_dtype=torch_dtype)

pipeline = MLXStableDiffusionPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler, feature_extractor=None)
prompt = "a fluffy pi in a verdant forest"
seed = 0
prompt_ids = pipeline.prepare_inputs(prompt)

image = pipeline(prompt_ids, seed).images
B, H, W, C = image.shape
image = image.reshape(H, B * W, C)
image = (image * 255).astype(np.uint8)
image = Image.fromarray(image)
image.save("fluffy_pi.png")
