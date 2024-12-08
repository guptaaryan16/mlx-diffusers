from diffusers import MLXAutoencoderKL, MLXUNet2DConditionModel, MLXStableDiffusionPipeline, MLXDDIMScheduler 
from transformers import CLIPImageProcessor, CLIPTokenizer, CLIPTextModel
import mlx.core as mx
import torch
from PIL import Image
import numpy as np

model_id = "CompVis/stable-diffusion-v1-4"
dtype = mx.bfloat16

## MLX based Components
scheduler = MLXDDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
vae = MLXAutoencoderKL.from_pretrained(model_id, subfolder="vae", dtype=dtype, from_pt=True)
unet = MLXUNet2DConditionModel.from_pretrained(model_id, subfolder="unet", dtype=dtype, from_pt=True)

torch_dtype = torch.float16
## PyTorch based parts(to be converted to MLX)
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch_dtype)
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer",  torch_dtype=torch_dtype)
feature_extractor = CLIPImageProcessor.from_pretrained(model_id, subfolder="feature_extractor",  torch_dtype=torch_dtype)

pipeline = MLXStableDiffusionPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler, feature_extractor=feature_extractor)

prompt = "a photo of an astronaut riding a horse on mars"
seed = 0
prompt_ids = pipeline.prepare_inputs(prompt)

images = pipeline(prompt_ids, seed).images
images = pipeline.numpy_to_pil(images)
images.save("astronaut_rides_horse.png")
