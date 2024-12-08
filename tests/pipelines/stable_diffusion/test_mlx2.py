
from diffusers import MLXAutoencoderKL
from transformers import CLIPImageProcessor, CLIPTokenizer, CLIPTextModel
import mlx.core as mx
import torch
from PIL import Image
import numpy as np

def numpy_to_pil(images):
    """
    Convert a NumPy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255)
    images = images.round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

model_id = "CompVis/stable-diffusion-v1-4"
dtype = mx.bfloat16

vae = MLXAutoencoderKL.from_pretrained(model_id, subfolder="vae", dtype=dtype, from_pt=True)
vae.set_dtype(mx.bfloat16)
latents = mx.load("/Users/guptaaryan16/Desktop/OSS/diffusers/latents.npy").astype(mx.bfloat16)

image = vae.decode(
   latents
).sample

images = mx.transpose(mx.clip((image / 2 + 0.5), 0, 1), (0, 2, 3, 1))
images = np.array(images.astype(mx.float32))

np.save("/Users/guptaaryan16/Desktop/OSS/diffusers/tests/pipelines/stable_diffusion/images2.npy", images)
images = numpy_to_pil(images)

images.save("astronaut_rides_horse.png")
