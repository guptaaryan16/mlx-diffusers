from diffusers import MLXAutoencoderKL, MLXUNet2DConditionModel, MLXStableDiffusionPipeline, MLXDDIMScheduler 
from transformers import CLIPImageProcessor, CLIPTokenizer, CLIPTextModel
import mlx.core as mx
import torch
from PIL import Image
import numpy as np

images = np.load("/Users/guptaaryan16/Desktop/OSS/diffusers/tests/pipelines/stable_diffusion/image.npy").astype(np.float32)
images = np.asarray(images.reshape((1,) + images.shape[-3:]))

def dummy_input(self):
    batch_size = 4
    num_channels = 3
    sizes = (32, 32)

    prng_key = mx.random.key(0)
    image = mx.random.uniform(key=prng_key, shape=((batch_size, num_channels) + sizes), dtype=mx.float32)

    return {"sample": image}

def prepare_init_args_and_inputs_for_common(self):
    init_dict = {
        "block_out_channels": [32, 64],
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
        "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
        "latent_channels": 4,
    }
    inputs_dict = self.dummy_input
    return init_dict, inputs_dict

def test_from_pretrained_hub(self):
    model = MLXAutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", from_pt=True)
    self.assertIsNotNone(model)

    image = model(**self.dummy_input)

    assert image is not None, "Make sure output is not None"