from diffusers.models import MLXAutoencoderKL
import mlx.core as mx

batch_size = 4
num_channels = 3
sizes = (32, 32)

prng_key = mx.random.key(0)
image = mx.random.uniform(key=prng_key, shape=((batch_size, num_channels) + sizes), dtype=mx.float32) 
inputs_dict = {"sample": image}

init_dict = {
            "block_out_channels": [32, 64],
            "in_channels": 3,
            "out_channels": 3,
            "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
            "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
            "latent_channels": 4,
        }


model = MLXAutoencoderKL(**init_dict)
print(model)