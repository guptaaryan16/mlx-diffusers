import unittest

from diffusers import MLXAutoencoderKL
from diffusers.utils import is_mlx_available
from diffusers.utils.testing_utils import require_mlx

from ..test_modeling_common_mlx import MLXModelTesterMixin

if is_mlx_available():
    import mlx.core as mx


@require_mlx
class MLXAutoencoderKLTests(MLXModelTesterMixin, unittest.TestCase):
    model_class = MLXAutoencoderKL

    @property
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
    
    # TODO: Complete the model loading in mlx 
    # def test_from_pretrained_hub(self):
    #     model, loading_info = MLXAutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema-original", output_loading_info=True)
    #     self.assertIsNotNone(model)
    #     self.assertEqual(len(loading_info["missing_keys"]), 0)

    #     image = model(**self.dummy_input)

    #     assert image is not None, "Make sure output is not None"
