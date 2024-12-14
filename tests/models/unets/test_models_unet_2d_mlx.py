import gc
import unittest

from parameterized import parameterized

from diffusers import MLXUNet2DConditionModel
from diffusers.utils import is_mlx_available
from diffusers.utils.testing_utils import load_hf_numpy, require_mlx, slow


if is_mlx_available():
    import numpy as np
    import mlx.core as mx


@slow
@require_mlx
class MLXUNet2DConditionModelIntegrationTests(unittest.TestCase):
    def get_file_format(self, seed, shape):
        return f"gaussian_noise_s={seed}_shape={'_'.join([str(s) for s in shape])}.npy"

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()

    def get_latents(self, seed=0, shape=(4, 4, 64, 64), fp16=False):
        dtype = mx.bfloat16 if fp16 else mx.float32
        image = mx.array(load_hf_numpy(self.get_file_format(seed, shape)), dtype=dtype)
        return image

    def get_unet_model(self, fp16=False, model_id="CompVis/stable-diffusion-v1-4"):
        dtype = mx.bfloat16 if fp16 else mx.float32

        model = MLXUNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", dtype=dtype, from_pt=True
        )
        return model

    def get_encoder_hidden_states(self, seed=0, shape=(4, 77, 768), fp16=False):
        dtype = mx.bfloat16 if fp16 else mx.float32
        hidden_states = mx.array(
            load_hf_numpy(self.get_file_format(seed, shape)), dtype=dtype
        )
        return hidden_states

    @parameterized.expand(
        [
            # fmt: off
            [83, 4, [-0.8476, -0.7416, 0.0585, -0.3641, -0.1426, -0.9923, -0.2844 ,-1.7754]],
            [17, 0.55, [-0.2835, -0.1197, 0.1916, 0.08379, 0.6240, -0.4074, -0.8641, -0.4192]],
            [8, 0.89, [-1.1721, 0.1750, 0.2020, 0.1186, 1.6469, -0.3763, 0.6893, 0.1516]],
            [3, 1000, [-0.8655, 0.3523, -0.6838, 0.3321, 1.7616, -0.5872, 0.2899, -1.5459]],
            # fmt: on
        ]
    )
    def test_compvis_sd_v1_4_mlx_vs_torch_fp16(self, seed, timestep, expected_slice):
        model = self.get_unet_model(
            model_id="CompVis/stable-diffusion-v1-4", fp16=True
        )
        latents = self.get_latents(seed, fp16=True)
        encoder_hidden_states = self.get_encoder_hidden_states(seed, fp16=True)

        sample = model(
            latents,
            mx.array(timestep, dtype=mx.int32),
            encoder_hidden_states=encoder_hidden_states,
        ).sample 
        
        mx.eval(sample)
        assert sample.shape == latents.shape

        output_slice = mx.array(sample[-1, -2:, -2:, :2].flatten(), dtype=mx.float32)
        expected_output_slice = mx.array(expected_slice, dtype=mx.float32)

        # Found torch (float16) and mlx (bfloat16) outputs to be within this tolerance, in the same hardware
        assert mx.allclose(output_slice, expected_output_slice, atol=1e-2)

    # @parameterized.expand(
    #     [
    #         # fmt: off
    #         [83, 4, [0.1672, 0.1216, 0.2495, 0.2096, -0.3479, 0.1208, -0.3948, -0.4423]],
    #         [17, 0.55, [0.1276, 0.1466, 0.2401, 0.1438, -0.0427, -0.3950, -0.3642, -0.1445]],
    #         [8, 0.89, [-0.3872, -0.1293, -0.1552, 0.0510, 0.0456, -0.5277, -0.1444, -0.4731]],
    #         [3, 1000, [0.0679, 0.2061, 0.1910, 0.2753, -0.0798, -0.3515, -0.0809, -0.4194]],
    #         # fmt: on
    #     ]
    # )
    # def test_stabilityai_sd_v2_mlx_vs_torch_fp16(self, seed, timestep, expected_slice):
    #     model = self.get_unet_model(
    #         model_id="stabilityai/stable-diffusion-2", fp16=True
    #     )
    #     latents = self.get_latents(seed, shape=(4, 4, 96, 96), fp16=True)
    #     encoder_hidden_states = self.get_encoder_hidden_states(
    #         seed, shape=(4, 77, 1024), fp16=True
    #     )

    #     sample = model(
    #         latents,
    #         mx.array(timestep, dtype=mx.int32),
    #         encoder_hidden_states=encoder_hidden_states
    #     ).sample

    #     assert sample.shape == latents.shape

    #     output_slice = mx.array(
    #         sample[-1, -2:, -2:, :2].flatten(), dtype=mx.float32
    #     )
    #     expected_output_slice = mx.array(expected_slice, dtype=mx.float32)

    #     # Found torch (float16) and mlx (bfloat16) outputs to be within this tolerance, on the same hardware
    #     assert mx.allclose(output_slice, expected_output_slice, atol=1e-2)
