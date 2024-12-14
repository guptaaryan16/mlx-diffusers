
from diffusers.utils import is_mlx_available
from diffusers.utils.testing_utils import require_mlx

if is_mlx_available():
    import mlx.core as mx
    import mlx.nn as nn


@require_mlx
class MLXModelTesterMixin:
    def test_forward_with_norm_groups(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        
        model = self.model_class(**init_dict)
        output = model(**inputs_dict)

        if isinstance(output, dict):
            output = output.to_tuple()[0]
 
        self.assertIsNotNone(output)
        expected_shape = inputs_dict["sample"].shape
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")
