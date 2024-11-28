# coding=utf-8
# Copyright 2024 Apple and HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import tempfile
import unittest
from typing import Dict, List, Tuple

from diffusers import MLXDDPMScheduler, MLXDDIMScheduler
from diffusers.utils import is_mlx_available
from diffusers.utils import logging
from diffusers.utils.testing_utils import require_mlx

if is_mlx_available():
    import mlx.core as mx

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@require_mlx
class MLXSchedulerCommonTest(unittest.TestCase):
    scheduler_classes = ()
    forward_default_kwargs = ()

    @property
    def default_num_inference_steps(self):
        return 50

    @property
    def default_timestep(self):
        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.get("num_inference_steps", self.default_num_inference_steps)

        try:
            scheduler_config = self.get_scheduler_config()
            scheduler = self.scheduler_classes[0](**scheduler_config)

            scheduler.set_timesteps(num_inference_steps)
            timestep = scheduler.timesteps[0]
        except NotImplementedError:
            logger.warning(
                f"The scheduler {self.__class__.__name__} does not implement a `get_scheduler_config` method."
                f" `default_timestep` will be set to the default value of 1."
            )
            timestep = 1

        return timestep

    @property
    def dummy_sample(self):
        batch_size = 4
        num_channels = 3
        height = 8
        width = 8

        key1, key2 = mx.random.split(mx.random.key(0))
        sample = mx.random.uniform(key=key1, shape=(batch_size, num_channels, height, width))

        return sample, key2

    @property
    def dummy_sample_deter(self):
        batch_size = 4
        num_channels = 3
        height = 8
        width = 8

        num_elems = batch_size * num_channels * height * width
        sample = mx.array(mx.arange(num_elems))
        sample = mx.reshape(sample, (num_channels, height, width, batch_size))
        sample = sample / num_elems
        return mx.transpose(sample, (3, 0, 1, 2))

    def get_scheduler_config(self):
        raise NotImplementedError

    def dummy_model(self):
        def model(sample, t, *args):
            return sample * t / (t + 1)

        return model

    def check_over_configs(self, time_step=0, **config):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)
        time_step = time_step if time_step is not None else self.default_timestep

        for scheduler_class in self.scheduler_classes:
            sample, key = self.dummy_sample
            residual = 0.1 * sample

            scheduler_config = self.get_scheduler_config(**config)
            scheduler = scheduler_class(**scheduler_config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
                new_scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(
                scheduler, "set_timesteps"
            ):
                kwargs["num_inference_steps"] = num_inference_steps

            output = scheduler.step(
                residual, time_step, sample, key, **kwargs
            ).prev_sample
            new_output = new_scheduler.step(
                residual, time_step, sample, key, **kwargs
            ).prev_sample
            
            assert (
                mx.sum(mx.abs(output - new_output)) < 1e-5
            ), "Scheduler outputs are not identical"

    def check_over_forward(self, time_step=0, **forward_kwargs):
        kwargs = dict(self.forward_default_kwargs)
        kwargs.update(forward_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)
        time_step = time_step if time_step is not None else self.default_timestep

        for scheduler_class in self.scheduler_classes:
            sample, key = self.dummy_sample
            residual = 0.1 * sample

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
                new_scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(
                scheduler, "set_timesteps"
            ):
                kwargs["num_inference_steps"] = num_inference_steps

            output = scheduler.step(
                residual, time_step, sample, key, **kwargs
            ).prev_sample
            new_output = new_scheduler.step(
                residual, time_step, sample, key, **kwargs
            ).prev_sample

            assert (
                mx.sum(mx.abs(output - new_output)) < 1e-5
            ), "Scheduler outputs are not identical"

    def test_from_save_pretrained(self):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            sample, key = self.dummy_sample
            residual = 0.1 * sample

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
                new_scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(
                scheduler, "set_timesteps"
            ):
                kwargs["num_inference_steps"] = num_inference_steps

            output = scheduler.step(
                residual, 1, sample, key, **kwargs
            ).prev_sample
            new_output = new_scheduler.step(
                residual, 1, sample, key, **kwargs
            ).prev_sample
            assert (
                mx.sum(mx.abs(output - new_output)) < 1e-5
            ), "Scheduler outputs are not identical"

    def test_step_shape(self):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            sample, key = self.dummy_sample
            residual = 0.1 * sample

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(
                scheduler, "set_timesteps"
            ):
                kwargs["num_inference_steps"] = num_inference_steps

            output_0 = scheduler.step(
                residual, 0, sample, key, **kwargs
            ).prev_sample
            output_1 = scheduler.step(
                residual, 1, sample, key, **kwargs
            ).prev_sample

            self.assertEqual(output_0.shape, sample.shape)
            self.assertEqual(output_0.shape, output_1.shape)

    def test_scheduler_outputs_equivalence(self):
        def recursive_check(tuple_object, dict_object):
            if isinstance(tuple_object, (List, Tuple)):
                for tuple_iterable_value, dict_iterable_value in zip(
                    tuple_object, dict_object.values()
                ):
                    recursive_check(tuple_iterable_value, dict_iterable_value)
            elif isinstance(tuple_object, Dict):
                for tuple_iterable_value, dict_iterable_value in zip(
                    tuple_object.values(), dict_object.values()
                ):
                    recursive_check(tuple_iterable_value, dict_iterable_value)
            elif tuple_object is None:
                return
            else:
                self.assertTrue(
                    mx.allclose(
                        mx.nan_to_num(tuple_object),
                        mx.nan_to_num(dict_object),
                        atol=1e-5,
                    ),
                    msg=(
                        "Tuple and dict output are not equal. Difference:"
                        f" {mx.max(mx.abs(tuple_object - dict_object))}. Tuple has `nan`:"
                        f" {mx.isnan(tuple_object).any()} and `inf`: {mx.isinf(tuple_object)}. Dict has"
                        f" `nan`: {mx.isnan(dict_object).any()} and `inf`: {mx.isinf(dict_object)}."
                    ),
                )

        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            sample, key = self.dummy_sample
            residual = 0.1 * sample

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(
                scheduler, "set_timesteps"
            ):
                kwargs["num_inference_steps"] = num_inference_steps

            outputs_dict = scheduler.step(residual, 0, sample, key, **kwargs)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(
                scheduler, "set_timesteps"
            ):
                kwargs["num_inference_steps"] = num_inference_steps

            outputs_tuple = scheduler.step(
                residual, 0, sample, key, return_dict=False, **kwargs
            )

            recursive_check(outputs_tuple[0], outputs_dict.prev_sample)


@require_mlx
class MLXDDPMSchedulerTest(MLXSchedulerCommonTest):
    scheduler_classes = (MLXDDPMScheduler,)

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "variance_type": "fixed_small",
            "clip_sample": True,
        }

        config.update(**kwargs)
        return config

    def test_timesteps(self):
        # TODO: Check for timestep = 1
        for timesteps in [2, 5, 100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_betas(self):
        for beta_start, beta_end in zip(
            [0.0001, 0.001, 0.01, 0.1], [0.002, 0.02, 0.2, 2]
        ):
            self.check_over_configs(beta_start=beta_start, beta_end=beta_end)

    def test_schedules(self):
        for schedule in ["linear", "squaredcos_cap_v2"]:
            self.check_over_configs(beta_schedule=schedule)

    def test_variance_type(self):
        for variance in ["fixed_small", "fixed_large", "other"]:
            self.check_over_configs(variance_type=variance)

    def test_clip_sample(self):
        for clip_sample in [True, False]:
            self.check_over_configs(clip_sample=clip_sample)

    def test_time_indices(self):
        for t in [0, 500, 999]:
            self.check_over_forward(time_step=t)

    def test_variance(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        assert mx.sum(mx.abs(scheduler._get_variance(0) - 0.0)) < 1e-5
        assert mx.sum(mx.abs(scheduler._get_variance(487) - 0.00979)) < 1e-5
        assert mx.sum(mx.abs(scheduler._get_variance(999) - 0.02)) < 1e-5

    def test_full_loop_no_noise(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        num_trained_timesteps = len(scheduler)

        model = self.dummy_model()
        sample = self.dummy_sample_deter
        key1, key2 = mx.random.split(mx.random.key(0))

        for t in reversed(range(num_trained_timesteps)):
            # 1. predict noise residual
            residual = model(sample, t)

            # 2. predict previous mean of sample x_t-1
            output = scheduler.step(residual, t, sample, key1)
            pred_prev_sample = output.prev_sample
            key1, key2 = mx.random.split(key2)

            if t > 0:
                noise = self.dummy_sample_deter
                variance = scheduler._get_variance(t) ** (0.5) * noise
            
            sample = pred_prev_sample + variance
            sample = pred_prev_sample

        result_sum = mx.sum(mx.abs(sample))
        result_mean = mx.mean(mx.abs(sample))

        assert abs(result_sum - 263.2113) < 1e-1
        assert abs(result_mean - 0.342176) < 1e-3



@require_mlx
class MLXDDIMSchedulerTest(MLXSchedulerCommonTest):
    scheduler_classes = (MLXDDIMScheduler,)
    forward_default_kwargs = (("num_inference_steps", 50),)

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
        }

        config.update(**kwargs)
        return config

    def full_loop(self, **config):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(**config)
        scheduler = scheduler_class(**scheduler_config)

        num_inference_steps = 10

        model = self.dummy_model()
        sample = self.dummy_sample_deter

        scheduler.set_timesteps(num_inference_steps)

        for t in scheduler.timesteps:
            residual = model(sample, t)
            output = scheduler.step(residual, t, sample)
            sample = output.prev_sample

        return sample

    def check_over_configs(self, time_step=0, **config):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            sample, _ = self.dummy_sample
            residual = 0.1 * sample

            scheduler_config = self.get_scheduler_config(**config)
            scheduler = scheduler_class(**scheduler_config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
                new_scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            output = scheduler.step(residual, time_step, sample, **kwargs).prev_sample
            new_output = new_scheduler.step(residual, time_step, sample, **kwargs).prev_sample

            assert mx.sum(mx.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def test_from_save_pretrained(self):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            sample, _ = self.dummy_sample
            residual = 0.1 * sample

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
                new_scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            output = scheduler.step(residual, 1, sample, **kwargs).prev_sample
            new_output = new_scheduler.step(residual, 1, sample, **kwargs).prev_sample

            assert mx.sum(mx.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def check_over_forward(self, time_step=0, **forward_kwargs):
        kwargs = dict(self.forward_default_kwargs)
        kwargs.update(forward_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            sample, _ = self.dummy_sample
            residual = 0.1 * sample

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
                new_scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            output = scheduler.step(residual, time_step, sample, **kwargs).prev_sample
            new_output = new_scheduler.step(residual, time_step, sample, **kwargs).prev_sample

            assert mx.sum(mx.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def test_scheduler_outputs_equivalence(self):

        def recursive_check(tuple_object, dict_object):
            if isinstance(tuple_object, (List, Tuple)):
                for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object.values()):
                    recursive_check(tuple_iterable_value, dict_iterable_value)
            elif isinstance(tuple_object, Dict):
                for tuple_iterable_value, dict_iterable_value in zip(tuple_object.values(), dict_object.values()):
                    recursive_check(tuple_iterable_value, dict_iterable_value)
            elif tuple_object is None:
                return
            else:
                self.assertTrue(
                    mx.allclose(mx.nan_to_num(tuple_object), mx.nan_to_num(dict_object), atol=1e-5),
                    msg=(
                        "Tuple and dict output are not equal. Difference:"
                        f" {mx.max(mx.abs(tuple_object - dict_object))}. Tuple has `nan`:"
                        f" {mx.isnan(tuple_object).any()} and `inf`: {mx.isinf(tuple_object)}. Dict has"
                        f" `nan`: {mx.isnan(dict_object).any()} and `inf`: {mx.isinf(dict_object)}."
                    ),
                )

        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            sample, _ = self.dummy_sample
            residual = 0.1 * sample

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            outputs_dict = scheduler.step(residual, 0, sample, **kwargs)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            outputs_tuple = scheduler.step(residual, 0, sample, return_dict=False, **kwargs)

            recursive_check(outputs_tuple[0], outputs_dict.prev_sample)

    def test_step_shape(self):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            sample, _ = self.dummy_sample
            residual = 0.1 * sample

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            output_0 = scheduler.step(residual, 0, sample, **kwargs).prev_sample
            output_1 = scheduler.step(residual, 1, sample, **kwargs).prev_sample

            self.assertEqual(output_0.shape, sample.shape)
            self.assertEqual(output_0.shape, output_1.shape)

    def test_timesteps(self):
        for timesteps in [100, 500, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_steps_offset(self):
        for steps_offset in [0, 1]:
            self.check_over_configs(steps_offset=steps_offset)

        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(steps_offset=1)
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(5)
        assert mx.equal(scheduler.timesteps, mx.array([801, 601, 401, 201, 1])).all()

    def test_betas(self):
        for beta_start, beta_end in zip([0.0001, 0.001, 0.01, 0.1], [0.002, 0.02, 0.2, 2]):
            self.check_over_configs(beta_start=beta_start, beta_end=beta_end)

    def test_schedules(self):
        for schedule in ["linear", "squaredcos_cap_v2"]:
            self.check_over_configs(beta_schedule=schedule)

    def test_time_indices(self):
        for t in [1, 10, 49]:
            self.check_over_forward(time_step=t)

    def test_inference_steps(self):
        for t, num_inference_steps in zip([1, 10, 50], [10, 50, 500]):
            self.check_over_forward(time_step=t, num_inference_steps=num_inference_steps)

    def test_variance(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        assert mx.sum(mx.abs(scheduler._get_variance(0, 0) - 0.0)) < 1e-5
        assert mx.sum(mx.abs(scheduler._get_variance(420, 400) - 0.14771)) < 1e-5
        assert mx.sum(mx.abs(scheduler._get_variance(980, 960) - 0.32460)) < 1e-5
        assert mx.sum(mx.abs(scheduler._get_variance(0, 0) - 0.0)) < 1e-5
        assert mx.sum(mx.abs(scheduler._get_variance(487, 486) - 0.00979)) < 1e-5
        assert mx.sum(mx.abs(scheduler._get_variance(999, 998) - 0.02)) < 1e-5

    def test_full_loop_no_noise(self):
        sample = self.full_loop()

        result_sum = mx.sum(mx.abs(sample))
        result_mean = mx.mean(mx.abs(sample))

        assert abs(result_sum - 172.0067) < 1e-2
        assert abs(result_mean - 0.223967) < 1e-3

    def test_full_loop_with_set_alpha_to_one(self):
        # We specify different beta, so that the first alpha is 0.99
        sample = self.full_loop(set_alpha_to_one=True, beta_start=0.01)
        result_sum = mx.sum(mx.abs(sample))
        result_mean = mx.mean(mx.abs(sample))
        
        assert abs(result_sum - 149.8295) < 1e-2
        assert abs(result_mean - 0.1951) < 1e-3

    def test_full_loop_with_no_set_alpha_to_one(self):
        # We specify different beta, so that the first alpha is 0.99
        sample = self.full_loop(set_alpha_to_one=False, beta_start=0.01)
        result_sum = mx.sum(mx.abs(sample))
        result_mean = mx.mean(mx.abs(sample))

        assert abs(result_sum - 149.0784) < 1e-2
        assert abs(result_mean - 0.1941) < 1e-3

    def test_prediction_type(self):
        for prediction_type in ["epsilon", "sample", "v_prediction"]:
            self.check_over_configs(prediction_type=prediction_type)
