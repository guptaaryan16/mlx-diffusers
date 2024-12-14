# Copyright 2024 Apple and The HuggingFace Team. All rights reserved.
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

import warnings 
from typing import Dict, List, Optional, Union

import mlx.core as mx
import numpy as np
import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPTokenizer, CLIPTextModel
from ...models import MLXAutoencoderKL, MLXUNet2DConditionModel
from ...callbacks import MultiPipelineCallbacks, PipelineCallback

from ...schedulers import MLXEulerDiscreteScheduler, MLXDDPMScheduler
from ...utils import deprecate, logging, replace_example_docstring
from ..pipeline_mlx_utils import MLXDiffusionPipeline
from .pipeline_output import MLXStableDiffusionPipelineOutput
from PIL import Image
from tqdm.auto import tqdm

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

#TODO: Implement the model loading for MLXDiffusionPipeline through `pipeline_mlx_utils`
class MLXStableDiffusionPipeline:
    r"""
    MLX-based pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`MLXDiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`MLXAutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.MLXCLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`MLXUNet2DConditionModel`]):
            A `MLXUNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`MLXDDIMScheduler`], [`MLXLMSDiscreteScheduler`], [`MLXPNDMScheduler`], or
            [`MLXDPMSolverMultistepScheduler`].
        safety_checker ([`MLXStableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    def __init__(
        self,
        vae: MLXAutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: MLXUNet2DConditionModel,
        scheduler: Union[MLXEulerDiscreteScheduler, MLXDDPMScheduler],
        # safety_checker: MLXStableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        dtype: mx.Dtype = mx.float32,
    ):
        super().__init__()
        self.dtype = dtype
        self.vae=vae
        self.text_encoder=text_encoder
        self.tokenizer=tokenizer
        self.unet=unet
        self.scheduler=scheduler
        self.feature_extractor=feature_extractor
        self.safety_checker = None
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def prepare_inputs(self, prompt: Union[str, List[str]]):
        if not isinstance(prompt, (str, list)):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        return text_input.input_ids

    def _get_has_nsfw_concepts(self, features, params):
        has_nsfw_concepts = self.safety_checker(features, params)
        return has_nsfw_concepts

    def _run_safety_checker(self, images, safety_model_params, jit=False):
        # safety_model_params should already be replicated when jit is True
        pil_images = [Image.fromarray(image) for image in images]
        features = self.feature_extractor(pil_images, return_tensors="np").pixel_values

        has_nsfw_concepts = self._get_has_nsfw_concepts(
            features, safety_model_params
        )

        images_was_copied = False
        for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
            if has_nsfw_concept:
                if not images_was_copied:
                    images_was_copied = True
                    images = images.copy()

                images[idx] = np.zeros(images[idx].shape, dtype=np.uint8)  # black image

            if any(has_nsfw_concepts):
                warnings.warn(
                    "Potential NSFW content was detected in one or more images. A black image will be returned"
                    " instead. Try again with a different prompt and/or seed."
                )

        return images, has_nsfw_concepts

    def _generate(
        self,
        prompt_ids: np.array,
        prng_seed: int,
        num_inference_steps: int,
        height: int,
        width: int,
        guidance_scale: float,
        latents: Optional[mx.array] = None,
        neg_prompt_ids: Optional[mx.array] = None,
        callback=None,
        callback_steps=None, 
        callback_on_step_end = None
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        # get prompt text embeddings
        prompt_embeds = mx.array(self.text_encoder(prompt_ids)[0]).astype(self.dtype)

        # TODO: currently it is assumed `do_classifier_free_guidance = guidance_scale > 1.0`
        # implement this conditional `do_classifier_free_guidance = guidance_scale > 1.0`
        batch_size = prompt_ids.shape[0]

        max_length = prompt_ids.shape[-1]

        if neg_prompt_ids is None:
            uncond_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=max_length,
                return_tensors="np",
            ).input_ids
        else:
            uncond_input = neg_prompt_ids
        negative_prompt_embeds = mx.array(self.text_encoder(
            uncond_input
        )[0]).astype(self.dtype)
        context = mx.concatenate([negative_prompt_embeds, prompt_embeds])

        # Ensure model output will be `float32` before going into the scheduler
        guidance_scale = mx.array([guidance_scale], dtype=mx.float32)

        latents_shape = (
            batch_size,
            self.unet.config.in_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        if latents is None:
            mx.random.seed(prng_seed)
            latents = mx.random.normal(shape=latents_shape, dtype=mx.float32)
        else:
            if latents.shape != latents_shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}"
                )

        def loop_body(step, latents):
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            latents_input = mx.concatenate([latents]*2)

            t = mx.array(self.scheduler.timesteps, dtype=mx.int32)[step]
            timestep = mx.broadcast_to(t, shape=[latents_input.shape[0]])

            latents_input = self.scheduler.scale_model_input(
                latents_input, t
            )
            
            # predict the noise residual
            noise_pred = self.unet(
                latents_input,
                timestep,
                context
            ).sample
  
            # perform guidance
            noise_pred_uncond, noise_prediction_text = mx.split(noise_pred, 2, axis=0)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_prediction_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred, t, latents
            ).prev_sample
           
            return latents
                
        self.scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
        )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        num_warmup_steps = len(self.scheduler.timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            
            for i in range(num_inference_steps):
                latents = loop_body(i, latents)
                mx.eval(latents)
                # call the callback, if provided
                if i == len(self.scheduler.timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()


        # scale and decode the image latents with vae
        latents = 1 / self.vae.config.scaling_factor * latents

        image = self.vae.decode(
            latents
        ).sample
        
        image = mx.transpose(mx.clip((image / 2 + 0.5), 0, 1), (0, 2, 3, 1))
        return image

    def __call__(
        self,
        prompt_ids: np.array,
        prng_seed: int = 0,
        num_inference_steps: int = 50,
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: Union[float, mx.array] = 7.5,
        latents: mx.array = None,
        neg_prompt_ids: mx.array = None,
        return_dict: bool = True,
        jit: bool = False,
        callback_on_step_end=None,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            latents (`jnp.ndarray`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                array is generated by sampling using the supplied random `generator`.
            jit (`bool`, defaults to `False`):
                Whether to run `pmap` versions of the generation and safety scoring functions.

                    <Tip warning={true}>

                    This argument exists because `__call__` is not yet end-to-end pmap-able. It will be removed in a
                    future release.

                    </Tip>

            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.MLXStableDiffusionPipelineOutput`] instead of
                a plain tuple.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.MLXStableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.MLXStableDiffusionPipelineOutput`] is
                returned, otherwise a `tuple` is returned where the first element is a list with the generated images
                and the second element is a list of `bool`s indicating whether the corresponding generated image
                contains "not-safe-for-work" (nsfw) content.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        if isinstance(guidance_scale, float):
            # Convert to a tensor so each device gets a copy. Follow the prompt_ids for
            # shape information, as they may be sharded (when `jit` is `True`), or not.
            guidance_scale = mx.array([guidance_scale] * prompt_ids.shape[0])
            if len(prompt_ids.shape) > 2:
                # Assume sharded
                guidance_scale = guidance_scale[:, None]

        images = self._generate(
                prompt_ids,
                prng_seed,
                num_inference_steps,
                height,
                width,
                guidance_scale,
                latents,
                neg_prompt_ids,
                callback=callback,
                callback_steps=callback_steps 
        )

        # if self.safety_checker is not None:
        #     # safety_params = params["safety_checker"]
        #     # images_uint8_casted = (images * 255).round().astype("uint8")
        #     # num_devices, batch_size = images.shape[:2]

        #     # images_uint8_casted = np.asarray(images_uint8_casted).reshape(
        #     #     num_devices * batch_size, height, width, 3
        #     # )
        #     # images_uint8_casted, has_nsfw_concept = self._run_safety_checker(
        #     #     images_uint8_casted, safety_params, jit
        #     # )
        #     # images = np.asarray(images).copy()

        #     # # block images
        #     # if any(has_nsfw_concept):
        #     #     for i, is_nsfw in enumerate(has_nsfw_concept):
        #     #         if is_nsfw:
        #     #             images[i, 0] = np.asarray(images_uint8_casted[i])

        #     images = images.reshape(num_devices, batch_size, height, width, 3)
        # else:
        images = np.asarray(images)
        has_nsfw_concept = False

        if not return_dict:
            return (images, has_nsfw_concept)

        return MLXStableDiffusionPipelineOutput(
            images=images, nsfw_content_detected=has_nsfw_concept
        )
   
    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a NumPy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config = kwargs
