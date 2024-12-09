# coding=utf-8
# Copyright 2024 Apple and The HuggingFace Inc. team.
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

import os
from typing import Union

import mlx.core as mx
import mlx.nn as nn

from mlx.nn.utils import tree_flatten, tree_unflatten
from huggingface_hub import create_repo, hf_hub_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    validate_hf_hub_args,
)
from requests import HTTPError

from .. import __version__
from ..utils import (
    CONFIG_NAME,
    MLX_WEIGHTS_NAME,
    SAFETENSORS_WEIGHTS_NAME,
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    PushToHubMixin,
    logging,
)
from .modeling_mlx_pytorch_utils import convert_pytorch_state_dict_to_mlx


logger = logging.get_logger(__name__)

# Inspired from https://github.com/ml-explore/mlx-examples/blob/main/stable_diffusion/stable_diffusion/model_io.py
class MLXModelMixin(nn.Module, PushToHubMixin):
    r"""
    Base class for all MLX models.

    [`MLXModelMixin`] takes care of storing the model configuration and provides methods for loading, downloading and
    saving models.

        - **config_name** ([`str`]) -- Filename to save a model to when calling [`~MLXModelMixin.save_pretrained`].
    """

    config_name = CONFIG_NAME
    _automatically_saved_args = ["_diffusers_version", "_class_name", "_name_or_path"]
    _mlx_internal_args = []

    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initialized under go here.
        """
        return cls(config, **kwargs)
    
    def __init__(self):
        super().__init__() 

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        dtype: mx.Dtype = mx.float32,
        *model_args,
        **kwargs,
    ):
        r"""
        Instantiate a pretrained MLX model from a pretrained model configuration.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                    - A string, the *model id* (for example `runwayml/stable-diffusion-v1-5`) of a pretrained model
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      using [`~MLXModelMixin.save_pretrained`].
            dtype (`mx.dtype`, *optional*, defaults to `mx.float32`):
                The data type of the computation. Can be one of `mx.float32` or `mx.float16`.
            model_args (sequence of positional arguments, *optional*):
                All remaining positional arguments are passed to the underlying model's `__init__` method.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the cached versions if they exist.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            from_pt (`bool`, *optional*, defaults to `False`):
                Load the model weights from a PyTorch checkpoint save file.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it is loaded) and initiate the model (for
                example, `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
                automatically loaded:

                    - If a configuration is provided with `config`, `kwargs` are directly passed to the underlying
                      model's `__init__` method (we assume all relevant updates to the configuration have already been
                      done).
                    - If a configuration is not provided, `kwargs` are first passed to the configuration class
                      initialization function [`~ConfigMixin.from_config`]. Each key of the `kwargs` that corresponds
                      to a configuration attribute is used to override said attribute with the supplied `kwargs` value.
                      Remaining keys that do not correspond to any configuration attribute are passed to the underlying
                      model's `__init__` function.

        Examples:

        ```python
        >>> from diffusers import MLXUNet2DConditionModel

        >>> # Download model and configuration from huggingface.co and cache.
        >>> model = MLXUNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> # Model was saved using *save_pretrained('./test/saved_model/')* (for example purposes, not runnable).
        >>> model = MLXUNet2DConditionModel.from_pretrained("./test/saved_model/")
        ```

        If you get the error message below, you need to finetune the weights for your downstream task:

        ```bash
        Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
        - conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
        You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
        ```
        """
        config = kwargs.pop("config", None)
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        from_pt = kwargs.pop("from_pt", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        
        user_agent = {
            "diffusers": __version__,
            "file_type": "model",
            "framework": "mlx",
        }

        # Load config if we don't provide one
        if config is None:
            config, unused_kwargs = cls.load_config(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                **kwargs,
            )

        model, model_kwargs = cls.from_config(
            config, dtype=dtype, return_unused_kwargs=True, **unused_kwargs
        )
        # Load model
        pretrained_path_with_subfolder = (
            pretrained_model_name_or_path
            if subfolder is None
            else os.path.join(pretrained_model_name_or_path, subfolder)
        )
        if os.path.isdir(pretrained_path_with_subfolder):
            if from_pt:
                if not os.path.isfile(
                    os.path.join(pretrained_path_with_subfolder, SAFETENSORS_WEIGHTS_NAME)
                ):
                    raise EnvironmentError(
                        f"Error no file named {SAFETENSORS_WEIGHTS_NAME} found in directory {pretrained_path_with_subfolder} "
                    )
                model_file = os.path.join(pretrained_path_with_subfolder, SAFETENSORS_WEIGHTS_NAME)
            elif os.path.isfile(
                os.path.join(pretrained_path_with_subfolder, MLX_WEIGHTS_NAME)
            ):
                # Load from a SAFETENSORS checkpoint
                model_file = os.path.join(
                    pretrained_path_with_subfolder, MLX_WEIGHTS_NAME
                )
            else:
                raise EnvironmentError(
                    f"Error no file named {MLX_WEIGHTS_NAME} or {SAFETENSORS_WEIGHTS_NAME} found in directory "
                    f"{pretrained_path_with_subfolder}."
                )
        else:
            try:
                model_file = hf_hub_download(
                    pretrained_model_name_or_path,
                    filename=SAFETENSORS_WEIGHTS_NAME if from_pt else MLX_WEIGHTS_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    subfolder=subfolder,
                    revision=revision,
                )

            except RepositoryNotFoundError:
                raise EnvironmentError(
                    f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier "
                    "listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a "
                    "token having permission to this repo with `token` or log in with `huggingface-cli "
                    "login`."
                )
            except RevisionNotFoundError:
                raise EnvironmentError(
                    f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for "
                    "this model name. Check the model page at "
                    f"'https://huggingface.co/{pretrained_model_name_or_path}' for available revisions."
                )
            except EntryNotFoundError:
                raise EnvironmentError(
                    f"{pretrained_model_name_or_path} does not appear to have a file named {SAFETENSORS_WEIGHTS_NAME}."
                )
            except HTTPError as err:
                raise EnvironmentError(
                    f"There was a specific connection error when trying to load {pretrained_model_name_or_path}:\n"
                    f"{err}"
                )
            except ValueError:
                raise EnvironmentError(
                    f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it"
                    f" in the cached files and it looks like {pretrained_model_name_or_path} is not the path to a"
                    f" directory containing a file named {SAFETENSORS_WEIGHTS_NAME} or {MLX_WEIGHTS_NAME}.\nCheckout your"
                    " internet connection or see how to run the library in offline mode at"
                    " 'https://huggingface.co/docs/transformers/installation#offline-mode'."
                )
            except EnvironmentError:
                raise EnvironmentError(
                    f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                    "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                    f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                    f"containing a file named {SAFETENSORS_WEIGHTS_NAME} or {MLX_WEIGHTS_NAME}."
                )
       
        # Used https://github.com/ml-explore/mlx/blob/main/python/mlx/nn/layers/base.py
        # Didn't use mlx.nn.Module.load_weights method directly for enhanced debugging within the diffusers library
        if from_pt:
            # Step 1: Get the pytorch file
            state = dict(mx.load(model_file))
            state = convert_pytorch_state_dict_to_mlx(state, model)
        else:
            state = dict(mx.load(model_file))
        
        required_params = dict(tree_flatten(model.parameters()))
        unexpected_keys = (state.keys() - required_params.keys())
        missing_keys = (required_params.keys() - state.keys())
    
        if missing_keys:
            logger.warning(
                f"The checkpoint {pretrained_model_name_or_path} is missing required keys: {missing_keys}. "
                "Make sure to call model.init_weights to initialize the missing weights."
            )
            cls._missing_keys = missing_keys

        for key in required_params.keys():
            v_new = state[key]
            if not isinstance(v_new, mx.array):
                raise ValueError(
                    "Expected mx.array but received "
                    f"{type(v_new)} for parameter {key}"
                ) 
            if key in required_params and required_params[key].shape != state[key].shape:
                raise ValueError(
                    f"Trying to load the pretrained weight for {key} failed: checkpoint has shape "
                    f"{state[key].shape} which is incompatible with the model shape {required_params[key].shape}. "
                )

        # remove unexpected keys to not be saved again
        for unexpected_key in unexpected_keys:
            del state[unexpected_key]

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
                " with another architecture."
            )
        else:
            logger.info(
                f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n"
            )

        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        else:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
                f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
                " training."
            ),
        state = [(k, v.astype(dtype)) for k,v in state.items()] 
        model.update(tree_unflatten(state))
        
        # Eval the weights due to lazy loading in SAFETENSORS
        mx.eval(model.parameters())
        model.eval()

        return model

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Save a model and its configuration file to a directory so that it can be reloaded using the
        [`~SAFETENSORSModelMixin.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save a model and its configuration file to. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful during distributed training and you
                need to call this function on all processes. In this case, set `is_main_process=True` only on the main
                process to avoid race conditions.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        if os.path.isfile(save_directory):
            logger.error(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )
            return

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            private = kwargs.pop("private", False)
            create_pr = kwargs.pop("create_pr", False)
            token = kwargs.pop("token", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = create_repo(
                repo_id, exist_ok=True, private=private, token=token
            ).repo_id

        model_to_save = self

        # Attach architecture to the config
        # Save the config
        if is_main_process:
            model_to_save.save_config(save_directory)

        # save model
        output_model_file = os.path.join(save_directory, MLX_WEIGHTS_NAME)
        self.save_weights(output_model_file)
        
        logger.info(f"Model weights saved in {output_model_file}")

        if push_to_hub:
            self._upload_folder(
                save_directory,
                repo_id,
                token=token,
                commit_message=commit_message,
                create_pr=create_pr,
            )
