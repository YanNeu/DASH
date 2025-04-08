# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math

import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_invisible_watermark_available,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline, retrieve_timesteps, rescale_noise_cfg
from diffusers import DiffusionPipeline, DDIMScheduler, TCDScheduler, AutoencoderKL, EulerAncestralDiscreteScheduler, UNet2DConditionModel, LCMScheduler
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def make_loss_dict(loss_function, name, weight):
    loss_dict = {
        'loss': loss_function,
        'name': name,
        'weight': weight
    }
    return loss_dict

def load_diffusion_pipeline(diffusion_model, steps, diffusion_gpu, vae_gpu):
    if diffusion_model == 'sdxl_hyper_lora':
        eta = 0.
        resolution = 1024
        timesteps = None
        if steps == 1:
            lora_repo_name: str = "ByteDance/Hyper-SD"
            lora_chkpt_name: str = "Hyper-SDXL-1step-lora.safetensors"
        elif steps == 2:
            lora_repo_name: str = "ByteDance/Hyper-SD"
            lora_chkpt_name: str = "Hyper-SDXL-2steps-lora.safetensors"
        elif steps == 4:
            lora_repo_name: str = "ByteDance/Hyper-SD"
            lora_chkpt_name: str = "Hyper-SDXL-4steps-lora.safetensors"
        else:
            raise ValueError()

        pipe = StableDiffusionXLPipelineWithGrad.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
                                                                 variant="fp16").to(diffusion_gpu)
        pipe.load_lora_weights(hf_hub_download(lora_repo_name, lora_chkpt_name))
        pipe.fuse_lora()

        if steps == 1:
            pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
        else:
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    elif diffusion_model == 'sdxl_hyper':
        eta = 0.
        resolution = 1024
        timesteps = [800]
        assert steps == 1
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        repo_name = "ByteDance/Hyper-SD"
        ckpt_name = "Hyper-SDXL-1step-Unet.safetensors"
        # Load model.
        unet = UNet2DConditionModel.from_config(base_model_id, subfolder="unet").to(diffusion_gpu, torch.float16)
        unet.load_state_dict(load_file(hf_hub_download(repo_name, ckpt_name)))
        unet = unet.to(diffusion_gpu, torch.float16)
        pipe = StableDiffusionXLPipelineWithGrad.from_pretrained(base_model_id, unet=unet, torch_dtype=torch.float16,
                                                 variant="fp16").to(diffusion_gpu)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    elif diffusion_model == 'sdxl_turbo':
        eta = 0.
        resolution = 512
        timesteps = None
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
        )
        pipe = StableDiffusionXLPipelineWithGrad.from_pretrained("stabilityai/sdxl-turbo",
                                                                 vae=vae,
                                                                 torch_dtype=torch.float16,
                                                         variant="fp16").to(diffusion_gpu)
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
    else:
        raise ValueError()

    pipe.vae = pipe.vae.to(vae_gpu)

    return pipe, resolution, eta, timesteps

@dataclass
class OptimParams:
    steps: int = 25
    stepsize: float = 0.1
    optim: str = 'adam'
    adam_eps: float = 1e-8
    gradient_clipping: Optional[float] = 0.1
    sgd_scheduler: Optional[str] = None
    warmup_steps: Optional[int] = 2
    latent_lr_factor: float = 0.1
    conditioning_lr_factor: float = 1.0
    add_text_embeds_lr_factor: float = 1.0
    latent_norm_reg: Optional[float] = 1.0
    optimize_latents: bool = True
    optimize_conditioning: bool = True
    optimize_add_text_embeds: bool = True

def create_scheduler(optim, params: OptimParams = OptimParams()):
    if params.sgd_scheduler is None:
        def schedule(current_step: int):
            return 1.
    elif params.sgd_scheduler == 'cosine':
        def schedule(current_step: int):
            return 0.5 * (1. + math.cos(math.pi * current_step / params.steps))
    else:
        raise NotImplementedError()

    def warmup(current_step: int):
        if params.warmup_steps is not None and current_step < params.warmup_steps:  # current_step / warmup_steps * base_lr
            return float(current_step / params.warmup_steps)
        else:  # (num_training_steps - current_step) / (num_training_steps - warmup_steps) * base_lr
            return schedule(current_step)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup)
    return lr_scheduler

def create_optimizer(initial_latents=None, prompt_embeds=None, add_text_embeds=None, params: OptimParams = OptimParams()):
    optim_variables = []
    assert params.optimize_conditioning or params.optimize_latents or params.optimize_add_text_embeds
    if params.optimize_latents:
        assert initial_latents is not None
        initial_latents.requires_grad_(True)
        optim_variables.append({'params': [initial_latents], "lr": params.stepsize * params.latent_lr_factor})
    if params.optimize_conditioning:
        assert prompt_embeds is not None
        prompt_embeds.requires_grad_(True)
        optim_variables.append({'params': [prompt_embeds], "lr": params.stepsize * params.conditioning_lr_factor})
    if params.optimize_add_text_embeds:
        assert add_text_embeds is not None
        add_text_embeds.requires_grad_(True)
        optim_variables.append({'params': [add_text_embeds], "lr": params.stepsize * params.add_text_embeds_lr_factor})

    if params.optim == 'adam':
        #Adam eps leads to NAN in fp16 with default eps
        optim = torch.optim.Adam(optim_variables, lr=params.stepsize, eps=params.adam_eps)
    elif params.optim == 'adamw':
        optim = torch.optim.AdamW(optim_variables, lr=params.stepsize, eps=params.adam_eps)
    elif params.optim == 'sgd':
        optim = torch.optim.SGD(optim_variables, lr=params.stepsize, momentum=0.9)
    else:
        raise NotImplementedError()

    return optim


def freeze_params(params):
    for param in params:
        param.requires_grad = False


class StableDiffusionXLPipelineWithGrad(StableDiffusionXLPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion XL.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion XL uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([` CLIPTextModelWithProjection`]):
            Second frozen text-encoder. Stable Diffusion XL uses the text and pool portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):
            Whether the negative prompt embeddings shall be forced to always be set to 0. Also see the config of
            `stabilityai/stable-diffusion-xl-base-1-0`.
        add_watermarker (`bool`, *optional*):
            Whether to use the [invisible_watermark library](https://github.com/ShieldMnt/invisible-watermark/) to
            watermark output images. If not defined, it will default to True if the package is installed, otherwise no
            watermarker will be used.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->unet->vae"
    _optional_components = [
        "tokenizer",
        "tokenizer_2",
        "text_encoder",
        "text_encoder_2",
        "image_encoder",
        "feature_extractor",
    ]
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "add_text_embeds",
        "add_time_ids",
        "negative_pooled_prompt_embeds",
        "negative_add_time_ids",
    ]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            force_zeros_for_empty_prompt=force_zeros_for_empty_prompt,
            add_watermarker=add_watermarker
        )

        #gradient checkpointing
        self.unet.enable_gradient_checkpointing()
        self.vae.enable_gradient_checkpointing()
        self.text_encoder.eval()
        self.text_encoder_2.eval()
        self.unet.eval()
        self.vae.eval()

        #
        freeze_params(self.vae.parameters())
        freeze_params(self.unet.parameters())
        freeze_params(self.text_encoder.parameters())
        freeze_params(self.text_encoder_2.parameters())

        #

    @torch.no_grad()
    def __call__(
        self,
        losses_dict,
        targets_dict=None,
        optim_params: OptimParams = OptimParams(),
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.unet.device

        #move to gpu
        self.text_encoder = self.text_encoder.to(device)
        self.text_encoder_2 = self.text_encoder_2.to(device)

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )


        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )


        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        #offload to cpu
        self.text_encoder = self.text_encoder.to('cpu')
        self.text_encoder_2 = self.text_encoder_2.to('cpu')

        # 8.1 Apply denoising_end
        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self._num_timesteps = len(timesteps)

        #keep in fp32 or adam dies :/
        latents_initial = latents.detach().clone().to(torch.float32)
        prompt_embeds_initial = prompt_embeds.detach().clone().to(torch.float32)
        add_text_embeds_initial = add_text_embeds.detach().clone().to(torch.float32)
        optim = create_optimizer(initial_latents=latents_initial, prompt_embeds=prompt_embeds_initial,
                                 add_text_embeds=add_text_embeds_initial, params=optim_params)
        scheduler = create_scheduler(optim, optim_params)

        imgs_cpu = torch.zeros((optim_params.steps + 1, 3, height, width))
        loss_scores = {}


        for optim_step in range(optim_params.steps + 1):
            with torch.enable_grad():
                latents = latents_initial.to(self.unet.dtype)
                prompt_embeds = prompt_embeds_initial.to(self.unet.dtype)
                add_text_embeds = add_text_embeds_initial.to(self.unet.dtype)
                if generator is not None:
                    inner_generator = generator.clone_state()
                else:
                    inner_generator = None
                self.scheduler._step_index = None
                extra_step_kwargs = self.prepare_extra_step_kwargs(inner_generator, eta)
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                        added_cond_kwargs["image_embeds"] = image_embeds
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_dtype = latents.dtype
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                    if latents.dtype != latents_dtype:
                        if torch.backends.mps.is_available():
                            # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                            latents = latents.to(latents_dtype)

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                        add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                        negative_pooled_prompt_embeds = callback_outputs.pop(
                            "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                        )
                        add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                        negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)

                    # if XLA_AVAILABLE:
                    #     xm.mark_step()

                # make sure the VAE is in float32 mode, as it overflows in float16
                self.vae = self.vae.to(torch.float32)
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
                latents = latents.to(self.vae.device)

                # unscale/denormalize the latents
                # denormalize with the mean and std if available and not None
                has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
                has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
                if has_latents_mean and has_latents_std:
                    latents_mean = (
                        torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                    )
                    latents_std = (
                        torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                    )
                    latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
                else:
                    latents = latents / self.vae.config.scaling_factor

                image = self.vae.decode(latents, return_dict=False)[0]
                image = torch.clamp(0.5 * image + 0.5, 0.0, 1.0)

                imgs_cpu[optim_step] = image.detach().cpu()

                with torch.set_grad_enabled(optim_step < optim_params.steps):
                    loss = calculate_loss(image, losses_dict, targets_dict, device, loss_scores=loss_scores, augment=True)

                if optim_step < optim_params.steps:
                    loss_string = f'{optim_step} - loss {loss.detach().item():.3f}'

                    if optim_params.latent_norm_reg is not None and optim_params.optimize_latents:
                        reg = calculate_latent_norm_reg(latents_initial)
                        loss = loss + (optim_params.latent_norm_reg * reg).to(loss.device).to(loss.dtype)
                        loss_string += f' Latent norm reg {reg.detach().item():.3f}'

                    loss_string += f' Total loss {loss.detach().item():.3f}'
                    print(loss_string)

                    scheduler.step()
                    loss.backward()

                    if optim_params.gradient_clipping is not None:
                        for param_group in optim.param_groups:
                            torch.nn.utils.clip_grad_norm_(param_group['params'], optim_params.gradient_clipping)

                    optim.step()
                    optim.zero_grad()
                    del loss, image
                else:
                    print(f'Final - Total loss {loss.detach().item():.3f}')
                    break

        # Offload all models
        self.maybe_free_model_hooks()

        return_values = {
            'imgs': imgs_cpu,
            'loss_scores': loss_scores,
        }
        return return_values

#https://arxiv.org/pdf/2402.14017 eq10
def calculate_latent_norm_reg(latents, loss_scores=None):
    latents_flat = latents.view(latents.shape[0], -1)
    log_norm = torch.log(torch.linalg.vector_norm(latents_flat, dim=1))
    norm_sqr = torch.sum(latents**2)
    d = math.prod(latents.shape[1:])

    # log density is (d-1) log( norm ) - norm^2 / 2

    # since we minimize, we need to minimize the negative log density
    lhs = (d - 1) * log_norm
    rhs = norm_sqr / 2
    nll = torch.sum(rhs - lhs)

    # largest log density is achieved for d - 1 = norm^2 with value:
    max_density = 0.5 * (d-1) * (math.log(d-1) - 1)
    #normalize with this to get abs(nll) close to 1
    nll = nll / max_density


    if loss_scores is not None:
        if 'latent_norm_reg' not in loss_scores:
            loss_scores['latent_norm_reg'] = []

        loss_scores['latent_norm_reg'].append(nll.detach().item())

    return nll


def calculate_loss(image, losses_dict, targets_dict, device, loss_scores=None, augment=False):
    loss = torch.zeros((1,), device=device)
    if not isinstance(losses_dict, (tuple, list)):
        losses_dict = [losses_dict]
    for loss_dict in losses_dict:
        loss_function = loss_dict['loss']
        loss_name = loss_dict['name']
        loss_w = loss_dict['weight']

        loss_target = targets_dict[loss_name] if targets_dict is not None else None
        loss_value = loss_function(image, loss_target, augment=augment).to(device)
        loss = loss + loss_w * loss_value

        if loss_scores is not None:
            if loss_name not in loss_scores:
                loss_scores[loss_name] = []
            loss_scores[loss_name].append(loss_value.detach().item())
    if loss_scores is not None:
        if 'total' not in loss_scores:
            loss_scores['total'] = []

        loss_scores['total'].append(loss.detach().item())

    return loss
