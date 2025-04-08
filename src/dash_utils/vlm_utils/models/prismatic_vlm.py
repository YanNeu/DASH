import torch
from dataclasses import dataclass
from typing import Optional, Dict

from .base_model import Model, BaseConfig, DEFAULT_GENERATION_KWARGS
from .processor import ProcessorFunction
from .hf_access_token import ACCESS_TOKEN

from transformers import BatchEncoding, BatchFeature
from types import MethodType

@dataclass
class PrismaticConfig(BaseConfig):
    name: str = ''
    model_id: str =  ''

@dataclass
class PrismaticSigLIPConfig(PrismaticConfig):
    name: str = 'PrismaticSigLIP'
    model_id: str =  "siglip-224px+7b"

@dataclass
class PrismaticDinoV2Config(PrismaticConfig):
    name: str = 'PrismaticDinoV2'
    model_id: str =  "dinov2-224px+7b"

@dataclass
class PrismaticCLIPConfig(PrismaticConfig):
    name: str = 'PrismaticCLIP'
    model_id: str =  "clip-224px+7b"


def get_prismatic_config_from_name(vlm: str) -> PrismaticConfig:
    assert 'prismatic' in vlm.lower()
    if 'siglip' in vlm.lower():
        return PrismaticSigLIPConfig()
    elif 'dinov2' in vlm.lower():
        return PrismaticDinoV2Config()
    elif 'clip' in vlm.lower():
        return PrismaticCLIPConfig()
    else:
        raise NotImplementedError()


class PrismaticProcessor:
    def __init__(self, image_transform, tokenizer):
        self.image_transform = image_transform
        self.tokenizer = tokenizer

    def __call__(self, images, prompt_texts):
        tokenizer_out = self.tokenizer(prompt_texts, truncation=True, return_tensors="pt")
        pixel_values = self.image_transform(images)

        return BatchFeature(
            data={"pixel_values": pixel_values, "input_ids": tokenizer_out['input_ids'], 'attention_mask': tokenizer_out['attention_mask']},
        )

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

class PrismaticProcessorFunction(ProcessorFunction):
    def __init__(self, model, processor: PrismaticProcessor):
        super().__init__(processor)
        self.model = model

    def __call__(self, batch: Dict, padding: str = "longest", max_length: Optional[int] = None, *args, **kwargs):
        assert len(batch['image']) == 1
        image = batch['image'][0]
        user_prompt = batch['prompt'][0]
        prompt_builder = self.model.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=user_prompt)
        prompt_text = prompt_builder.get_prompt()

        inputs = self.processor(image, prompt_text)
        return inputs

def get_prismatic(device: torch.device, vlm: Optional[str] = None, config: Optional[PrismaticConfig] = None):
    assert vlm is not None or config is not None
    assert not (vlm is not None and config is not None)

    if vlm is not None:
        config = get_prismatic_config_from_name(vlm)

    return Prismatic(device, config)

#right now this only works with timm 0.9.x not timm 1.x.x, pray for prismatic fix
@torch.inference_mode()
def monkey_patch_generate(self, input_ids, pixel_values, attention_mask, **kwargs: str):
    # Prepare Inputs
    if isinstance(pixel_values, torch.Tensor):
        pixel_values = pixel_values[None, ...].to(self.device)
    elif isinstance(pixel_values, dict):
        pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
    else:
        raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

    # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
    self._supports_cache_class = False
    output = super(self.__class__, self).generate(
        input_ids=input_ids,# Shape: [1, seq]
        attention_mask=attention_mask,
        pixel_values=pixel_values,  # Shape: [1, 3, res, res] or Dict[str, Shape[1, 3, res, res]]
        **kwargs
    )

    return output


class Prismatic(Model):
    def __init__(self, device: torch.device, config: PrismaticConfig):
        #lazy import
        from prismatic import load

        model = load(config.model_id, hf_token=ACCESS_TOKEN)
        model.to(device, dtype=torch.bfloat16)

        model.generate = MethodType(monkey_patch_generate, model)

        image_transform = model.vision_backbone.image_transform
        tokenizer = model.llm_backbone.tokenizer

        model.vision_backbone.image_transform = None
        model.llm_backbone.tokenizer = None

        processor = PrismaticProcessor(image_transform, tokenizer)
        processor_function = PrismaticProcessorFunction(model, processor)

        super().__init__(model, processor, processor_function, config)

    @property
    def dtype(self):
        return torch.bfloat16



