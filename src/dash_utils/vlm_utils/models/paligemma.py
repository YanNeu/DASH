import torch
from dataclasses import dataclass
from typing import Optional, Dict
from transformers import (PaliGemmaForConditionalGeneration, AutoProcessor)

from .base_model import Model, BaseConfig
from .processor import ProcessorFunction
from .hf_access_token import ACCESS_TOKEN

@dataclass
class PaliGemmaConfig(BaseConfig):
    name: str = 'Paligemma-3b'
    model_id: str =  "google/paligemma-3b-mix-224"
    revision: str = 'bfloat16'

@dataclass
class PaliGemma2_3BConfig(BaseConfig):
    name: str = 'Paligemma2-3b'
    model_id: str =  "google/paligemma2-3b-mix-448"
    revision: str = 'bfloat16'

@dataclass
class PaliGemma2_10BConfig(BaseConfig):
    name: str = 'Paligemma2-10b'
    model_id: str =  "google/paligemma2-10b-mix-448"
    revision: str = 'bfloat16'

def get_paligemma_config_from_name(vlm: str) -> PaliGemmaConfig:
    if 'paligemma2' in vlm.lower():
        if '3b' in vlm.lower():
            return PaliGemma2_3BConfig()
        elif '10b' in vlm.lower():
            return PaliGemma2_10BConfig()
        else:
            raise ValueError()
    elif 'paligemma' in vlm.lower():
        return PaliGemmaConfig()
    else:
        raise NotImplementedError()


class PaliGemmaProcessorFunction(ProcessorFunction):
    def __init__(self, processor):
        super().__init__(processor)

    def __call__(self, batch: Dict, padding: str = "longest", max_length: Optional[int] = None, *args, **kwargs):
        if 'target' in batch:
            inputs = self.processor(images=batch['image'], text=batch['prompt'], suffix=batch['target'], padding=padding, max_length=max_length, return_tensors="pt")
        else:
            inputs = self.processor(images=batch['image'], text=batch['prompt'], padding="longest", return_tensors="pt")
        return inputs

class PaliGemma2ProcessorFunction(ProcessorFunction):
    def __init__(self, processor):
        super().__init__(processor)

    def __call__(self, batch: Dict, padding: str = "longest", max_length: Optional[int] = None, *args, **kwargs):
        for i, text in enumerate(batch['prompt']):
            batch['prompt'][i] = f"answer en {text}"

        if 'target' in batch:
            inputs = self.processor(images=batch['image'], text=batch['prompt'], suffix=batch['target'], padding=padding, max_length=max_length, return_tensors="pt")
        else:
            inputs = self.processor(images=batch['image'], text=batch['prompt'], padding="longest", return_tensors="pt")
        return inputs


def get_paligemma(device: torch.device, vlm: Optional[str] = None, config: Optional[PaliGemmaConfig] = None):
    assert vlm is not None or config is not None
    assert not (vlm is not None and config is not None)

    if vlm is not None:
        config = get_paligemma_config_from_name(vlm)

    return PaliGemma(device, config)


class PaliGemma(Model):
    def __init__(self, device: torch.device, config: PaliGemmaConfig):
        if config.revision == 'bfloat16':
            dtype = torch.bfloat16
        else:
            raise NotImplementedError()

        if isinstance(config, PaliGemmaConfig):
            model = PaliGemmaForConditionalGeneration.from_pretrained(
                config.model_id,
                torch_dtype=dtype,
                device_map=device,
                revision=config.revision,
                token=ACCESS_TOKEN,
            ).eval()
        elif isinstance(config, (PaliGemma2_10BConfig,PaliGemma2_3BConfig)):
            model = PaliGemmaForConditionalGeneration.from_pretrained(
                config.model_id,
                torch_dtype=dtype,
                device_map=device,
                token=ACCESS_TOKEN,
            ).eval()

        processor = AutoProcessor.from_pretrained(config.model_id, token=ACCESS_TOKEN)
        processor.tokenizer.padding_side = 'left'

        for param in model.parameters():
            param.requires_grad = False

        if isinstance(config, PaliGemmaConfig):
            processor_function = PaliGemmaProcessorFunction(processor)
        elif isinstance(config, (PaliGemma2_10BConfig,PaliGemma2_3BConfig)):
            processor_function = PaliGemma2ProcessorFunction(processor)
        super().__init__(model, processor, processor_function, config)
