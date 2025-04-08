import torch
from dataclasses import dataclass
from typing import Optional, Dict
from transformers import (Blip2Processor, Blip2ForConditionalGeneration,
                          InstructBlipProcessor, InstructBlipForConditionalGeneration)

from .base_model import Model, BaseConfig
from .processor import ProcessorFunction
from .hf_access_token import ACCESS_TOKEN
from abc import ABC

@dataclass
class Blip2Config(BaseConfig):
    name: str = ''
    model_id: str =  ''

@dataclass
class Blip2T5XXLConfig(Blip2Config):
    name: str = 'Blip2-T5-XXL'
    model_id: str =  "Salesforce/blip2-flan-t5-xxl"

@dataclass
class Blip2InstructT5XXLConfig(Blip2Config):
    name: str = 'Blip2-T5-XXL-Instruct'
    model_id: str =  "Salesforce/instructblip-flan-t5-xxl"

@dataclass
class Blip2InstructVicunaConfig(Blip2Config):
    name: str = 'Blip2-Vicuna-Instruct'
    model_id: str =  "Salesforce/instructblip-vicuna-7b"

def get_blip2_config_from_name(vlm: str) -> Blip2Config:
    if 'instruct' in vlm.lower():
        if 't5' in vlm.lower():
            return Blip2InstructT5XXLConfig()
        elif 'vic' in vlm.lower():
            return Blip2InstructVicunaConfig()
    elif 't5' in vlm.lower():
        return Blip2T5XXLConfig()
    else:
        raise NotImplementedError()

class Blip2ProcessorFunction(ProcessorFunction):
    def __init__(self, processor):
        super().__init__(processor)

    def __call__(self, batch: Dict, *args, **kwargs):
        inputs = self.processor(images=batch['image'], text=batch['prompt'], padding="longest", return_tensors="pt")
        return inputs


def get_blip2(device: torch.device, vlm: Optional[str] = None, config: Optional[Blip2Config] = None):
    assert vlm is not None or config is not None
    assert not (vlm is not None and config is not None)

    if vlm is not None:
        config = get_blip2_config_from_name(vlm)

    if isinstance(config, Blip2T5XXLConfig):
        return Blip2(device, config)
    elif isinstance(config, (Blip2InstructVicunaConfig,Blip2InstructT5XXLConfig)):
        return InstructBlip(device, config)


class Blip2(Model):
    def __init__(self, device: torch.device, config: Blip2Config):
        processor = Blip2Processor.from_pretrained(config.model_id)
        model = Blip2ForConditionalGeneration.from_pretrained(config.model_id,
                                                              torch_dtype=torch.float16,
                                                              device_map=device,
                                                              ).eval()
        processor.tokenizer.padding_side = 'left'

        for param in model.parameters():
            param.requires_grad = False

        processor_function = Blip2ProcessorFunction(processor)
        super().__init__(model, processor, processor_function, config, decode_cut_off_prompt=False)



class InstructBlip(Model):
    def __init__(self, device: torch.device, config: Blip2Config):
        model = InstructBlipForConditionalGeneration.from_pretrained(config.model_id,
                                                                     torch_dtype=torch.float16,
                                                                     device_map=device,
                                                                     ).eval()
        processor = InstructBlipProcessor.from_pretrained(config.model_id)
        processor.tokenizer.padding_side = 'left'

        for param in model.parameters():
            param.requires_grad = False

        processor_function = Blip2ProcessorFunction(processor)
        super().__init__(model, processor, processor_function, config, decode_cut_off_prompt=False)
