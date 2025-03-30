import torch
from dataclasses import dataclass
from typing import Optional, Dict, Union
from transformers import (MllamaForConditionalGeneration, AutoProcessor)

from .base_model import Model, BaseConfig
from .processor import ProcessorFunction
from .hf_access_token import ACCESS_TOKEN


@dataclass
class LlamaConfig(BaseConfig):
    name: str = ''
    model_id: str =  ''

@dataclass
class Llama32VisionInstructConfig(LlamaConfig):
    name: str = 'Llama-3.2-11B-Vision-Instruct'
    model_id: str =  "meta-llama/Llama-3.2-11B-Vision-Instruct"


@dataclass
class Llama32VisionConfig(LlamaConfig):
    name: str = 'Llama-3.2-11B-Vision'
    model_id: str =  "meta-llama/Llama-3.2-11B-Vision"


@dataclass
class Llama32VisionInstruct90BConfig(LlamaConfig):
    name: str = 'Llama-3.2-90B-Vision-Instruct'
    model_id: str =  "meta-llama/Llama-3.2-90B-Vision-Instruct"


@dataclass
class Llama32Vision90BConfig(LlamaConfig):
    name: str = 'Llama-3.2-90B-Vision'
    model_id: str =  "meta-llama/Llama-3.2-90B-Vision"


def get_llama_config_from_name(vlm: str) -> LlamaConfig:
    if '90b' in vlm.lower():
        if 'instruct' in vlm.lower():
            return Llama32VisionInstruct90BConfig()
        else:
            return Llama32Vision90BConfig()
    else:
        if 'instruct' in vlm.lower():
            return Llama32VisionInstructConfig()
        else:
            return Llama32VisionConfig()


class Llama32VisionInstructProcessorFunction(ProcessorFunction):
    def __init__(self, processor):
        super().__init__(processor)

    def __call__(self, batch: Dict, *args, **kwargs):
        conversations = []
        for prompt in batch['prompt']:
            conversation = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]}
            ]
            text_conversation = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            conversations.append(text_conversation)

        inputs = self.processor(images=batch['image'], text=conversations, padding="longest", add_special_tokens=False,  return_tensors="pt")
        return inputs

class Llama32VisionProcessorFunction(ProcessorFunction):
    def __init__(self, processor):
        super().__init__(processor)

    def __call__(self, batch: Dict, *args, **kwargs):
        conversations = []
        for prompt in batch['prompt']:
            text_conversation = f"<|image|><|begin_of_text|>{prompt}"
            conversations.append(text_conversation)

        inputs = self.processor(images=batch['image'], text=conversations, return_tensors="pt")
        return inputs


def get_llama(device: torch.device, vlm: Optional[str] = None, config: Optional[LlamaConfig] = None):
    assert vlm is not None or config is not None
    assert not (vlm is not None and config is not None)

    if vlm is not None:
        config = get_llama_config_from_name(vlm)

    return Llama32Vision(device, config)


class Llama32Vision(Model):
    def __init__(self, device: torch.device, config: LlamaConfig):
        model = MllamaForConditionalGeneration.from_pretrained(
            config.model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            token = ACCESS_TOKEN
        )
        processor = AutoProcessor.from_pretrained(config.model_id, token=ACCESS_TOKEN)
        #processor.tokenizer.padding_side = 'left'

        for param in model.parameters():
            param.requires_grad = False

        if isinstance(config, (Llama32VisionInstructConfig, Llama32VisionInstruct90BConfig)):
            processor_function = Llama32VisionInstructProcessorFunction(processor)
        elif isinstance(config, (Llama32VisionConfig, Llama32Vision90BConfig)):
            processor_function = Llama32VisionProcessorFunction(processor)
        else:
            raise NotImplementedError()

        super().__init__(model, processor, processor_function, config)
