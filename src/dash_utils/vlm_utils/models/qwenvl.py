import torch
from dataclasses import dataclass
from typing import Optional, Dict, Union, List
from transformers import (Qwen2VLForConditionalGeneration, AutoProcessor, DataCollatorForLanguageModeling, BatchFeature)

from .base_model import Model, BaseConfig
from .processor import ProcessorFunction
from .hf_access_token import ACCESS_TOKEN
from PIL.Image import Image, Resampling



@dataclass
class QwenVLConfig(BaseConfig):
    name: str = ''
    model_id: str =  ""

@dataclass
class Qwen2VL7BConfig(QwenVLConfig):
    name: str = 'Qwen2-VL-7B-Instruct'
    model_id: str =  "Qwen/Qwen2-VL-7B-Instruct"

@dataclass
class Qwen2VL72BConfig(QwenVLConfig):
    name: str = 'Qwen2-VL-72B-Instruct'
    model_id: str =  "Qwen/Qwen2-VL-72B-Instruct"


def get_qwen2_config_from_name(vlm: str) -> QwenVLConfig:
    if 'qwen2' in vlm.lower():
        if '72b' in vlm.lower():
            return Qwen2VL72BConfig()
        else:
            return Qwen2VL7BConfig()
    else:
        raise NotImplementedError()


class Qwen2VLProcessorFunction(ProcessorFunction):
    def __init__(self, processor, max_size=1024):
        super().__init__(processor)
        self.max_size = max_size

        processor_patch_size = self.processor.image_processor.patch_size * self.processor.image_processor.merge_size
        self.processor_patch_size = processor_patch_size
        self.processor.image_processor.max_pixels = processor_patch_size * (round(max_size / processor_patch_size)) * (
                    processor_patch_size ** 2)

    def make_conversations(self, batch, add_generation_prompt=None):
        conversations = []
        if 'target' in batch:
            add_generation_prompt = False if add_generation_prompt is None else add_generation_prompt
            for prompt, target in zip(batch['prompt'], batch['target']):
                template = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image"},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": target},
                        ],
                    }
                ]
                text_conversation = self.processor.apply_chat_template(template, add_generation_prompt=add_generation_prompt)
                conversations.append(text_conversation)
        else:
            conversations = []
            for prompt in batch['prompt']:
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                text_conversation = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                conversations.append(text_conversation)

        return conversations

    def __call__(self, batch: Dict, *args, **kwargs):
        conversations = self.make_conversations(batch)

        for i, image in enumerate(batch['image']):
            if min(image.size) < self.processor_patch_size:
                #edge case handling for the one out of a million images that is smaller than 28...
                width, height = image.size

                new_width = max(width, self.processor_patch_size)
                new_height = max(height, self.processor_patch_size)

                resized_image = image.resize((new_width, new_height), Resampling.BICUBIC)

                batch['image'][i] = resized_image

        inputs = self.processor(
            text=conversations, images=batch['image'], padding="longest", return_tensors="pt"
        )

        return inputs


def get_qwen2vl(device: torch.device, vlm: Optional[str] = None, config: Optional[Qwen2VL7BConfig] = None):
    assert vlm is not None or config is not None
    assert not (vlm is not None and config is not None)

    if vlm is not None:
        config = get_qwen2_config_from_name(vlm)

    return Qwen2VL(device, config)


class Qwen2VL(Model):
    def __init__(self, device: torch.device, config: QwenVLConfig):
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.model_id, torch_dtype="auto", device_map=device
        ).eval()
        processor = AutoProcessor.from_pretrained(config.model_id)
        processor.tokenizer.padding_side = 'left'

        for param in model.parameters():
            param.requires_grad = False

        processor_function = Qwen2VLProcessorFunction(processor)
        super().__init__(model, processor, processor_function, config)

    @property
    def image_size(self):
        return self.processor_function.max_size
