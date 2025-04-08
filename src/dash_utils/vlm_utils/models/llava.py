import torch
from dataclasses import dataclass
from typing import Optional, Union, Dict
from abc import ABC

from transformers import (LlavaForConditionalGeneration, LlavaNextForConditionalGeneration,
                          LlavaOnevisionForConditionalGeneration, AutoProcessor, )

from .base_model import Model, BaseConfig
from .processor import ProcessorFunction
from .hf_access_token import ACCESS_TOKEN


@dataclass
class LLaVAConfig(BaseConfig):
    name: str = ''
    model_id: str =  ''

@dataclass
class LLaVA15Config(LLaVAConfig):
    name: str = 'LLaVA-v1.5-7B'
    model_id: str = 'llava-hf/llava-1.5-7b-hf'

@dataclass
class LLaVA16VicunaConfig(LLaVAConfig):
    name: str = 'LLaVa-1.6-Vicuna'
    model_id: str = 'llava-hf/llava-v1.6-vicuna-7b-hf'

@dataclass
class LLaVA16MistralConfig(LLaVAConfig):
    name: str = 'LLaVa-1.6-Mistral'
    model_id: str = 'llava-hf/llava-v1.6-mistral-7b-hf'

@dataclass
class LLaVA16LlamaConfig(LLaVAConfig):
    name: str = 'LLaVa-1.6-Llama'
    model_id: str = 'llava-hf/llama3-llava-next-8b-hf'

@dataclass
class LLaVAOnevision7BConfig(LLaVAConfig):
    name: str = 'LLaVA-Onevision-7b'
    model_id: str =  'llava-hf/llava-onevision-qwen2-7b-ov-hf'

@dataclass
class LLaVAOnevision05BConfig(LLaVAConfig):
    name: str = 'LLaVA-Onevision-05b'
    model_id: str =  'llava-hf/llava-onevision-qwen2-0.5b-ov-hf'

def get_llava_config_from_name(vlm: str) -> LLaVAConfig:
    if '1.5' in vlm:
        return LLaVA15Config()
    elif '1.6'  in vlm:
        #Vicuna
        if 'vic' in vlm.lower():
            return LLaVA16VicunaConfig()
        #Mistral
        elif 'mis' in vlm.lower():
            return LLaVA16MistralConfig()
        #Llama
        elif 'llama' in vlm.lower():
            return LLaVA16LlamaConfig()
    elif 'onevision' in vlm.lower():
        if '7b' in vlm.lower():
            return LLaVAOnevision7BConfig()
        else:
            return LLaVAOnevision05BConfig()
    else:
        raise NotImplementedError()


def get_llava(device: torch.device, vlm: Optional[str] = None, config: Optional[LLaVAConfig] = None):
    assert vlm is not None or config is not None
    assert not (vlm is not None and config is not None)

    if vlm is not None:
        config = get_llava_config_from_name(vlm)

    if isinstance(config, LLaVA15Config):
        return LLaVA15(device, config)
    if isinstance(config, (LLaVA16VicunaConfig, LLaVA16MistralConfig, LLaVA16LlamaConfig)):
        return LLaVA16(device, config)
    if isinstance(config, (LLaVAOnevision05BConfig, LLaVAOnevision7BConfig)):
        return LLaVAOnevision(device, config)
    else:
        raise NotImplementedError()


#Generic Processor for the LLaVA version where chat_template is ACTUALLY correct
class LLaVAProcessorFunction(ProcessorFunction):
    def __init__(self, processor):
        super().__init__(processor)

    def __call__(self, batch, *args, **kwargs):
        conversations = []
        for prompt in batch['prompt']:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            text_conversation = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            conversations.append(text_conversation)

        inputs = self.processor(images=batch['image'], text=conversations, padding="longest", return_tensors="pt")
        return inputs

#################
#LLaVA 1.5 / Next
#################
class LLaVA15(Model):
    def __init__(self, device: torch.device, config: LLaVA15Config):
        model = LlavaForConditionalGeneration.from_pretrained(
            config.model_id,
            torch_dtype=torch.float16,
            attn_implementation = "flash_attention_2",
            token=ACCESS_TOKEN
        ).to(device)
        model.eval()
        processor = AutoProcessor.from_pretrained(config.model_id)

        for param in model.parameters():
            param.requires_grad = False

        processor_function = LLaVAProcessorFunction(processor)
        super().__init__(model, processor, processor_function, config)

#################
#LLaVA 1.6 / Next
#################
class LLaVA16ProcessorFunction(LLaVAProcessorFunction):
    def __init__(self, processor):
        super().__init__(processor)

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
            for prompt in batch['prompt']:
                add_generation_prompt = True if add_generation_prompt is None else add_generation_prompt
                template = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image"},
                        ],
                    },
                ]
                text_conversation = self.processor.apply_chat_template(template, add_generation_prompt=add_generation_prompt)
                conversations.append(text_conversation)

        return conversations


    @property
    def image_size(self):
        #The resolution for square inputs
        return 672

#Wrong template in huggingface...
class LLaVA16VicunaProcessorFunction(LLaVA16ProcessorFunction):
    def __init__(self, processor):
        super().__init__(processor)

    def make_conversations(self, batch, add_generation_prompt=None):
        conversations = []
        if 'target' in batch:
            for prompt, target in zip(batch['prompt'], batch['target']):
                text_conversation = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{prompt} ASSISTANT: {target}"
                conversations.append(text_conversation)
        else:
            for prompt in batch['prompt']:
                text_conversation = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{prompt} ASSISTANT:"
                conversations.append(text_conversation)

        return conversations

    def __call__(self, batch, *args, **kwargs):
        conversations = self.make_conversations(batch)
        inputs = self.processor(images=batch['image'], text=conversations, padding="longest", return_tensors="pt")
        return inputs

#no system prompt in chat template
class LLaVA16LLamaProcessorFunction(LLaVA16ProcessorFunction):
    def __init__(self, processor):
        super().__init__(processor)

    #SYSTEM_PROMPT = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."

    def make_conversations(self, batch, add_generation_prompt=None):
        conversations = []

        if 'target' in batch:
            add_generation_prompt = False if add_generation_prompt is None else add_generation_prompt
            for prompt, target in zip(batch['prompt'], batch['target']):
                template = [
                    # {
                    #     "role": "system",
                    #     "content": [
                    #         {"type": "text", "text": self.SYSTEM_PROMPT},
                    #     ],
                    # },
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
            for prompt in batch['prompt']:
                add_generation_prompt = True if add_generation_prompt is None else add_generation_prompt
                template = [

                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image"},
                        ],
                    },
                ]
                text_conversation = self.processor.apply_chat_template(template, add_generation_prompt=add_generation_prompt)
                conversations.append(text_conversation)

        return conversations

    def __call__(self, batch: Dict, *args, **kwargs):
        conversations = self.make_conversations(batch)
        inputs = self.processor(images=batch['image'], text=conversations, padding="longest", return_tensors="pt")
        return inputs

class LLaVA16(Model):
    def __init__(self, device: torch.device, config: Union[LLaVA16VicunaConfig,LLaVA16MistralConfig, LLaVA16LlamaConfig]):
        model = LlavaNextForConditionalGeneration.from_pretrained(
            config.model_id,
            torch_dtype=torch.float16,
            attn_implementation = "flash_attention_2",
            token = ACCESS_TOKEN,
        ).to(device)
        processor = AutoProcessor.from_pretrained(config.model_id, padding_side='left')

        for param in model.parameters():
            param.requires_grad = False

        if isinstance(config, LLaVA16VicunaConfig):
            processor_function = LLaVA16VicunaProcessorFunction(processor)
        elif isinstance(config, LLaVA16MistralConfig):
            processor_function = LLaVA16ProcessorFunction(processor)
        elif isinstance(config, LLaVA16LlamaConfig):
            processor_function = LLaVA16LLamaProcessorFunction(processor)
        else:
            raise NotImplementedError()

        super().__init__(model, processor, processor_function, config)

    @property
    def image_size(self):
        #The resolution for square inputs
        return 672

class LLaVAOnevisionProcessorFunction(LLaVAProcessorFunction):
    def __init__(self, processor):
        super().__init__(processor)

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
                text_conversation = self.processor.apply_chat_template(template, add_generation_prompt=add_generation_prompt, )
                conversations.append(text_conversation)
        else:
            for prompt in batch['prompt']:
                add_generation_prompt = True if add_generation_prompt is None else add_generation_prompt
                template = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image"},
                        ],
                    },
                ]
                text_conversation = self.processor.apply_chat_template(template, add_generation_prompt=add_generation_prompt)
                conversations.append(text_conversation)

        return conversations

    def __call__(self, batch: Dict, *args, **kwargs):
        conversations = self.make_conversations(batch)
        inputs = self.processor(images=batch['image'], text=conversations, padding="longest", return_tensors="pt")
        return inputs

class LLaVAOnevision(Model):
    def __init__(self, device: torch.device, config: LLaVAOnevision05BConfig):
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            config.model_id,
            torch_dtype=torch.float16,
            attn_implementation = "flash_attention_2",
            token=ACCESS_TOKEN
        ).to(device)
        model.eval()
        processor = AutoProcessor.from_pretrained(config.model_id)

        processor.tokenizer.padding_side = "left"
        for param in model.parameters():
            param.requires_grad = False

        processor_function = LLaVAOnevisionProcessorFunction(processor)
        super().__init__(model, processor, processor_function, config)

