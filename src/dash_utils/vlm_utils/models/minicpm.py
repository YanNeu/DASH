import torch
from dataclasses import dataclass
from typing import Optional
from copy import deepcopy
import json

from PIL.Image import Image
from transformers import (AutoModel, AutoProcessor, BatchEncoding)

from .base_model import Model, BaseConfig
from .processor import ProcessorFunction
from .hf_access_token import ACCESS_TOKEN


@dataclass
class MiniCPMConfig(BaseConfig):
    name: str = 'MiniCPM-V2.6'
    model_id: str =  "openbmb/MiniCPM-V-2_6"


def get_minicpm_config_from_name(vlm: str) -> MiniCPMConfig:
    if 'minicpm' in vlm.lower():
        return MiniCPMConfig()
    else:
        raise NotImplementedError()


class MiniCPMProcessorFunction(ProcessorFunction):
    def __init__(self, processor):
        super().__init__(processor)

    def __call__(self, batch, *args, **kwargs):
        #https://huggingface.co/openbmb/MiniCPM-V
        msgs_list = []
        for image, prompt in zip(batch['image'], batch['prompt']):
            msgs_list.append([{'role': 'user', 'content': [image, prompt]},])

        images_list = [None] * len(msgs_list)

        #from https://huggingface.co/openbmb/MiniCPM-V-2_6/blob/main/modeling_minicpmv.py
        prompts_lists = []
        input_images_lists = []
        for image, msgs in zip(images_list, msgs_list):
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            copy_msgs = deepcopy(msgs)

            assert len(msgs) > 0, "msgs is empty"
            system_prompt = ''

            if image is not None and isinstance(copy_msgs[0]["content"], str):
                copy_msgs[0]["content"] = [image, copy_msgs[0]["content"]]

            images = []
            for i, msg in enumerate(copy_msgs):
                role = msg["role"]
                content = msg["content"]
                assert role in ["user", "assistant"]
                if i == 0:
                    assert role == "user", "The role of first msg should be user"
                if isinstance(content, str):
                    content = [content]
                cur_msgs = []
                for c in content:
                    if isinstance(c, Image):
                        images.append(c)
                        cur_msgs.append("(<image>./</image>)")
                    elif isinstance(c, str):
                        cur_msgs.append(c)
                msg["content"] = "\n".join(cur_msgs)

            if system_prompt:
                sys_msg = {'role': 'system', 'content': system_prompt}
                copy_msgs = [sys_msg] + copy_msgs

            prompts_lists.append(self.processor.tokenizer.apply_chat_template(copy_msgs, tokenize=False, add_generation_prompt=True))
            input_images_lists.append(images)

        inputs = self.processor(
            prompts_lists,
            input_images_lists,
            max_slice_nums=None,
            use_image_id=None,
            return_tensors="pt",
            max_length=None
        )

        return inputs


def get_minicpm(device: torch.device, vlm: Optional[str] = None, config: Optional[MiniCPMConfig] = None):
    assert vlm is not None or config is not None
    assert not (vlm is not None and config is not None)

    if vlm is not None:
        config = get_minicpm_config_from_name(vlm)

    return MiniCPM(device, config)


class MiniCPM(Model):
    def __init__(self, device: torch.device, config: MiniCPMConfig):
        model = AutoModel.from_pretrained(config.model_id, trust_remote_code=True,
                                          attn_implementation='sdpa',
                                          device_map=device,
                                          torch_dtype=torch.bfloat16)  # sdpa or flash_attention_2, no eager
        model = model.eval()
        processor = AutoProcessor.from_pretrained(config.model_id, trust_remote_code=True)
        processor.tokenizer.padding_side = 'left'

        for param in model.parameters():
            param.requires_grad = False

        processor_function = MiniCPMProcessorFunction(processor)
        super().__init__(model, processor, processor_function, config)
