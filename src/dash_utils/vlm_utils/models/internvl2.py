import math
from typing import Optional, Dict, Union
from PIL import Image

import torch
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel, BatchEncoding

from .base_model import Model, BaseConfig, DEFAULT_GENERATION_KWARGS
from .processor import ProcessorFunction


@dataclass
class InternVL2Config(BaseConfig):
    name: str = ''
    model_id: str = ''

@dataclass
class InternVL21BConfig(InternVL2Config):
    name: str = 'InternVL2-1B'
    model_id: str =  "OpenGVLab/InternVL2-1B"

@dataclass
class InternVL22BConfig(InternVL2Config):
    name: str = 'InternVL2-2B'
    model_id: str =  "OpenGVLab/InternVL2-2B"

@dataclass
class InternVL24BConfig(InternVL2Config):
    name: str = 'InternVL2-4B'
    model_id: str =  "OpenGVLab/InternVL2-4B"

@dataclass
class InternVL28BConfig(InternVL2Config):
    name: str = 'InternVL2-8B'
    model_id: str =  "OpenGVLab/InternVL2-8B"

@dataclass
class InternVL226BConfig(InternVL2Config):
    name: str = 'InternVL2-26B'
    model_id: str =  "OpenGVLab/InternVL2-26B"

@dataclass
class InternVL240BConfig(InternVL2Config):
    name: str = 'InternVL2-40B'
    model_id: str =  "OpenGVLab/InternVL2-40B"

@dataclass
class InternVL276BConfig(InternVL2Config):
    name: str = 'InternVL2-76B'
    model_id: str =  "OpenGVLab/InternVL2-76B"

def get_internvl2_config_from_name(vlm: str) -> InternVL2Config:
    if '1b' in vlm.lower():
        return InternVL21BConfig()
    elif '2b' in vlm.lower():
        return InternVL22BConfig()
    elif '4b' in vlm.lower():
        return InternVL24BConfig()
    elif '8b' in vlm.lower():
        return InternVL28BConfig()
    elif '26b' in vlm.lower():
        return InternVL226BConfig()
    elif '40b' in vlm.lower():
        return InternVL240BConfig()
    elif '76b' in vlm.lower():
        return InternVL276BConfig()
    else:
        raise NotImplementedError()


class InternVL2ProcessorFunction(ProcessorFunction):
    def __init__(self, tokenizer, input_size=448, max_num=12):
        super().__init__(None)
        self.tokenizer = tokenizer
        self.input_size = input_size
        self.max_num = max_num

        self.transform = build_transform(input_size=self.input_size)

    def get_tokenizer(self):
        return self.tokenizer

    def __call__(self, batch: Dict, *args, **kwargs):
        assert len(batch['prompt']) == 1

        prompt =batch['prompt'][0]

        image = batch['image']
        images = dynamic_preprocess(image, image_size=self.input_size, use_thumbnail=True, max_num=self.max_num)
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        return BatchEncoding({'prompt': prompt, 'pixel_values': pixel_values})


class InternVL2(Model):
    def __init__(self, device: torch.device, config: InternVL2Config):
        model = AutoModel.from_pretrained(
            config.model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True).to(device)
        model.eval()

        for param in model.parameters():
            param.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True, use_fast=False)
        processor_function = InternVL2ProcessorFunction(self.tokenizer)
        super().__init__(model, None, processor_function, config)

    def __call__(self, inputs: BatchEncoding, generation_kwargs: None, *args, **kwargs):
        if generation_kwargs is None:
            generation_kwargs = DEFAULT_GENERATION_KWARGS

        response = self.model.chat(self.tokenizer, inputs['pixel_values'], inputs['prompt'], generation_kwargs)
        return response


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map
