import warnings

import PIL
import torch
from torch.utils.data import DataLoader
from PIL import Image
from typing import List, Dict, Optional
from datasets import Dataset
from transformers import BatchFeature
import torchvision.transforms.functional as TF

from .models.base_model import Model
from .models.processor import ProcessorFunction


def make_vlm_datasets_dataset_dataloader(data_dicts: List[Dict], processor_function: ProcessorFunction,
                                         batch_size=32, num_workers=4, shuffle=False):
    """
    data_dicts are dictionaries that are assumed to have the entries "image_path" and "prompt"
    """
    assert len(data_dicts) > 0

    #transform from List[dict] to dict[List]
    dict_first_data = {k: [] for k in data_dicts[0].keys()}
    for data_dict in data_dicts:
        for k, v in data_dict.items():
            dict_first_data[k].append(v)

    dataset = Dataset.from_dict(dict_first_data)
    dataset = dataset.to_iterable_dataset()

    def load_image(row):
        from PIL import PngImagePlugin
        LARGE_ENOUGH_NUMBER = 100
        PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 2)

        image_path = row['image_path']
        if isinstance(image_path, str):
            try:
                img = Image.open(image_path)
                img = img.convert('RGB')
            except Exception as e:
                print(f"Could not load {row['image_path']}: {e}")
                img = Image.new('RGB', (512, 512))
        else:
            raise ValueError()

        row['image'] = img
        return row

    dataset = dataset.map(load_image)

    def apply_processor_and_batch(rows):
        processed = processor_function(rows)
        new_row = {'batch': [processed]}
        return new_row

    dataset = dataset.map(apply_processor_and_batch, batched=True, batch_size=batch_size,
                          remove_columns=['image_path', 'prompt', 'image'])

    def collate_batches(batches):
        assert len(batches) == 1
        return batches[0]['batch']

    dataloader = DataLoader(
        dataset,
        collate_fn=collate_batches,
        shuffle=shuffle,
        batch_size=1,
        num_workers=1, # > 1 only works with sharding, but then order gets swapped..
    )

    return dataset, dataloader


@torch.inference_mode()
def forward_dataset(data_loader: DataLoader, model: Model, generation_kwargs: Optional[dict]=None):
    all_outputs = {}
    for inputs in data_loader:
        inputs = inputs.to(model.device, model.dtype)
        outputs = model(inputs, generation_kwargs=generation_kwargs)
        outputs = model.decode(inputs, outputs)

        for key, values in outputs.items():
            if key not in all_outputs:
                all_outputs[key] = []
            all_outputs[key].extend(values)

    return all_outputs

@torch.inference_mode()
def forward_in_memory_data(images, prompts, model: Model, generation_kwargs: Optional[dict]=None):
    all_outputs = {}
    processor = model.get_processor_function()
    for image, prompt in zip(images, prompts):
        batch = {'image': [image], 'prompt': [prompt]}
        inputs = processor(batch)
        inputs = inputs.to(model.device, model.dtype)
        outputs = model(inputs, generation_kwargs)
        outputs = model.decode(inputs, outputs)

        for key, values in outputs.items():
            if key not in all_outputs:
                all_outputs[key] = []
            all_outputs[key].extend(values)

    return all_outputs


@torch.inference_mode()
def get_yes_no_decisions_probabilities(return_dict, processor, k=20, accumulate='max'):
    decisions = []
    for response in return_dict['response']:
        if 'yes' in response.lower():
            decision = 1
        elif 'no' in response.lower():
            decision = 0
        else:
            decision = -1

        decisions.append(decision)

    yes_probs = []
    no_probs = []

    if accumulate == 'max':
        accu_f = lambda x, y: max(x,y)
    elif accumulate == 'sum':
        accu_f = lambda x, y: sum(x,y)
    else:
        raise ValueError()

    for logits in return_dict['logits']:
        yes_prob = -1
        no_prob = -1

        probs = torch.nn.functional.softmax(logits, dim=1)
        topk_results = torch.topk(probs, k=k, dim=1)
        for logit_i in range(len(probs)):
            for top_idx in topk_results.indices[logit_i]:
                decoded = processor.decode(top_idx)
                if 'yes' in decoded.lower():
                    yes_prob = accu_f(yes_prob, probs[logit_i, top_idx].item())
                elif 'no' in decoded.lower():
                    no_prob = accu_f(no_prob, probs[logit_i, top_idx].item())

        yes_probs.append(yes_prob)
        no_probs.append(no_prob)

    return_dict = {
        'decision': decisions,
        'yes_prob': yes_probs,
        'no_prob': no_probs
    }

    return return_dict

def get_standard_spurious_prompt(target_label, prompt_id=0):
    if prompt_id == 0:
        prompt = f'Does this image contain a {target_label}? Please answer only with yes or no.'
    elif prompt_id == 1:
        prompt = f'Is {target_label} in the image? Please answer only with yes or no.'
    elif prompt_id == 2:
        prompt = f'Is {target_label} visible in the image? Please answer only with yes or no.'
    elif prompt_id == 3:
        prompt = f'Does the image show a {target_label}? Please answer only with yes or no.'
    elif prompt_id == 4:
        prompt = f'Can you see a {target_label} in this image? Please answer only with yes or no.'
    elif prompt_id == 5:
        prompt = f'Is there a {target_label} present in the image? Please answer only with yes or no.'
    elif prompt_id == 6:
        prompt = f'Does this picture include a {target_label}? Please answer only with yes or no.'
    elif prompt_id == 7:
        prompt = f'Is a {target_label} depicted in this image? Please answer only with yes or no.'
    elif prompt_id == 8:
        prompt = f'Is a {target_label} shown in the image? Please answer only with yes or no.'
    elif prompt_id == 9:
        prompt = f'Does this image have a {target_label}? Please answer only with yes or no.'
    elif prompt_id == 10:
        prompt = f'Is a {target_label} present in the picture? Please answer only with yes or no.'
    elif prompt_id == 11:
        prompt = f'Does the image feature a {target_label}? Please answer only with yes or no.'
    elif prompt_id == 12:
        prompt = f'Is there a {target_label} in this image? Please answer only with yes or no.'
    elif prompt_id == 13:
        prompt = f'Does this image display a {target_label}? Please answer only with yes or no.'
    elif prompt_id == 14:
        prompt = f'Can a {target_label} be seen in the image? Please answer only with yes or no.'
    elif prompt_id == 15:
        prompt = f'Is a {target_label} observable in this image? Please answer only with yes or no.'
    elif prompt_id == 16:
        prompt = f'Does this image portray a {target_label}? Please answer only with yes or no.'
    elif prompt_id == 17:
        prompt = f'Is the {target_label} present in the image? Please answer only with yes or no.'
    elif prompt_id == 18:
        prompt = f'Does this photo include a {target_label}? Please answer only with yes or no.'
    elif prompt_id == 19:
        prompt = f'Is there any {target_label} in the image? Please answer only with yes or no.'
    else:
        raise ValueError(f'Prompt id unknown {prompt_id}')
    return prompt
