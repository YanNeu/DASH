import warnings
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.folder import pil_loader
from typing import List, Dict
from datasets import Dataset
from transformers import BatchFeature
from PIL import Image
from .models import DetectionModel
from .models.base_model import DetectionProcessorFunction


def make_detection_dataset_dataloader(data_dicts: List[Dict], processor_function: DetectionProcessorFunction,
                                batch_size=32, num_workers=16, shuffle=False):
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
        try:
            img = Image.open(row['image_path'])
            img = img.convert('RGB')
        except Exception as e:
            print(f"Could not load {row['image_path']}: {e}")
            img = Image.new('RGB', (512, 512))

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
        num_workers=num_workers,
    )

    return dataset, dataloader



@torch.inference_mode()
def foward_detection_dataset(data_loader: DataLoader, model: DetectionModel, box_threshold=0.0, text_threshold=0.0):
    all_outputs = []
    for inputs in data_loader:
        inputs = inputs.to(model.device)
        outputs = model(inputs)
        outputs = model.decode(inputs, outputs, box_threshold=box_threshold, text_threshold=text_threshold)
        for output in outputs:
            for k,v in output.items():
                if isinstance(v, torch.Tensor):
                    output[k] = v.detach().cpu()
        all_outputs.extend(outputs)

    return all_outputs


@torch.inference_mode()
def get_max_detection_score(all_outputs):
    max_scores = []
    for output in all_outputs:
        if len(output['scores']) > 0:
            max_score = torch.max(output['scores']).item()
        else:
            max_score = 0.
        max_scores.append(max_score)

    return max_scores



