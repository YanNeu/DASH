import torch
from dataclasses import dataclass
from typing import Optional, Dict
from .base_model import DifferentiableDetectionProcessorFunction
from transformers.image_processing_utils import get_size_dict
from transformers.image_utils import ChannelDimension, make_list_of_images, pil_torch_interpolation_mapping
from torchvision.transforms import functional as TF

class OwlV2DifferentiableProcessorFunctionFunction(DifferentiableDetectionProcessorFunction):
    def __init__(self, processor):
        super().__init__(processor)

    def __call__(self, batch: Dict, *args, **kwargs):
        text = []
        for prompt in batch['prompt']:
            if isinstance(prompt, str):
                prompt = [prompt]
            else:
                prompt = list(prompt)

            text.append(prompt)

        pil_image_list = []
        for image in batch['image']:
            pil_image_list.append(TF.to_pil_image(image))

        inputs = self.processor(images=pil_image_list, text=text, padding="longest", return_tensors="pt")
        differentiable_images = self._image_processing(batch['image'])
        assert inputs['pixel_values'].shape == differentiable_images.shape
        inputs['pixel_values'] = differentiable_images
        return inputs

    def _image_processing(self, images):
        do_rescale = False  # processor.do_rescale
        rescale_factor = 1.0
        do_pad = self.processor.image_processor.do_pad
        do_resize = self.processor.image_processor.do_resize
        do_normalize = self.processor.image_processor.do_normalize
        image_mean = self.processor.image_processor.image_mean
        image_std = self.processor.image_processor.image_std
        resample = self.processor.image_processor.resample
        size = self.processor.image_processor.size

        # All transformations expect numpy arrays.
        images = make_list_of_images(images)

        if do_resize:
            if do_resize:
                width, height = size["width"], size["height"]
                images = [
                    TF.resize(image, size=(width, height), interpolation=pil_torch_interpolation_mapping[resample])
                    for image in images
                ]

        if do_rescale:
            images = [
                image * rescale_factor
                for image in images
            ]

        if do_normalize:
            images = [
                TF.normalize(image, mean=image_mean, std=image_std)
                for image in images
            ]


        data = torch.stack(images, dim=0)
        return data

