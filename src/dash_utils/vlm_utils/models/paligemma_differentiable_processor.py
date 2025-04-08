import torch
from dataclasses import dataclass
from typing import Optional, Dict
from .processor import DifferentiableProcessorMixin
from .paligemma import PaliGemmaProcessorFunction
from transformers.image_processing_utils import get_size_dict
from transformers.image_utils import ChannelDimension, make_list_of_images, pil_torch_interpolation_mapping
from torchvision.transforms import functional as TF

class PaliGemmaDifferentiableProcessorFunctionFunction(DifferentiableProcessorMixin,PaliGemmaProcessorFunction):
    def __init__(self, processor):
        DifferentiableProcessorMixin.__init__(self, processor)  # explicit calls without super
        PaliGemmaProcessorFunction.__init__(self, processor)

    def __call__(self, batch: Dict, *args, **kwargs):
        pil_image_list = []
        for image in batch['image']:
            pil_image_list.append(TF.to_pil_image(image))
        inputs = self.processor(images=pil_image_list, text=batch['prompt'], suffix=batch['target'], padding="longest", return_tensors="pt")
        differentiable_images = self._image_processing(batch['image'])
        assert inputs['pixel_values'].shape == differentiable_images.shape
        inputs['pixel_values'] = differentiable_images
        return inputs

    def get_target_str(self, target_str):
        if target_str in ['yes', 'Yes', 'YES']:
            target_str = 'yes'
        elif target_str in ['no', 'No', 'NO']:
            target_str = 'no'
        else:
            target_str = target_str
        return target_str

    def _image_processing(self, images):
        processor = self.processor.image_processor
        do_resize = processor.do_resize
        size = processor.size
        size = get_size_dict(size, param_name="size", default_to_square=False)
        resample = processor.resample
        do_rescale = False #the original work in [1,255] so rescaling is done with 1/255, not needed here
        rescale_factor = processor.rescale_factor
        do_normalize = processor.do_normalize
        image_mean = processor.image_mean
        image_std = processor.image_std
        do_convert_rgb = processor.do_convert_rgb
        data_format = ChannelDimension.FIRST
        images = make_list_of_images(images)


        for image in images:
            #CxHxW
            assert image.dim() == 3
            assert image.shape[0] == 3

        if do_resize:
            height, width = size["height"], size["width"]
            images = [
                TF.resize(image, size=(height, width), interpolation=pil_torch_interpolation_mapping[resample], antialias=True)
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
