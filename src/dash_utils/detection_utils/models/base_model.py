import torch
from transformers import BatchEncoding, DataCollatorForLanguageModeling, BatchFeature
from typing import Dict
from abc import ABC
from dataclasses import dataclass

@dataclass
class DetectionBaseConfig(ABC):
    name: str = ''
    model_id: str =  ''


class DetectionProcessorFunction:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch: Dict, *args, **kwargs):
        raise NotImplementedError()

    def get_differentiable_processor(self):
        raise NotImplementedError()

class DifferentiableDetectionProcessorFunction(DetectionProcessorFunction):
    def __init__(self, processor):
        super().__init__(processor)

    #some models prefer Lower/UpperCase outputs
    def get_target_str(self, target_str):
        raise NotImplementedError()



class DetectionModel:
    def __init__(self, model, processor, processor_function: DetectionProcessorFunction, config: DetectionBaseConfig):
        self.model = model
        self.processor = processor
        self.processor_function = processor_function
        self.config = config

    @property
    def name(self):
        return self.config.name

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def device(self):
        return self.model.device

    @property
    def image_size(self):
        return self.model.config.vision_config.image_size

    def get_processor_function(self):
        return self.processor_function

    def __call__(self, inputs: BatchEncoding, *args, **kwargs):
        return_dict = self.model(**inputs)
        return return_dict

    def decode(self, inputs, outputs, box_threshold=0.0, text_threshold=0.0, reverse_mode=False):
        raise NotImplementedError()


