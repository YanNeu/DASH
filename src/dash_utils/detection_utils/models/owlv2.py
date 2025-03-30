import torch
from dataclasses import dataclass
from typing import Optional, Dict
from transformers import (Owlv2ForObjectDetection, AutoProcessor)

from .base_model import DetectionModel, DetectionProcessorFunction, DetectionBaseConfig
from .owlv2_differentiable_processor import OwlV2DifferentiableProcessorFunctionFunction

@dataclass
class OWLv2BaseConfig(DetectionBaseConfig):
    name: str = 'OWLv2-Base'
    model_id: str = "google/owlv2-base-patch16-ensemble"

@dataclass
class OWLv2LargeConfig(DetectionBaseConfig):
    name: str = 'OWLv2-Large'
    model_id: str = "google/owlv2-large-patch14-ensemble"


def get_owlv2_config_from_name(name: str) -> DetectionBaseConfig:
    if 'large' in name:
        return OWLv2LargeConfig()
    else:
        return OWLv2BaseConfig()

class OwlV2ProcessorFunction(DetectionProcessorFunction):
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

        inputs = self.processor(images=batch['image'], text=text, padding="longest", return_tensors="pt")
        return inputs

    def get_differentiable_processor(self):
        return OwlV2DifferentiableProcessorFunctionFunction(self.processor)


def get_owlv2(device: torch.device, name: Optional[str] = None, config: Optional[DetectionBaseConfig] = None):
    assert name is not None or config is not None
    assert not (name is not None and config is not None)

    if name is not None:
        config = get_owlv2_config_from_name(name)

    return OWLv2(device, config)


class OWLv2(DetectionModel):
    def __init__(self, device: torch.device, config: DetectionBaseConfig):
        model = Owlv2ForObjectDetection.from_pretrained(
            config.model_id,
            device_map=device,
        ).eval()

        processor = AutoProcessor.from_pretrained(config.model_id)

        for param in model.parameters():
            param.requires_grad = False

        processor_function = OwlV2ProcessorFunction(processor)
        super().__init__(model, processor, processor_function, config)

    def decode(self, inputs, outputs, box_threshold=0.0, text_threshold=0.0, reverse_mode=False):
        post_processed_outputs = self.processor.post_process_object_detection(
            outputs,
            threshold=0.0,
            target_sizes=None
        )

        results = []
        for output in post_processed_outputs:
            # results.append({"scores": score, "labels": label, "boxes": box})
            output_scores = output['scores']
            output_labels = output['labels']
            output_boxes = output['boxes']

            #if we want to use the object detector to enforce the object being present in the image
            if reverse_mode:
                pass_threshold = output_scores < text_threshold
            else:
                pass_threshold = output_scores >= text_threshold

            score = output_scores[pass_threshold]
            label = output_labels[pass_threshold]
            box = output_boxes[pass_threshold]
            results.append({"scores": score, "labels": label, "boxes": box})

        return post_processed_outputs



