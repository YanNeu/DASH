import torch
from dataclasses import dataclass
from typing import Optional, Dict
from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor)
from transformers.models.grounding_dino import GroundingDinoForObjectDetection
from .base_model import DetectionModel, DetectionProcessorFunction, DetectionBaseConfig


@dataclass
class GroundingDinoConfig(DetectionBaseConfig):
    name: str = 'GroundingDino-Base'
    model_id: str =  "IDEA-Research/grounding-dino-base"


def get_groundingdino_config_from_name(name: str) -> GroundingDinoConfig:
    return GroundingDinoConfig()


class GroundingDinoProcessorFunction(DetectionProcessorFunction):
    def __init__(self, processor):
        super().__init__(processor)

    def __call__(self, batch: Dict, *args, **kwargs):
        # VERY important: text queries need to be lowercased + end with a dot
        text = []
        for prompt in batch['prompt']:
            if isinstance(prompt, str):
                prompt = [prompt]
            elif not isinstance(prompt, (list, tuple)):
                raise ValueError()

            text_prompt = '.'.join(prompt) + '.'
            text.append(text_prompt.lower())

        inputs = self.processor(images=batch['image'], text=text, padding="longest", return_tensors="pt")
        return inputs


def get_groundingdino(device: torch.device, name: Optional[str] = None, config: Optional[GroundingDinoConfig] = None):
    assert name is not None or config is not None
    assert not (name is not None and config is not None)

    if name is not None:
        config = get_groundingdino_config_from_name(name)

    return GroundingDino(device, config)


class GroundingDino(DetectionModel):
    def __init__(self, device: torch.device, config: GroundingDinoConfig):
        model = AutoModelForZeroShotObjectDetection.from_pretrained(config.model_id).to(device)
        processor = AutoProcessor.from_pretrained(config.model_id)

        for param in model.parameters():
            param.requires_grad = False

        processor_function = GroundingDinoProcessorFunction(processor)
        super().__init__(model, processor, processor_function, config)


    def decode(self, inputs, outputs, box_threshold=0.0, text_threshold=0.0, reverse_mode=False):
        if reverse_mode:
            raise NotImplementedError()

        post_processed_outputs = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=None
        )

        for output in post_processed_outputs:
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    output[key] = value

        return post_processed_outputs
