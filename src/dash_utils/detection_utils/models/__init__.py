import torch

from .base_model import DetectionModel, DetectionBaseConfig
from .grounding_dino import get_groundingdino, get_groundingdino_config_from_name
from .owlv2 import get_owlv2, get_owlv2_config_from_name

DETECTION_MODEL_NAME_TO_CONSTRUCTOR = {
    'groundingdino': get_groundingdino,
    'grounding_dino': get_groundingdino,
    'owlv2': get_owlv2,
}

DETECTION_MODEL_NAME_TO_CONFIG = {
    'groundingdino': get_groundingdino_config_from_name,
    'grounding_dino': get_groundingdino_config_from_name,
    'owlv2': get_owlv2_config_from_name,
}

def load_detection_model(detection_model: str, device: torch.device) -> DetectionModel:
    for name, constructor in DETECTION_MODEL_NAME_TO_CONSTRUCTOR.items():
        if name in detection_model.lower():
            return constructor(device, name=detection_model)

    raise ValueError(f'Detection Model not recognized {detection_model}')

def get_detection_config_from_name(detection_model: str) -> DetectionBaseConfig:
    for name, config_getter in DETECTION_MODEL_NAME_TO_CONFIG.items():
        if name in detection_model.lower():
            return config_getter(detection_model)

    raise ValueError(f'Detection Model not recognized {detection_model}')
