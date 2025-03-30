import torch

from .paligemma import get_paligemma, get_paligemma_config_from_name, PaliGemmaProcessorFunction
from .llava import get_llava, get_llava_config_from_name, LLaVA16ProcessorFunction, LLaVA16LLamaProcessorFunction, LLaVA16VicunaProcessorFunction
from .processor import ProcessorFunction
from .qwenvl import get_qwen2vl, get_qwen2_config_from_name, Qwen2VLProcessorFunction
from .minicpm import get_minicpm, get_minicpm_config_from_name
from .blip2 import get_blip2, get_blip2_config_from_name
from .llama import get_llama, get_llama_config_from_name
from .base_model import Model, BaseConfig
from .prismatic_vlm import get_prismatic, get_prismatic_config_from_name

from .llava_differentiable_processor import LLaVA16DifferentiableProcessorFunction, LLaVA16LlamaDifferentiableProcessorFunction, LLaVA16VicunaDifferentiableProcessorFunction
from .paligemma_differentiable_processor import PaliGemmaDifferentiableProcessorFunctionFunction
from .qwenvl_differentiable_processor import Qwen2VLDifferentiableProcessorFunction

VLM_NAME_TO_CONSTRUCTOR = {
    'paligemma': get_paligemma,
    'llava': get_llava,
    'qwen2vl': get_qwen2vl,
    'minicpm': get_minicpm,
    'blip2': get_blip2,
    'llama32vision': get_llama,
    'llama3.2vision': get_llama,
    'prismatic': get_prismatic,
}

VLM_NAME_TO_CONFIG = {
    'paligemma': get_paligemma_config_from_name,
    'llava': get_llava_config_from_name,
    'qwen2vl': get_qwen2_config_from_name,
    'minicpm': get_minicpm_config_from_name,
    'blip2': get_blip2_config_from_name,
    'llama32vision': get_llama_config_from_name,
    'llama3.2vision': get_llama_config_from_name,
    'prismatic': get_prismatic_config_from_name,
}

def load_vlm_model(vlm: str, device: torch.device) -> Model:
    for name, constructor in VLM_NAME_TO_CONSTRUCTOR.items():
        if name in vlm.lower():
            return constructor(device, vlm=vlm)

    raise ValueError(f'VLM not recognized {vlm}')


def get_config_from_name(vlm: str) -> BaseConfig:
    for name, config_getter in VLM_NAME_TO_CONFIG.items():
        if name in vlm.lower():
            return config_getter(vlm)

    raise ValueError(f'VLM not recognized {vlm}')

def get_differentiable_processor_function(processor: ProcessorFunction):
    if isinstance(processor, PaliGemmaProcessorFunction):
        return PaliGemmaDifferentiableProcessorFunctionFunction(processor.processor)
    elif isinstance(processor, LLaVA16ProcessorFunction):
        return LLaVA16DifferentiableProcessorFunction(processor.processor)
    elif isinstance(processor, LLaVA16LlamaDifferentiableProcessorFunction):
        return LLaVA16LlamaDifferentiableProcessorFunction(processor.processor)
    elif isinstance(processor, LLaVA16VicunaProcessorFunction):
        return LLaVA16VicunaDifferentiableProcessorFunction(processor.processor)
    elif isinstance(processor, Qwen2VLProcessorFunction):
        return Qwen2VLDifferentiableProcessorFunction(processor.processor)
    else:
        raise ValueError(f'Processor type not recognized {processor}')