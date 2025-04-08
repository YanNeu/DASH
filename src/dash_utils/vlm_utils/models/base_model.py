import torch
from dataclasses import dataclass
from abc import ABC
from transformers import BatchEncoding
from .processor import ProcessorFunction

DEFAULT_GENERATION_KWARGS = {
    'max_new_tokens': 15,
    'do_sample': False,
    'num_beams': 1,
    'output_logits': True,
    'return_dict_in_generate': True
}

@dataclass
class BaseConfig(ABC):
    name: str = ''
    model_id: str =  ''

class Model:
    def __init__(self, model, processor, processor_function: ProcessorFunction, config: BaseConfig,
                 decode_cut_off_prompt: bool = True):
        self.model = model
        self.processor = processor
        self.processor_function = processor_function
        self.config = config
        self.decode_cut_off_prompt = decode_cut_off_prompt

    @property
    def name(self):
        return self.config.name

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def device(self):
        return self.model.device

    def get_processor_function(self) -> ProcessorFunction:
        return self.processor_function

    @property
    def image_size(self):
        return self.model.config.vision_config.image_size

    #calls underlying generate
    def __call__(self, inputs: BatchEncoding, generation_kwargs = None, *args, **kwargs):
        if generation_kwargs is None:
            generation_kwargs = DEFAULT_GENERATION_KWARGS

        return_dict = self.model.generate(**inputs, **generation_kwargs)
        return return_dict

    #forward without generate for loss computation
    def forward(self, inputs, *args, **kwargs):
        return self.model(**inputs)

    def decode(self, inputs, outputs):
        return_dict = {}
        full_response = self.processor.batch_decode(outputs['sequences'], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if self.decode_cut_off_prompt:
            input_length_padded = inputs['input_ids'].shape[1]
            response = self.processor.batch_decode(outputs['sequences'][:,input_length_padded:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        else:
            response = full_response

        return_dict['full_response'] = full_response
        return_dict['response'] = response

        if 'logits' in outputs:
            #logits come in format: ( logits_first_generated, logits_second_generated, ...)
            bs = len(response)
            return_dict['logits'] = []

            logits_cpu = [outputs['logits'][seq_i].detach().cpu() for seq_i in range(len(outputs['logits']))]

            for i in range(bs):
                return_dict['logits'].append( [logits_cpu[seq_i][i, :] for seq_i in range(len(outputs['logits'])) ] )

        for i in range(len(return_dict['logits'])):
            return_dict['logits'][i] = torch.stack(return_dict['logits'][i], dim=0)

        return return_dict

