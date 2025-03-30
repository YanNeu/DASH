from typing import Dict, List

import torch


class ProcessorFunction:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch: Dict, *args, **kwargs):
        raise NotImplementedError()

    def get_tokenizer(self):
        return self.processor.tokenizer


class DifferentiableProcessorMixin:
    def __init__(self, processor):
        self.processor = processor

    #some models prefer Lower/UpperCase outputs
    def get_target_str(self, target_str) -> str:
        raise NotImplementedError()

    #some models have different yes/Yes/yes. outputs so we can return all and search for the best
    def get_target_str_options(self, target_str) -> List[str]:
        return [self.get_target_str(target_str)]

    def create_masked_labels(self, batch, inputs, num_sos_tokens=1):
        tokenizer_target = self.processor.tokenizer(batch['target'], padding="longest")
        target_encoding = tokenizer_target['input_ids']

        inputs_ids = inputs['input_ids']

        labels = torch.zeros_like(inputs_ids)
        labels.fill_(-100)

        for i in range(len(labels)):
            # ignore start of sentence
            target_encoding_i = target_encoding[i][num_sos_tokens:]
            assert self.processor.tokenizer.decode(target_encoding_i) == batch['target'][i]
            target_found = -1
            for j in reversed(range(labels.shape[1])):
                if inputs_ids[i, j] == target_encoding_i[-1]:
                    target_found = j
                    break

            assert target_found != -1
            for j, target_encoding_j in enumerate(reversed(target_encoding_i)):
                labels[i, (target_found - j)] = target_encoding_j

            assert self.processor.tokenizer.decode(labels[i, labels[i] != - 100]) == batch['target'][i]

        return labels
