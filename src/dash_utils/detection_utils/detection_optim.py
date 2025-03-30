import torch
import torch.nn as nn
import torch.nn.functional as F
from .models.base_model import DetectionModel, DetectionProcessorFunction, DifferentiableDetectionProcessorFunction


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cut_power=1.0):
        super().__init__()

        self.cut_size = cut_size
        self.cut_power = cut_power

    def forward(self, pixel_values, num_cutouts):
        sideY, sideX = pixel_values.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(num_cutouts):
            size = int(torch.rand([]) ** self.cut_power * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = pixel_values[:, :, offsety : offsety + size, offsetx : offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


def get_augmentation_function(size, num_cutouts, noise_sd, cut_power=1.0):
    cutout = MakeCutouts(size, cut_power=cut_power)

    def augment(x):
        if num_cutouts > 0:
            x = cutout(x, num_cutouts)
        else:
            x = x
        if noise_sd > 0:
            x = x + noise_sd * torch.randn_like(x)
        return x

    return augment


def get_detection_loss(prompt, model: DetectionModel, detection_threshold=0.1, reduction='max', num_cutouts=0,
                       cut_power=1.0, noise_sd=0, reverse_mode=False):
    processor_function: DetectionProcessorFunction = model.get_processor_function()
    diff_processor_function: DifferentiableDetectionProcessorFunction = processor_function.get_differentiable_processor()
    augment_function = get_augmentation_function(model.image_size, num_cutouts, noise_sd, cut_power=cut_power)

    def loss_function(image, targets, augment=True):
        image = image.to(model.device)

        if augment:
            augmented_batch = augment_function(image)
        else:
            augmented_batch = image

        batch = {
            'image': augmented_batch,
            'prompt': [prompt] * len(augmented_batch),
        }

        inputs = diff_processor_function(batch)
        inputs = inputs.to(model.device)
        outputs = model(inputs)

        decoded_bb_losses = model.decode(inputs, outputs, detection_threshold, detection_threshold, reverse_mode=reverse_mode)

        loss = torch.zeros((1,), device=model.device)
        for loss_bb in decoded_bb_losses:
            if loss_bb['scores'].numel() == 0:
                continue

            if reduction == 'mean':
                if reverse_mode:
                    loss +=  (1 / len(augmented_batch)) * torch.sum(-torch.log(loss_bb['scores']))
                else:
                    loss +=  (1 / len(augmented_batch)) * torch.sum(-torch.log(1. - loss_bb['scores']))
            elif reduction == 'max':
                if reverse_mode:
                    loss += (1/ len(augmented_batch)) * torch.min(-torch.log(loss_bb['scores']))
                else:
                    loss += (1/ len(augmented_batch)) * torch.max(-torch.log(1. - loss_bb['scores']))
            else:
                raise ValueError()


        return loss

    return loss_function
