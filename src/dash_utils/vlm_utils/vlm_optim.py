import torch
import torch.nn as nn
import torch.nn.functional as F
from .models.base_model import Model
from .models.processor import ProcessorFunction, DifferentiableProcessorMixin
from .models import get_differentiable_processor_function


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


def get_vlm_loss(prompt, target_str, model: Model, num_cutouts=0, cut_power=1.0, noise_sd=0):
    processor_function: ProcessorFunction = model.get_processor_function()
    diff_processor_function = get_differentiable_processor_function(processor_function)
    augment_function = get_augmentation_function(model.image_size, num_cutouts, noise_sd, cut_power=cut_power)

    target_str = diff_processor_function.get_target_str(target_str)

    def loss_function(image, targets, augment=True):
        image = image.to(model.device)

        if augment:
            augmented_batch = augment_function(image)
        else:
            augmented_batch = image

        batch = {
            'image': augmented_batch,
            'prompt': [prompt] * len(augmented_batch),
            'target': [target_str] * len(augmented_batch),
        }

        inputs = diff_processor_function(batch)
        inputs = inputs.to(model.device)
        return_dict = model.forward(inputs)
        loss = return_dict['loss']
        return loss

    return loss_function

class VLMLossFunctionWithTargetOptions:
    def __init__(self, model, prompt, target_str, augment_function, diff_processor_function):
        self.model = model
        self.prompt = prompt
        self.target_str = target_str
        self.augment_function = augment_function
        self.diff_processor_function: DifferentiableProcessorMixin = diff_processor_function
        self.target_option = None

    def _find_best_target_option(self, image, targets):
        if self.target_option is None:
            target_options = self.diff_processor_function.get_target_str_options(self.target_str)
            if len(target_options) == 1:
                self.target_option = target_options[0]
            else:
                with torch.inference_mode():
                    target_opt_losses = torch.zeros(len(target_options))
                    for i, target_opt in enumerate(target_options):
                        batch = {
                            'image': image,
                            'prompt': [self.prompt],
                            'target': [target_opt],
                        }

                        inputs = self.diff_processor_function(batch)
                        inputs = inputs.to(self.model.device)
                        return_dict = self.model.forward(inputs)
                        loss = return_dict['loss']
                        target_opt_losses[i] = loss.item()

                    best_loss_idx = torch.argmin(target_opt_losses).item()
                    self.target_option = target_options[best_loss_idx]

        return self.target_option

    def __call__(self, image, targets, augment=True):
        image = image.to(self.model.device)

        if augment:
            augmented_batch = self.augment_function(image)
        else:
            augmented_batch = image

        target_str = self._find_best_target_option(image, targets)
        batch = {
            'image': augmented_batch,
            'prompt': [self.prompt] * len(augmented_batch),
            'target': [target_str] * len(augmented_batch),
        }

        inputs = self.diff_processor_function(batch)
        inputs = inputs.to(self.model.device)
        return_dict = self.model.forward(inputs)
        loss = return_dict['loss']
        return loss


def get_vlm_loss_target_options(prompt, target_str, model: Model, num_cutouts=0, cut_power=1.0, noise_sd=0):
    processor_function: ProcessorFunction = model.get_processor_function()
    diff_processor_function = get_differentiable_processor_function(processor_function)
    augment_function = get_augmentation_function(model.image_size, num_cutouts, noise_sd, cut_power=cut_power)

    loss_function = VLMLossFunctionWithTargetOptions(model, prompt, target_str, augment_function, diff_processor_function)

    return loss_function
