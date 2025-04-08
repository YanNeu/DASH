import torch
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Dict, List, Union, Tuple, Iterable

from transformers import LlavaNextImageProcessor
from .processor import DifferentiableProcessorMixin
from .llava import LLaVA16LLamaProcessorFunction, LLaVA16VicunaProcessorFunction, LLaVA16ProcessorFunction

from transformers.image_processing_utils import get_size_dict, select_best_resolution, BatchFeature
from transformers.image_utils import (ChannelDimension,
                                      make_list_of_images,
                                      pil_torch_interpolation_mapping,
                                      OPENAI_CLIP_MEAN,
                                      OPENAI_CLIP_STD,
                                      ChannelDimension,
                                      ImageInput,
                                      PILImageResampling,)
from transformers.models.llava_next.image_processing_llava_next import divide_to_patches
from transformers.utils import TensorType, is_vision_available, logging

from torchvision.transforms import functional as TF

from transformers.image_transforms import (
    PaddingMode,
    get_resize_output_image_size,
)

class LLaVA16DifferentiableProcessorFunction(DifferentiableProcessorMixin, LLaVA16ProcessorFunction):
    def __init__(self, processor):
        super().__init__(processor)
        self.diff_image_processor = LlavaNextDifferentiableImageProcessor.from_llava_next_image_processor(
            self.processor.image_processor)

    def get_target_str(self, target_str):
        if target_str in ['yes', 'Yes', 'YES']:
            target_str = 'Yes'
        elif target_str in ['no', 'No', 'NO']:
            target_str = 'No'
        else:
            target_str = target_str
        return target_str

    def __call__(self, batch: Dict, *args, **kwargs):
        pil_image_list = []
        for image in batch['image']:
            pil_image_list.append(TF.to_pil_image(image))

        conversations = self.make_conversations(batch)
        inputs = self.processor(images=pil_image_list, text=conversations, padding="longest", return_tensors="pt")

        labels = self.create_masked_labels(batch, inputs)
        inputs['labels'] = labels

        differentiable_images = self.diff_image_processor.preprocess(batch['image'])
        diff_pixel_values = differentiable_images['pixel_values']
        assert inputs['pixel_values'].shape == diff_pixel_values.shape
        inputs['pixel_values'] = diff_pixel_values
        return inputs

class LLaVA16VicunaDifferentiableProcessorFunction(DifferentiableProcessorMixin, LLaVA16VicunaProcessorFunction):
    def __init__(self, processor):
        super().__init__(processor)
        self.diff_image_processor = LlavaNextDifferentiableImageProcessor.from_llava_next_image_processor(self.processor.image_processor)

    def get_target_str(self, target_str):
        if target_str in ['yes', 'Yes', 'YES']:
            target_str = 'Yes'
        elif target_str in ['no', 'No', 'NO']:
            target_str = 'No'
        else:
            target_str = target_str
        return target_str

    def __call__(self, batch: Dict, *args, **kwargs):
        pil_image_list = []
        for image in batch['image']:
            pil_image_list.append(TF.to_pil_image(image))

        conversations = self.make_conversations(batch)
        inputs = self.processor(images=pil_image_list, text=conversations, padding="longest", return_tensors="pt")

        labels = self.create_masked_labels(batch, inputs)
        inputs['labels'] = labels

        differentiable_images = self.diff_image_processor.preprocess(batch['image'])
        diff_pixel_values = differentiable_images['pixel_values']
        assert inputs['pixel_values'].shape == diff_pixel_values.shape
        inputs['pixel_values'] = diff_pixel_values
        return inputs


class LLaVA16LlamaDifferentiableProcessorFunction(DifferentiableProcessorMixin, LLaVA16LLamaProcessorFunction):
    def __init__(self, processor):
        super().__init__(processor)
        self.diff_image_processor = LlavaNextDifferentiableImageProcessor.from_llava_next_image_processor(self.processor.image_processor)


    def get_target_str(self, target_str):
        if target_str in ['yes', 'Yes', 'YES']:
            target_str = 'Yes'
        elif target_str in ['no', 'No', 'NO']:
            target_str = 'No'
        else:
            target_str = target_str
        return target_str

    def __call__(self, batch: Dict, *args, **kwargs):
        pil_image_list = []
        for image in batch['image']:
            pil_image_list.append(TF.to_pil_image(image))

        conversations = self.make_conversations(batch)
        inputs = self.processor(images=pil_image_list, text=conversations, padding="longest", return_tensors="pt")

        labels = self.create_masked_labels(batch, inputs)
        inputs['labels'] = labels

        differentiable_images = self.diff_image_processor.preprocess(batch['image'])
        diff_pixel_values = differentiable_images['pixel_values']
        assert inputs['pixel_values'].shape == diff_pixel_values.shape
        inputs['pixel_values'] = diff_pixel_values
        return inputs


#https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_next/image_processing_llava_next.py
def get_image_size(image: torch.Tensor):
    assert image.ndim == 3
    _, h, w = image.shape
    return h,w

    #Nested functionality removed
def make_batched_images(images) -> List[List[ImageInput]]:
    """
    Accepts images in list or nested list format, and makes a list of images for preprocessing.

    Args:
        images (`Union[List[List[ImageInput]], List[ImageInput], ImageInput]`):
            The input image.

    Returns:
        list: A list of images.
    """
    image_list = []

    #Nested functionality removed
    if isinstance(images, list):
        for image in images:
            assert isinstance(image, torch.Tensor)
            assert image.ndim == 3
            image_list.append(image)
    elif isinstance(images, torch.Tensor):
        if images.ndim == 3:
            image_list = [images]
        elif images.ndim == 4:
            image_list = [images[k] for k in range(len(images))]
        else:
            raise ValueError()
    else:
        raise ValueError()

    return image_list



def divide_to_patches(image: torch.Tensor, patch_size: int) -> List[torch.Tensor]:
    """
    Divides an image into patches of a specified size.
    """
    patches = []
    height, width = get_image_size(image)
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = image[:, i : i + patch_size, j : j + patch_size]
            patches.append(patch)

    return patches


def expand_to_square(image: torch.Tensor, background_color) -> torch.Tensor:
    """
    Expands an image to a square by adding a background color.
    """

    height, width = get_image_size(image)

    if width == height:
        return image
    elif width > height:
        result = torch.ones((width, width, image.shape[2]), dtype=image.dtype, device=image.device) * background_color
        result[(width - height) // 2 : (width - height) // 2 + height, :] = image
        return result
    else:
        result = torch.ones((height, height, image.shape[2]), dtype=image.dtype, device=image.device) * background_color
        result[:, (height - width) // 2 : (height - width) // 2 + width] = image
        return result


def _get_patch_output_size(image, target_resolution):
    original_height, original_width = get_image_size(image)

    target_height, target_width = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    return new_height, new_width


def get_resize_output_image_size(
    input_image: torch.Tensor,
    size: Union[int, Tuple[int, int], List[int], Tuple[int]],
    default_to_square: bool = True,
    max_size: Optional[int] = None,
) -> tuple:
    """
    Find the target (height, width) dimension of the output image after resizing given the input image and the desired
    size.
    """
    if isinstance(size, (tuple, list)):
        if len(size) == 2:
            return tuple(size)
        elif len(size) == 1:
            # Perform same logic as if size was an int
            size = size[0]
        else:
            raise ValueError("size must have 1 or 2 elements if it is a list or tuple")

    if default_to_square:
        return (size, size)

    height, width = get_image_size(input_image)
    short, long = (width, height) if width <= height else (height, width)
    requested_new_short = size

    new_short, new_long = requested_new_short, int(requested_new_short * long / short)

    if max_size is not None:
        if max_size <= requested_new_short:
            raise ValueError(
                f"max_size = {max_size} must be strictly greater than the requested "
                f"size for the smaller edge size = {size}"
            )
        if new_long > max_size:
            new_short, new_long = int(max_size * new_short / new_long), max_size

    return (new_long, new_short) if width <= height else (new_short, new_long)

def resize(
    image: torch.Tensor,
    size: Tuple[int, int],
    resample: "PILImageResampling" = None,
) -> torch.Tensor:
    """
    Resizes `image` to `(height, width)` specified by `size` using the TORCH library.
    """
    resample = resample if resample is not None else PILImageResampling.BILINEAR

    if not len(size) == 2:
        raise ValueError("size must have 2 elements")

    resized_image = TF.resize(image, size, interpolation=pil_torch_interpolation_mapping[resample], antialias=True)
    return resized_image

def pad(
    image: torch.Tensor,
    padding: Union[int, Tuple[int, int], Iterable[Tuple[int, int]]],
    mode: PaddingMode = PaddingMode.CONSTANT,
    constant_values: Union[float, Iterable[float]] = 0.0,
    data_format: Optional[Union[str, ChannelDimension]] = None,
) -> torch.Tensor:
    """
    Pads the `image` with the specified (height, width) `padding` and `mode`.

    """

    def _expand_for_data_format(values):
        """
        Convert values to be in the format expected by np.pad based on the data format.
        """
        if isinstance(values, (int, float)):
            values = ((values, values), (values, values))
        elif isinstance(values, tuple) and len(values) == 1:
            values = ((values[0], values[0]), (values[0], values[0]))
        elif isinstance(values, tuple) and len(values) == 2 and isinstance(values[0], int):
            values = (values, values)
        elif isinstance(values, tuple) and len(values) == 2 and isinstance(values[0], tuple):
            values = values
        else:
            raise ValueError(f"Unsupported format: {values}")

        # add 0 for channel dimension
        values = ((0, 0), *values)

        # Add additional padding if there's a batch dimension
        values = (0, *values) if image.ndim == 4 else values
        return values

    padding = _expand_for_data_format(padding)

    padding =  [item for tup in padding for item in tup]

    if mode == PaddingMode.CONSTANT:
        image = F.pad(image, padding, mode="constant", value=constant_values)
    elif mode == PaddingMode.REFLECT:
        image = F.pad(image, padding, mode="reflect")
    elif mode == PaddingMode.REPLICATE:
        image = F.pad(image, padding, mode="edge")
    elif mode == PaddingMode.SYMMETRIC:
        image = F.pad(image, padding, mode="symmetric")
    else:
        raise ValueError(f"Invalid padding mode: {mode}")

    return image


class LlavaNextDifferentiableImageProcessor:
    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        image_grid_pinpoints: List = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_center_crop: bool = True,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1.0,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = True,
        do_convert_rgb: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"shortest_edge": 224}
        size = get_size_dict(size, default_to_square=False)
        image_grid_pinpoints = (
            image_grid_pinpoints
            if image_grid_pinpoints is not None
            else [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]
        )
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        self.do_resize = do_resize
        self.size = size
        self.image_grid_pinpoints = image_grid_pinpoints
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.do_pad = do_pad
        self.do_convert_rgb = do_convert_rgb

        #NO rescaling since we work in float [0,1]
        self.do_rescale = False
        self.rescale_factor = 1.0

    @staticmethod
    def from_llava_next_image_processor(processor: LlavaNextImageProcessor):
        return LlavaNextDifferentiableImageProcessor(
        do_resize=processor.do_resize,
        size=processor.size,
        image_grid_pinpoints=processor.image_grid_pinpoints,
        resample=processor.resample,
        do_center_crop=processor.do_center_crop,
        crop_size=processor.crop_size,
        do_rescale=processor.do_rescale,
        rescale_factor=processor.rescale_factor,
        do_normalize=processor.do_normalize,
        image_mean=processor.image_mean,
        image_std=processor.image_std,
        do_pad=processor.do_pad,
        do_convert_rgb=processor.do_convert_rgb,
        )

    def resize(self,
        image: torch.Tensor,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
    ) -> torch.Tensor:
        """
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.
        """

        default_to_square = True
        if "shortest_edge" in size:
            size = size["shortest_edge"]
            default_to_square = False
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
        else:
            raise ValueError("Size must contain either 'shortest_edge' or 'height' and 'width'.")

        output_size = get_resize_output_image_size(
            image,
            size=size,
            default_to_square=default_to_square,
        )

        return resize(image, output_size, resample=resample)

    def pad(self,
        image: torch.Tensor,
        padding: Union[int, Tuple[int, int], Iterable[Tuple[int, int]]],
        mode: PaddingMode = PaddingMode.CONSTANT,
        constant_values: Union[float, Iterable[float]] = 0.0,
    ) -> torch.Tensor:
        """
        Pads the `image` with the specified `padding` and `mode`. Padding can be in the (`height`, `width`)
        dimension of in the (`num_patches`) dimension. In the second case an iterable if tuples is expected
        as input.
        """

        # call the general `pad` if padding on `height/width`, otherwise it's the `num_patched` dim
        if isinstance(padding, int) or len(padding) != 4:
            return pad(image, padding, mode, constant_values)

        padding = [item for tup in padding for item in tup]
        if mode == PaddingMode.CONSTANT:
            image = F.pad(image, padding, mode="constant", value=constant_values)
        elif mode == PaddingMode.REFLECT:
            image = F.pad(image, padding, mode="reflect")
        elif mode == PaddingMode.REPLICATE:
            image = F.pad(image, padding, mode="edge")
        elif mode == PaddingMode.SYMMETRIC:
            image = F.pad(image, padding, mode="symmetric")
        else:
            raise ValueError(f"Invalid padding mode: {mode}")

        return image

    def _preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: int = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
    ) -> torch.Tensor:
        """
        Preprocess an image or batch of images. Copy of the `preprocess` method from `CLIPImageProcessor`.
        """
        images = make_list_of_images(images)

        all_images = []
        for image in images:
            assert image.dim() == 3
            assert image.shape[0] == 3

            if do_resize:
                image = self.resize(image=image, size=size, resample=resample)

            if do_center_crop:
                if isinstance(crop_size, int):
                    crop_size = (crop_size, crop_size)
                elif isinstance(crop_size, Dict):
                    crop_size = (crop_size["height"], crop_size["width"])
                elif isinstance(crop_size, Tuple | List):
                    assert len(crop_size) == 2
                    crop_size = (crop_size[0], crop_size[1])
                else:
                    raise ValueError()
                image = TF.center_crop(img=image, output_size=crop_size)

            if do_rescale:
                raise NotImplementedError()

            if do_normalize:
                image = TF.normalize(image, mean=image_mean, std=image_std)

            all_images.append(image)

        return all_images

    def _resize_for_patching(
        self, image: torch.Tensor, target_resolution: tuple, resample
    ) -> torch.Tensor:
        """
        Resizes an image to a target resolution while maintaining aspect ratio.

        Returns:
            np.array: The resized and padded image.
        """
        new_height, new_width = _get_patch_output_size(image, target_resolution)

        # Resize the image
        resized_image = resize(image, size=(new_height, new_width), resample=resample)
        return resized_image

    def _pad_for_patching(
        self, image: torch.Tensor, target_resolution: tuple
    ) -> torch.Tensor:
        """
        Pad an image to a target resolution while maintaining aspect ratio.
        """
        target_height, target_width = target_resolution
        new_height, new_width = _get_patch_output_size(image, target_resolution)

        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2

        padded_image = self.pad(image, padding=((paste_y, paste_y), (paste_x, paste_x)))

        return padded_image

    def get_image_patches(
        self,
        image: torch.Tensor,
        grid_pinpoints,
        size: tuple,
        patch_size: int,
        resample: PILImageResampling,
        data_format: ChannelDimension,
        input_data_format: ChannelDimension,
    ) -> List[torch.Tensor]:
        """
        Process an image with variable resolutions by dividing it into patches.

        """
        if not isinstance(grid_pinpoints, list):
            raise TypeError("grid_pinpoints must be a list of possible resolutions.")

        possible_resolutions = grid_pinpoints

        image_size = get_image_size(image)
        best_resolution = select_best_resolution(image_size, possible_resolutions)
        resized_image = self._resize_for_patching(
            image, best_resolution, resample=resample
        )
        padded_image = self._pad_for_patching(resized_image, best_resolution)

        patches = divide_to_patches(padded_image, patch_size=patch_size)

        resized_original_image = resize(
            image,
            size=size,
            resample=resample,
        )

        image_patches = [resized_original_image] + patches

        return image_patches

    def _pad_for_batching(
        self,
        pixel_values: List[torch.Tensor],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Pads images on the `num_of_patches` dimension with zeros to form a batch of same number of patches.
        """
        max_patch = max(len(x) for x in pixel_values)
        pixel_values = [
            self.pad(
                image,
                padding=((0, max_patch - image.shape[0]), (0, 0), (0, 0), (0, 0)),
            )
            for image in pixel_values
        ]

        return pixel_values

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        image_grid_pinpoints: List = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: int = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = None,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Args:
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(size, param_name="size", default_to_square=False)
        image_grid_pinpoints = image_grid_pinpoints if image_grid_pinpoints is not None else self.image_grid_pinpoints
        resample = resample if resample is not None else self.resample
        do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
        crop_size = crop_size if crop_size is not None else self.crop_size
        crop_size = get_size_dict(crop_size, param_name="crop_size", default_to_square=True)
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_pad = do_pad if do_pad is not None else self.do_pad
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        images = make_batched_images(images)

        new_images = []
        image_sizes = [get_image_size(image) for image in images]
        for image in images:
            # convert image into a list of patches
            # we intentially use the same data format as the input data format
            image_patches = self.get_image_patches(
                image,
                image_grid_pinpoints,
                size=(size["shortest_edge"], size["shortest_edge"])
                if "shortest_edge" in size
                else (min(size["height"], size["width"]), min(size["height"], size["width"])),
                patch_size=crop_size["height"],
                resample=resample,
                data_format=input_data_format,
                input_data_format=input_data_format,
            )

            # preprocess patches
            pixel_values = self._preprocess(
                image_patches,
                do_resize=do_resize,
                size=size,
                resample=resample,
                do_center_crop=do_center_crop,
                crop_size=crop_size,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
            )
            pixel_values = torch.stack(pixel_values, dim=0)
            new_images.append(pixel_values)

        if do_pad:
            processed_images = self._pad_for_batching(new_images)

        return BatchFeature(
            data={"pixel_values": torch.stack(processed_images, dim=0), "image_sizes": image_sizes}, tensor_type=return_tensors
        )
