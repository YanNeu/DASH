import torch
from dataclasses import dataclass
import math
from typing import Optional, Dict, List, Union, Tuple
from .processor import DifferentiableProcessorMixin
from .qwenvl import Qwen2VLProcessorFunction
from transformers.image_processing_utils import get_size_dict, BatchFeature
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor, smart_resize
from transformers.image_utils import (ChannelDimension,
                                      make_list_of_images,
                                      pil_torch_interpolation_mapping,
                                      OPENAI_CLIP_MEAN,
                                      OPENAI_CLIP_STD,
                                      ChannelDimension,
                                      ImageInput,
                                      VideoInput,
                                      PILImageResampling,)
from transformers.utils import TensorType
from torchvision.transforms import functional as TF
import numpy as np

class Qwen2VLDifferentiableProcessorFunction(DifferentiableProcessorMixin,Qwen2VLProcessorFunction):
    def __init__(self, processor, max_size=896):
        DifferentiableProcessorMixin.__init__(self, processor)  # explicit calls without super
        Qwen2VLProcessorFunction.__init__(self, processor)
        self.diff_image_processor = Qwen2VLDifferentiableImageProcessor.from_qwen2vl_image_processor(
            self.processor.image_processor)

        #this needs a smaller max-size for memory constraints during backprop
        processor_patch_size = self.diff_image_processor.patch_size * self.diff_image_processor.merge_size
        self.diff_image_processor.max_pixels = processor_patch_size*(round(max_size / processor_patch_size)) * (processor_patch_size**2)
        self.processor.image_processor.max_pixels = self.diff_image_processor.max_pixels
        self.processor.max_size = max_size
        self.max_size = max_size

    #longest edge
    def _resize_torch(self, images: Union[torch.Tensor], max_size: int) -> List[torch.Tensor]:
        images_scaled = []
        for img in images:
            # need to rescale?
            h, w = get_image_size(img)
            original_size = max(h, w)

            if original_size >= max_size:
                if h==w:
                    resized_w = resized_h = max_size
                elif h > w:
                    resized_h = max_size
                    resized_w = int(round((max_size / h) * w))
                else:
                    resized_w = max_size
                    resized_h = int(round((max_size / w) * h))

                img = TF.resize(img, (resized_w, resized_h), interpolation=pil_torch_interpolation_mapping[self.diff_image_processor.resample], antialias=True)

            images_scaled.append(img)

        return images_scaled


    def __call__(self, batch: Dict, *args, **kwargs):
        pil_image_list = []
        for image in batch['image']:
            pil_image = TF.to_pil_image(image)
            # width, height = pil_image.size
            #
            # resized_height, resized_width = smart_resize(
            #     height,
            #     width,
            #     factor=self.diff_image_processor.patch_size * self.diff_image_processor.merge_size,
            #     min_pixels=self.diff_image_processor.min_pixels,
            #     max_pixels=self.diff_image_processor.max_pixels,
            # )
            #
            # pil_image = pil_image.resize((resized_width, resized_height))
            pil_image_list.append(pil_image)

        conversations = self.make_conversations(batch)
        inputs = self.processor(images=pil_image_list, text=conversations, padding="longest", return_tensors="pt")

        labels = self.create_masked_labels(batch, inputs, num_sos_tokens=0)
        inputs['labels'] = labels

        #differentiable_images = self._resize_torch(batch['image'], self.max_size)
        differentiable_images = self.diff_image_processor.preprocess(batch['image'])
        diff_pixel_values = differentiable_images['pixel_values']
        assert inputs['pixel_values'].shape == diff_pixel_values.shape
        inputs['pixel_values'] = diff_pixel_values
        return inputs

    def get_target_str(self, target_str):
        if target_str in ['yes', 'Yes', 'YES']:
            target_str = 'Yes'
        elif target_str in ['no', 'No', 'NO']:
            target_str = 'No'
        else:
            target_str = target_str
        return target_str

    def get_target_str_options(self, target_str):
        if target_str in ['yes', 'Yes', 'YES']:
            return ['yes', 'Yes', 'Yes.', 'yes.']
        elif target_str in ['no', 'No', 'NO']:
            return ['no', 'No', 'No.', 'no.']

        return [target_str]

#https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py
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

# Copied from transformers.models.llava_next_video.image_processing_llava_next_video.make_batched_videos
def make_batched_videos(videos):
    raise NotImplementedError()

def get_image_size(image: torch.Tensor):
    assert image.ndim == 3
    _, h, w = image.shape
    return h,w

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

class Qwen2VLDifferentiableImageProcessor:
    model_input_names = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]

    def __init__(
        self,
        do_resize: bool = True,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        min_pixels: int = 56 * 56,
        max_pixels: int = 28 * 28 * 1280,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.size = {"min_pixels": min_pixels, "max_pixels": max_pixels}
        self.do_convert_rgb = do_convert_rgb

        #NO rescaling since we work in float [0,1]
        self.do_rescale = False
        self.rescale_factor = 1.0


    @staticmethod
    def from_qwen2vl_image_processor(processor: Qwen2VLImageProcessor):
        return Qwen2VLDifferentiableImageProcessor(
            do_resize=processor.do_resize,
            resample=processor.resample,
            do_rescale=processor.do_rescale,
            rescale_factor=processor.rescale_factor,
            do_normalize=processor.do_normalize,
            image_mean=processor.image_mean,
            image_std=processor.image_std,
            do_convert_rgb=processor.do_convert_rgb,
            min_pixels=processor.min_pixels,
            max_pixels=processor.max_pixels,
            patch_size=processor.patch_size,
            temporal_patch_size=processor.temporal_patch_size,
            merge_size=processor.merge_size,
        )

    def _preprocess(
        self,
        images: Union[ImageInput],
        do_resize: bool = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):

        images = make_list_of_images(images)

        height, width = get_image_size(images[0])
        resized_height, resized_width = height, width
        processed_images = []
        for image in images:
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=self.patch_size * self.merge_size,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                )
                image = resize(
                    image, size=(resized_height, resized_width), resample=resample
                )

            if do_rescale:
                raise NotImplementedError()

            if do_normalize:
                image = TF.normalize(image, mean=image_mean, std=image_std)

            processed_images.append(image)

        patches = torch.stack(processed_images)

        # Repeat the patches along the time dimension if there's only one image
        if patches.shape[0] == 1:
            patches = patches.repeat(self.temporal_patch_size, 1, 1, 1)

        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h = resized_height // self.patch_size
        grid_w = resized_width // self.patch_size

        # Reshape the tensor to prepare for patch extraction
        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )

        # Permute the axes to rearrange the tensor dimensions
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)

        # Flatten the patches into the desired shape
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * self.temporal_patch_size * self.patch_size * self.patch_size
        )

        return flatten_patches, (grid_t, grid_h, grid_w)

    def preprocess(
        self,
        images: ImageInput,
        videos: VideoInput = None,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        if images is not None:
            images = make_batched_images(images)
        if videos is not None:
            videos = make_batched_videos(videos)

        if images is not None:
            pixel_values, vision_grid_thws = [], []
            for image in images:
                patches, image_grid_thw = self._preprocess(
                    image,
                    do_resize=do_resize,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    data_format=data_format,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                )
                pixel_values.extend(patches)
                vision_grid_thws.append(image_grid_thw)
            pixel_values = torch.stack(pixel_values)
            vision_grid_thws = np.array(vision_grid_thws)
            data = {"pixel_values": pixel_values, "image_grid_thw": vision_grid_thws}

        if videos is not None:
            pixel_values, vision_grid_thws = [], []
            for images in videos:
                patches, video_grid_thw = self._preprocess(
                    images,
                    do_resize=do_resize,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    data_format=data_format,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                )
                pixel_values.extend(patches)
                vision_grid_thws.append(video_grid_thw)
            pixel_values = np.array(pixel_values)
            vision_grid_thws = np.array(vision_grid_thws)
            data = {"pixel_values_videos": pixel_values, "video_grid_thw": vision_grid_thws}

        return BatchFeature(data=data, tensor_type=return_tensors)