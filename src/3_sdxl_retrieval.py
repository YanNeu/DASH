import os
import warnings
from typing import Optional, List
import torch
from dataclasses import dataclass, field
from omegaconf import OmegaConf
from tqdm import tqdm

from dash_utils.utils import load_results, get_full_image_folder_path
from dash_utils.common_elements_utils import load_object_image_folders
from dash_utils.vlm_utils import get_config_from_name
from dash_utils.detection_utils import get_detection_config_from_name
from dash.nn_retrieval_utils import Retriever

@dataclass
class Args:
    prompt_type: str = 'llama_only_follow_up'
    datasets: List[str] = field(default_factory=lambda:[
        'spurious_imagenet',
        'openimages_uncommon',
        'openimages_10_quantile',
        'openimages_median',
        'openimages_common',
        'coco',
        'objects_365_100'
    ])
    spurious_prompt_id: int = 4

    result_dir: str = 'output/3_optim_sdxl_retrieval'
    source_dir: str = 'output/2_optim_sdxl'

    num_workers: int = 64
    request_nns: int = 1000
    download_nns: int = 20

    # VLM parameters
    model_id: str = "paligemma"

    # detection thresholding
    detection_model: str = 'owlv2-base'

    success_only: bool = False

    detection_threshold: float = 0.1
    min_success_per_prompt: int = 1
    min_success_prompts: int = 1

    clip_similarity_threshold: float = 0.95
    dreamsim_similarity_threshold: float = 0.9

    resume: bool = False

def setup() -> Args:
    default_config: Args = OmegaConf.structured(Args)
    cli_args = OmegaConf.from_cli()
    config: Args = OmegaConf.merge(default_config, cli_args)
    return config


def main():
    args = setup()

    print('#######################################################')
    print('Please remember to increase your ulimit: ulimit -n 8192')
    print('#######################################################')

    vlm_config = get_config_from_name(args.model_id)

    detection_config = get_detection_config_from_name(args.detection_model)
    retriever = Retriever(args.num_workers, clip_similarity_threshold=args.clip_similarity_threshold,
                          dreamsim_similarity_threshold=args.dreamsim_similarity_threshold)

    for dataset in args.datasets:
        #load previous results
        source_dir = get_full_image_folder_path(args.source_dir, dataset, args.prompt_type, args.spurious_prompt_id, vlm_config, detection_config)
        object_img_dirs = load_object_image_folders(source_dir)
        results_dict = load_results(object_img_dirs, detection_config.name, vlm_config.name, args.detection_threshold, prompt_id=args.spurious_prompt_id)

        #make result dirs
        result_dir = get_full_image_folder_path(args.result_dir, dataset, args.prompt_type, args.spurious_prompt_id, vlm_config, detection_config)
        os.makedirs(result_dir, exist_ok=args.resume)
        with open(os.path.join(result_dir, 'config.yaml'), "w") as f:
            OmegaConf.save(args, f)


        for obj_idx, obj_label in enumerate(tqdm(results_dict)):
            obj_results = results_dict[obj_label]

            success_source_ids_to_nns = {}
            for source_id in obj_results['prompt_to_idx']:
                nn_ids = obj_results['source_to_nns'][source_id]

                # collect ids of success nns
                if args.success_only:
                    success_ids = [nn['id'] for nn in nn_ids if nn['id'] in obj_results['success_ids']]
                else:
                    success_ids = [nn['id'] for nn in nn_ids]

                if len(success_ids) >= args.min_success_per_prompt:
                    success_source_ids_to_nns[source_id] = success_ids

            #exploit good prompts
            if len(success_source_ids_to_nns) >= args.min_success_prompts:
                obj_retrieval_dir = os.path.join(result_dir, obj_label)
                os.makedirs(obj_retrieval_dir, exist_ok=True)

                success_ids = set()
                for nn_ids in success_source_ids_to_nns.values():
                    success_ids.update(nn_ids)

                success_ids = list(success_ids)
                img_paths = [obj_results['img_paths'][id] for id in success_ids]
                retriever.download_from_images(img_paths, args.download_nns, args.request_nns, obj_retrieval_dir, resume=args.resume)


if __name__ == '__main__':
    main()