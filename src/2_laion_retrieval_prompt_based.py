import os
import torch
from omegaconf import OmegaConf
from typing import List, Optional
from dataclasses import dataclass, field
from os import listdir
from os.path import isfile, join
import json
from tqdm import tqdm
from dash_utils.nn_retrieval_utils import Retriever, NNInfo
from dash_utils.utils import load_prompts, get_full_image_folder_path, get_full_prompt_folder_path

@dataclass
class Args:
    prompt_dir: str = "output/1_prompts"
    result_dir: str = 'output/2_prompt_retrieval'
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

    num_workers: int = 64
    request_nns: int = 1000
    download_nns: int = 20
    skip_existing: bool = False
    clip_similarity_threshold: float = 0.95
    dreamsim_similarity_threshold: float = 0.9
    resume: bool = False

def setup() -> Args:
    default_config: Args = OmegaConf.structured(Args)
    cli_args = OmegaConf.from_cli()
    config: Args = OmegaConf.merge(default_config, cli_args)
    assert config.result_dir is not None
    return config

def main():
    args = setup()

    print('#######################################################')
    print('Please remember to increase your ulimit: ulimit -n 8192')
    print('#######################################################')

    #load prompts
    for dataset in args.datasets:
        source_dir = get_full_prompt_folder_path(args.prompt_dir, dataset)
        classname_to_prompts = load_prompts(source_dir, args.prompt_type)

        #make result dir
        result_dir = get_full_image_folder_path(args.result_dir, dataset, args.prompt_type, None, None, None, agnostic=True)
        os.makedirs(result_dir, exist_ok=args.resume)
        with open(os.path.join(result_dir, 'config.yaml'), "w") as f:
            OmegaConf.save(args, f)

        retriever = Retriever(args.num_workers, clip_similarity_threshold=args.clip_similarity_threshold,
                              dreamsim_similarity_threshold=args.dreamsim_similarity_threshold)

        for class_name, prompts in tqdm(classname_to_prompts.items()):
            class_subdir = os.path.join(result_dir, class_name)
            if args.skip_existing and os.path.isfile(os.path.join(class_subdir, 'nn_infos.pt')):
                continue
            os.makedirs(class_subdir, exist_ok=True)

            retriever.download_from_text(prompts, args.download_nns, args.request_nns, class_subdir, resume=args.resume)




if __name__=="__main__":
    main()
