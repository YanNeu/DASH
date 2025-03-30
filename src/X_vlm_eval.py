#Can be run in parallel
#CUDA_VISIBLE_DEVICES=5,6,7,8 torchrun --nproc-per-node 4 --standalone src/X_vlm_eval.py
#

import os
import torch
from dataclasses import dataclass, field
from omegaconf import OmegaConf
from typing import Optional, List

from setuptools.config.setupcfg import configuration_to_dict
from tqdm import tqdm

from dash_utils.vlm_utils import load_vlm_model, make_vlm_datasets_dataset_dataloader, forward_dataset, get_yes_no_decisions_probabilities, get_standard_spurious_prompt, get_config_from_name
from dash_utils.detection_utils import load_detection_model, foward_detection_dataset, make_detection_dataset_dataloader, get_max_detection_score, get_detection_config_from_name
from dash_utils.utils import load_id_to_image_paths, get_detection_results_filepath, get_vlm_results_filepath, get_full_image_folder_path

@dataclass
class PaliGemmaEvalArgs:
    gpu: int = 0

    source_dir: str = 'output_cvpr/2_prompt_retrieval'
    prompt_type: str = 'llama_only_follow_up'
    datasets: List[str] = field(default_factory=lambda:[
        'spurious_imagenet',
        'openimages_common',
        'openimages_10_quantile',
        'openimages_median',
        'openimages_uncommon',
        'coco',
        'objects_365_100'
    ])
    use_agnostic_folder: bool = False

    spurious_prompt_id: int = 4

    nn_filter: Optional[str] = '_sd'
    skip_existing: bool = False

    local_id: int = 0
    world_size: int = 0

    spurious_prompt_ids: list[int] = field(default_factory=lambda:[4])
    #
    vlm_batchsize: int = 8
    detection_batchsize: int = 8

    num_workers: int = 1

    #VLM parameters
    model_id: str =  "paligemma"
    vlm_calculation: bool = True

    eval_model_id: Optional[str] = None

    #detection thresholding
    detection_model: str = 'owlv2-base'
    detection_calculation: bool = True

    eval_detection_model: Optional[str] = None

def setup() -> PaliGemmaEvalArgs:
    default_config: PaliGemmaEvalArgs = OmegaConf.structured(PaliGemmaEvalArgs)
    cli_args = OmegaConf.from_cli()
    config: PaliGemmaEvalArgs = OmegaConf.merge(default_config, cli_args)
    return config

def main():
    args = setup()
    
    if "WORLD_SIZE" in os.environ:
        args.local_id = int(os.environ["LOCAL_RANK"])
        args.world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        args.gpu = args.local_id + args.gpu
    else:
        local_rank = args.local_id
        world_size = args.world_size

    device = torch.device(f'cuda:{args.gpu}')
    print(f'Rank {args.local_id} / {args.world_size} - VLM GPU: {device}')


    source_vlm_config = get_config_from_name(args.model_id)
    source_detection_config = get_detection_config_from_name(args.detection_model)

    #vlm model
    if args.vlm_calculation:
        eval_model_id = args.model_id if args.eval_model_id is None else args.eval_model_id
        model = load_vlm_model(eval_model_id, device)
        eval_vlm_name = model.name
    else:
        model = None
        eval_vlm_name = ''

    #detection threshold model
    if args.detection_calculation:
        eval_detection_model = args.detection_model if args.eval_detection_model is None else args.eval_detection_model
        detection_model = load_detection_model(eval_detection_model, device)
        eval_detection_name = detection_model.name
    else:
        detection_model = None
        eval_detection_name = ''


    for dataset in args.datasets:
        source_dir = get_full_image_folder_path(args.source_dir, dataset, args.prompt_type, args.spurious_prompt_id,
                                                source_vlm_config, source_detection_config, agnostic=args.use_agnostic_folder)
        result_dir = source_dir

        subdirs = sorted(next(os.walk(source_dir))[1])

        for todo_idx, object_label in enumerate(tqdm(subdirs)):
            if args.world_size > 1 and (todo_idx + 1) % args.world_size != args.local_id:
                continue

            object_subdir = os.path.join(result_dir, object_label)

            if not os.path.exists(object_subdir):
                os.makedirs(object_subdir)

            for prompt_id in args.spurious_prompt_ids:
                vlm_output_file = get_vlm_results_filepath(object_subdir, eval_vlm_name, prompt_id=prompt_id)
                if args.skip_existing and  os.path.isfile(vlm_output_file):
                    old_vlm_output = torch.load(vlm_output_file)
                else:
                    old_vlm_output = {}
                if args.vlm_calculation:
                    id_to_img_path = load_id_to_image_paths(object_subdir, file_filter=args.nn_filter)

                    if not id_to_img_path:
                        print(f'Warning empty directory: {object_subdir} - skipping')
                        continue

                    query = get_standard_spurious_prompt(object_label, prompt_id=prompt_id)

                    #evaluate responses for all NNS
                    all_vlm_outputs = {}

                    nn_image_paths = []
                    id_to_lin_idx = {}
                    for id, img_path in id_to_img_path.items():
                        if id in old_vlm_output:
                            id_dict = old_vlm_output[id]
                            all_vlm_outputs[id] = id_dict
                        else:
                            id_to_lin_idx[id] = len(nn_image_paths)
                            nn_image_paths.append(img_path)

                    if len(id_to_lin_idx) > 0:
                        data_dicts = [{'image_path': img_path, 'prompt': query} for img_path in nn_image_paths]
                        dataset, data_loader = make_vlm_datasets_dataset_dataloader(data_dicts, model.get_processor_function(),
                                                                           batch_size=args.vlm_batchsize, num_workers=args.num_workers)

                        outputs = forward_dataset(data_loader, model)
                        decisions_probabilities = get_yes_no_decisions_probabilities(outputs, model.processor)

                        for id, lin_idx in id_to_lin_idx.items():
                            id_dict = {
                                'response': outputs['response'][lin_idx],
                                'full_response': outputs['full_response'][lin_idx],
                                'decision': decisions_probabilities['decision'][lin_idx],
                                'yes_prob': decisions_probabilities['yes_prob'][lin_idx],
                                'no_prob': decisions_probabilities['no_prob'][lin_idx],
                            }
                            all_vlm_outputs[id] = id_dict

                    torch.save(all_vlm_outputs, vlm_output_file)

            detection_output_file = get_detection_results_filepath(object_subdir, eval_detection_name)
            if args.skip_existing and os.path.isfile(detection_output_file):
                old_detection_output = torch.load(detection_output_file)
            else:
                old_detection_output = {}

            if args.detection_calculation:
                all_detections = {}
                id_to_img_path = load_id_to_image_paths(object_subdir, file_filter=args.nn_filter)

                nn_image_paths = []
                id_to_lin_idx = {}
                for id, img_path in id_to_img_path.items():
                    if id in old_detection_output:
                        id_dict = old_detection_output[id]
                        all_detections[id] = id_dict
                    else:
                        id_to_lin_idx[id] = len(nn_image_paths)
                        nn_image_paths.append(img_path)

                if len(nn_image_paths) > 0:
                    data_dicts = [{'image_path': img_path, 'prompt': object_label} for img_path in nn_image_paths]
                    dataset, data_loader = make_detection_dataset_dataloader(data_dicts, detection_model.get_processor_function(),
                                                                             batch_size=args.detection_batchsize, num_workers=args.num_workers)

                    outputs = foward_detection_dataset(data_loader, detection_model)
                    max_scores = get_max_detection_score(outputs)

                    for id, lin_idx in id_to_lin_idx.items():
                        id_dict = {
                            'max_score': max_scores[lin_idx]
                        }
                        all_detections[id] = id_dict

                torch.save(all_detections, detection_output_file)


if __name__ == '__main__':
    main()