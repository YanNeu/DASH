#Can be run in parallel
#CUDA_VISIBLE_DEVICES=5,6,7,8 torchrun --nproc-per-node 4 --standalone src/2_sdxl_optim.py
#

import os
import json
from collections import OrderedDict

import torch

import torchvision.transforms.functional as TF

from datetime import datetime
from tqdm import tqdm

from dataclasses import dataclass, field
from omegaconf import OmegaConf
from typing import Optional, Dict, List

from dash_utils.detection_utils import load_detection_model
from dash_utils.detection_utils.detection_optim import get_detection_loss
from dash_utils.sdxl_pipeline import load_diffusion_pipeline, OptimParams, make_loss_dict
from dash_utils.vlm_utils import load_vlm_model, forward_in_memory_data, get_yes_no_decisions_probabilities, get_standard_spurious_prompt
from dash_utils.vlm_utils.vlm_optim import get_vlm_loss_target_options
from dash_utils.plotting_utils import plot
from dash_utils.utils import load_prompts, get_nn_infos_filepath, get_full_image_folder_path, get_model_dir, get_full_prompt_folder_path

import multiprocessing
import wandb

def make_wandb_run(project, name=None, config=None):
    wandb.init(
        project=project,
        name=name,
        config=config)

@dataclass
class VLMOptimArgs:
    diffusion_model: str = "sdxl_hyper"
    guidance_scale: float = 0.
    num_diffusion_steps: int = 1

    optim_params: OptimParams = field(default_factory=lambda:
        OptimParams()
    )

    max_prompts_per_class: Optional[int] = None

    #augmentation
    num_cutouts: int = 0
    noise_sd: float = 0.005

    det_num_cutouts: Optional[int] = 0
    det_noise_sd: Optional[float] = 0.0

    cut_power: float = 0.3

    #devices
    gpu: int = 0
    diffusion_gpu: Optional[int] = 1
    reg_gpu: Optional[int] = None
    vae_gpu: Optional[int] = None

    local_rank: int = 0
    world_size: int = 0

    prompt_dir: str = "output/1_prompts"
    result_dir: str = 'output/2_optim_sdxl'
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

    images_per_prompt: int = 1

    skip_existing: bool = True

    target_str: str = "yes"
    eval_dir: str = 'output_cvpr'

    #VLM parameters
    checkpoint_vlm: bool = False
    checkpoint_det: bool = False

    spurious_prompt_id: int = 4

    #vlm
    vlm_weight: float = 1.0
    model_id: str =  "paligemma"

    #detection thresholding
    detection_model: str = 'owlv2-base'
    detection_weight: float = 1.0
    detection_threshold: float = 0.075
    detection_reduction: str = 'max'

    wandb_project: str = 'VLMOptimization'
    log_wandb: bool = False

    reverse_mode: bool = False

    access_token: str = 'hf_<YOUR_TOKEN>'

def setup() -> VLMOptimArgs:
    default_config: VLMOptimArgs = OmegaConf.structured(VLMOptimArgs)
    cli_args = OmegaConf.from_cli()
    config: VLMOptimArgs = OmegaConf.merge(default_config, cli_args)
    return config

def make_id(prompt_idx, prompt_sub_index, images_per_prompt):
    num_post_digits = len(str(images_per_prompt))
    new_id = int(f'{prompt_idx}{f'{prompt_sub_index}'.zfill(num_post_digits)}')
    return new_id

def make_nn_infos(idx_to_prompt, images_per_prompt):
    file_infos = {}
    source_dicts = {}

    for prompt_idx, prompt in idx_to_prompt.items():
        source_dicts[prompt_idx] = []
        for prompt_sub_index in range(images_per_prompt):
            id = make_id(prompt_idx, prompt_sub_index, images_per_prompt)

            info_dict = {
                'url': None,
                'caption': prompt,
                'id': id,
                'idcs_similarity': {prompt_idx: 1.0},
                'clip_embedding': None
            }

            file_infos[id] = info_dict

            source_dicts[prompt_idx].append({
                'id': id,
                'similarity': 1.0
            })

    info_out_dict = {
        'file_infos': file_infos,
        'source_nn_infos': source_dicts
    }

    return info_out_dict


def main():
    args: VLMOptimArgs = setup()

    gpus_per_threat = set()
    for gpu in [args.gpu, args.vae_gpu, args.diffusion_gpu, args.reg_gpu]:
        if gpu is not None:
            gpus_per_threat.add(gpu)

    if "WORLD_SIZE" in os.environ:
        local_device_rank =  int(os.environ["LOCAL_RANK"])
        args.local_rank = args.local_rank + int(os.environ["LOCAL_RANK"])
        if args.world_size == 0:
            args.world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        gpu_offset = len(gpus_per_threat)
    else:
        gpu_offset = 0
        local_device_rank = 0

    #load diffusion
    vlm_device = torch.device(f'cuda:{args.gpu + local_device_rank * gpu_offset}')
    reg_device = torch.device(f'cuda:{args.reg_gpu + local_device_rank * gpu_offset}') if args.reg_gpu else vlm_device

    diffusion_device = torch.device(f'cuda:{args.diffusion_gpu + local_device_rank * gpu_offset}') if args.diffusion_gpu else vlm_device
    vae_device = torch.device(f'cuda:{args.vae_gpu + local_device_rank * gpu_offset}') if args.vae_gpu else diffusion_device

    print(f'Rank {args.local_rank} / {args.world_size} - VLM GPU: {vlm_device} - Diffusion GPU {diffusion_device} - VAE GPU {vae_device} - Reg GPU {reg_device}')

    pipe, resolution, eta, timesteps = load_diffusion_pipeline(args.diffusion_model, args.num_diffusion_steps, diffusion_device, vae_device)

    #load vlm

    model = load_vlm_model(args.model_id, vlm_device)
    for param in model.model.parameters():
        param.requires_grad = False
    if args.checkpoint_vlm:
        model.model.gradient_checkpointing_enable()
        model.model.train()  # checkpointing only enabled in train

    #detection regularisation
    if args.detection_weight > 0:
        detection_model = load_detection_model(args.detection_model, reg_device)
        for param in detection_model.model.parameters():
            param.requires_grad = False
        if args.checkpoint_det:
            detection_model.model.gradient_checkpointing_enable()
            detection_model.model.train() #checkpointing only enabled in train
    else:
        detection_model = None

    queue = multiprocessing.JoinableQueue()
    writer_process = multiprocessing.Process(target=consumer_process, args=(queue,))
    writer_process.start()

    for dataset in args.datasets:
        source_dir = get_full_prompt_folder_path(args.prompt_dir, dataset)
        classname_to_prompts = load_prompts(source_dir, args.prompt_type)
        #

        num_cutouts = args.num_cutouts
        noise_sd = args.noise_sd

        det_num_cutouts = args.det_num_cutouts if args.det_num_cutouts is not None else num_cutouts
        det_noise_sd = args.det_noise_sd if args.det_noise_sd is not None else noise_sd

        cut_power = args.cut_power

        model_dir = get_model_dir(model, detection_model)
        wandb_name = f'{dataset}_{model_dir}_vlm_{num_cutouts}_{noise_sd}_det_{det_num_cutouts}_{det_noise_sd}_cut_{cut_power}_{args.optim_params.optim}_{args.optim_params.steps}_{args.optim_params.stepsize}_split_{args.local_rank}_{args.world_size}'

        #output folders
        result_dir = get_full_image_folder_path(args.result_dir, dataset, args.prompt_type, args.spurious_prompt_id, model, detection_model)
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, 'config.yaml'), "w") as f:
            OmegaConf.save(args, f)

        if args.log_wandb:
            wandb.init(
                project=args.wandb_project,
                name=wandb_name,
                config=dict(args))

        for job_idx, (class_name, prompts) in enumerate(tqdm(classname_to_prompts.items())):
            if args.world_size > 1 and (job_idx + 1) % args.world_size != args.local_rank:
                continue

            class_subdir = os.path.join(result_dir, class_name)
            os.makedirs(class_subdir, exist_ok=True)

            #setup queue
            idx_sub_prompt_queue = []
            idx_to_prompt = {}
            for prompt_idx, prompt in enumerate(prompts):
                if args.max_prompts_per_class is not None and prompt_idx >= args.max_prompts_per_class:
                    break

                idx_to_prompt[prompt_idx] = prompt
                for prompt_sub_index in range(args.images_per_prompt):
                    idx_sub_prompt_queue.append((prompt_idx, prompt_sub_index, prompt))

            prompt_info_dict = {
                'prompt_to_idx': idx_to_prompt,
            }

            #nn info
            info_out_dict = make_nn_infos(idx_to_prompt, args.images_per_prompt)
            torch.save(info_out_dict, get_nn_infos_filepath(class_subdir))

            #prompt infos
            with open(os.path.join(class_subdir, 'prompt_infos.json'), 'w') as fp:
                json.dump(prompt_info_dict, fp)

            #optimization
            while idx_sub_prompt_queue:
                losses_dicts = []
                #vlm loss
                if args.vlm_weight > 0:
                    if args.reverse_mode and args.target_str == 'yes':
                        args.target_str = 'no'

                    vlm_prompt = get_standard_spurious_prompt(class_name, prompt_id=args.spurious_prompt_id)
                    vlm_loss = get_vlm_loss_target_options(vlm_prompt, args.target_str, model,
                                                           num_cutouts=num_cutouts,
                                                           noise_sd=noise_sd,
                                                           cut_power=cut_power,
                                                           )
                    losses_dicts.append(make_loss_dict(vlm_loss, 'vlm_loss', args.vlm_weight))

                # detection reg
                if args.detection_weight > 0:
                    detection_loss = get_detection_loss(class_name, detection_model,
                                                        reverse_mode=args.reverse_mode,
                                                        detection_threshold=args.detection_threshold,
                                                        reduction=args.detection_reduction, num_cutouts=det_num_cutouts,
                                                        cut_power=cut_power, noise_sd=det_noise_sd)
                    losses_dicts.append(make_loss_dict(detection_loss, 'detection_loss', args.detection_weight))

                targets_dict = {'vlm_loss': None, 'detection_loss': None}

                prompt_idx, prompt_sub_index, prompt = idx_sub_prompt_queue.pop(0)

                image_id = make_id(prompt_idx, prompt_sub_index, args.images_per_prompt)
                plot_file = os.path.join(class_subdir, f'{image_id}.pdf')
                ours_file = os.path.join(class_subdir, f'{image_id}.png')
                sd_file = os.path.join(class_subdir, f'{image_id}_sd.png')
                if args.skip_existing and (os.path.isfile(plot_file) and os.path.isfile(ours_file) and os.path.isfile(sd_file)):
                    continue
                generator = torch.Generator(device=diffusion_device)
                generator.manual_seed(prompt_idx * args.images_per_prompt + prompt_sub_index)
                return_values = pipe(
                                    losses_dict=losses_dicts,
                                    targets_dict=targets_dict,
                                    optim_params=args.optim_params,
                                    prompt=prompt,
                                    guidance_scale=args.guidance_scale,
                                    height=resolution,
                                    width=resolution,
                                    num_inference_steps=args.num_diffusion_steps,
                                    eta=eta,
                                    generator=generator,
                                    timesteps=timesteps
                )

                with torch.no_grad():
                    img_grid = return_values['imgs']
                    loss_scores = return_values['loss_scores']
                    initial_img = img_grid[0]

                    #predict:
                    pil_list = [TF.to_pil_image(image) for image in img_grid]
                    prompt_list = [vlm_prompt] * len(pil_list)
                    outputs = forward_in_memory_data(pil_list, prompt_list, model)
                    decisions_probabilities = get_yes_no_decisions_probabilities(outputs, model.processor)
                    answer_yes_no = decisions_probabilities['decision']
                    yes_p = decisions_probabilities['yes_prob']
                    response = outputs['response']

                    if args.log_wandb:
                        log_wandb(answer_yes_no, response, yes_p, class_name, img_grid, loss_scores, prompt)

                    queue.put((answer_yes_no, response, yes_p, class_name, img_grid, initial_img, loss_scores, ours_file, plot_file, prompt, sd_file))

        wandb.finish()

    for _ in range(10):
        queue.put(None)
    writer_process.join()


def consumer_process(queue: multiprocessing.Queue):
    while True:
        msg = queue.get(block=True)  #read from queue
        if msg is None:
            break
        else:
            save_results(*msg)

def log_wandb(answer_yes_no, response, yes_p, class_name, img_grid, loss_scores, prompt):
    min_loss_idx = torch.argmin(torch.FloatTensor(loss_scores['total'])).item()
    min_loss_img = img_grid[min_loss_idx]

    pil_sd = TF.to_pil_image(torch.clamp(img_grid[0], 0, 1))
    pil_ours = TF.to_pil_image(torch.clamp(min_loss_img, 0, 1))

    start = wandb.Image(pil_sd, caption=f'{prompt}\n{class_name}: {response[0]}\nyes p.: {yes_p[0]:.3f}')
    best = wandb.Image(pil_ours, caption=f'{prompt}\n{class_name}: {response[min_loss_idx]}\nyes p.: {yes_p[min_loss_idx]:.3f}')
    wandb_log_dict = {
        'start_best': [start, best],
        'decision': answer_yes_no[min_loss_idx],
        'yes_p': yes_p[min_loss_idx],
    }
    for loss_i_name, loss_i_scores in loss_scores.items():
        wandb_log_dict[loss_i_name] = loss_i_scores[min_loss_idx]

    wandb.log(wandb_log_dict)

def save_results(answer_yes_no, response, yes_p, class_name, img_grid, initial_img, loss_scores, ours_file, plot_file, prompt, sd_file):
    min_loss_idx = torch.argmin(torch.FloatTensor(loss_scores['total'])).item()
    min_loss_img = img_grid[min_loss_idx]
    pil_sd = TF.to_pil_image(torch.clamp(img_grid[0], 0, 1))
    pil_sd.save(sd_file)
    pil_ours = TF.to_pil_image(torch.clamp(min_loss_img, 0, 1))
    pil_ours.save(ours_file)
    title_attributes = {
        class_name: answer_yes_no,
        'VLM': response,
        'yes p.': yes_p,
    }


    plot(img_grid, title_attributes, plot_file,
         original_image=initial_img, original_title=prompt,
         loss_scores=loss_scores)


if __name__ == '__main__':
    main()
