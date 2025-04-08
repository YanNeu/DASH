import os
import json
from collections import OrderedDict
from typing import Optional
import torch
from tqdm import tqdm
from PIL import Image
import hashlib
import atexit

def get_nn_infos_filepath(source_dir):
    return os.path.join(source_dir, 'nn_infos.pt')

def load_nn_infos_file(source_dir):
    return torch.load(get_nn_infos_filepath(source_dir))

def get_vlm_results_filepath(source_dir, vlm_name, prompt_id=0):
    return os.path.join(source_dir, f"{vlm_name}_results_prompt_{prompt_id}.pt")

def load_vlm_results_file(source_dir, vlm_name, prompt_id=0):
    return torch.load(get_vlm_results_filepath(source_dir, vlm_name, prompt_id=prompt_id))

def get_detection_results_filepath(source_dir, detection_name):
    return os.path.join(source_dir, f"{detection_name}_scores.pt")

def load_detection_results_file(source_dir, detection_name):
    return torch.load(get_detection_results_filepath(source_dir, detection_name))

def get_prompt_infos_filepath(source_dir):
    return os.path.join(source_dir, "prompt_infos.json")

def get_nn_filepath(dir, id):
    return os.path.join(dir, f'{id}.png')

def get_model_dir(model_config, detection_config):
    return f'{model_config.name}' if detection_config is None else f'{model_config.name}_{detection_config.name}'

def get_full_image_folder_path(root, dataset, prompt_type, spurious_prompt_id, model_config, detection_config, agnostic=False):

    #stage 2 prompt based retrieval folders are agnostic of model/spurious prompt id
    if agnostic:
        full_path = os.path.join(root, dataset, prompt_type)
    else:
        model_dir = get_model_dir(model_config, detection_config)
        full_path = os.path.join(root, dataset, prompt_type, f'prompt_{spurious_prompt_id}', model_dir)
    return full_path

def load_cluster_info(dir, linkage, distance_threshold):
    return torch.load(get_cluster_info_filepath(dir, linkage, distance_threshold))


def get_cluster_info_filepath(dir, linkage, distance_threshold):
    file_name = f'clusters_{linkage}_{distance_threshold:.2f}.pt'
    return os.path.join(dir, file_name)

def load_reference_knn_file(dir, obj_label, knn_k):
    return torch.load(get_reference_knn_filepath(dir, obj_label, knn_k))

def get_reference_knn_filepath(dir, obj_label, knn_k):
    file_name = f'{obj_label}_knn_results_{knn_k}.pt'
    return os.path.join(dir, file_name)

def get_full_prompt_folder_path(root, dataset):
    return os.path.join(root, dataset)

def get_dataset_embedding_filepath(dir, dataset):
    if 'coco' in dataset.lower():
        return os.path.join(dir, f'coco.npy')
    elif 'objects_365' in dataset.lower():
        return os.path.join(dir, f'objects365.npy')
    else:
        raise ValueError()

def load_idx_to_prompt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            idx_to_prompt = json.load(f)

        if 'prompt_to_idx' in idx_to_prompt:
            idx_to_prompt = {int(k): v for k, v in idx_to_prompt['prompt_to_idx'].items()}
        else:
            idx_to_prompt = {int(k): v for k, v in idx_to_prompt.items()}
    except Exception as e:
        raise Exception(f'{file_path} - {e}' )

    return idx_to_prompt

def load_prompts(source_dir, prompt_type):
    files = sorted(next(os.walk(source_dir))[2])
    classname_to_prompts = OrderedDict()

    #{SORT_IDX}_{IDX}_result.txt
    for file in files:
        if f'prompts_{prompt_type}' in file and file.endswith('.json'):
            class_name = file.split('_')[0]
            idx_to_prompt = load_idx_to_prompt(os.path.join(source_dir, file))
            prompt_list = [idx_to_prompt[idx] for idx in sorted(idx_to_prompt.keys())]
            classname_to_prompts[class_name] = prompt_list

    return classname_to_prompts


def load_id_to_image_paths(dir, file_filter=None):
    files = sorted(next(os.walk(dir))[2])
    id_to_image_paths = {}
    for file in files:
        if file.endswith('.png') or file.endswith('.jpg'):
            #remove files with the file_filter in them
            if file_filter is not None and file_filter in file: continue
            base_filenmae = os.path.splitext(file)[0]
            if base_filenmae.isdecimal():
                id = int(base_filenmae)
            else:
                id = base_filenmae
            id_to_image_paths[id] = (os.path.join(dir, file))

    return id_to_image_paths


def load_results(object_img_dirs, detector_name, vlm_name, detection_threshold, prompt_id=0, load_incomplete=True, reverse=False):
    print(f"Loading results for {vlm_name}... - Prompt type {prompt_id}")
    results_dict = OrderedDict()
    for object_label, nns_folder in tqdm(object_img_dirs.items()):
        results_dict[object_label] = {}
        
        # load results
        detection_path = get_detection_results_filepath(nns_folder, detector_name)
        vlm_path = get_vlm_results_filepath(nns_folder, vlm_name, prompt_id=prompt_id)
        nn_info_path = get_nn_infos_filepath(nns_folder)
        prompt_path = get_prompt_infos_filepath(nns_folder)

        id_to_img_path = load_id_to_image_paths(nns_folder)
        results_dict[object_label]['img_paths'] = id_to_img_path

        all_detections = None
        all_vlm_outputs = None
        nn_info = None
        prompt_infos = None
        if os.path.exists(detection_path):
            all_detections = torch.load(detection_path)
            results_dict[object_label]['detection_results'] = all_detections
        
        if os.path.exists(vlm_path):
            all_vlm_outputs = torch.load(vlm_path)
            results_dict[object_label]['vlm_results'] = all_vlm_outputs
        
        if os.path.exists(nn_info_path):
            nn_info = torch.load(nn_info_path)
            results_dict[object_label]['nn_info'] = nn_info
            results_dict[object_label]['source_to_nns'] = nn_info['source_nn_infos']

        if os.path.exists(prompt_path):
            prompt_infos = load_idx_to_prompt(prompt_path)
            results_dict[object_label]['prompt_to_idx'] = prompt_infos

        if not (all_detections is None or all_vlm_outputs is None or nn_info is None):
            nn_ids = list(nn_info['file_infos'].keys())

            vlm_successes = torch.zeros(len(nn_ids), dtype=torch.bool)
            detection_successes = torch.zeros(len(nn_ids), dtype=torch.bool)

            total_success_ids = []
            for lin_idx, id in enumerate(nn_ids):
                if reverse:
                    vlm_successes[lin_idx] = all_vlm_outputs[id]['decision'] == 0
                    detection_successes[lin_idx] = all_detections[id]['max_score'] >= detection_threshold
                else:
                    vlm_successes[lin_idx] = all_vlm_outputs[id]['decision'] == 1
                    detection_successes[lin_idx] = all_detections[id]['max_score'] < detection_threshold
                if vlm_successes[lin_idx] and detection_successes[lin_idx]:
                    total_success_ids.append(id)

            total_successes = vlm_successes & detection_successes

            stats = {
                'total': len(nn_ids),
                'vlm_successes': torch.sum(vlm_successes).item(),
                'detection_successes': torch.sum(detection_successes).item(),
                'successes': torch.sum(total_successes).item()
            }

            results_dict[object_label]['stats'] = stats
            results_dict[object_label]['success_ids'] = total_success_ids
        elif not load_incomplete:
            raise ValueError(f'Could not load detection, vlm and nn_info file')

    return results_dict


def load_object_categories(file_path):
    classes = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            classes.append(line.strip())

    print(f'Found {len(classes)} objects')
    return classes


def load_image(image_path):
    from PIL import PngImagePlugin
    LARGE_ENOUGH_NUMBER = 100
    PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 2)

    if isinstance(image_path, str):
        try:
            img = Image.open(image_path)
            img = img.convert('RGB')
        except Exception as e:
            print(f"Could not load {image_path}: {e}")
            img = Image.new('RGB', (512, 512))
    else:
        raise ValueError()

    return img

class ReLAIONIndex:
    def __init__(self, db_filename):
        if db_filename is not None:
            import plyvel
            print(f'Loading ReLAION Index from {db_filename}')
            self.db = plyvel.DB(db_filename, create_if_missing=False)
        else:
            self.db = None
        # Register the close method to be called at program exit
        atexit.register(self.close)

    def md5_hash(self, url):
        return hashlib.md5(url.encode('utf-8')).hexdigest().encode('utf-8')

    def url_exists(self, url):
        if self.db is None:
            return True
        else:
            h = self.md5_hash(url)
            exists = self.db.get(h) is not None
            return exists

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None  # Prevent further operations on the closed DB

    def __enter__(self):
        # Allows use of 'with' statements
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Close the database when exiting the 'with' block
        self.close()
        # Unregister the atexit handler since we've already closed the DB
        atexit.unregister(self.close)

    def __del__(self):
        # Attempt to close the database when the object is garbage collected
        self.close()
