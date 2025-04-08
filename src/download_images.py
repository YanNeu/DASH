import os
import json
from dash_utils.nn_retrieval_utils import download_image

URL_ROOT_DIR = "urls"

VERSIONS = ['optim', 'prompt']

SOURCE_DATASETS = [
    "coco",
    "objects_365_100",
    "openimages_10_quantile",
    "openimages_common",
    "openimages_median",
    "openimages_uncommon",
    "spurious_imagenet",
]

SOURCE_MODELS = ["Paligemma-3b_OWLv2-Base", "LLaVa-1.6-Vicuna_OWLv2-Base", "LLaVa-1.6-Mistral_OWLv2-Base"]

if __name__ == "__main__":

    target_dir = "output"
    for version in VERSIONS:
        target_dir = os.path.join(target_dir, f"4_exploitation_{version}_retrieval")
        
        for dataset in SOURCE_DATASETS:
            target_dir = os.path.join(target_dir, dataset, "llava_only_follow_up", "prompt_4")

            for source_model in SOURCE_MODELS:
                target_dir = os.path.join(target_dir, source_model)

                url_file = os.path.join(URL_ROOT_DIR, version, source_model, f"{dataset}.json")
                url_lists = json.load(open(url_file, "r"))

                for object_name in url_lists:
                    obj_dir = os.path.join(target_dir, object_name)
                    os.makedirs(obj_dir, exist_ok=True)
                    
                    for cluster_id in url_lists[object_name]:

                        for img_id, url in url_lists[object_name][cluster_id]:
                            
                            img_path = os.path.join(obj_dir, f"{img_id}.png")
                            download_image(url, img_path)

                            break
                        break
                    break
                break
            break
        break