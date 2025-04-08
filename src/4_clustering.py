import os
import torch

from omegaconf import OmegaConf
from dataclasses import dataclass, field
from typing import Optional, List, Set
from tqdm import tqdm
from scipy.spatial.distance import cdist
import numpy as np

from dash_utils.utils import load_results, get_full_image_folder_path, load_image, get_cluster_info_filepath, ReLAIONIndex
from dash_utils.common_elements_utils import load_object_image_folders
from dash_utils.vlm_utils import get_config_from_name
from dash_utils.detection_utils import get_detection_config_from_name


@dataclass
class Args:
    source_dir: str = 'output/4_exploitation_prompt_retrieval'
    prompt_type: str = 'llama_only_follow_up'
    use_agnostic_folder: bool = False
    datasets: List[str] = field(default_factory=lambda:[
        'spurious_imagenet',
        'openimages_common',
        'openimages_10_quantile',
        'openimages_median',
        'openimages_uncommon',
        'coco',
        'objects_365_100'
    ])

    # VLM parameters
    model_id: str = "paligemma"

    # detection thresholding
    detection_model: str = 'owlv2-base'
    detection_threshold: float = 0.1

    spurious_prompt_id: int = 4
    skip_existing: bool = False

    min_cluster_size: int = 5
    distance_threshold: float = 0.6
    linkage: str = 'average'

    embedding_type: str = 'dreamsim_embedding'
    relaion_index: Optional[str] = '../combined_relaion.db'

def setup() -> Args:
    default_config: Args = OmegaConf.structured(Args)
    cli_args = OmegaConf.from_cli()
    config: Args = OmegaConf.merge(default_config, cli_args)
    return config


class NNInfosForClustering:
    def __init__(self, id, img_path, caption, embedding):
        self.id = id
        self.img_path = img_path
        self.caption = caption
        self.embedding = embedding



def merge_pre_clusters(
        success_infos: List[NNInfosForClustering],
        pre_clusters: List[List[int]],
        linkage_method: str = 'average',
        distance_threshold: float = 0.5,
        min_cluster_size: int = 5
) -> List[List[int]]:
    """
    Merges pre_clusters based on the specified linkage method and distance threshold,
    handling duplicates and updating distances dynamically.

    Parameters:
        success_infos (List[NNInfosForClustering]): List of NNInfosForClustering objects.
        pre_clusters (List[List[int]]): List of pre_clusters, each containing indices into success_infos.
        linkage_method (str): Linkage method to use ('average', 'single', 'complete').
        distance_threshold (float): Distance threshold for merging clusters.
        min_cluster_size (int): Minimum cluster size to keep after merging.

    Returns:
        List[List[int]]: Merged clusters containing indices into success_infos.
    """
    import heapq

    # Convert pre_clusters to sets to remove duplicates within clusters
    clusters = [set(cluster) for cluster in pre_clusters]

    # Precompute embeddings for all success_infos
    embeddings = np.array([info.embedding for info in success_infos])

    # Initialize a priority queue for cluster pairs based on their distances
    # Each item is a tuple: (distance, cluster_index_a, cluster_index_b)
    heap = []

    # Create a mapping from cluster indices to their current positions
    cluster_indices = list(range(len(clusters)))

    # Initialize the next unique cluster index
    next_cluster_idx = max(cluster_indices) + 1

    # Build initial active clusters dictionary
    active_clusters = {idx: clusters[idx] for idx in cluster_indices}

    # Compute initial distances between clusters
    cluster_indices_list = list(active_clusters.keys())
    for i in range(len(cluster_indices_list)):
        idx_i = cluster_indices_list[i]
        for j in range(i + 1, len(cluster_indices_list)):
            idx_j = cluster_indices_list[j]
            dist = compute_cluster_distance(
                active_clusters[idx_i], active_clusters[idx_j], embeddings, linkage_method
            )
            heapq.heappush(heap, (dist, idx_i, idx_j))

    # While there are distances below the threshold
    while heap:
        dist, idx_a, idx_b = heapq.heappop(heap)
        # Check if clusters are still active (may have been merged)
        if idx_a not in active_clusters or idx_b not in active_clusters:
            continue
        if dist > distance_threshold:
            break  # No more clusters within the threshold

        # Merge clusters
        new_cluster = active_clusters[idx_a].union(active_clusters[idx_b])

        # Remove old clusters
        del active_clusters[idx_a]
        del active_clusters[idx_b]

        # Add new cluster with a unique index
        new_idx = next_cluster_idx
        next_cluster_idx += 1
        active_clusters[new_idx] = new_cluster

        # Update distances involving the new cluster
        for idx_other in list(active_clusters.keys()):
            if idx_other == new_idx:
                continue
            dist = compute_cluster_distance(
                new_cluster, active_clusters[idx_other], embeddings, linkage_method
            )
            heapq.heappush(heap, (dist, new_idx, idx_other))

    # Collect clusters and remove those smaller than min_cluster_size
    merged_clusters = [
        list(cluster) for cluster in active_clusters.values()
        if len(cluster) >= min_cluster_size
    ]

    return merged_clusters


def compute_cluster_distance(
    cluster_a: Set[int],
    cluster_b: Set[int],
    embeddings: np.ndarray,
    linkage_method: str
) -> float:
    """
    Computes the distance between two clusters based on the specified linkage method.

    Parameters:
        cluster_a (Set[int]): Indices of elements in the first cluster.
        cluster_b (Set[int]): Indices of elements in the second cluster.
        embeddings (np.ndarray): Array of embeddings for all data points.
        linkage_method (str): Linkage method to use ('average', 'single', 'complete').

    Returns:
        float: The distance between the two clusters.
    """
    # Get embeddings for the clusters
    embeddings_a = embeddings[list(cluster_a)]
    embeddings_b = embeddings[list(cluster_b)]

    # Compute pairwise cosine distances between points in clusters
    distances = cdist(embeddings_a, embeddings_b, metric='cosine')

    if linkage_method == 'single':
        return np.min(distances)
    elif linkage_method == 'complete':
        return np.max(distances)
    elif linkage_method == 'average':
        return np.mean(distances)
    else:
        raise ValueError(f"Unsupported linkage method: {linkage_method}")

if __name__ == '__main__':
    args = setup()

    vlm_config = get_config_from_name(args.model_id)
    vlm_name = vlm_config.name

    detection_config = get_detection_config_from_name(args.detection_model)
    detection_name = detection_config.name

    with ReLAIONIndex(args.relaion_index) as relaion_index:
        for dataset in args.datasets:
            source_dir = get_full_image_folder_path(args.source_dir, dataset, args.prompt_type, args.spurious_prompt_id,
                                                    vlm_config, detection_config, agnostic=args.use_agnostic_folder)

            object_img_dirs = load_object_image_folders(source_dir)
            results_dict = load_results(object_img_dirs, detection_name, vlm_name, args.detection_threshold,
                                        prompt_id=args.spurious_prompt_id)


            total_images = 0
            clustered_images = 0
            total_clusters = 0
            laion_failed_images = 0
            total_pre_cluster_images = 0
            for obj_label in tqdm(results_dict):
                obj_results = results_dict[obj_label]
                obj_subdir = os.path.join(source_dir, obj_label)
                cluster_file_path = get_cluster_info_filepath(obj_subdir, args.linkage, args.distance_threshold)

                clusters_dict = {
                    'source_to_cluster' : {source_id: [] for source_id in obj_results['prompt_to_idx']},
                    'clusters' : [],
                }

                success_infos = []
                success_info_id_to_idx = {}
                source_id_to_pre_cluster = {}

                total_images += len(obj_results['success_ids'])

                source_pre_clusters = {}
                non_relaion_ids = set()

                for nn_id in obj_results['success_ids']:
                    nn_info = obj_results['nn_info']['file_infos'][nn_id]
                    url = nn_info['url']

                    if not relaion_index.url_exists(url):
                        # print(f'Warning: {url} not in relaion index - skipping file')
                        non_relaion_ids.add(nn_id)
                        continue

                    img_path = obj_results['img_paths'][nn_id]
                    caption = nn_info['caption']
                    embedding = nn_info[args.embedding_type]
                    cluster_info = NNInfosForClustering(nn_id, img_path, caption, embedding)
                    success_info_id_to_idx[nn_id] = len(success_infos)
                    success_infos.append(cluster_info)

                    for source_id in nn_info['idcs_similarity']:
                        if source_id not in source_id_to_pre_cluster:
                            source_id_to_pre_cluster[source_id] = []
                        source_id_to_pre_cluster[source_id].append(success_info_id_to_idx[nn_id])

                laion_failed_images += len(non_relaion_ids)
                total_pre_cluster_images += len(success_info_id_to_idx)

                if len(success_infos) < args.min_cluster_size:
                    clusters_list = []
                else:
                    clusters_list = merge_pre_clusters(success_infos, list(source_id_to_pre_cluster.values()), linkage_method=args.linkage, distance_threshold=args.distance_threshold, min_cluster_size=args.min_cluster_size)

                clustered_ids = set()

                if clusters_list:
                    success_info_idx_to_id = {v: k for (k, v) in success_info_id_to_idx.items()}
                    clusters_nn_ids = []
                    for cluster in clusters_list:
                        c_nn_ids = [success_info_idx_to_id[idx] for idx in cluster]
                        clusters_nn_ids.append(c_nn_ids)
                        clustered_ids.update(c_nn_ids)

                    source_to_cluster = {}
                    for source_id in obj_results['prompt_to_idx']:
                        source_to_cluster[source_id] = []
                        if source_id in source_id_to_pre_cluster:
                            pre_cluster = source_id_to_pre_cluster[source_id]
                            for cluster_idx, cluster in enumerate(clusters_list):
                                if all(idx in cluster for idx in pre_cluster):
                                    source_to_cluster[source_id].append(cluster_idx)

                    clusters_dict = {
                        'source_to_cluster': source_to_cluster,
                        'clusters': clusters_nn_ids,
                    }

                    clustered_images += len(clustered_ids)
                    total_clusters += len(clusters_list)

                torch.save(clusters_dict, cluster_file_path)

            clustered_images_ratio = 0 if total_images == 0 else clustered_images / total_images
            print(
                f'{dataset}: total clusters {total_clusters} - total images: {total_images} - pre_cluster images {total_pre_cluster_images} - clustered images {clustered_images} - ratio {clustered_images_ratio:.3f} - relaion failed: {laion_failed_images}')

