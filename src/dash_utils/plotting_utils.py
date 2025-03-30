import math
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import wandb
from PIL import Image
from sklearn.metrics.pairwise import cosine_distances

def order_images_by_similarity(cluster_indices, success_infos):
    """
    Order images so that neighboring images are similar.

    Parameters:
        cluster_indices (List[int]): Indices of elements in the cluster.
        success_infos (List[NNInfosForClustering]): List of NNInfosForClustering objects.

    Returns:
        List[int]: Ordered indices from cluster_indices.
    """
    # Extract embeddings for the cluster
    embeddings = np.array([success_infos[idx].embedding for idx in cluster_indices])

    # Compute pairwise cosine distances
    distance_matrix = cosine_distances(embeddings)

    num_images = len(cluster_indices)
    unvisited = set(range(num_images))
    current_idx = 0  # Start from the first image
    unvisited.remove(current_idx)
    ordering = [current_idx]

    while unvisited:
        last_idx = ordering[-1]
        # Find the nearest unvisited image
        distances_to_last = [(distance_matrix[last_idx][i], i) for i in unvisited]
        next_distance, next_idx = min(distances_to_last)
        ordering.append(next_idx)
        unvisited.remove(next_idx)

    # Map ordering indices back to cluster_indices
    ordered_cluster_indices = [cluster_indices[idx] for idx in ordering]

    return ordered_cluster_indices


def select_images_at_intervals(ordered_cluster_indices, max_images):
    """
    Select images at regular intervals from the ordered list.

    Parameters:
        ordered_cluster_indices (List[int]): Ordered indices from cluster_indices.
        max_images (int): Number of images to select.

    Returns:
        List[int]: Selected indices from cluster_indices.
    """
    total_images = len(ordered_cluster_indices)
    if total_images <= max_images:
        return ordered_cluster_indices
    else:
        # Calculate step size to sample images at regular intervals
        step = total_images / max_images
        selected_indices = [ordered_cluster_indices[int(i * step)] for i in range(max_images)]
        return selected_indices


def select_diverse_indices(cluster_indices, success_infos, max_images):
    """
    Select a subset of indices from cluster_indices of size max_images, maximizing diversity
    while ensuring neighboring images are similar.

    Parameters:
        cluster_indices (List[int]): Indices of elements in the cluster.
        success_infos (List[NNInfosForClustering]): List of NNInfosForClustering objects.
        max_images (int): Maximum number of images to select.

    Returns:
        List[int]: Selected indices from cluster_indices.
    """
    if len(cluster_indices) <= max_images:
        return cluster_indices

    # Order images by similarity
    ordered_cluster_indices = order_images_by_similarity(cluster_indices, success_infos)

    # Select images at regular intervals
    selected_indices = select_images_at_intervals(ordered_cluster_indices, max_images)

    return selected_indices


def resize_image(image: Image.Image, max_size: int = 512) -> Image.Image:
    """
    Resizes the image to have the longest edge equal to max_size pixels, preserving aspect ratio.

    Parameters:
        image (Image.Image): The original PIL image.
        max_size (int): The maximum size for the longest edge.

    Returns:
        Image.Image: The resized image.
    """
    original_size = image.size
    ratio = max_size / max(original_size)
    if ratio < 1:
        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
        image = image.resize(new_size)
    return image

def plot(trajectory_imgs, title_attributes, filename, original_image=None, original_title=None,
         original_reconstructed=None, reconstructed_title=None, loss_scores=None, regularizer_scores=None,
         model_grad_cams=None, wandb_name=None, wandb_step=None):
    scale_factor = 4.0
    num_cols = max(2, math.ceil(math.sqrt(len(trajectory_imgs))))
    num_rows = math.ceil(len(trajectory_imgs) / num_cols)

    if model_grad_cams is not None:
        num_sub_rows = 1 + model_grad_cams.shape[0]
    else:
        num_sub_rows = 1

    total_rows = 1 + num_sub_rows * num_rows

    fig, axs = plt.subplots(total_rows, num_cols, figsize=(scale_factor * num_cols, total_rows * 1.3 * scale_factor))

    # plot original:
    axs[0, 0].axis('off')
    if original_image is not None:
        img = original_image.permute(1, 2, 0).cpu().detach()
        axs[0, 0].imshow(img, interpolation='lanczos')

        if original_title is not None:
            axs[0, 0].set_title(original_title)
        else:
            axs[0, 0].set_title('Original')

    axs[0, 1].axis('off')
    if original_reconstructed is not None:
        img = original_reconstructed.permute(1, 2, 0).cpu().detach()
        axs[0, 1].imshow(img, interpolation='lanczos')

        if reconstructed_title is not None:
            axs[0, 1].set_title(reconstructed_title)
        if reconstructed_title is not None:
            axs[0, 1].set_title('Original Null-Reconstructed')

    for j in range(2, num_cols):
        axs[0, j].axis('off')

    # plot counterfactuals
    for outer_row_idx in range(0, num_rows):
        row_idx = 1 + outer_row_idx * num_sub_rows
        for sub_row_idx in range(num_sub_rows):
            for col_idx in range(num_cols):
                img_idx = outer_row_idx * num_cols + col_idx
                ax = axs[row_idx + sub_row_idx, col_idx]
                if img_idx >= len(trajectory_imgs):
                    ax.axis('off')
                    continue
                if sub_row_idx == 0:
                    img = trajectory_imgs[img_idx]
                    img = torch.clamp(img.permute(1, 2, 0), min=0.0, max=1.0)

                    ax.axis('off')
                    ax.imshow(img, interpolation='lanczos')

                    title = ''
                    if title_attributes is not None:
                        #should be a dict with tensor inside
                        for attribute_idx, (title_attribute, attribute_values) in enumerate(title_attributes.items()):
                            if attribute_idx == 0:
                                try:
                                    title += f'{img_idx} - {title_attribute}: {attribute_values[img_idx]:.5f}'
                                except:
                                    title += f'{img_idx} - {title_attribute}: {attribute_values[img_idx]}'
                            else:
                                try:
                                    title += f'\n{title_attribute}: {attribute_values[img_idx]:.5f}'
                                except:
                                    title += f'\n{img_idx} - {title_attribute}: {attribute_values[img_idx]}'

                    if loss_scores is not None:
                        for loss_name, loss_values in loss_scores.items():
                            title += f'\n{loss_name}: {loss_values[img_idx]:.5f}'

                    if regularizer_scores is not None:
                        for reg_name, reg_s in regularizer_scores.items():
                            title += f'\n{reg_name}: {reg_s[img_idx]:.5f}'

                    ax.set_title(title)
                else:
                    #heatmap
                    img = trajectory_imgs[img_idx]
                    img = torch.clamp(img.permute(1, 2, 0), min=0.0, max=1.0)
                    cam = model_grad_cams[sub_row_idx - 1, img_idx]
                    #chw to hwc
                    img_np = img.numpy()
                    cam_np = cam.permute(1, 2, 0).numpy()

                    colormap = cv2.COLORMAP_JET
                    heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), colormap)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    heatmap = np.float32(heatmap) / 255

                    image_weight = 0.5
                    cam = (1 - image_weight) * heatmap + image_weight * img_np
                    cam = cam / np.max(cam)

                    ax.axis('off')
                    ax.set_title(f'CAM classifier {sub_row_idx - 1}')

                    ax.imshow(cam, interpolation='lanczos')

    plt.tight_layout()
    fig.savefig(filename)

    if wandb_name is not None:
        wandb.log({wandb_name: fig}, step=wandb_step)

    plt.close(fig)
