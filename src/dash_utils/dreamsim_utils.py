import torch
from tqdm import tqdm
from dreamsim import dreamsim
from .clip_utils import make_clip_dataset_dataloader

def load_dreamsim_model(device):
    model, preprocess = dreamsim(pretrained=True, device=device)
    return model, preprocess

@torch.inference_mode()
def compute_dreamsim_embeddings(model, preprocess, image_paths, batch_size, device, num_workers=16):
    preprocess_squeeze = lambda x: preprocess(x).squeeze(dim=0)
    bs = min(batch_size, len(image_paths))
    num_workers = min(num_workers, bs)
    dataset, dataloader = make_clip_dataset_dataloader(image_paths, preprocess_squeeze, bs, num_workers=num_workers)
    embeddings = compute_dreamsim_embeddings_for_dataloader(model, dataloader, device)
    return embeddings

@torch.inference_mode()
def compute_dreamsim_embeddings_for_dataloader(model, dataloader, device, progress=False):
    embeddings_list = []
    with torch.no_grad():
        with tqdm(disable= not progress) as pbar:
            for batch_inputs in dataloader:
                if isinstance(batch_inputs, list):
                    batch_inputs = batch_inputs[0]
                batch_inputs = batch_inputs.to(device)
                batch_embeddings = model.embed(batch_inputs)
                embeddings_list.append(batch_embeddings.cpu())
                pbar.update(1)

    embeddings = torch.cat(embeddings_list, dim=0)
    return embeddings
