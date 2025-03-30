import torch
from open_clip import create_model_from_pretrained
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class ClipDataset(Dataset):
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        from PIL import PngImagePlugin
        LARGE_ENOUGH_NUMBER = 100
        PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 2)

        try:
            img = Image.open(image_path)
            img = img.convert('RGB')
        except Exception as e:
            print(f"Could not load {image_path}: {e}")
            img = Image.new('RGB', (512, 512))

        processed = self.preprocess(img)
        return processed


def load_clip_model(device, clip_model='hf-hub:apple/DFN5B-CLIP-ViT-H-14'):
    model, preprocess = create_model_from_pretrained(clip_model)
    model.to(device)
    model.eval()
    return model, preprocess


def make_clip_dataset_dataloader(image_paths, preprocess, batch_size, num_workers=16, shuffle=False):
    dataset = ClipDataset(image_paths, preprocess)

    dataloader = DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return dataset, dataloader

@torch.inference_mode()
def compute_clip_embeddings(model, preprocess, image_paths, batch_size, device, num_workers=16):
    dataset, dataloader = make_clip_dataset_dataloader(image_paths, preprocess, batch_size, num_workers=num_workers)
    embeddings_list = []
    with torch.no_grad():
        for batch_inputs in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_embeddings = model.encode_image(batch_inputs)
            # Normalize embeddings
            batch_embeddings = batch_embeddings / batch_embeddings.norm(p=2, dim=-1, keepdim=True)
            embeddings_list.append(batch_embeddings.cpu())
    embeddings = torch.cat(embeddings_list, dim=0)
    return embeddings
