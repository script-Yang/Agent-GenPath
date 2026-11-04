"""Read-only semantic segmentation dataset."""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from common.config import read_path
from torchvision.transforms import functional as TF


class CellDataset(Dataset):
    def __init__(self, root, fold, size=256, num_classes=6):
        root = Path(root)
        self.samples = []
        self.size = size
        self.num_classes = num_classes
        images, masks = root / "images" / fold, root / "masks" / fold
        if not images.is_dir() or not masks.is_dir():
            raise FileNotFoundError(f"Missing image/mask fold: {root}, {fold}")
        for image in sorted(images.glob("*.npy")):
            mask = masks / image.name
            if not mask.is_file():
                raise FileNotFoundError(mask)
            self.samples.append((image, mask))
        if not self.samples:
            raise ValueError(f"Empty fold: {fold}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        image = np.load(image_path, allow_pickle=False)
        mask = np.load(mask_path, allow_pickle=False)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.max() > 1:
            image = image / 255.0
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        if mask.ndim != 2:
            raise ValueError(f"Expected semantic HxW mask, got {mask.shape}: {mask_path}")
        if (
            mask.min() < 0
            or mask.max() >= self.num_classes
            or not np.equal(mask, np.floor(mask)).all()
        ):
            raise ValueError(f"Mask class IDs outside 0..{self.num_classes-1}: {mask_path}")
        x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1)
        y = torch.from_numpy(np.asarray(mask, dtype=np.float32)).unsqueeze(0)
        x = TF.resize(x, [self.size] * 2, interpolation=TF.InterpolationMode.BILINEAR)
        y = TF.resize(y, [self.size] * 2, interpolation=TF.InterpolationMode.NEAREST)[0].long()
        return TF.normalize(x, [0.5] * 3, [0.5] * 3), y


def build_loader(cfg, fold, train=False):
    dataset = CellDataset(
        read_path(cfg["data"]["root"]),
        fold,
        cfg["model"]["backbone"]["input_size"],
        cfg["model"]["num_classes"],
    )
    return DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size" if train else "eval_batch_size"],
        shuffle=train,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
