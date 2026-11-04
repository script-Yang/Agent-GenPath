"""Image, ground-truth, and prediction previews."""

import torch
from torchvision.utils import save_image

from .model import predict


@torch.no_grad()
def save_preview(model, loader, device, cfg, path):
    images, masks = next(iter(loader))
    images, masks = images[:4].to(device), masks[:4].to(device)
    pred = predict(model, images, cfg).argmax(1)
    palette = (
        torch.tensor(
            [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255]],
            device=device,
        )
        / 255
    )
    if cfg["model"]["num_classes"] > len(palette):
        ids = torch.arange(cfg["model"]["num_classes"], device=device)
        palette = torch.stack([(ids * 71) % 256, (ids * 137) % 256, (ids * 223) % 256], dim=1) / 255
    colored = [palette[m].permute(0, 3, 1, 2) for m in [masks, pred]]
    save_image(torch.cat([(images + 1).clamp(0, 2) / 2] + colored, dim=3), path, nrow=1)
