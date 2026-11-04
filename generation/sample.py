"""Sample pathology images with the pretrained flow-matching DiT."""

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from common.checkpoint import load_weights
from common.config import check_inputs, create_parser, load_config, read_path, write_json
from common.runtime import device_for, start_run
from models.dit import MFDiT


@torch.no_grad()
def euler_sample(model, device, size, count, steps):
    model.eval()
    x = torch.randn(count, 3, size, size, device=device)
    for i in range(steps):
        t = torch.full((count,), 1 - i / steps, device=device)
        x = x - model(x, t, torch.zeros_like(t), y=None) / steps
    return (x.clamp(-1, 1) + 1) / 2


def main():
    parser = create_parser("sample", "Sample pathology images with flow matching")
    parser.add_argument("--checkpoint", "--ckpt")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set, command="sample", output_dir=args.output_dir)
    for flag, key in [
        ("checkpoint", "checkpoint"),
        ("num_samples", "samples"),
        ("steps", "steps"),
        ("batch_size", "batch_size"),
    ]:
        value = getattr(args, flag)
        if value is not None:
            cfg[key] = value
    checkpoint = read_path(cfg["checkpoint"])
    if min(cfg["steps"], cfg["samples"], cfg["batch_size"]) < 1:
        raise ValueError("steps, samples and batch_size must be positive")
    if args.check_only:
        check_inputs(cfg, {"checkpoint": checkpoint})
        return
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if cfg["device"] not in {"auto", "cpu"}:
        raise ValueError("device must be auto or cpu; select GPUs with CUDA_VISIBLE_DEVICES")
    device = device_for(cfg)
    with start_run(cfg) as out:
        model = MFDiT(**cfg["model"]["backbone"])
        load_weights(model, checkpoint)
        model.to(device).eval()
        for offset in tqdm(range(0, cfg["samples"], cfg["batch_size"]), desc="Sampling"):
            count = min(cfg["batch_size"], cfg["samples"] - offset)
            images = euler_sample(
                model, device, cfg["model"]["backbone"]["input_size"], count, cfg["steps"]
            )
            for i, image in enumerate(images.cpu()):
                save_image(image, out / f"sample_{offset + i + 1:05d}.png")
        write_json(out / "summary.json", {"samples": cfg["samples"], "checkpoint": str(checkpoint)})


if __name__ == "__main__":
    main()
