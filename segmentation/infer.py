"""Infer semantic masks for an image or directory; no ground-truth masks needed."""

import json
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF
from tqdm import tqdm
from common.checkpoint import load_weights
from common.config import create_parser, load_config, read_path, output_path, write_json
from common.runtime import device_for, start_run
from .model import build_model, predict


def main():
    parser = create_parser("infer", "Predict segmentation masks (decoder_v2 by default)")
    parser.add_argument("--input", help="Image file or directory")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set, command="infer", output_dir=args.output_dir)
    if args.checkpoint:
        cfg["checkpoint"] = args.checkpoint
    out = output_path(cfg["output_dir"])
    checkpoint = read_path(cfg["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.check_only:
        print(
            json.dumps(
                {"checkpoint": str(checkpoint), "exists": True, "output_dir": str(out)},
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if not args.input:
        parser.error("--input is required for inference")
    source = read_path(args.input)
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if source.is_dir():
        files = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)
        base = source
    elif source.is_file() and source.suffix.lower() in suffixes:
        files, base = [source], source.parent
    else:
        raise ValueError("Input must be a supported image or directory: " + str(source))
    if not files:
        raise ValueError("No images found under " + str(source))
    device = device_for(cfg)
    model = build_model(cfg)
    load_weights(model, checkpoint)
    model.to(device).eval()
    cfg["input"] = str(source)
    with start_run(cfg) as out:
        size = cfg["model"]["backbone"]["input_size"]
        classes = cfg["model"]["num_classes"]
        if classes > 256:
            raise ValueError("PNG mask export supports up to 256 classes")
        palette = np.array(
            [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255]],
            dtype=np.uint8,
        )
        if classes > len(palette):
            ids = np.arange(classes)
            palette = np.stack(
                [(ids * 71) % 256, (ids * 137) % 256, (ids * 223) % 256], axis=1
            ).astype(np.uint8)
        with torch.inference_mode():
            for path in tqdm(files, desc="Segmentation"):
                with Image.open(path) as opened:
                    image = opened.convert("RGB")
                    original_size = image.size
                    tensor = TF.to_tensor(image)
                tensor = TF.resize(
                    tensor, [size, size], interpolation=TF.InterpolationMode.BILINEAR
                )
                tensor = TF.normalize(tensor, [0.5] * 3, [0.5] * 3).unsqueeze(0).to(device)
                mask = predict(model, tensor, cfg).argmax(1)[0].cpu().numpy().astype(np.uint8)
                # Nearest-neighbor resizing preserves class IDs.
                mask_image = Image.fromarray(mask).resize(original_size, Image.Resampling.NEAREST)
                relative = path.relative_to(base)
                # Retain the input extension to avoid output-name collisions.
                name = relative.parent / (relative.name + ".png")
                mask_path, color_path = out / "masks" / name, out / "colors" / name
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                color_path.parent.mkdir(parents=True, exist_ok=True)
                mask_image.save(mask_path)
                Image.fromarray(palette[np.asarray(mask_image)]).save(color_path)
        write_json(
            out / "summary.json",
            {
                "images": len(files),
                "checkpoint": str(checkpoint),
                "num_classes": classes,
                "rate": cfg["training"]["rate"],
                "mask_values": "class IDs, 0 is background",
            },
        )


if __name__ == "__main__":
    main()
