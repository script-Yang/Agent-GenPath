"""Evaluate segmentation checkpoints on held-out folds."""

import gc
import torch
from tqdm import tqdm

from common.checkpoint import load_weights
from common.config import check_inputs, create_parser, load_config, read_path, write_json
from common.runtime import device_for, start_run
from .dataset import build_loader
from .metrics import SegmentationMetrics, aggregate
from .model import build_model, predict
from .visualization import save_preview


@torch.no_grad()
def evaluate_model(model, loader, device, cfg, boundary=False, desc="Evaluation"):
    model.eval()
    metrics = SegmentationMetrics(cfg["model"]["num_classes"], boundary)
    precision = cfg.get("mixed_precision", "no")
    amp = device.type == "cuda" and precision in {"fp16", "bf16"}
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    for images, target in tqdm(loader, desc=desc, mininterval=1.0, dynamic_ncols=True):
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
            prediction = predict(model, images.to(device), cfg).argmax(1).cpu().numpy()
        metrics.update(prediction, target.numpy())
    return metrics.compute()


def evaluate_checkpoints(cfg):
    device = device_for(cfg)
    paths = [
        read_path(cfg["weights_dir"]) / cfg["weight_pattern"].format(split=i)
        for i in range(1, len(cfg["splits"]) + 1)
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    with start_run(cfg) as out:
        results = []
        for i, (path, split) in enumerate(zip(paths, cfg["splits"]), 1):
            model = build_model(cfg)
            load_weights(model, path)
            model.to(device)
            loader = build_loader(cfg, split[2])
            result = evaluate_model(
                model,
                loader,
                device,
                cfg,
                cfg.get("boundary_metrics", False),
                desc=f"Test split {i}",
            )
            results.append(result)
            write_json(out / f"split{i}_metrics.json", result)
            save_preview(model, loader, device, cfg, out / f"split{i}.png")
            del model, loader
            gc.collect()
        write_json(out / "metrics.json", {"splits": results, "summary": aggregate(results)})


def main():
    parser = create_parser("evaluate", "Evaluate segmentation checkpoints (decoder_v2 by default)")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set, command="evaluate", output_dir=args.output_dir)
    if args.check_only:
        paths = {"data.root": cfg["data"]["root"]}
        for i in range(1, len(cfg["splits"]) + 1):
            paths[f"split{i}"] = read_path(cfg["weights_dir"]) / cfg["weight_pattern"].format(
                split=i
            )
        check_inputs(cfg, paths)
        return
    evaluate_checkpoints(cfg)


if __name__ == "__main__":
    main()
