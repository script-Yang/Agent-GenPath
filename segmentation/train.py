"""Train and validate independent segmentation splits."""

import gc
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm

from common.checkpoint import load_weights
from common.config import check_inputs, create_parser, load_config, read_path, write_json
from common.runtime import accelerator_for, start_run
from .dataset import build_loader
from .evaluate import evaluate_model
from .losses import dice_loss
from .metrics import aggregate
from .model import build_model, predict
from .visualization import save_preview


def train(cfg):
    accel = accelerator_for(cfg)
    args = cfg["training"]
    if args["epochs"] < 1:
        raise ValueError("epochs must be positive")
    for split in cfg["splits"]:
        if len(split) != 3 or len(set(split)) != 3:
            raise ValueError(f"Train/validation/test folds must be distinct: {split}")
    if cfg.get("pretrained") and not read_path(cfg["pretrained"]).is_file():
        raise FileNotFoundError(cfg["pretrained"])
    with start_run(cfg, accel) as out:
        accel.print(f"Output: {out}; device: {accel.device}; precision: {accel.mixed_precision}")
        all_results = []
        for split_idx, (train_fold, val_fold, test_fold) in enumerate(cfg["splits"], 1):
            model = build_model(cfg, initialize=True)
            params = [
                {"params": model.backbone.parameters(), "lr": args["backbone_lr"]},
                {"params": model.decoder.parameters(), "lr": args["lr"]},
            ]
            optimizer = torch.optim.AdamW(params, lr=args["lr"], weight_decay=args["weight_decay"])
            train_loader = build_loader(cfg, train_fold, True)
            # Shard training only; rank zero evaluates complete folds.
            val_loader = build_loader(cfg, val_fold)
            test_loader = build_loader(cfg, test_fold)
            model, optimizer, train_loader = accel.prepare(model, optimizer, train_loader)
            split_out = out / f"split{split_idx}"
            if accel.is_main_process:
                split_out.mkdir()
                write_json(
                    split_out / "folds.json",
                    {
                        "train": train_fold,
                        "validation": val_fold,
                        "test": test_fold,
                        "samples": {
                            "train": len(train_loader.dataset),
                            "validation": len(val_loader.dataset),
                            "test": len(test_loader.dataset),
                        },
                    },
                )
            best = float("-inf")
            history = []
            for epoch in range(1, args["epochs"] + 1):
                model.train()
                losses = torch.zeros(2, device=accel.device)
                progress = tqdm(
                    train_loader,
                    desc=f'Train split {split_idx} epoch {epoch}/{args["epochs"]}',
                    disable=not accel.is_main_process,
                    mininterval=1.0,
                    dynamic_ncols=True,
                )
                for images, masks in progress:
                    optimizer.zero_grad(set_to_none=True)
                    logits = predict(model, images, cfg)
                    loss = F.cross_entropy(logits.float(), masks) + dice_loss(
                        logits, masks, cfg["model"]["num_classes"]
                    )
                    accel.backward(loss)
                    optimizer.step()
                    losses += torch.tensor(
                        [loss.detach().item() * len(images), len(images)], device=accel.device
                    )
                    progress.set_postfix(loss=float(loss.detach()), refresh=False)
                losses = accel.reduce(losses, reduction="sum")
                accel.wait_for_everyone()
                if accel.is_main_process:
                    plain = accel.unwrap_model(model)
                    metrics = evaluate_model(
                        plain, val_loader, accel.device, cfg, desc=f"Validation split {split_idx}"
                    )
                    history.append(
                        {
                            "epoch": epoch,
                            "train_loss": float(losses[0] / losses[1]),
                            "validation": metrics,
                        }
                    )
                    write_json(split_out / "history.json", history)
                    torch.save(plain.state_dict(), split_out / "latest.pth")
                    if metrics["mIoU"] > best:
                        best = metrics["mIoU"]
                        torch.save(plain.state_dict(), split_out / "best_mIoU.pth")
                    save_preview(plain, val_loader, accel.device, cfg, split_out / "validation.png")
                    print(json.dumps(history[-1]), flush=True)
                accel.wait_for_everyone()
            if accel.is_main_process:
                plain = accel.unwrap_model(model)
                load_weights(plain, split_out / "best_mIoU.pth")
                result = evaluate_model(
                    plain,
                    test_loader,
                    accel.device,
                    cfg,
                    cfg.get("boundary_metrics", False),
                    desc=f"Test split {split_idx}",
                )
                all_results.append(result)
                write_json(split_out / "test_metrics.json", result)
                print(json.dumps({"split": split_idx, "test": result}), flush=True)
                save_preview(plain, test_loader, accel.device, cfg, split_out / "test.png")
                del plain
            accel.wait_for_everyone()
            del model, optimizer, train_loader, val_loader, test_loader, params
            accel.free_memory()
            gc.collect()
        if accel.is_main_process:
            write_json(
                out / "metrics.json", {"splits": all_results, "summary": aggregate(all_results)}
            )


def main():
    parser = create_parser("train", "Train segmentation (decoder_v2 by default)")
    parser.add_argument("--gpu", type=int, help="Physical GPU index from nvidia-smi")
    parser.add_argument("--background", action="store_true", help="Detach with one run.log")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set, command="train", output_dir=args.output_dir)
    if args.check_only:
        paths = {"data.root": cfg["data"]["root"]}
        if cfg.get("pretrained"):
            paths["pretrained"] = cfg["pretrained"]
        check_inputs(cfg, paths)
        return
    from common.launch import select_gpu, launch_background

    if args.gpu is not None:
        select_gpu(args.gpu)
    if args.background:
        print(json.dumps(launch_background(cfg, "segmentation.train"), indent=2))
        return
    train(cfg)


if __name__ == "__main__":
    main()
