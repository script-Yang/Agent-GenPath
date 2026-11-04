"""Repository-relative configuration and output paths."""

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def output_path(value):
    path = read_path(value)
    allowed = (ROOT / "outputs").resolve()
    if path == allowed or allowed not in path.parents:
        raise ValueError(f"Output must be a run directory under {allowed}: {path}")
    return path


def next_run_path(cfg):
    command = cfg["command"]
    if command == "sample":
        parent, name = ROOT / "outputs/generation", "samples"
    else:
        if command not in {"train", "infer", "evaluate"}:
            raise ValueError(f"Unknown command: {command}")
        head = cfg["model"].get("head", "decoder_v2")
        if head not in {"decoder_v2", "decoder_v1"}:
            raise ValueError(f"Unknown model.head: {head!r}")
        rate = float(cfg["training"]["rate"])
        if not math.isfinite(rate):
            raise ValueError("training.rate must be finite")
        parent = ROOT / "outputs/segmentation"
        name = f"{command}_{head}_t{rate:g}"
    path = parent / name
    number = 2
    while path.exists():
        path = parent / f"{name}_{number}"
        number += 1
    return path


def load_config(path, overrides=(), *, command=None, output_dir=None):
    cfg = json.loads(read_path(path).read_text(encoding="utf-8"))
    for item in overrides:
        key, value = item.split("=", 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        parent = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in parent or not isinstance(parent[part], dict):
                raise ValueError(f"Unknown configuration section: {key}")
            parent = parent[part]
        if parts[-1] not in parent:
            raise ValueError(f"Unknown configuration key: {key}")
        parent[parts[-1]] = value
    if command is not None:
        cfg["command"] = command
    if output_dir is not None:
        cfg["output_dir"] = output_dir
    if cfg.get("output_dir", "auto") == "auto":
        cfg["output_dir"] = str(next_run_path(cfg))
    output_path(cfg["output_dir"])
    return cfg


def write_json(path, data):
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def create_parser(command, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(ROOT / "configs" / f"{command}.json"))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--output-dir", help="A fresh run directory under outputs/")
    parser.add_argument("--check-only", action="store_true")
    return parser


def check_inputs(cfg, paths):
    inputs = {
        key: {"path": str(read_path(value)), "exists": read_path(value).exists()}
        for key, value in paths.items()
    }
    print(
        json.dumps({"output_dir": str(output_path(cfg["output_dir"])), "inputs": inputs}, indent=2)
    )
    if any(not item["exists"] for item in inputs.values()):
        raise SystemExit(1)
