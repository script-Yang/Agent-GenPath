"""Detached task execution with one log and one runtime metadata file."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import uuid

from .config import ROOT, output_path, write_json

_background_processes = []


def select_gpu(index):
    if index < 0:
        raise ValueError("GPU index must be non-negative")
    identity = subprocess.check_output(
        ["nvidia-smi", "-i", str(index), "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    if not identity.startswith("GPU-") or "\n" in identity:
        raise ValueError(f"Cannot select physical GPU {index}")
    os.environ.update(
        CUDA_VISIBLE_DEVICES=identity,
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        AGENT_GENPATH_GPU_INDEX=str(index),
    )


def launch_background(cfg, module):
    """Reserve a fresh run before starting; never overwrite an existing run."""
    out = output_path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", cfg)
    token = uuid.uuid4().hex
    env = os.environ.copy()
    env.update(
        AGENT_GENPATH_RUN_LOG=str(out / "run.log"),
        AGENT_GENPATH_LAUNCH_TOKEN=token,
        PYTHONUNBUFFERED="1",
    )
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    for key in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        env.pop(key, None)
    command = [sys.executable, "-u", "-m", "common.launch", "--worker", str(out / "config.json")]
    record = {
        "status": "starting",
        "launch_token": token,
        "launcher_pid": os.getpid(),
        "launch_command": command,
        "module": module,
        "output_dir": str(out),
        "gpu_index": env.get("AGENT_GENPATH_GPU_INDEX"),
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        "launched_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out / "runtime.json", record)
    try:
        with (out / "run.log").open("x", encoding="utf-8", buffering=1) as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _background_processes.append(process)
    except BaseException as exc:
        record.update(
            status="failed", error=str(exc), finished_at=datetime.now(timezone.utc).isoformat()
        )
        record.pop("launch_token", None)
        write_json(out / "runtime.json", record)
        raise
    # The worker owns runtime.json from this point; avoid racing its updates.
    return {
        "pid": process.pid,
        "output_dir": str(out),
        "log": str(out / "run.log"),
        "module": module,
    }


def worker(config_path):
    path = Path(config_path).resolve()
    cfg = json.loads(path.read_text())
    out = output_path(cfg["output_dir"])
    if path != out / "config.json":
        raise ValueError("Worker requires the reserved run configuration")
    record_path = out / "runtime.json"
    record = json.loads(record_path.read_text())
    if record.get("launch_token") != os.environ.get("AGENT_GENPATH_LAUNCH_TOKEN"):
        raise RuntimeError("Invalid background run reservation")
    record["pid"] = os.getpid()
    write_json(record_path, record)
    module = record["module"]
    sys.argv = [module, "--config", str(path)]
    try:
        runpy.run_module(module, run_name="__main__")
    except BaseException as exc:
        record = json.loads(record_path.read_text())
        record.pop("launch_token", None)
        record.update(
            status="failed", error=str(exc), finished_at=datetime.now(timezone.utc).isoformat()
        )
        write_json(record_path, record)
        raise
    else:
        record = json.loads(record_path.read_text())
        if record.get("status") in {"starting", "running"}:
            record.pop("launch_token", None)
            record.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat())
            write_json(record_path, record)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("Launch background runs through a task entry point")
    worker(sys.argv[2])
