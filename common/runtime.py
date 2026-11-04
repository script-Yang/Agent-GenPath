"""Run lifecycle, logging, and device selection."""

import json
import os
import sys
import traceback
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path

from .config import output_path, write_json


class _Tee:
    def __init__(self, stream, log):
        self.stream = stream
        self.log = log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text.replace("\r", "\n"))
        self.log.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def external_run_log(out):
    value = os.environ.get("AGENT_GENPATH_RUN_LOG")
    if not value:
        return False
    path = out / "run.log"
    if Path(value).resolve() != path.resolve():
        raise RuntimeError("Background log does not match the selected output directory")
    expected = path.stat()
    for fd in (1, 2):
        actual = os.fstat(fd)
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            raise RuntimeError("Background stdout/stderr must point to run.log")
    return True


@contextmanager
def run_logging(out, rank=0):
    if external_run_log(out):
        print(f"[{datetime.now(timezone.utc).isoformat()}] Run started", flush=True)
        try:
            yield
        except BaseException:
            # The uncaught traceback is already captured by the same log descriptor.
            print(f"[{datetime.now(timezone.utc).isoformat()}] Run failed", flush=True)
            raise
        else:
            print(f"[{datetime.now(timezone.utc).isoformat()}] Run completed", flush=True)
        return
    name = "run.log" if rank == 0 else f"rank{rank}.log"
    with (out / name).open("x", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
            print(f"[{datetime.now(timezone.utc).isoformat()}] Run started", flush=True)
            try:
                yield
            except BaseException:
                traceback.print_exc(file=log)
                print(f"[{datetime.now(timezone.utc).isoformat()}] Run failed", flush=True)
                raise
            else:
                print(f"[{datetime.now(timezone.utc).isoformat()}] Run completed", flush=True)


@contextmanager
def start_run(cfg, accelerator=None):
    out = output_path(cfg["output_dir"])
    # Broadcast setup failures before synchronizing ranks.
    error = None
    if accelerator is None or accelerator.is_main_process:
        try:
            launch = {}
            if external_run_log(out):
                launch = json.loads((out / "runtime.json").read_text())
                if (
                    launch.get("status") != "starting"
                    or launch.pop("launch_token", None)
                    != os.environ.get("AGENT_GENPATH_LAUNCH_TOKEN")
                    or json.loads((out / "config.json").read_text()) != cfg
                ):
                    raise RuntimeError("Invalid or already-used background run reservation")
            else:
                out.mkdir(parents=True, exist_ok=False)
                write_json(out / "config.json", cfg)
            import torch

            device = (
                accelerator.device
                if accelerator is not None
                else torch.device(
                    "cpu" if cfg["device"] == "cpu" or not torch.cuda.is_available() else "cuda"
                )
            )
            write_json(
                out / "runtime.json",
                {
                    **launch,
                    "status": "running",
                    "pid": os.getpid(),
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "argv": sys.argv,
                    "working_directory": str(Path.cwd()),
                    "python_executable": sys.executable,
                    "python_version": sys.version.split()[0],
                    "torch_version": str(torch.__version__),
                    "cuda_version": torch.version.cuda,
                    "device": str(device),
                    "gpu_name": (
                        torch.cuda.get_device_name(device) if device.type == "cuda" else None
                    ),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "mixed_precision": (
                        accelerator.mixed_precision
                        if accelerator is not None
                        else (
                            cfg.get("mixed_precision", "no")
                            if device.type == "cuda" and cfg.get("command") == "evaluate"
                            else "no"
                        )
                    ),
                    "num_processes": accelerator.num_processes if accelerator is not None else 1,
                },
            )
        except Exception as exc:
            error = str(exc)
    if accelerator is not None and accelerator.num_processes > 1:
        from accelerate.utils import broadcast_object_list

        message = broadcast_object_list([error])
        error = message[0]
    if error:
        raise RuntimeError(f"Cannot create a fresh run: {error}. Choose a new output_dir.")
    if accelerator is not None:
        accelerator.wait_for_everyone()
    main_process = accelerator is None or accelerator.is_main_process
    status = "completed"
    try:
        with run_logging(out, accelerator.process_index if accelerator is not None else 0):
            yield out
    except BaseException:
        status = "failed"
        raise
    finally:
        if main_process:
            path = out / "runtime.json"
            runtime = json.loads(path.read_text())
            runtime.update(status=status, finished_at=datetime.now(timezone.utc).isoformat())
            write_json(path, runtime)


def accelerator_for(cfg):
    import torch
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, set_seed

    cpu = cfg["device"] == "cpu" or not torch.cuda.is_available()
    if cfg["device"] not in {"auto", "cpu"}:
        raise ValueError("Use device=cpu/auto and CUDA_VISIBLE_DEVICES to select GPUs")
    accelerator = Accelerator(
        cpu=cpu,
        mixed_precision="no" if cpu else cfg["mixed_precision"],
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    set_seed(cfg["seed"], device_specific=True)
    return accelerator


def device_for(cfg):
    import torch
    from accelerate.utils import set_seed

    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError("This command is single-process; launch it with python, not accelerate.")
    set_seed(cfg["seed"])
    return torch.device(
        "cpu" if cfg["device"] == "cpu" or not torch.cuda.is_available() else "cuda"
    )
