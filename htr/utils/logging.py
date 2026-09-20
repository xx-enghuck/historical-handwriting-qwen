"""Local JSON/JSONL experiment tracking without external accounts."""

import importlib.metadata
import logging
import platform
import subprocess
import time
from pathlib import Path

import torch

from htr.config import Config
from htr.utils.io import read_json, write_json


def environment(root: Path) -> dict:
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            dist.metadata["Name"]: dist.version
            for dist in importlib.metadata.distributions()
            if dist.metadata["Name"]
        },
        "cuda": torch.version.cuda,
        "gpus": [
            {
                "name": torch.cuda.get_device_name(i),
                "total_memory": torch.cuda.get_device_properties(i).total_memory,
            }
            for i in range(torch.cuda.device_count())
        ],
    }


class RunLogger:
    def __init__(self, cfg: Config, stage: str, split_hashes: dict, parameters: dict):
        self.directory = cfg.path(cfg.train.output_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        result_path = self.directory / "result.json"
        self.previous_runtime = (
            read_json(result_path).get("runtime_seconds", 0.0)
            if cfg.train.resume and result_path.exists()
            else 0.0
        )
        meta = {
            "stage": stage,
            "config": cfg.to_dict(),
            "seed": cfg.seed,
            "split_hashes": split_hashes,
            "parameters": parameters,
            "environment": environment(Path(cfg.project_root)),
        }
        self.metadata = meta
        name = "resume_metadata.json" if cfg.train.resume else "metadata.json"
        write_json(self.directory / name, meta)

    def event(self, values: dict) -> None:
        import json

        values = {**values, "session_runtime_seconds": time.perf_counter() - self.started}
        with (self.directory / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(values, allow_nan=False) + "\n")
        logging.getLogger(__name__).info("%s", values)

    def finish(self, values: dict) -> None:
        write_json(
            self.directory / "result.json",
            {
                **self.metadata,
                **values,
                "runtime_seconds": self.previous_runtime + time.perf_counter() - self.started,
            },
        )
