"""Adapter/merger bundles and epoch-boundary optimizer/RNG resume state."""

import random
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from safetensors.torch import load_file, save_file

from htr.config import Config, LoraConfig
from htr.models.lora import inspect_layout, merger_module, set_trainable, unwrap
from htr.models.qwen import load_qwen
from htr.utils.io import file_hash, read_json, write_json


def bundle_hash(directory: Path) -> str:
    from htr.utils.io import digest

    return digest(
        {
            p.relative_to(directory).as_posix(): file_hash(p)
            for p in sorted(directory.rglob("*"))
            if p.is_file()
            and (
                p.name
                in {
                    "adapter_model.safetensors",
                    "adapter_config.json",
                    "merger.safetensors",
                    "bundle.json",
                }
                or p.parent.name in {"processor", "base_config"}
            )
        }
    )


def save_bundle(
    model, processor, cfg: Config, directory: Path, stage: str, split_hash: str
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True)
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in merger_module(model).state_dict().items()},
        str(directory / "merger.safetensors"),
    )
    processor.save_pretrained(directory / "processor")
    unwrap(model).config.save_pretrained(directory / "base_config")
    write_json(
        directory / "bundle.json",
        {
            "stage": stage,
            "base_model": cfg.model.name,
            "base_revision": getattr(unwrap(model).config, "_commit_hash", None)
            or cfg.model.revision,
            "train_split_hash": split_hash,
            "model_config": asdict(cfg.model),
            "crop": asdict(cfg.crop),
            "lora": asdict(cfg.lora),
            "training_config": cfg.to_dict(),
        },
    )


def load_bundle(cfg: Config, directory: Path, stage: str):
    meta = read_json(directory / "bundle.json")
    if meta["base_model"] != cfg.model.name:
        raise ValueError("Checkpoint base model differs from model.name")
    # Debug destinations do not change pixels.
    current_crop = {k: v for k, v in asdict(cfg.crop).items() if k != "debug_dir"}
    stored_crop = {k: v for k, v in meta["crop"].items() if k != "debug_dir"}
    if current_crop != stored_crop:
        raise ValueError("Checkpoint and current crop differ; use the training preprocessing")
    for key in ("prompt", "min_pixels", "max_pixels"):
        if getattr(cfg.model, key) != meta["model_config"][key]:
            raise ValueError(f"Checkpoint and current model.{key} differ")
    if stage != "inference" and asdict(cfg.lora) != meta["lora"]:
        raise ValueError("Checkpoint LoRA config differs from requested training config")
    model, processor = load_qwen(
        replace(cfg.model, revision=meta["base_revision"]), directory / "processor"
    )
    layout = inspect_layout(model, LoraConfig(**meta["lora"]))
    adapter_config = read_json(directory / "adapter_config.json")
    if set(adapter_config["target_modules"]) != set(layout.targets):
        raise ValueError("Saved adapter target modules do not match inspected Qwen LLM projections")
    model = PeftModel.from_pretrained(model, directory, is_trainable=stage != "inference")
    merger_module(model).load_state_dict(
        load_file(str(directory / "merger.safetensors")), strict=True
    )
    set_trainable(model, stage)
    return model, processor, meta


def save_training_state(path: Path, optimizer, epoch: int, best: float, stale: int) -> None:
    numpy_state = np.random.get_state()
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best": best,
            "stale": stale,
            "python_rng": random.getstate(),
            "numpy_rng": [
                numpy_state[0],
                numpy_state[1].tolist(),
                numpy_state[2],
                numpy_state[3],
                numpy_state[4],
            ],
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        path,
    )


def restore_training_state(path: Path, optimizer) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    random.setstate(state["python_rng"])
    ns = state["numpy_rng"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state
