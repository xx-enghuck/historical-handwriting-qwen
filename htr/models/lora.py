"""Inspect the actual Qwen module tree and enforce the research freeze policy."""

import logging
from dataclasses import dataclass

from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model
from torch import nn

from htr.config import LoraConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelLayout:
    vision: str
    merger: str
    language: str
    targets: tuple[str, ...]


def unwrap(model: nn.Module) -> nn.Module:
    return model.get_base_model() if hasattr(model, "peft_config") else model


def inspect_layout(model: nn.Module, cfg: LoraConfig) -> ModelLayout:
    """Discover full module names; never match projection suffixes in the vision tower."""
    model = unwrap(model)
    modules = dict(model.named_modules())
    visual = [n for n in modules if n.split(".")[-1] == "visual"]
    language = [n for n in modules if n.split(".")[-1] == "language_model"]
    if len(visual) != 1 or len(language) != 1:
        raise ValueError(
            "Expected one Qwen visual and language_model module; check pinned Transformers API"
        )
    merger = visual[0] + ".merger"
    if merger not in modules:
        raise ValueError(f"Visual merger {merger} is missing; inspect the installed model")
    targets = tuple(
        n
        for n, module in modules.items()
        if n.startswith(language[0] + ".")
        and n.split(".")[-1] in cfg.targets
        and isinstance(module, nn.Linear)
    )
    missing = set(cfg.targets) - {n.split(".")[-1] for n in targets}
    if missing or not targets:
        raise ValueError(f"Missing LLM LoRA projections: {sorted(missing)}; found {targets}")
    return ModelLayout(visual[0], merger, language[0], targets)


def add_lora(model: nn.Module, cfg: LoraConfig) -> nn.Module:
    layout = inspect_layout(model, cfg)
    model.requires_grad_(False)
    adapted = get_peft_model(
        model,
        PeftLoraConfig(
            task_type="CAUSAL_LM",
            r=cfg.r,
            lora_alpha=cfg.alpha,
            lora_dropout=cfg.dropout,
            bias=cfg.bias,
            target_modules=list(layout.targets),
        ),
    )
    set_trainable(adapted, "sft")
    return adapted


def is_adapter(name: str) -> bool:
    return ".lora_A." in name or ".lora_B." in name


def set_trainable(model: nn.Module, stage: str) -> dict:
    """SFT: LoRA + merger. MWER: LoRA only. Frozen base parameters are verified."""
    if stage not in {"sft", "mwer", "inference"}:
        raise ValueError(f"Unknown stage: {stage}")
    base = unwrap(model)
    merger_names = [n for n, _ in base.named_modules() if n.endswith("visual.merger")]
    if len(merger_names) != 1:
        raise ValueError("Cannot uniquely locate visual.merger")
    merger_name = merger_names[0]
    adapters = 0
    for name, parameter in base.named_parameters():
        adapter = is_adapter(name)
        if adapter:
            if ".language_model." not in "." + name:
                raise ValueError(f"LoRA outside LLM: {name}")
            adapters += 1
        allowed = stage != "inference" and (
            adapter or (stage == "sft" and name.startswith(merger_name + "."))
        )
        parameter.requires_grad_(allowed)
    if stage != "inference" and not adapters:
        raise ValueError("No LoRA adapter parameters found")
    return parameter_report(model)


def parameter_report(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    count = sum(trainable.values())
    for name, size in trainable.items():
        logger.info("Trainable: %s (%d)", name, size)
    report = {
        "total_parameters": total,
        "trainable_parameters": count,
        "trainable_percentage": 100 * count / max(total, 1),
        "trainable_modules": trainable,
    }
    logger.info(
        "Parameters: %d / %d trainable (%.4f%%)", count, total, report["trainable_percentage"]
    )
    return report


def merger_module(model: nn.Module) -> nn.Module:
    matches = [
        module for name, module in unwrap(model).named_modules() if name.endswith("visual.merger")
    ]
    if len(matches) != 1:
        raise ValueError("Cannot uniquely locate merger")
    return matches[0]


def train_mode(model: nn.Module, stage: str) -> None:
    """Disable frozen vision stochastic layers; retain SFT merger gradients."""
    model.train()
    for name, module in unwrap(model).named_modules():
        if name.endswith("visual"):
            module.eval()
    if stage == "sft":
        merger_module(model).train()
    # MWER uses deterministic teacher-forced probabilities (dropout disabled),
    # while autograd remains enabled in eval mode.
    if stage == "mwer":
        model.eval()
