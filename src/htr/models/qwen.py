"""Current Transformers Qwen2.5-VL loading and multimodal response-only encoding."""

import logging
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

from htr.config import ModelConfig

logger = logging.getLogger(__name__)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "mps":
        raise ValueError("MPS is not supported; use CPU or CUDA")
    return device


def load_qwen(cfg: ModelConfig, processor_path: Path | None = None):
    device = resolve_device(cfg.device)
    dtype = cfg.dtype
    if dtype == "auto":
        dtype = (
            "bfloat16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "float32"
        )
    if dtype not in {"float32", "bfloat16"}:
        raise ValueError(
            "Supported training dtypes: float32, bfloat16, auto (fp16 needs loss scaling)"
        )
    if dtype == "bfloat16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("bfloat16 requires a supported CUDA GPU; use float32 for CPU tests")
    base_config = AutoConfig.from_pretrained(
        cfg.name, revision=cfg.revision, trust_remote_code=False
    )
    revision = getattr(base_config, "_commit_hash", None) or cfg.revision
    processor = AutoProcessor.from_pretrained(
        str(processor_path) if processor_path else cfg.name,
        **({} if processor_path else {"revision": revision}),
        min_pixels=cfg.min_pixels,
        max_pixels=cfg.max_pixels,
        trust_remote_code=False,
    )
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.name,
        revision=revision,
        config=base_config,
        dtype=getattr(torch, dtype),
        attn_implementation=cfg.attn_implementation,
        trust_remote_code=False,
    ).to(device)
    logger.info(
        "Loaded %s revision=%s dtype=%s device=%s",
        cfg.name,
        getattr(model.config, "_commit_hash", cfg.revision),
        dtype,
        device,
    )
    return model, processor


def response_labels(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, prompt_lengths: list[int]
) -> torch.Tensor:
    """Mask by positions, including left padding; never mask EOS by token ID."""
    labels = torch.full_like(input_ids, -100)
    for row, prompt_length in enumerate(prompt_lengths):
        active = attention_mask[row].nonzero(as_tuple=True)[0]
        if prompt_length >= len(active):
            raise ValueError("No response tokens available for language-model loss")
        labels[row, active[prompt_length:]] = input_ids[row, active[prompt_length:]]
    return labels


class QwenEncoder:
    """Encode image+prompt separately, append exact response tokens and one EOS.

    This avoids guessing the assistant span from text-template substring matches.
    SFT includes the assistant end-of-turn token, but no following template newline.
    Candidate token IDs can be supplied unchanged for exact generation rescoring.
    """

    def __init__(self, processor, cfg: ModelConfig):
        self.processor, self.cfg = processor, cfg
        self.processor.tokenizer.padding_side = "left"

    def prompt(self, image: Image.Image) -> dict[str, torch.Tensor]:
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": self.cfg.prompt}],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        result = dict(
            self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=False,
                return_mm_token_type_ids=True,
            )
        )
        if result["input_ids"].shape[1] >= self.cfg.max_sequence_length:
            raise ValueError("Image+prompt exceeds max_sequence_length; adjust pixel budget")
        return result

    def response_ids(self, text: str) -> list[int]:
        tokenizer = self.processor.tokenizer
        ids = tokenizer.encode(text, add_special_tokens=False)
        if set(ids) & set(tokenizer.all_special_ids):
            raise ValueError("Transcription contains reserved tokenizer control tokens")
        if tokenizer.eos_token_id is None:
            raise ValueError("Processor must define an assistant end-of-turn EOS token")
        return ids + [tokenizer.eos_token_id]

    def batch(
        self,
        images: list[Image.Image],
        texts: list[str] | None = None,
        candidate_ids: list[list[int]] | None = None,
    ) -> dict[str, torch.Tensor]:
        if (texts is None) == (candidate_ids is None):
            raise ValueError("Supply exactly one of texts or candidate_ids")
        targets = (
            candidate_ids if candidate_ids is not None else [self.response_ids(t) for t in texts]
        )
        if len(images) != len(targets) or not images:
            raise ValueError("Expected one nonempty target sequence per image")
        prompts = [self.prompt(image) for image in images]
        lengths = [p["input_ids"].shape[1] for p in prompts]
        sequences, token_types = [], []
        for prompt, ids in zip(prompts, targets, strict=True):
            if not ids:
                raise ValueError("Cannot score an empty token sequence (EOS is a valid target)")
            sequence = prompt["input_ids"][0].tolist() + ids
            if len(sequence) > self.cfg.max_sequence_length:
                raise ValueError("Response exceeds max_sequence_length; refusing silent truncation")
            sequences.append(sequence)
            types = prompt.get("mm_token_type_ids", torch.zeros_like(prompt["input_ids"]))[
                0
            ].tolist()
            token_types.append(types + [0] * len(ids))
        padded = self.processor.tokenizer.pad(
            {"input_ids": sequences}, padding=True, return_tensors="pt"
        )
        result = dict(padded)
        width = result["input_ids"].shape[1]
        result["mm_token_type_ids"] = torch.tensor(
            [[0] * (width - len(t)) + t for t in token_types]
        )
        for key in ("pixel_values", "image_grid_thw"):
            result[key] = torch.cat([p[key] for p in prompts], dim=0)
        result["labels"] = response_labels(result["input_ids"], result["attention_mask"], lengths)
        return result


def to_device(batch: dict[str, torch.Tensor], model) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    return {key: value.to(device) for key, value in batch.items()}
