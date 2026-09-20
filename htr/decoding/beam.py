"""Deterministic model-score-only search; no references or risk arguments."""

import time
from dataclasses import asdict, replace

import torch
from PIL import Image

from htr.config import DecodeConfig
from htr.models.qwen import QwenEncoder, to_device


def greedy_decode(model, encoder, image, cfg: DecodeConfig) -> dict:
    return generate_candidates(
        model, encoder, image, replace(cfg, num_beams=1, num_return_sequences=1)
    )


def synchronize(model) -> None:
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def generate_candidates(model, encoder: QwenEncoder, image: Image.Image, cfg: DecodeConfig) -> dict:
    """Keep exact generated IDs through first EOS; truncated candidates get no added EOS."""
    if not 1 <= cfg.num_return_sequences <= cfg.num_beams:
        raise ValueError("Require 1 <= num_return_sequences <= num_beams")
    model.eval()
    prompt = to_device(encoder.prompt(image), model)
    prefix_length = prompt["input_ids"].shape[1]
    if prefix_length + cfg.max_new_tokens > encoder.cfg.max_sequence_length:
        raise ValueError("Prompt + max_new_tokens exceeds max_sequence_length")
    synchronize(model)
    started = time.perf_counter()
    outputs = model.generate(
        **prompt,
        do_sample=False,
        num_beams=cfg.num_beams,
        num_return_sequences=cfg.num_return_sequences,
        max_new_tokens=cfg.max_new_tokens,
        length_penalty=cfg.length_penalty,
        return_dict_in_generate=True,
        output_scores=True,
        use_cache=True,
        pad_token_id=encoder.processor.tokenizer.pad_token_id,
    )
    synchronize(model)
    elapsed = time.perf_counter() - started
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    rows = []
    for sequence in outputs.sequences:
        ids = sequence[prefix_length:].tolist()
        ended = False
        for i, token in enumerate(ids):
            if token in eos_ids:
                ids, ended = ids[: i + 1], True
                break
        text = encoder.processor.tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        rows.append(
            {
                "text": text,
                "token_ids": ids,
                "ended_with_eos": ended,
                "output_tokens": len(ids),
                "output_chars": len(text),
            }
        )
    return {
        "candidates": rows,
        "generation_seconds": elapsed,
        "generation_config": {
            k: v for k, v in asdict(cfg).items() if k not in {"input_csv", "output"}
        },
    }
