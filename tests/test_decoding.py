from copy import deepcopy

import torch
from PIL import Image

from htr.config import DecodeConfig, ModelConfig
from htr.decoding.beam import generate_candidates
from htr.models.qwen import QwenEncoder


def test_greedy_beam_no_reference_and_no_gradient(tiny_components):
    torch.set_num_threads(1)
    model, processor = tiny_components
    encoder = QwenEncoder(processor, ModelConfig(prompt="Transcribe."))
    image = Image.new("RGB", (32, 16), "gray")
    for beams in (1, 2, 4):
        cfg = DecodeConfig(num_beams=beams, num_return_sequences=beams, max_new_tokens=3)
        first = generate_candidates(model, encoder, image, cfg)
        second = generate_candidates(model, encoder, image, deepcopy(cfg))
        assert first["candidates"] == second["candidates"]
        assert len(first["candidates"]) == beams
        assert all(p.grad is None for p in model.parameters())
