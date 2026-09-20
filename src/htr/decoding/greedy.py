"""Greedy decoding is beam width one without sampling."""

from dataclasses import replace

from htr.config import DecodeConfig
from htr.decoding.beam import generate_candidates


def greedy_decode(model, encoder, image, cfg: DecodeConfig) -> dict:
    return generate_candidates(
        model, encoder, image, replace(cfg, num_beams=1, num_return_sequences=1)
    )
