"""Supervised transcription NLL; shared loop lives in engine.py."""

from htr.models.qwen import QwenEncoder, to_device


def sft_loss(model, encoder: QwenEncoder, images: list, texts: list):
    batch = to_device(encoder.batch(images, texts=texts), model)
    return model(**batch, use_cache=False).loss
