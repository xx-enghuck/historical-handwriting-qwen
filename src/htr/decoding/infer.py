"""Image-only inference entry point, kept separate from reference-based evaluation."""

import time
from pathlib import Path

from htr.config import Config
from htr.data.dataset import image_set_hash, read_samples, samples_hash
from htr.data.split import load_splits
from htr.decoding.beam import generate_candidates
from htr.models.lora import parameter_report
from htr.models.qwen import QwenEncoder, load_qwen
from htr.training.checkpoint import bundle_hash, load_bundle
from htr.training.engine import prepare_image
from htr.utils.io import file_hash, write_json, write_jsonl
from htr.utils.logging import environment
from htr.utils.seed import seed_everything


def infer(cfg: Config, split: str = "validation", components: tuple | None = None) -> list[dict]:
    seed_everything(cfg.seed, cfg.deterministic)
    started = time.perf_counter()
    if cfg.decode.num_return_sequences != 1:
        raise ValueError("Final inference returns one hypothesis; use generate-nbest for MWER")
    if cfg.decode.input_csv:
        # Deliberately do not read the transcription column even when it exists.
        samples = read_samples(
            cfg.path(cfg.decode.input_csv), cfg.path(cfg.data.image_root), require_labels=False
        )
        split = "external"
    else:
        samples = load_splits(cfg)[split]
    if not samples:
        raise ValueError(f"Empty inference split: {split}")
    if components:
        model, processor = components
    elif cfg.train.checkpoint:
        model, processor, _ = load_bundle(cfg, cfg.path(cfg.train.checkpoint), "inference")
    else:
        model, processor = load_qwen(cfg.model)
        model.requires_grad_(False)
    encoder = QwenEncoder(processor, cfg.model)
    # Fixed one-image warm-up, excluded from reported per-image generation time.
    generate_candidates(model, encoder, prepare_image(samples[0], cfg), cfg.decode)
    predictions = []
    for sample in samples:
        result = generate_candidates(model, encoder, prepare_image(sample, cfg), cfg.decode)
        candidate = result["candidates"][0]
        predictions.append(
            {
                "image_id": sample.image_id,
                "image_path": sample.image_path,
                "prediction": candidate["text"],
                "output_tokens": candidate["output_tokens"],
                "output_chars": candidate["output_chars"],
                "ended_with_eos": candidate["ended_with_eos"],
                "generation_seconds": result["generation_seconds"],
            }
        )
    output = cfg.path(cfg.decode.output)
    write_jsonl(output, predictions)
    checkpoint = cfg.path(cfg.train.checkpoint) if cfg.train.checkpoint else None
    write_json(
        output.with_suffix(".metadata.json"),
        {
            "config": cfg.to_dict(),
            "seed": cfg.seed,
            "split": split,
            "dataset_split_hash": samples_hash(samples),
            "image_set_hash": image_set_hash(samples),
            "predictions_hash": file_hash(output),
            "environment": environment(Path(cfg.project_root)),
            "model_checkpoint": str(checkpoint) if checkpoint else cfg.model.name,
            "checkpoint_hash": bundle_hash(checkpoint) if checkpoint else None,
            "base_revision": getattr(model.config, "_commit_hash", None),
            "parameters": parameter_report(model),
            "runtime_seconds": time.perf_counter() - started,
            "timing_protocol": "one-image warmup; synchronized generation only; batch size one",
        },
    )
    return predictions
