"""Offline, fixed SFT N-best generation over ORIGINAL training samples only."""

from dataclasses import asdict, replace

from htr.config import Config
from htr.data.dataset import samples_hash
from htr.data.split import load_splits
from htr.decoding.beam import generate_candidates
from htr.models.qwen import QwenEncoder
from htr.training.checkpoint import bundle_hash, load_bundle
from htr.training.engine import prepare_image
from htr.utils.io import write_json, write_jsonl
from htr.utils.seed import seed_everything


def generate_nbest(cfg: Config) -> list[dict]:
    seed_everything(cfg.seed, cfg.deterministic)
    if not cfg.train.checkpoint:
        raise ValueError("N-best generation requires an SFT checkpoint")
    samples = load_splits(cfg)["train"]
    source = cfg.path(cfg.train.checkpoint)
    model, processor, bundle = load_bundle(cfg, source, "inference")
    if bundle["train_split_hash"] != samples_hash(samples) or bundle["stage"] != "sft":
        raise ValueError("N-best source must be an SFT bundle from this training split")
    encoder = QwenEncoder(processor, cfg.model)
    decode = replace(
        cfg.decode, num_beams=cfg.mwer.num_beams, num_return_sequences=cfg.mwer.num_return_sequences
    )
    rows = []
    for sample in samples:
        generated = generate_candidates(model, encoder, prepare_image(sample, cfg), decode)
        candidates, seen = [], set()
        for candidate in generated["candidates"]:
            key = tuple(candidate["token_ids"])
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)
        if len(candidates) < 2:
            raise ValueError(
                f"Fewer than two distinct token hypotheses for {sample.image_id}; increase beams"
            )
        rows.append(
            {
                "image_id": sample.image_id,
                "ground_truth": sample.transcription,
                "candidates": candidates,
                "generation_seconds": generated["generation_seconds"],
            }
        )
    output = cfg.path(cfg.mwer.nbest_path)
    if output.exists():
        raise ValueError("N-best cache exists; choose a new mwer.nbest_path to preserve provenance")
    write_jsonl(output, rows)
    from htr.utils.io import file_hash

    write_json(
        output.with_suffix(".metadata.json"),
        {
            "train_split_hash": samples_hash(samples),
            "checkpoint_hash": bundle_hash(source),
            "checkpoint": str(source),
            "seed": cfg.seed,
            "decode": asdict(decode),
            "model": asdict(cfg.model),
            "crop": asdict(cfg.crop),
            "role": "original_train_only",
            "nbest_hash": file_hash(output),
        },
    )
    return rows
