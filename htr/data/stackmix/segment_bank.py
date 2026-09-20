"""Cache aligned segments with auditable, train-only provenance."""

import logging
from dataclasses import asdict

import numpy as np

from htr.config import Config
from htr.data.dataset import samples_hash
from htr.data.split import load_splits
from htr.data.stackmix.aligner import Aligner, AlignmentError
from htr.data.stackmix.ctc import crop_signature, load_aligner
from htr.utils.io import file_hash, read_json, write_json

logger = logging.getLogger(__name__)


def bank_signature(cfg: Config, samples: list) -> dict:
    return {
        "train_split_hash": samples_hash(samples),
        "crop": crop_signature(cfg),
        "ctc_hash": file_hash(cfg.path(cfg.stackmix.ctc_checkpoint)),
        "min_quality": cfg.stackmix.min_quality,
        "max_alignment_cer": cfg.stackmix.max_alignment_cer,
        "height": cfg.stackmix.height,
    }


def verify_bank(bank: dict, train_samples: list) -> None:
    allowed = {s.image_id: s for s in train_samples}
    if bank["signature"]["train_split_hash"] != samples_hash(train_samples):
        raise ValueError("StackMix bank train split hash differs")
    if bank["sources"] != {key: asdict(value) for key, value in allowed.items()}:
        raise ValueError("StackMix bank source records differ from train manifest")
    for segment in bank["segments"]:
        if segment["source_id"] not in allowed:
            raise ValueError("StackMix bank contains a non-training source")
        source = allowed[segment["source_id"]]
        offset = segment["offset"]
        if (
            not 0 <= offset < len(source.transcription)
            or source.transcription[offset] != segment["text"]
        ):
            raise ValueError("StackMix segment label differs from its training source")
        if segment["style"] != source.group:
            raise ValueError("StackMix segment style differs from its training source")


def build_bank(cfg: Config, aligner: Aligner | None = None) -> dict:
    """An injected aligner is for tests; production uses a provenance-checked CTC model."""
    from htr.training.engine import prepare_image

    samples = load_splits(cfg)["train"]
    signature = bank_signature(cfg, samples)
    directory = cfg.path(cfg.stackmix.bank_dir)
    index = directory / "bank.json"
    if index.exists():
        bank = read_json(index)
        verify_bank(bank, samples)
        if bank["signature"] != signature:
            raise ValueError("Cached bank settings changed; use a new bank_dir")
        return bank
    aligner = aligner or load_aligner(cfg, samples)
    directory.mkdir(parents=True, exist_ok=True)
    segments, rejected = [], []
    for sample in samples:
        image = prepare_image(sample, cfg)
        try:
            alignment = aligner.align(image, sample.transcription)
        except AlignmentError as error:
            rejected.append({"image_id": sample.image_id, "reason": str(error)})
            continue
        if "".join(s.text for s in alignment) != sample.transcription:
            raise ValueError("Aligner changed transcription; refusing segment extraction")
        for offset, segment in enumerate(alignment):
            if segment.text.isspace() or segment.quality < cfg.stackmix.min_quality:
                continue
            if not 0 <= segment.left < segment.right <= image.width:
                raise ValueError("Invalid alignment bounds")
            crop = image.crop((segment.left, 0, segment.right, image.height))
            ink_fraction = float((np.asarray(crop.convert("L")) < cfg.crop.threshold).mean())
            if ink_fraction < 0.001:
                continue
            relative = f"segments/{sample.image_id}_{offset:04d}.png"
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(path)
            segments.append(
                {
                    **asdict(segment),
                    "source_id": sample.image_id,
                    "offset": offset,
                    "style": sample.group,
                    "image_path": relative,
                    "image_hash": file_hash(path),
                    "ink_fraction": ink_fraction,
                }
            )
    if not segments:
        raise ValueError(
            "No segments passed quality filtering; improve/train the CTC recognizer and inspect its errors"
        )
    bank = {
        "algorithm": "train-only CTC Viterbi character StackMix-style",
        "signature": signature,
        "sources": {s.image_id: asdict(s) for s in samples},
        "segments": segments,
        "rejected_lines": rejected,
    }
    verify_bank(bank, samples)
    write_json(index, bank)
    logger.info("StackMix bank: %d segments, %d rejected lines", len(segments), len(rejected))
    return bank


def load_bank(cfg: Config, samples: list) -> dict:
    directory = cfg.path(cfg.stackmix.bank_dir)
    bank = read_json(directory / "bank.json")
    verify_bank(bank, samples)
    if bank["signature"] != bank_signature(cfg, samples):
        raise ValueError("Bank config/provenance differs")
    for segment in bank["segments"]:
        path = (directory / segment["image_path"]).resolve()
        if not path.is_relative_to(directory) or file_hash(path) != segment["image_hash"]:
            raise ValueError("Corrupted or out-of-directory bank segment")
    return bank
