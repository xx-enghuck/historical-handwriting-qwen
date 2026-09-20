"""Deterministic synthesis from train-only words and aligned character images."""

import math
import random
from dataclasses import asdict

from PIL import Image

from htr.config import Config
from htr.data.dataset import Sample, samples_hash
from htr.data.split import load_splits
from htr.data.stackmix.segment_bank import load_bank
from htr.utils.io import file_hash, read_json, write_json


def synthesize(cfg: Config) -> dict:
    train = load_splits(cfg)["train"]
    bank = load_bank(cfg, train)
    directory = cfg.path(cfg.stackmix.synthetic_dir)
    root = cfg.path(cfg.data.image_root)
    if not directory.is_relative_to(root):
        raise ValueError(
            "stackmix.synthetic_dir must be under data.image_root for portable image metadata"
        )
    expected = {
        "train_split_hash": samples_hash(train),
        "seed": cfg.seed,
        "settings": asdict(cfg.stackmix),
        "bank_hash": file_hash(cfg.path(cfg.stackmix.bank_dir) / "bank.json"),
    }
    manifest_path = directory / "synthetic.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["signature"] != expected:
            raise ValueError("Synthetic cache differs; choose a new synthetic_dir")
        return manifest
    rng = random.Random(cfg.seed)
    styles = sorted({s["style"] for s in bank["segments"] if s["style"] is not None})
    if cfg.stackmix.style_consistent and not styles:
        raise ValueError("Style-consistent augmentation requires group metadata")
    pools = {}
    for style in styles if cfg.stackmix.style_consistent else [None]:
        chars = {}
        for segment in bank["segments"]:
            if style is None or segment["style"] == style:
                chars.setdefault(segment["text"], []).append(segment)
        # Both words and their sampling frequency come exclusively from train.
        words = [
            word
            for sample in train
            if style is None or sample.group == style
            for word in sample.transcription.split()
            if all(char in chars for char in word)
        ]
        if words:
            pools[style] = (chars, words)
    if not pools:
        raise ValueError(
            "Bank does not cover any complete training word; improve alignment coverage"
        )
    records = []
    count = math.floor(len(train) * cfg.stackmix.synthetic_ratio)
    if count < 1:
        raise ValueError("synthetic_ratio produces no synthetic samples")
    for index in range(count):
        style = rng.choice(list(pools))
        chars, words = pools[style]
        text = " ".join(
            rng.choices(words, k=rng.randint(cfg.stackmix.min_words, cfg.stackmix.max_words))
        )
        pieces, source_ids = [], []
        for char in text:
            if char.isspace():
                pieces.append(
                    Image.new("RGB", (cfg.stackmix.word_spacing, cfg.stackmix.height), "white")
                )
                continue
            segment = rng.choice(chars[char])
            with Image.open(cfg.path(cfg.stackmix.bank_dir) / segment["image_path"]) as image:
                width = max(1, round(image.width * cfg.stackmix.height / image.height))
                pieces.append(
                    image.convert("RGB").resize(
                        (width, cfg.stackmix.height), Image.Resampling.BILINEAR
                    )
                )
            source_ids.append(segment["source_id"])
        width = sum(p.width for p in pieces) + cfg.stackmix.spacing * max(0, len(pieces) - 1)
        canvas = Image.new("RGB", (width, cfg.stackmix.height), "white")
        x = 0
        for piece in pieces:
            canvas.paste(piece, (x, 0))
            x += piece.width + cfg.stackmix.spacing
        path = directory / f"synthetic_{index:06d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path)
        sample = Sample(
            f"synthetic_{index:06d}",
            path.relative_to(root).as_posix(),
            text,
            style,
            file_hash(path),
        )
        records.append({"sample": asdict(sample), "source_ids": sorted(set(source_ids))})
    manifest = {"signature": expected, "records": records}
    write_json(manifest_path, manifest)
    return manifest


def load_synthetic(cfg: Config, train: list[Sample]) -> list[Sample]:
    bank = load_bank(cfg, train)
    manifest = read_json(cfg.path(cfg.stackmix.synthetic_dir) / "synthetic.json")
    expected = {
        "train_split_hash": samples_hash(train),
        "seed": cfg.seed,
        "settings": asdict(cfg.stackmix),
        "bank_hash": file_hash(cfg.path(cfg.stackmix.bank_dir) / "bank.json"),
    }
    if manifest["signature"] != expected:
        raise ValueError("Synthetic data settings/provenance differ")
    allowed = set(bank["sources"])
    samples = []
    for row in manifest["records"]:
        if not set(row["source_ids"]) <= allowed:
            raise ValueError("Synthetic data contains a non-training source")
        sample = Sample(**row["sample"])
        allowed_words = {
            word
            for source in train
            if not cfg.stackmix.style_consistent or source.group == sample.group
            for word in source.transcription.split()
        }
        if not set(sample.transcription.split()) <= allowed_words:
            raise ValueError("Synthetic transcription contains non-training words")
        from htr.data.dataset import image_path

        if (
            file_hash(image_path(cfg.path(cfg.data.image_root), sample.image_path))
            != sample.image_hash
        ):
            raise ValueError("Synthetic image hash mismatch")
        samples.append(sample)
    if len(samples) != math.floor(len(train) * cfg.stackmix.synthetic_ratio):
        raise ValueError("Synthetic sample count differs from requested ratio")
    return samples
