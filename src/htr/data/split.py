"""Leakage checks and portable, immutable split manifests."""

import random
from dataclasses import asdict

from htr.config import Config
from htr.data.dataset import Sample, image_path, read_samples, samples_hash
from htr.utils.io import file_hash, read_json, write_json


def assert_disjoint(train: list[Sample], validation: list[Sample]) -> None:
    for field in ("image_id", "image_path", "image_hash", "group"):
        left = {getattr(x, field) for x in train if getattr(x, field)}
        right = {getattr(x, field) for x in validation if getattr(x, field)}
        overlap = left & right
        if overlap:
            raise ValueError(f"Dataset leakage: overlapping {field}: {sorted(overlap)[:3]}")


def split_samples(samples: list[Sample], fraction: float, seed: int) -> tuple[list, list]:
    """Split entire groups; fraction refers to groups if writer/document IDs exist."""
    groups = sorted({sample.group or sample.image_id for sample in samples})
    if len(groups) < 2:
        raise ValueError("At least two groups are required for train/validation splitting")
    random.Random(seed).shuffle(groups)
    count = min(len(groups) - 1, max(1, round(len(groups) * fraction)))
    heldout = set(groups[:count])
    train = [s for s in samples if (s.group or s.image_id) not in heldout]
    validation = [s for s in samples if (s.group or s.image_id) in heldout]
    assert_disjoint(train, validation)
    return train, validation


def prepare_splits(cfg: Config) -> dict:
    root = cfg.path(cfg.data.image_root)
    source = read_samples(cfg.path(cfg.data.source_csv), root, cfg.data.group_column)
    if cfg.data.validation_csv:
        train = source
        validation = read_samples(cfg.path(cfg.data.validation_csv), root, cfg.data.group_column)
    else:
        train, validation = split_samples(source, cfg.data.validation_fraction, cfg.seed)
    assert_disjoint(train, validation)
    test = []
    if cfg.data.test_csv:
        test = read_samples(cfg.path(cfg.data.test_csv), root, cfg.data.group_column, False)
        assert_disjoint(train, test)
        assert_disjoint(validation, test)
    manifest = {
        "schema_version": 1,
        "seed": cfg.seed,
        "group_column": cfg.data.group_column,
        "splits": {
            name: [asdict(s) for s in rows]
            for name, rows in (("train", train), ("validation", validation), ("test", test))
        },
        "hashes": {
            "train": samples_hash(train),
            "validation": samples_hash(validation),
            "test": samples_hash(test),
        },
    }
    target = cfg.path(cfg.data.split_dir) / "manifest.json"
    if target.exists() and read_json(target) != manifest:
        raise ValueError("Split manifest differs; use a new split_dir to preserve provenance")
    write_json(target, manifest)
    return manifest


def load_splits(cfg: Config) -> dict[str, list[Sample]]:
    manifest = read_json(cfg.path(cfg.data.split_dir) / "manifest.json")
    result = {name: [Sample(**row) for row in rows] for name, rows in manifest["splits"].items()}
    for name, rows in result.items():
        if samples_hash(rows) != manifest["hashes"][name]:
            raise ValueError(f"Corrupted {name} manifest")
        for sample in rows:
            if (
                file_hash(image_path(cfg.path(cfg.data.image_root), sample.image_path))
                != sample.image_hash
            ):
                raise ValueError(f"Image changed since splitting: {sample.image_path}")
    assert_disjoint(result["train"], result["validation"])
    assert_disjoint(result["train"], result.get("test", []))
    assert_disjoint(result["validation"], result.get("test", []))
    return result
